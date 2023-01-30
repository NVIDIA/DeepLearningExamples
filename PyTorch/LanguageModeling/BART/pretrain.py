# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import glob
import logging
import os
from tabnanny import check
import time
import datetime
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from training_base import BaseTransformer, add_generic_args, generic_train
from bart.tokenization.tokenization_mbart import MBartTokenizer

from bart.configuration.configuration_bart import BartConfig
from bart.tokenization.tokenization_bart import BartTokenizer
from bart.modeling.modeling_bart import BartForConditionalGeneration

from utils.utils import (
    PretrainingSeq2SeqDataset,
    Seq2SeqDataset,
    assert_all_frozen,
    freeze_params,
    get_git_info,
    label_smoothed_nll_loss,
    lmap,
    pickle_save,
    save_git_info,
    save_json,
    use_task_specific_params,
    format_step
)
from utils.data_collator import DataCollatorForBART
from utils.gpu_affinity import set_affinity
from utils.distributed_utils import get_rank, get_device_count, get_world_size
import dllogger

import lddl.torch
from lddl.utils import get_all_parquets_under

logger = logging.getLogger(__name__)

class BartForConditionalGenerationWrapper(torch.nn.Module):
    def __init__(self, model, args):
        super(BartForConditionalGenerationWrapper, self).__init__()
        if args.fp16:
            model.half()
        elif args.bf16:
            model.bfloat16()

        model.train()

        self.module = model

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        outputs = self.module.forward(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, use_cache=False)

        return outputs

class PretrainingModule(BaseTransformer):
    mode = "pretraining"
    loss_names = ["loss"]

    def __init__(self, hparams, **kwargs):

        super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
        use_task_specific_params(self.model, "pretraining")
        save_git_info(self.hparams.output_dir)
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)

        self.dataset_kwargs: dict = dict(
            max_source_length=self.hparams.max_source_length,
            prefix=self.model.config.prefix or "",
        )
        self.n_obs = {
            "train": self.hparams.n_train if self.hparams.n_train >= 0 else None
        }

        #@todo should you freeze?
        if self.hparams.freeze_embeds:
            self.freeze_embeds()
        if self.hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())

        self.hparams.git_sha = get_git_info()["repo_sha"]
        self.num_workers = hparams.num_workers
        self.decoder_start_token_id = None  # default to config

        if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
            self.decoder_start_token_id = self.tokenizer.lang_code_to_id[hparams.tgt_lang]
            self.model.config.decoder_start_token_id = self.decoder_start_token_id

        self.collate_fn = DataCollatorForBART(
            tokenizer=self.tokenizer,
            mlm_probability=self.hparams.mlm_probability,
            permute_sentence_ratio=self.hparams.permute_sentence_ratio,
            decoder_start_token_id=self.model.config.decoder_start_token_id
        )

        self.dataset_class = (
            PretrainingSeq2SeqDataset
        )

        self.conig = self.model.config

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        except AttributeError:
            freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                freeze_params(d.embed_tokens)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)


    def _step(self, batch: dict) -> Tuple:
        pad_token_id = self.tokenizer.pad_token_id
        src_ids, src_mask, decoder_input_ids = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]
        tgt_ids = batch["labels"]

        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs[0]
        if self.hparams.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id) #@should you ignore unmasked tokens? Check!

            assert lm_logits.shape[-1] == self.config.vocab_size
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=pad_token_id
            )
        return (loss,), lm_logits

    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch) -> Dict:
        loss_tensors, logits = self._step(batch)
        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        # tokens per batch
        logs["ip_tpb"] = batch["input_ids"].numel()
        logs["op_tpb"] = batch["labels"].numel()
        logs["tpb"] = batch["input_ids"].ne(self.pad).sum() + batch["labels"].ne(self.pad).sum()
        logs["bs"] = batch["input_ids"].shape[0]
        logs["src_pad_tok"] = batch["input_ids"].eq(self.pad).sum()
        logs["src_pad_frac"] = batch["input_ids"].eq(self.pad).float().mean()
        # TODO(SS): make a wandb summary metric for this

        # self.log("train_loss_ddp_avg", loss_tensors[0], on_step=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)

        return {"loss": loss_tensors[0], "log": logs}

    # Can remove after pytorch lightning fix
    def training_epoch_end(self, outputs) -> None:
        return

    def save_metrics(self, latest_metrics, type_path) -> None:
        self.metrics[type_path].append(latest_metrics)
        save_json(self.metrics, self.metrics_save_path)

    def get_dataset(self, type_path, src_file,
                    shuffle_buffer_size=1000,
                    shuffle_buffer_warmup_factor=16,
                    max_shards_per_node=1048576) -> Seq2SeqDataset:

        lddl_dataset_kwargs = {
          'transform':lambda x:x,
          'local_rank': get_rank(),
          'shuffle_buffer_size': shuffle_buffer_size,
          'shuffle_buffer_warmup_factor': shuffle_buffer_warmup_factor,
          'base_seed': self.hparams.seed,
          'max_shards_per_node': max_shards_per_node
        }

        n_obs = self.n_obs[type_path]
        dataset = self.dataset_class(
            get_all_parquets_under(src_file),
            self.tokenizer,
            n_obs=n_obs,
            type_path=type_path,
            **self.dataset_kwargs, **lddl_dataset_kwargs,
        )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path, self.hparams.data_dir)

        dataloader_args = {"collate_fn":self.collate_fn}

        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.num_workers,
            sampler=None,
            pin_memory=True
        )

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        return dataloader

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        add_generic_args(parser, root_dir)
        parser.add_argument(
            "--max_source_length",
            default=1024,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument("--load_model_weights_only", action="store_true", help="Only load model weights, ignoring other ckpt states. useful at the start of phase2 training")
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--logger_name", type=str, choices=["default", "wandb", "wandb_shared"], default="default")
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--buffer_size", type=int, default=128, required=False, help="Buffer size for shuffling dataset")
        parser.add_argument(
            "--task", type=str, default="pretraining", required=False, help="# examples. -1 means use all."
        )
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument("--mlm_probability", type=float, default=0.3, required=False)
        parser.add_argument("--permute_sentence_ratio", type=float, default=1.0, required=False)
        parser.add_argument("--src_lang", type=str, default="", required=False)
        parser.add_argument("--tgt_lang", type=str, default="", required=False)

        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )
        parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', 0), help="local_rank for distributed training on gpus")
        parser.add_argument('--json-summary', type=str, default="results/dllogger.json",
                            help='If provided, the json summary will be written to'
                                 'the specified file.')
        return parser

def set_seed(args):
    random.seed(args.seed + get_rank())
    np.random.seed(args.seed + get_rank())
    torch.manual_seed(args.seed + get_rank())

def load_checkpoint(args, path, model, optimizer, scaler):
    checkpoint = torch.load(path, map_location=args.device)
    model.load_state_dict(checkpoint["model"])

    if not args.load_model_weights_only:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

def main(args, model=None) -> PretrainingModule:
    print(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.device = device

    # Set GPU affinity
    if args.affinity != 'disabled':
        affinity = set_affinity(
            get_rank(),
            get_device_count(),
            args.affinity
        )
        logger.warning(f'{get_rank()}: thread affinity: {affinity}')

    # Set seed
    set_seed(args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if get_rank() in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process global rank: %s, device: %s, distributed training: %s, 16-bits training: %s",
        get_rank(),
        device,
        bool(get_rank() != -1),
        (args.fp16 or args.bf16),
    )

    if model is None:
        if "pretraining" in args.task:
            ### Define BART model
            # Config from "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large-cnn/config.json
            # Vocab modified to 50265 to be consistent with facebook/bart-large default
            config = BartConfig(**json.load(open(args.config_path, "r")))
            if args.fp16:
                config.dtype = torch.float16
            elif args.bf16:
                config.dtype = torch.bfloat16
            else:
                config.dtype = None
            config.pre_ln = args.pre_ln

            model = BartForConditionalGeneration(config=config)
            tokenizer = BartTokenizer.from_pretrained(
                'facebook/bart-large')  # Downloads vocab and merges file automatically
            trainer: PretrainingModule = PretrainingModule(args, model=model, config=config, tokenizer=tokenizer)
        else:
            raise ValueError("Only pretraining supported!")

    dataset = Path(args.data_dir).name
    trainer.model.to(device)

    # Set up optimizer and scheduler
    optimizer, scheduler = trainer.configure_optimizers()
    optimizer = optimizer[0]
    scheduler = scheduler[0]['scheduler']
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "_step*.ckpt"), recursive=True),
                key=lambda x:int(x.split("step")[1].split(".")[0])))

    step = 0
    if args.resume_from_checkpoint:
        if ".ckpt" in args.resume_from_checkpoint:
            checkpoint = args.resume_from_checkpoint
        else:
            if len(checkpoints) > 0: #No checkpoints available
                checkpoint = checkpoints[-1]
                args.resume_from_checkpoint = checkpoint
            else:
                args.resume_from_checkpoint = None
                checkpoint = None

        if checkpoint is None:
            logger.info("Pretraining from scratch")

        else:
            logger.info("Loading BART model checkpoint using %s", checkpoint)
            checkpoint_suffix = checkpoint.split("step")[-1].split(".")[0]
            step = int(checkpoint_suffix) + 1
            load_checkpoint(args, checkpoint, trainer.model, optimizer, scaler)

    if args.load_model_weights_only:
        args.resume_from_checkpoint = None
        step = 0

    if args.fp16 and args.allreduce_post_accumulation_half_precision:
        trainer.model.half()
    if args.bf16 and args.allreduce_post_accumulation_half_precision:
        trainer.model.bfloat16()

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        trainer.model = torch.nn.parallel.DistributedDataParallel(
            trainer.model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    generic_train(args, trainer, optimizer, scheduler, scaler, checkpoints, step)

    pickle_save(trainer.hparams, trainer.output_dir / "hparams.pkl")
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = PretrainingModule.add_model_specific_args(parser, os.getcwd())

    args = parser.parse_args()

    if get_rank() == 0:
        dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                                           filename=args.json_summary)])

    main(args)

    dllogger.flush()

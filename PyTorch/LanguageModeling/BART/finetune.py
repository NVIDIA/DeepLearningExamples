# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from training_base import BaseTransformer, add_generic_args, generic_test, generic_train
from bart.tokenization.tokenization_mbart import MBartTokenizer
from bart.modeling.modeling_t5 import T5ForConditionalGeneration

from bart.configuration.configuration_bart import BartConfig
from bart.tokenization.tokenization_bart import BartTokenizer
from bart.modeling.modeling_bart import BartForConditionalGeneration, shift_tokens_right

from utils.utils import (
    ROUGE_KEYS,
    LegacySeq2SeqDataset,
    Seq2SeqDataset,
    assert_all_frozen,
    calculate_bleu,
    calculate_rouge,
    flatten_list,
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
from utils.gpu_affinity import set_affinity
from utils.distributed_utils import get_rank, get_device_count, get_world_size
import dllogger
import time

logger = logging.getLogger(__name__)


class SummarizationModule(BaseTransformer):
    mode = "summarization"
    loss_names = ["loss"]
    metric_names = ROUGE_KEYS
    default_val_metric = "rouge2"

    def __init__(self, hparams, **kwargs):
        if hparams.sortish_sampler and hparams.gpus > 1:
            hparams.replace_sampler_ddp = False
        elif hparams.max_tokens_per_batch is not None:
            if hparams.gpus > 1:
                raise NotImplementedError("Dynamic Batch size does not work for multi-gpu training")
            if hparams.sortish_sampler:
                raise ValueError("--sortish_sampler and --max_tokens_per_batch may not be used simultaneously")

        super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
        use_task_specific_params(self.model, "summarization")
        save_git_info(self.hparams.output_dir)
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            prefix=self.model.config.prefix or "",
        )
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
        }
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"
        if self.hparams.freeze_embeds:
            self.freeze_embeds()
        if self.hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())

        self.hparams.git_sha = get_git_info()["repo_sha"]
        self.num_workers = hparams.num_workers
        self.sync_dist = True if hparams.gpus > 1 else False
        self.decoder_start_token_id = None  # default to config
        if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
            self.decoder_start_token_id = self.tokenizer.lang_code_to_id[hparams.tgt_lang]
            self.model.config.decoder_start_token_id = self.decoder_start_token_id
        self.dataset_class = (
            LegacySeq2SeqDataset
        )
        self.eval_beams = self.model.config.num_beams if self.hparams.eval_beams is None else self.hparams.eval_beams
        assert self.eval_beams >= 0, f"got self.eval_beams={self.eval_beams}. Need an integer >= 0"
        if self.hparams.eval_max_gen_length is not None:
            self.eval_max_length = self.hparams.eval_max_gen_length
        else:
            self.eval_max_length = self.model.config.max_length
        self.val_metric = self.default_val_metric if self.hparams.val_metric is None else self.hparams.val_metric

        self.config = self.model.config

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
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["labels"]
        if isinstance(self.model, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(tgt_ids)
        else:
            decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id, self.config.decoder_start_token_id)

        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs[0]
        if self.hparams.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

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

    def validation_step(self, batch) -> Dict:
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]
        generative_metrics = {
            k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names + ["gen_time", "gen_len"]
        }
        metric_val = (
            generative_metrics[self.val_metric] if self.val_metric in generative_metrics else losses[self.val_metric]
        )
        metric_tensor: torch.FloatTensor = torch.tensor(metric_val).type_as(loss)
        generative_metrics.update({k: v.item() for k, v in losses.items()})
        losses.update(generative_metrics)
        all_metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        all_metrics["step_count"] = self.step_count
        self.save_metrics(all_metrics, prefix)  # writes to self.metrics_save_path
        preds = flatten_list([x["preds"] for x in outputs])

        self.log(f"{prefix}_{self.val_metric}", metric_tensor, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        return {
            "log": all_metrics,
            "preds": preds,
            f"{prefix}_loss": loss,
            f"{prefix}_{self.val_metric}": metric_tensor,
        }

    def save_metrics(self, latest_metrics, type_path) -> None:
        self.metrics[type_path].append(latest_metrics)
        save_json(self.metrics, self.metrics_save_path)

    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_rouge(preds, target)

    def _generative_step(self, batch: dict) -> dict:
        t0 = time.time()
        loss_tensors, logits = self._step(batch)

        if self.eval_beams == 0:
            generated_ids = torch.argmax(logits.detach(), axis=-1)
        else:
            generated_ids = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=True,
                decoder_start_token_id=self.decoder_start_token_id,
                num_beams=self.eval_beams,
                max_length=self.eval_max_length,
                num_beam_groups=1, output_scores=False,
                return_dict_in_generate=False,
                encoder_no_repeat_ngram_size=0,
                diversity_penalty=0.0
            )
        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(batch["labels"])
        base_metrics = {name: loss.detach() for name, loss in zip(self.loss_names, loss_tensors)}
        rouge: Dict = self.calc_generative_metrics(preds, target)
        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=target, **rouge)
        return base_metrics

    def test_step(self, batch):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def get_dataset(self, type_path) -> Seq2SeqDataset:
        n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        dataset = self.dataset_class(
            self.tokenizer,
            type_path=type_path,
            n_obs=n_obs,
            max_target_length=max_target_length,
            **self.dataset_kwargs,
        )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path)

        if self.hparams.sortish_sampler and type_path != "test":
            sampler = dataset.make_sortish_sampler(batch_size, distributed=self.hparams.gpus > 1)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=False,
                num_workers=self.num_workers,
                sampler=sampler,
                pin_memory=True,
            )

        elif self.hparams.max_tokens_per_batch is not None and type_path != "test":
            batch_sampler = dataset.make_dynamic_sampler(
                self.hparams.max_tokens_per_batch, distributed=self.hparams.gpus > 1
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=dataset.collate_fn,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=shuffle,
                num_workers=self.num_workers,
                sampler=None,
                pin_memory=True,
            )

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

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
        parser.add_argument(
            "--max_target_length",
            default=56,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_max_target_length",
            default=142,  # these defaults are optimized for CNNDM. For xsum, see README.md.
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=142,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--sortish_sampler", action="store_true", default=False)
        parser.add_argument("--max_tokens_per_batch", type=int, default=None)
        parser.add_argument("--logger_name", type=str, choices=["default", "wandb", "wandb_shared"], default="default")
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_val", type=int, default=500, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument(
            "--task", type=str, default="summarization", required=False, help="# examples. -1 means use all."
        )
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument("--src_lang", type=str, default="", required=False)
        parser.add_argument("--tgt_lang", type=str, default="", required=False)
        parser.add_argument("--eval_beams", type=int, default=None, required=False, help="# beams to use. 0 corresponds to not using beam search.")
        parser.add_argument(
            "--val_metric", type=str, default=None, required=False, choices=["bleu", "rouge2", "loss", None]
        )
        parser.add_argument("--eval_max_gen_length", type=int, default=None, help="never generate more than n tokens")
        parser.add_argument("--save_top_k", type=int, default=1, required=False, help="How many checkpoints to save")
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )
        parser.add_argument('--json-summary', type=str, default="results/dllogger.json",
                            help='If provided, the json summary will be written to'
                                 'the specified file.')
        parser.add_argument('--distill', type=str, default=None, help="string indicating distillation to perform, only sft supported", choices=["sft", None])
        parser.add_argument('--layers', type=str, default=None, help="string indicating which layers to distill for SFT, split by '-' (ex. 0-6-11)")
        parser.add_argument('--do_encoder', action="store_true", default=False, help="if true distills the encoder")
        parser.add_argument('--do_decoder', action="store_true", default=False, help="if true distills the decoder")
        parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', 0), help="local_rank for distributed training on gpus")
        parser.add_argument("--gpus", type=int, default=1, help="number of gpus to train on applied per node")
        parser.add_argument("--load_model_weights_only", action="store_true", help="Only load model weights, ignoring other ckpt states. useful at the start of phase2 training")

        return parser


class TranslationModule(SummarizationModule):
    mode = "translation"
    loss_names = ["loss"]
    metric_names = ["bleu"]
    default_val_metric = "bleu"

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.dataset_kwargs["src_lang"] = hparams.src_lang
        self.dataset_kwargs["tgt_lang"] = hparams.tgt_lang

    def calc_generative_metrics(self, preds, target) -> dict:
        return calculate_bleu(preds, target)

def set_seed(args):
    random.seed(args.seed + get_rank())
    np.random.seed(args.seed + get_rank())
    torch.manual_seed(args.seed + get_rank())

def save_final_checkpoint(args, model):
    output_filename = os.path.join(args.output_dir, "final_step.ckpt")

    if get_rank() == 0:
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(),
                   output_filename)

def load_checkpoint(args, path, model, optimizer, scaler):
    checkpoint = torch.load(path, map_location=args.device)
    model.load_state_dict(checkpoint["model"])

    if not args.load_model_weights_only:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

def distill(layers, pick_layers):
    sft_layers = nn.ModuleList()
    delete_layers = []
    for i in range(len(layers)):
        if i in pick_layers:
            sft_layers.append(layers[i])
        else:
            delete_layers.append(i)

    # delete unnecessary layers
    for i in range(len(delete_layers)):
        del layers[delete_layers[i] - i]

    return sft_layers

def distill_sft(model):
    pick_layers = [int(s) for s in args.layers.split('-')]

    # if distilling encoder
    if args.do_encoder:
        layers = model.model.encoder.layers
        sft_layers = distill(layers, pick_layers)
        model.model.encoder.layers = sft_layers

    # if distilling decoder
    if args.do_decoder:
        layers = model.model.decoder.layers
        sft_layers = distill(layers, pick_layers)
        model.model.decoder.layers = sft_layers

    return model

def main(args, model=None) -> SummarizationModule:
    print(args)
    Path(args.output_dir).mkdir(exist_ok=True)

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

    checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "_step*.ckpt"), recursive=True),
                        key=lambda x:int(x.split("step")[1].split(".")[0])))

    if model is None:
        if "summarization" in args.task:
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

            if args.distill: # if distilling, start from finetuned checkpoint
                if Path(args.data_dir).name == "cnn_dm":
                    checkpoint = 'facebook/bart-large-cnn'
                else:
                    checkpoint = 'facebook/bart-large-xsum'
            else:
                checkpoint = 'facebook/bart-large' #Start from pretrained checkpoint otherwise

            if args.resume_from_checkpoint:
                print("Resuming from checkpoint, make sure checkpoint is finetuned for best results")
                if ".ckpt" in args.resume_from_checkpoint:
                    checkpoint = args.resume_from_checkpoint
                    if args.distill: # set resume from checkpoint to None (state dict is different)
                        args.resume_from_checkpoint = None
                    model = BartForConditionalGeneration(config=config)
                else:
                    if len(checkpoints) > 0: #No checkpoints available
                        checkpoint = checkpoints[-1]
                        args.resume_from_checkpoint = checkpoint
                        model = BartForConditionalGeneration(config=config)
                    else:
                        args.resume_from_checkpoint = None
                        print("No valid checkpoint to resume from. Using ", checkpoint)
                        model = BartForConditionalGeneration.from_pretrained(checkpoint, config=config)

            else:
                model = BartForConditionalGeneration.from_pretrained(checkpoint, config=config)

            print("Loading BART model checkpoint using ", checkpoint)

            if args.distill == "sft":
                model = distill_sft(model)

            tokenizer = BartTokenizer.from_pretrained(
                'facebook/bart-large')  # Downloads vocab and merges file automatically
            trainer: SummarizationModule = SummarizationModule(args, model=model, config=config, tokenizer=tokenizer)
        else:
            raise ValueError("Translation not supported at this time")
            model: SummarizationModule = TranslationModule(args)
    dataset = Path(args.data_dir).name
    trainer.model.to(device)

    # Set up optimizer and scheduler
    optimizer, scheduler = trainer.configure_optimizers()
    optimizer = optimizer[0]
    scheduler = scheduler[0]['scheduler']
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    step = 0
    if args.resume_from_checkpoint:
            logger.info("Loading BART model checkpoint using %s", checkpoint)
            checkpoint_suffix = checkpoint.split("step")[-1].split(".")[0]
            step = int(checkpoint_suffix) + 1
            load_checkpoint(args, checkpoint, trainer.model, optimizer, scaler)

    if args.distill or args.load_model_weights_only:
        args.resume_from_checkpoint = None
        step = 0

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        trainer.model = torch.nn.parallel.DistributedDataParallel(
            trainer.model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    generic_train(args, trainer, optimizer, scheduler, scaler, checkpoints, step)

    pickle_save(trainer.hparams, trainer.output_dir / "hparams.pkl")
    save_final_checkpoint(args, trainer.model)

    if args.do_predict:
        # Testing from a checkpoint
        generic_test(args, trainer)
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = SummarizationModule.add_model_specific_args(parser, os.getcwd())

    args = parser.parse_args()

    if get_rank() == 0:
        dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                                           filename=args.json_summary),
                                dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE, step_format=format_step)])

    main(args)

    dllogger.flush()

    torch.distributed.barrier()

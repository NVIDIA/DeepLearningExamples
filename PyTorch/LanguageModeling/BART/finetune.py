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

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from lightning_base import BaseTransformer, add_generic_args, generic_train
from bart.tokenization.tokenization_mbart import MBartTokenizer
from bart.modeling.modeling_t5 import T5ForConditionalGeneration

from bart.configuration.configuration_bart import BartConfig
from bart.tokenization.tokenization_bart import BartTokenizer
from bart.modeling.modeling_bart import BartForConditionalGeneration, shift_tokens_right

from utils.callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback, CheckpointEveryNSteps
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
    format_step,
)
from utils.distributed_utils import get_rank
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
            decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id, self.model.config.decoder_start_token_id)

        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs[0]
        if self.hparams.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

            assert lm_logits.shape[-1] == self.model.config.vocab_size
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

    def training_step(self, batch, batch_idx) -> Dict:
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

        self.log("train_loss_ddp_avg", loss_tensors[0], on_step=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        return {"loss": loss_tensors[0], "log": logs, "progress_bar":{"global_step":self.global_step}}

    # Can remove after pytorch lightning fix
    def training_epoch_end(self, outputs) -> None:
        return

    def validation_step(self, batch, batch_idx) -> Dict:
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

    def test_step(self, batch, batch_idx):
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
                # shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                # batch_size=None,
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

    if model is None:
        if "summarization" in args.task:
            ### Define BART model
            # Config from "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large-cnn/config.json
            # Vocab modified to 50265 to be consistent with facebook/bart-large default
            config = BartConfig(**json.load(open(args.config_path, "r")))
            config.fp16 = args.fp16

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
                else:
                    checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True)))
                    if len(checkpoints) > 0: #No checkpoints available
                        checkpoint = checkpoints[-1]
                        args.resume_from_checkpoint = checkpoint
                    else:
                        args.resume_from_checkpoint = None
                        print("No valid checkpoint to resume from. Using ", checkpoint)

            print("Loading BART model checkpoint using ", checkpoint)
            model = BartForConditionalGeneration.from_pretrained(checkpoint, config=config)

            if args.distill == "sft":
                model = distill_sft(model)

            tokenizer = BartTokenizer.from_pretrained(
                'facebook/bart-large')  # Downloads vocab and merges file automatically
            model: SummarizationModule = SummarizationModule(args, model=model, config=config, tokenizer=tokenizer)
        else:
            raise ValueError("Translation not supported at this time")
            model: SummarizationModule = TranslationModule(args)
    dataset = Path(args.data_dir).name
    if (
        args.logger_name == "default"
        or args.fast_dev_run
        or str(args.output_dir).startswith("/tmp")
        or str(args.output_dir).startswith("/var")
    ):
        logger = True  # don't pollute wandb logs unnecessarily
    elif args.logger_name == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        project = os.environ.get("WANDB_PROJECT", dataset)
        logger = WandbLogger(name=model.output_dir.name, project=project)

    elif args.logger_name == "wandb_shared":
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(name=model.output_dir.name, project=f"hf_{dataset}")

    # if args.early_stopping_patience >= 0:
    #     extra_callbacks = [get_early_stopping_callback(f"val_{model.val_metric}", args.early_stopping_patience)]
    # else:
    #     extra_callbacks = []
    extra_callbacks = [CheckpointEveryNSteps(args.output_dir, args.max_steps-1)]

    lower_is_better = args.val_metric == "loss"

    log_callback = Seq2SeqLoggingCallback()

    trainer: pl.Trainer = generic_train(
        model,
        args,
        logging_callback=log_callback,
        checkpoint_callback=get_checkpoint_callback(
            args.output_dir, model.val_metric, args.save_top_k, lower_is_better
        ),
        extra_callbacks=extra_callbacks,
        logger=logger,
    )

    if get_rank() == 0:
        dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                                           filename=args.json_summary),
                                dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE, step_format=format_step)])
        dllogger.metadata("avg_train_time", {"unit": "s"})
        dllogger.metadata("avg_train_throughput", {"unit": "tokens/s"})
        dllogger.log(step=tuple(), data={"avg_train_tokens":log_callback.tokens, "avg_train_time":log_callback.train_time, "total_train_epochs":log_callback.epochs, "avg_train_throughput":log_callback.tokens/log_callback.train_time})
        print("Avg Tokens = %d Avg Train Time = %.2f Total Epochs = %d"%(log_callback.tokens, log_callback.train_time, log_callback.epochs))
        print("Avg steps per sec = %d Avg Throughput (tok/s) = %d"%(log_callback.avg_steps_per_sec, log_callback.tokens/log_callback.train_time))
        dllogger.flush()

    pickle_save(model.hparams, model.output_dir / "hparams.pkl")
    if args.do_predict and not args.do_train:
        # Testing from a checkpoint
        trainer.test(model)
    elif args.do_predict and args.do_train:
        # test() without a model tests using the best checkpoint automatically
        model.hparams.test_checkpoint = ""
        checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True)))
        if checkpoints:
            model.hparams.test_checkpoint = checkpoints[-1]
            trainer.resume_from_checkpoint = checkpoints[-1]
        trainer.logger.log_hyperparams(model.hparams)
        trainer.test()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SummarizationModule.add_model_specific_args(parser, os.getcwd())

    args = parser.parse_args()

    main(args)


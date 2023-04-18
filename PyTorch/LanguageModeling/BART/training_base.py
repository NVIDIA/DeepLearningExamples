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
import logging
import os
from pathlib import Path
from typing import Any, Dict
import time

from bart.configuration.configuration_bart import BartConfig
from bart.tokenization.tokenization_bart import BartTokenizer
from bart.modeling.modeling_bart import *
from utils.optimization import (
    AdamW,
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from utils.gpu_affinity import set_affinity
from utils.distributed_utils import get_rank, get_device_count, get_world_size
from utils.utils import get_readable_time, Mean
from apex.optimizers import FusedAdam, FusedMixedPrecisionLamb
import dllogger
logger = logging.getLogger(__name__)



MODEL_MODES = {
    "question-answering": BartForQuestionAnswering,
    "pretraining": PretrainedBartModel,
    "token-classification": BartForSequenceClassification,
    "language-modeling": BartModel,
    "summarization": BartForConditionalGeneration,
    "translation": BartForConditionalGeneration,
}


# update this and the import above to support new schedulers from transformers.optimization
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    # '': get_constant_schedule,             # not supported for now
    # '': get_constant_schedule_with_warmup, # not supported for now
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"


class BaseTransformer():
    def __init__(
        self,
        hparams: argparse.Namespace,
        num_labels=None,
        mode="base",
        config=None,
        tokenizer=None,
        model=None,
        **config_kwargs
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()

        self.step_count = 0
        self.hparams = hparams
        self.output_dir = Path(self.hparams.output_dir)
        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None
        if config is None:
            self.config = AutoConfig.from_pretrained(
                self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
                **({"num_labels": num_labels} if num_labels is not None else {}),
                cache_dir=cache_dir,
                **config_kwargs,
            )
        else:
            self.config: BartConfig = config

        extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
        for p in extra_model_params:
            if getattr(self.hparams, p, None):
                assert hasattr(self.config, p), f"model config doesn't have a `{p}` attribute"
                setattr(self.config, p, getattr(self.hparams, p))

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
                cache_dir=cache_dir,
            )
        else:
            self.tokenizer: BartTokenizer = tokenizer
        # self.model_type = MODEL_MODES[mode]
        if model is None:
            self.model = self.model_type.from_pretrained(
                self.hparams.model_name_or_path,
                from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
                config=self.config,
                cache_dir=cache_dir,
            )
        else:
            self.model = model

    def __call__(self, input_ids, **kwargs):
        return self.forward(input_ids, **kwargs)

    def load_hf_checkpoint(self, *args, **kwargs):
        self.model = self.model_type.from_pretrained(*args, **kwargs)

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]

        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.hparams.lamb:
            optimizer_reduced_precision_type = self.config.dtype if self.hparams.allreduce_post_accumulation_half_precision else None
            optimizer = FusedMixedPrecisionLamb(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                eps=self.hparams.adam_epsilon,
                max_grad_norm=self.hparams.gradient_clip_val,
                reduced_precision_dtype=optimizer_reduced_precision_type)
        elif self.hparams.allreduce_post_accumulation_half_precision:
            raise ValueError("--allreduce_post_accumulation_half_precision is only supported on LAMB optimizer")
        elif self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False
            )
        else:
            optimizer = FusedAdam(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        return self.validation_end(outputs)

    @property
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        if self.hparams.max_steps:
            return self.hparams.max_steps
        else:
            assert self.hparams.max_epochs is not None
            num_devices = max(1, self.hparams.gpus * self.hparams.num_nodes)  # TODO: consider num_tpu_cores
            effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
            dataset_size = len(self.train_loader.dataset)
            return (dataset_size / effective_batch_size) * self.hparams.max_epochs

    def get_dataloader(self, type_path, batch_size, shuffle=False):
        raise NotImplementedError("You must implement this for your task")

    def train_dataloader(self):
        return self.get_dataloader("train", self.hparams.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.hparams.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.hparams.eval_batch_size, shuffle=False)

    def _feature_file(self, mode):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
                str(self.hparams.max_seq_length),
            ),
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("best_tfmr")
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument("--config_path", default="config.json", type=str, help="Config File for Bart model")
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
        parser.add_argument(
            "--resume_from_checkpoint",
            type=str,
            help="""Path/URL of the checkpoint from which training is resumed. If there is no checkpoint file at
                  the path, start from scratch. If resuming from mid-epoch checkpoint, training will start from
                  the beginning of the next epoch.""",
        )
        parser.add_argument(
            "--encoder_layerdrop",
            type=float,
            help="Encoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--decoder_layerdrop",
            type=float,
            help="Decoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            help="Dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--attention_dropout",
            type=float,
            help="Attention dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
        parser.add_argument(
            "--lr_scheduler",
            default="linear",
            choices=arg_to_scheduler_choices,
            metavar=arg_to_scheduler_metavar,
            type=str,
            help="Learning rate scheduler",
        )
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--gradient_clip_val", default=0.5, type=float, help="The value at which to clip gradients.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--max_steps", default=10, type=int, help="Stop training after this number of steps.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--num_workers", default=4, type=int, help="kwarg passed to DataLoader")
        parser.add_argument("--min_num_train_epochs", dest="min_epochs", default=0, type=int)
        parser.add_argument("--train_batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int)
        parser.add_argument("--adafactor", action="store_true")
        parser.add_argument("--lamb", action="store_true")
        parser.add_argument('--affinity', type=str,
                             default='socket_unique_interleaved',
                             choices=['socket', 'single', 'single_unique',
                                      'socket_unique_interleaved',
                                      'socket_unique_continuous',
                                      'disabled'],
                             help='type of CPU affinity')
        parser.add_argument('--allreduce_post_accumulation_half_precision',
                            default=False,
                            action='store_true',
                            help="Whether to do fp16/bf16 allreduce post accumulation.")

def add_generic_args(parser, root_dir) -> None:
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision instead of 32-bit",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to use BFloat 16 mixed precision instead of 32-bit",
    )
    parser.add_argument("--n_tpu_cores", dest="tpu_cores", type=int)
    parser.add_argument("--max_grad_norm", dest="gradient_clip_val", default=1.0, type=float, help="Max gradient norm")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        dest="accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument("--log_freq", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_checkpoint_steps", type=int, default=100, required=False, help="How many checkpoints to save")
    parser.add_argument(
        "--profile",
        action="store_true",
    )
    parser.add_argument("--pre_ln",
        default=True,
        action='store_true',
        help="Whether to use Pre-LN architecture."
    )

def save_checkpoint(args, checkpoints, model, optimizer, scaler, step):
    output_filename = os.path.join(args.output_dir, "_step{}.ckpt".format(step))

    if get_rank() == 0:
        model_to_save = model
        while(hasattr(model_to_save, "module")):
            model_to_save = model_to_save.module
        torch.save({"model": model_to_save.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict()},
                   output_filename)

def train_one_step(args, trainer, optimizer, scheduler, features, local_step, scaler):
    if args.fp16:
        cast_dtype = torch.float16
    elif args.bf16:
        cast_dtype = torch.bfloat16
    else:
        cast_dtype = None
    with torch.cuda.amp.autocast(dtype=cast_dtype, enabled=(args.fp16 or args.bf16) and not args.allreduce_post_accumulation_half_precision):
        result = trainer.training_step(features)
        total_loss = result["loss"]
        loss = total_loss
        if args.accumulate_grad_batches > 1:
            total_loss = total_loss / args.accumulate_grad_batches

    if local_step % args.accumulate_grad_batches == 0:
        scaler.scale(total_loss).backward()

        if not args.lamb:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), args.gradient_clip_val)

        scheduler.step()  # Update learning rate schedule
        scaler.step(optimizer)
        optimizer.zero_grad()

        skip_optimizer_step = scaler._found_inf_per_device(optimizer)[args.device] if scaler.is_enabled() else 0
        result["log"]["skip_optimizer_step"] = int(skip_optimizer_step)
        scaler.update()
    else:
        with trainer.model.no_sync():
            scaler.scale(total_loss).backward()

    return loss, result["log"]

def generic_train(
    args,
    trainer,
    optimizer,
    scheduler,
    scaler,
    checkpoints,
    step,
    **extra_train_kwargs
):
    device = args.device

    # Set up dataset
    dataloader = trainer.train_dataloader()

    # Set up metrics
    metrics = {}
    metrics["avg_train_throughput"] = Mean(name="train_perf")
    metrics["total_loss"] = Mean(name="total_loss")

    trainer.model.train()
    local_step = 0
    train_start, start_step = time.time(), step - 1
    resume_step = step
    skipped_optimizer_steps = 0

    if get_rank() == 0:
        dllogger.metadata("avg_train_time", {"unit": "s"})
        dllogger.metadata("avg_train_throughput", {"unit": "seq/s"})

    while step <= args.max_steps:
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            local_step += 1
            torch.cuda.synchronize()
            iter_start = time.time()

            total_loss, logs = train_one_step(args, trainer, optimizer, scheduler, batch, local_step, scaler)
            torch.cuda.synchronize()
            train_perf = logs["bs"] * get_world_size() / (time.time() - iter_start)


            metrics["total_loss"].update(total_loss)
            metrics["avg_train_throughput"].update(train_perf)
            if local_step % args.accumulate_grad_batches == 0:
                static_optimizer_step = local_step // args.accumulate_grad_batches
                skipped_optimizer_steps += logs["skip_optimizer_step"]
                opt_step = static_optimizer_step - skipped_optimizer_steps + resume_step

                if args.log_freq > 0 and step != opt_step and (
                        step % args.log_freq == 0 or step == args.max_steps):
                    log_info_dict = {k:v.result() for k, v in metrics.items()}
                    if get_rank() == 0:
                        dllogger.log(step=(step,), data=log_info_dict, verbosity=0)
                    print(
                        'Step:{step:6d}, Loss:{total_loss:10.6f}, Perf:{train_perf:4.2f}, Loss Scaler: {loss_scale}, '
                        'Elapsed: {elapsed}, ETA: {eta}'.format(
                            step=step,
                            total_loss=total_loss,
                            train_perf=train_perf,
                            loss_scale=scaler.get_scale(),
                            elapsed=get_readable_time(time.time() - train_start),
                            eta=get_readable_time(
                                (time.time() - train_start) / (step - start_step) * (args.max_steps - step))),
                            flush=True
                    )

                    if step == args.max_steps:
                        final_metrics = {}
                        log_info_dict['avg_train_time'] = time.time() - train_start
                        for key, v in log_info_dict.items():
                            val = torch.tensor(v, device=device)
                            torch.distributed.all_reduce(val, op=torch.distributed.ReduceOp.SUM)
                            val /= get_world_size()
                            final_metrics[key] = val.item()
                        if get_rank() == 0:
                            dllogger.log(step=(), data=log_info_dict, verbosity=0)
                        logger.info('<FINAL STEP METRICS> Step:{step:6d}, Loss:{total_loss:10.6f}, Perf:{avg_train_throughput:4.2f}, Train time:{avg_train_time}s'.format(
                                        step=step, **final_metrics))

                    for key, m in metrics.items():
                        if key != 'avg_train_throughput':
                            m.reset()

                    if get_rank() == 0:
                        dllogger.flush()

                if args.save_checkpoint_steps > 0 and step != opt_step and \
                        ((step % args.save_checkpoint_steps == 0 and step > 0) or step == args.max_steps):
                    save_checkpoint(args, checkpoints, trainer.model, optimizer, scaler, step)
                    logger.info(f" ** Saved model checkpoint for step {step}")

                step = opt_step
            if step > args.max_steps:
                break

def generic_test(
    args,
    trainer
):
    device = args.device

    # Set up dataset
    dataloader = trainer.test_dataloader()

    metrics = {k: Mean(name=k) for k in trainer.loss_names + trainer.metric_names}

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        result_metric = trainer.test_step(batch)
        for k, v in result_metric:
            metrics[k].update(v)

    log_info_dict = {k:v.result() for k, v in metrics.items()}
    final_metrics = {}
    for key, v in log_info_dict.items():
        val = torch.tensor(v, device=device)
        torch.distributed.all_reduce(val, op=torch.distributed.ReduceOp.SUM)
        val /= get_world_size()
        final_metrics[key] = val.item()
    if get_rank() == 0:
        dllogger.log(step=(), data=log_info_dict, verbosity=0)
    print(final_metrics)

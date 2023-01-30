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

import logging
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ProgressBar
from pytorch_lightning.utilities import rank_zero_only
from utils.utils import save_json
from utils.distributed_utils import all_reduce_item, get_world_size
import time

def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

logger = logging.getLogger(__name__)


class Seq2SeqLoggingCallback(pl.Callback):

    @rank_zero_only
    def on_batch_end(self, trainer, pl_module):
        lrs = {f"lr_group_{i}": param["lr"] for i, param in enumerate(pl_module.trainer.optimizers[0].param_groups)}
        pl_module.logger.log_metrics(lrs)

    @rank_zero_only
    def _write_logs(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, type_path: str, save_generations=True
    ) -> None:
        logger.info(f"***** {type_path} results at step {trainer.global_step:05d} *****")
        metrics = trainer.callback_metrics
        trainer.logger.log_metrics({k: v for k, v in metrics.items() if k not in ["log", "progress_bar", "preds"]})
        # Log results
        od = Path(pl_module.hparams.output_dir)
        if type_path == "test":
            results_file = od / "test_results.txt"
            generations_file = od / "test_generations.txt"
        else:
            # this never gets hit. I prefer not to save intermediate generations, and results are in metrics.json
            # If people want this it will be easy enough to add back.
            results_file = od / f"{type_path}_results/{trainer.global_step:05d}.txt"
            generations_file = od / f"{type_path}_generations/{trainer.global_step:05d}.txt"
            results_file.parent.mkdir(exist_ok=True)
            generations_file.parent.mkdir(exist_ok=True)
        with open(results_file, "a+") as writer:
            for key in sorted(metrics):
                if key in ["log", "progress_bar", "preds"]:
                    continue
                val = metrics[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                msg = f"{key}: {val:.6f}\n"
                writer.write(msg)

        if not save_generations:
            return

        if "preds" in metrics:
            content = "\n".join(metrics["preds"])
            generations_file.open("w+").write(content)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.train_tob_list.append(outputs[0][0]["log"]["tpb"])
        self.train_time_epoch_list.append(time.time() - self.t0) #Measures ~time for forward + backward + optimizer_step

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.t0 = time.time()


    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        try:
            npars = pl_module.model.model.num_parameters()
        except AttributeError:
            npars = pl_module.model.num_parameters()

        n_trainable_pars = count_trainable_parameters(pl_module)
        # mp stands for million parameters
        trainer.logger.log_metrics({"n_params": npars, "mp": npars / 1e6, "grad_mp": n_trainable_pars / 1e6})
        self.train_time_epoch_list = []
        self.train_tob_list = []
        self.tokens = 0
        self.train_time = 0.0
        self.avg_steps_per_sec = 0.0
        self.epochs = 0
        try:
            self.sync_dist = pl_module.sync_dist
        except:
            self.sync_dist = get_world_size() > 1

    def process_stats(self, train_times, outputs, filter_p=0.8):
        index_list = np.argsort(train_times) #sort based on train_times

        best_n = int(len(outputs) * 0.8)
        train_time = 0.0
        unpadded_tokens = 0

        for i in index_list[:best_n]:
            train_time += train_times[i]
            unpadded_tokens += outputs[i]
        avg_steps_per_sec = train_time / best_n
        return train_time, unpadded_tokens, best_n, avg_steps_per_sec

    def on_train_epoch_end(self, trainer, pl_module, outputs):

        try:

            outputs = self.train_tob_list
            train_time, unpadded_tokens, train_batches, avg_steps_per_sec = self.process_stats(self.train_time_epoch_list, outputs)
            pl_module.log("train_throughput", unpadded_tokens/train_time, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
            all_reduce_tokens = all_reduce_item(unpadded_tokens, "sum")
            all_reduce_time = all_reduce_item(train_time, "mean")
            all_reduce_avg_steps_per_sec = all_reduce_item(avg_steps_per_sec, "mean")

            #Accumulate
            self.tokens = ((self.tokens * self.epochs) + all_reduce_tokens) / (self.epochs + 1)
            self.train_time = ((self.train_time * self.epochs) + all_reduce_time) / (self.epochs + 1)
            self.avg_steps_per_sec = ((self.avg_steps_per_sec * self.epochs) + all_reduce_avg_steps_per_sec) / (self.epochs + 1.0)
            self.epochs +=1

            #Reset
            self.train_time_epoch_list = []
            self.train_tob_list = []
        except ZeroDivisionError:
            print("Train time is reported as 0? It's possible training is already complete!")
            pass

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):

        if self.epochs < 1:
            outputs = self.train_tob_list
            train_time, unpadded_tokens, train_batches, avg_steps_per_sec = self.process_stats(self.train_time_epoch_list, outputs)
            pl_module.log("train_throughput", unpadded_tokens/train_time, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
            all_reduce_tokens = all_reduce_item(unpadded_tokens, "sum")
            all_reduce_time = all_reduce_item(train_time, "mean")
            all_reduce_avg_steps_per_sec = all_reduce_item(avg_steps_per_sec, "mean")

            #Accumulate
            self.tokens = ((self.tokens * self.epochs) + all_reduce_tokens) / (self.epochs + 1)
            self.train_time = ((self.train_time * self.epochs) + all_reduce_time) / (self.epochs + 1)
            self.avg_steps_per_sec = ((self.avg_steps_per_sec * self.epochs) + all_reduce_avg_steps_per_sec) / (self.epochs + 1.0)

def get_checkpoint_callback(output_dir, metric, save_top_k=1):
    """Saves the best model by validation ROUGE2 score."""
    monitor = f"val_{metric}"
    if metric == "rouge2":
        exp = "{val_avg_rouge2:.4f}-{step_count}"
    elif metric == "bleu":
        exp = "{val_avg_bleu:.4f}-{step_count}"
    elif metric == "loss":
        exp = "{loss:.4f}-{epoch}"
        monitor = metric
    else:
        raise NotImplementedError(
            f"seq2seq callbacks only support rouge2, bleu and loss, got {metric}, You can make your own by adding to this function."
        )

    checkpoint_callback = ModelCheckpoint(
        filename=os.path.join(output_dir, exp),
        monitor=monitor,
        mode="min" if "loss" in metric else "max",
        save_top_k=save_top_k,
        period=1,  # maybe save a checkpoint every time val is run, not just end of epoch.
    )
    return checkpoint_callback


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        output_dir,
        save_step_frequency,
        prefix="",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.output_dir = output_dir
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_epoch{epoch}_step{global_step}.ckpt"
            ckpt_path = os.path.join(self.output_dir, filename)
            trainer.save_checkpoint(ckpt_path)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):

        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if self.use_modelcheckpoint_filename:
            filename = trainer.checkpoint_callback.filename
        else:
            filename = f"{self.prefix}_epoch{epoch}_step{global_step}.ckpt"
        ckpt_path = os.path.join(self.output_dir, filename)
        trainer.save_checkpoint(ckpt_path)

def get_early_stopping_callback(metric, patience):
    return EarlyStopping(
        monitor=metric,  # does this need avg?
        mode="min" if "loss" in metric else "max",
        patience=patience,
        verbose=True,
    )
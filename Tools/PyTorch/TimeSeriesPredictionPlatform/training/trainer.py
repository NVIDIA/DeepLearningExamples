# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from abc import ABC
from functools import partial

import dgl
import dllogger
import hydra
import numpy as np
import torch
import torch.nn as nn
from apex import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataloader import default_collate

from callbacks.ctl_callbacks import CTLCallbackContainer
from data.data_utils import TSBaseDataset, sample_data
from distributed_utils import (
    get_device,
    init_distributed,
    is_main_process,
    log,
    reduce_tensor,
)
from evaluators.evaluation_metrics import MetricEvaluator
from loggers.log_helper import setup_logger
from training.ema import ModelEmaV2
from training.utils import (
    maybe_restore_checkpoint,
    round_dict,
    save_checkpoint,
    to_device,
)


class Trainer(ABC):
    def train(self):
        return

    def evaluate(self):
        return


class CTLTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        train_dataset: TSBaseDataset,
        valid_dataset: TSBaseDataset,
        test_dataset: TSBaseDataset,
        optimizer,
        evaluator: MetricEvaluator,
        criterion,
        config,
    ):
        self.config = config

        self._stop_training = False

        self.metrics = {}

        callbacks = [hydra.utils.call(callback_config) for callback_config in self.config.trainer.callback.values()]
        self.callbacks = CTLCallbackContainer(self, callbacks)

        self.world_size = self.config.device.get("world_size", 1)
        train_dataset = sample_data(train_dataset, self.config.dataset.get("train_samples", -1))
        valid_dataset = sample_data(valid_dataset, self.config.dataset.get("valid_samples", -1))
        self.valid_dataset_len = len(valid_dataset)
        self.train_dataset_len = len(train_dataset)
        self.train_sampler = None
        self.valid_sampler = None
        if self.world_size > 1:
            local_rank = int(self.config.device.get("local_rank", os.environ.get("LOCAL_RANK", 0)))
            self.device = get_device(local_rank, self.config.device.get("name", "cpu"))
            self.is_distributed = init_distributed(
                int(self.config.device.get("world_size", os.environ.get("WORLD_SIZE", 1)))
            )
            torch.cuda.synchronize()
            self.train_sampler = DistributedSampler(
                train_dataset, config.device.world_size, seed=config.trainer.get("seed", 0), drop_last=True
            )
            self.valid_sampler = DistributedSampler(
                valid_dataset, config.device.world_size, seed=config.trainer.get("seed", 0), drop_last=False
            )
        elif self.config.device.get("local_rank", None):
            self.device = get_device(self.config.device.get("local_rank"), self.config.device.get("name", "cpu"))
        else:
            self.device = torch.device(self.config.device.get("name", "cpu"))
        self.logger = setup_logger(self.config)
        self.optimizer = optimizer
        self.amp_enabled = self.config.trainer.get("AMP", False)
        self.model = model.to(self.device)

        if config.trainer.get("ema", None) is not None:
            self.ema = ModelEmaV2(config, model, self.device)
        else:
            self.ema = None
        if self.amp_enabled:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O2", loss_scale="dynamic")
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        # TODO: line below has to go somewhere else. Or use default print. Logging module alters std streams which prevents us from
        # capturing their outputs.
        # log(config.pretty())

        # XXX: Not sure about this. Maybe this should be isolated in collate_fn inside a DataLoader. Or maybe we should abstract it away in data_utils?
        # For sure we have to rename this. This suggests that masked target is somehow different from
        # regular target.
        self.train_target = "target_masked" if config.model.get("train_target_mask", True) else "target"
        self.eval_target = "target_masked" if config.model.get("eval_target_mask", True) else "target"
        self.test_target = "target_masked" if config.model.get("test_target_mask", True) else "target"

        if self.config.dataset.get("graph", False) and self.config.model.get("graph_eligible", False):

            def _collate_graph(samples, target):
                batch = dgl.batch(samples)
                labels = batch.ndata["target"]
                # XXX: we need discuss how to do this neatly
                if target == "target_masked":
                    labels = labels[:, self.config.dataset.encoder_length :, :]

                return batch, labels

            _collate = _collate_graph
        else:

            def _collate_dict(samples, target):
                batch = default_collate(samples)
                labels = batch["target"]
                if target == "target_masked":
                    labels = labels[:, self.config.dataset.encoder_length :, :]
                return batch, labels

            _collate = _collate_dict

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.trainer.batch_size,
            num_workers=self.config.trainer.num_workers,
            sampler=self.train_sampler,
            shuffle=True if self.train_sampler is None else False,
            pin_memory=True,
            collate_fn=partial(_collate, target=self.train_target),
        )
        self.valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=self.config.trainer.batch_size,
            num_workers=self.config.trainer.num_workers,
            sampler=self.valid_sampler,
            shuffle=True if self.valid_sampler is None else False,
            pin_memory=True,
            collate_fn=partial(_collate, target=self.eval_target),
        )
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config.trainer.batch_size,
            num_workers=1,
            pin_memory=True,
            collate_fn=partial(_collate, target=self.test_target),
        )
        if self.config.get("scheduler", None):
            self.scheduler = hydra.utils.instantiate(self.config.scheduler, optimizer)
        else:
            self.scheduler = None

        self.evaluator = evaluator
        self.criterion = criterion

        self.log_path = self.config.get("log_path", os.getcwd())
        self.global_step = 0
        self.epoch = 0

        self.preds_train_output_selector = config.model.get("preds_train_output_selector", -1)
        self.preds_eval_output_selector = config.model.get("preds_eval_output_selector", -1)
        self.preds_test_output_selector = config.model.get("preds_test_output_selector", -1)

        model_ref = self.model.module if self.world_size > 1 else self.model
        test_method_name = config.model.get("test_method", "__call__")
        self.test_method = getattr(model_ref, test_method_name)

        checkpoint_path = config.trainer.get("checkpoint_path", None)
        maybe_restore_checkpoint(self, checkpoint_path)

    def assess_valid(self):
        self.model.eval()
        with torch.no_grad():
            running_losses = 0

            for i, (batch, labels) in enumerate(self.valid_dataloader):
                batch = to_device(batch, device=self.device)
                labels = to_device(labels, device=self.device)
                if self.ema:
                    preds = self.ema.module(batch)
                else:
                    preds = self.model(batch)
                if self.preds_eval_output_selector >= 0:
                    preds = preds[..., self.preds_eval_output_selector : self.preds_eval_output_selector + 1]

                losses = self.criterion(preds, labels)
                losses = reduce_tensor(losses, self.world_size).detach()
                running_losses += losses

        running_losses = running_losses / (len(self.valid_dataloader.dataset) / self.config.trainer.batch_size)
        if len(running_losses.size()) < 1:
            running_losses = running_losses.unsqueeze(0)
        running_losses = [loss.item() for loss in running_losses]
        data = {"val_loss": sum(running_losses)}
        for i, elem in enumerate(running_losses):
            data["val_loss_component_" + str(i)] = elem
        self.logger.log(step=self.global_step, data=data, verbosity=dllogger.Verbosity.VERBOSE)

        self.model.train()
        return sum(running_losses)

    def train(self):

        self.callbacks.on_train_begin()
        self.global_step = 0
        for epoch in range(self.epoch, self.config.trainer.num_epochs):
            self.callbacks.on_epoch_begin(epoch)

            self.logger.log(step=self.global_step, data={"epoch": epoch}, verbosity=dllogger.Verbosity.VERBOSE)

            for i, (batch, labels) in enumerate(self.train_dataloader):
                self.callbacks.on_batch_begin(i)

                self.optimizer.zero_grad()
                batch = to_device(batch, device=self.device)
                labels = to_device(labels, device=self.device)

                preds = self.model(batch)
                if self.preds_train_output_selector >= 0:
                    preds = preds[..., self.preds_train_output_selector : self.preds_train_output_selector + 1]

                losses = self.criterion(preds, labels)
                loss = losses.sum()

                if self.amp_enabled:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                self.optimizer.step()

                losses = reduce_tensor(losses, self.world_size, average=True)
                if len(losses.size()) < 1:
                    losses = [losses]
                losses = [loss.item() for loss in losses]
                data = {"loss": loss.item()}
                for k, v in enumerate(losses):
                    data["loss_component_" + str(k)] = v

                self.logger.log(step=self.global_step, data=data, verbosity=dllogger.Verbosity.VERBOSE)

                if self.config.optimizer.get("gradient_norm", 0.0) > 0:
                    nn.utils.clip_grad_norm(self.model.parameters(), self.config.optimizer.gradient_norm)
                # XXX: shouldn't we move logging to a callback?
                if self.global_step % self.config.trainer.log_interval == 0:
                    self.logger.flush()
                self.global_step += 1
                self.callbacks.on_batch_end(i, logs=data)
                if self.ema:
                    self.ema.update(self.model)
            if self.scheduler:
                self.scheduler.step()
            self.callbacks.on_valid_begin(epoch)
            validation_loss = self.assess_valid()
            data = {"val_loss": validation_loss}
            self.callbacks.on_valid_end(epoch, logs=data)

            if is_main_process():
                save_checkpoint(self, checkpoint_dir=self.log_path)

            if self.train_sampler:
                self.train_sampler.set_epoch(epoch)
                self.valid_sampler.set_epoch(epoch)

            self.callbacks.on_epoch_end(epoch, logs=data)

            if self._stop_training:
                break

        self.callbacks.on_train_end(logs=self.metrics)

    def evaluate(self):
        self.callbacks.on_evaluate_begin()
        maybe_restore_checkpoint(self, os.path.join(self.log_path, "best_checkpoint.pth.tar"))
        self.model.eval()

        with torch.no_grad():

            preds_full = []
            labels_full = []
            weights_full = []
            ids_full = []

            for i, (batch, labels) in enumerate(self.test_dataloader):
                batch = to_device(batch, device=self.device)
                labels = to_device(labels, device=self.device)

                if self.config.evaluator.get("use_weights", False):
                    weights = batch["weight"]
                else:
                    weights = None

                # XXX we should abstract this away
                ids = batch.ndata["id"] if isinstance(batch, dgl.DGLGraph) else batch["id"]
                ids = ids[
                    :, 0, ...
                ]  # Assumes that time dimension is at index 1. We don't check whether te examle is constructed correctly

                labels_full.append(labels)
                weights_full.append(weights)
                preds = self.test_method(batch)
                if self.preds_test_output_selector >= 0:
                    preds = preds[..., self.preds_test_output_selector : self.preds_test_output_selector + 1]
                ids_full.append(ids)
                preds_full.append(preds)

            preds_full = torch.cat(preds_full, dim=0).cpu().numpy()
            labels_full = torch.cat(labels_full, dim=0).cpu().numpy()

            if self.config.evaluator.get("use_weights", False):
                weights_full = torch.cat(weights_full).cpu().numpy()
            else:
                weights_full = np.zeros((0, 0))
            ids_full = torch.cat(ids_full).cpu().numpy()
            eval_metrics = self.evaluator(labels_full, preds_full, weights_full, ids_full)

            self.metrics.update(eval_metrics)

            self.logger.log(
                step=[], data={k: float(v) for k, v in self.metrics.items()}, verbosity=dllogger.Verbosity.VERBOSE
            )
            self.callbacks.on_evaluate_end(logs=round_dict(self.metrics, decimal=3))
            return round_dict(self.metrics, decimal=3)


class StatTrainer(Trainer):
    def __init__(self, dataset, evaluator: MetricEvaluator, config, model):
        self.config = config
        self.evaluator = evaluator
        self.dataloader = dataset
        self.global_step = 0
        self.epoch = 0
        self.model = model
        setup_logger(self.config)

    def evaluate(self):

        preds_full = []
        labels_full = []
        weights_full = []
        ids_full = []

        for train, test in self.dataloader:

            labels = test["endog"]
            if self.config.evaluator.get("use_weights", False):
                weights = test["weight"]
            else:
                weights = None
            ids = test["id"].iloc[0]
            self.model.fit(train["endog"], train["exog"])
            preds = self.model.predict(test["exog"])
            labels_full.append(labels)
            weights_full.append(weights)

            ids_full.append(ids)
            preds_full.append(preds)

        preds_full = np.stack(preds_full)
        labels_full = np.stack(labels_full)

        if self.config.evaluator.get("use_weights", False):
            weights_full = np.stack(weights_full)
        else:
            weights_full = np.zeros((0, 0))
        ids_full = np.stack(ids_full)
        metrics = self.evaluator(labels_full, preds_full, weights_full, ids_full)

        return metrics


def numpy_normalised_quantile_loss(y, y_pred, quantile):
    prediction_underflow = y - y_pred
    losses = []

    weighted_errors = quantile * np.maximum(prediction_underflow, 0.0) + (1.0 - quantile) * np.maximum(
        -prediction_underflow, 0.0
    )
    losses.append(weighted_errors.mean())

    normalizer = abs(y).mean()

    losses = 2 * losses / normalizer

    return losses

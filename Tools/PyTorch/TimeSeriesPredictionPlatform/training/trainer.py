# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

import dgl
import dllogger
import hydra
import numpy as np
import torch
import torch.nn as nn
import importlib
try:
    from apex import amp
except ImportError:
    print("Nvidia apex not available. Can't use apex Automatic Mixed Precision (AMP) for training.\
    Please check: https://github.com/NVIDIA/apex for installation")
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from callbacks.ctl_callbacks import CTLCallbackContainer
from data.datasets import TSBaseDataset, get_collate_fn
from distributed_utils import reduce_tensor, get_mp_context
from loggers.log_helper import setup_logger
from training.ema import ModelEmaV2
from criterion import TSPP_criterion_wrapper
from training.checkpoint_utils import maybe_continue_run
from training.utils import to_device


class Trainer(ABC):
    def train(self):
        return


class CTLTrainer(Trainer):
    def __init__(
            self,
            model: nn.Module,
            train_dataset: TSBaseDataset,
            valid_dataset: TSBaseDataset,
            optimizer,
            criterion,
            callbacks,
            config,
    ):
        self.config = config
        self._stop_training = False

        self.metrics = {}

        callbacks = callbacks.values()
        self.callbacks = CTLCallbackContainer(self, callbacks)

        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        self.device = next(model.parameters()).device

        self.valid_dataset_len = len(valid_dataset)
        self.train_dataset_len = len(train_dataset)
        self.train_sampler = None
        self.valid_sampler = None
        self.example_length = config.example_length
        self.encoder_length = config.encoder_length

        if self.world_size > 1:
            # XXX: is the seed argument here needed for reproducibility?
            # It should be set in launch_training.py with other seeds
            self.train_sampler = DistributedSampler(
                train_dataset, self.world_size, seed=config.get("seed", 1), drop_last=True
            )
            self.valid_sampler = DistributedSampler(
                valid_dataset, self.world_size, seed=config.get("seed", 1), drop_last=False
            )
        self.logger = setup_logger(self.config)
        self.optimizer = optimizer
        self.amp_enabled = self.config.get("amp", False)
        if not importlib.util.find_spec("apex"):
            self.amp_enabled = False
        self.model = model
        self.global_step = 0
        self.epoch = 0

        if not self.config.get('force_rerun'):
            maybe_continue_run(self)

        if config.get("ema", False):
            self.ema = ModelEmaV2(model, decay=self.config.get('ema_decay', 0.999), device=self.device)
        else:
            self.ema = None
        if self.amp_enabled:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O2", loss_scale="dynamic")
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

        mp_context = get_mp_context()

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            sampler=self.train_sampler,
            shuffle=True if self.train_sampler is None else False,
            pin_memory=True,
            collate_fn=get_collate_fn(config.model_type, config.encoder_length),
            multiprocessing_context=mp_context
        )
        self.valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            sampler=self.valid_sampler,
            pin_memory=True,
            collate_fn=get_collate_fn(config.model_type, config.encoder_length),
            multiprocessing_context=mp_context
        )

        # TODO: make it reccursively instantiated
        if self.config.get("scheduler", None):
            self.config.scheduler._target_ = self.config.scheduler.target
            del self.config.scheduler.target
            self.scheduler = hydra.utils.instantiate(self.config.scheduler, optimizer)
        else:
            self.scheduler = None

        cl_start_horizon = config.get("cl_start_horizon")
        cl_update = config.get("cl_update")

        self.criterion = TSPP_criterion_wrapper(criterion, cl_start_horizon, cl_update)

        self.log_path = self.config.get("log_path", os.getcwd())

    def prep_data(self, batch, labels, weights):
        batch = to_device(batch, device=self.device)
        labels = to_device(labels, device=self.device)
        weights = to_device(weights, device=self.device)

        return batch, labels, weights

    def validate(self):
        self.model.eval()
        self.criterion.eval()
        with torch.no_grad():
            running_losses = 0

            for i, (batch, labels, weights) in enumerate(self.valid_dataloader):
                batch, labels, weights = self.prep_data(batch, labels, weights)

                if self.ema:
                    preds = self.ema.module(batch)
                else:
                    preds = self.model(batch)

                losses = self.criterion(preds, labels, weights=weights)
                losses = reduce_tensor(losses, self.world_size).detach()
                running_losses += losses

        running_losses = running_losses / (len(self.valid_dataloader.dataset) / self.config.batch_size)
        if len(running_losses.size()) < 1:
            running_losses = running_losses.unsqueeze(0)
        running_losses = [loss.item() for loss in running_losses]
        data = {"val_loss": sum(running_losses)}
        for i, elem in enumerate(running_losses):
            data["val_loss_component_" + str(i)] = elem
        self.logger.log(step=self.global_step, data=data, verbosity=dllogger.Verbosity.VERBOSE)

        self.model.train()
        self.criterion.train()
        return sum(running_losses)

    def train(self):

        self.callbacks.on_train_begin()
        while self.epoch < self.config.num_epochs:
            self.callbacks.on_epoch_begin(self.epoch)

            self.logger.log(step=self.global_step, data={"epoch": self.epoch}, verbosity=dllogger.Verbosity.VERBOSE)

            for i, (batch, labels, weights) in enumerate(self.train_dataloader):
                self.callbacks.on_batch_begin(i)

                self.optimizer.zero_grad()
                batch, labels, weights = self.prep_data(batch, labels, weights)

                preds = self.model(batch)

                losses = self.criterion(preds, labels, weights=weights)
                loss = losses.sum()

                if self.amp_enabled:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if self.config.get("gradient_norm", 0.0) > 0:
                    nn.utils.clip_grad_norm(self.model.parameters(), self.config.gradient_norm)
                self.optimizer.step()

                losses = reduce_tensor(losses, self.world_size, average=True)
                if len(losses.size()) < 1:
                    losses = [losses]
                losses = [loss.item() for loss in losses]
                data = {"loss": loss.item()}
                for k, v in enumerate(losses):
                    data["loss_component_" + str(k)] = v

                self.logger.log(step=self.global_step, data=data, verbosity=dllogger.Verbosity.VERBOSE)

                self.callbacks.on_batch_end(i, logs=data)
                if self.ema:
                    self.ema.update(self.model)
                self.global_step += 1
            if self.scheduler:
                self.scheduler.step()
            self.callbacks.on_valid_begin(self.epoch)
            validation_loss = self.validate()
            if validation_loss != validation_loss: #NaN check
                self._stop_training = True
            data = {"val_loss": validation_loss}
            self.callbacks.on_valid_end(self.epoch, logs=data)

            if self.train_sampler:
                self.train_sampler.set_epoch(self.epoch)
                self.valid_sampler.set_epoch(self.epoch)

            self.callbacks.on_epoch_end(self.epoch, logs=data)

            if self._stop_training:
                break
            self.epoch += 1

        self.callbacks.on_train_end(logs=self.metrics)


class StatTrainer(Trainer):
    def __init__(self,
                 config,
                 model,
                 train_dataset,
                 valid_dataset
                 ):
        self.config = config
        self.train_dataset = train_dataset
        self.global_step = 0
        self.epoch = 0
        self.model = model
        self.logger = setup_logger(self.config)

    def train(self):
        for train_batch in self.train_dataset:
            self.model.fit(train_batch["endog"], train_batch["exog"])

        self.model.save()

    def validate(self):
        raise RuntimeError("Validation is not supported for StatTrainer")


class XGBTrainer(Trainer):
    def __init__(self, config, callbacks, model, train_dataset, valid_dataset):
        '''
        The idea behind this trainer is that we are given data at a time step t and want to create models to predict the value of a target
        from t+1 to t+n.  At time step t we have access to every feature including the target, and if we are trying to predict at time step
        t+i, we have access to the known and static values from there, using the function target_shift.  To aid in prediction and
        give the model access to the history, lag and moving features can be specified in the configs.
        Lag features can either be specifed by a min value and max value or a list of values.  If a min and max
        value are specified then the range(min, max+1) is used as the list.  Moving average (or rolling features) are specified
        by a window size.  These values are added with the feat_adder function.  A new model is trained for every step we want
        to predict.  The trainer is not recursive so each model is independent and does not rely on the previous trained models.
        '''
        self.config = config
        self.logger = setup_logger(config)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.patience = callbacks.early_stopping.patience
        self.log_interval = config.get('log_interval', 25)
        self.model = model

    def train(self):
        for i, ((train_step, labels), (valid_step, valid_labels)) in enumerate(zip(self.train_dataset, self.valid_dataset)):
            self.model.fit(train_step, labels, valid_step, valid_labels,
                           patience=self.patience,
                           log_interval=self.log_interval)
        self.model.save(os.getcwd())


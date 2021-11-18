# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch

from se3_transformer.runtime.loggers import Logger
from se3_transformer.runtime.metrics import MeanAbsoluteError


class BaseCallback(ABC):
    def on_fit_start(self, optimizer, args):
        pass

    def on_fit_end(self):
        pass

    def on_epoch_end(self):
        pass

    def on_batch_start(self):
        pass

    def on_validation_step(self, input, target, pred):
        pass

    def on_validation_end(self, epoch=None):
        pass

    def on_checkpoint_load(self, checkpoint):
        pass

    def on_checkpoint_save(self, checkpoint):
        pass


class LRSchedulerCallback(BaseCallback):
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger
        self.scheduler = None

    @abstractmethod
    def get_scheduler(self, optimizer, args):
        pass

    def on_fit_start(self, optimizer, args):
        self.scheduler = self.get_scheduler(optimizer, args)

    def on_checkpoint_load(self, checkpoint):
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def on_checkpoint_save(self, checkpoint):
        checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

    def on_epoch_end(self):
        if self.logger is not None:
            self.logger.log_metrics({'learning rate': self.scheduler.get_last_lr()[0]}, step=self.scheduler.last_epoch)
        self.scheduler.step()


class QM9MetricCallback(BaseCallback):
    """ Logs the rescaled mean absolute error for QM9 regression tasks """

    def __init__(self, logger, targets_std, prefix=''):
        self.mae = MeanAbsoluteError()
        self.logger = logger
        self.targets_std = targets_std
        self.prefix = prefix
        self.best_mae = float('inf')

    def on_validation_step(self, input, target, pred):
        self.mae(pred.detach(), target.detach())

    def on_validation_end(self, epoch=None):
        mae = self.mae.compute() * self.targets_std
        logging.info(f'{self.prefix} MAE: {mae}')
        self.logger.log_metrics({f'{self.prefix} MAE': mae}, epoch)
        self.best_mae = min(self.best_mae, mae)

    def on_fit_end(self):
        if self.best_mae != float('inf'):
            self.logger.log_metrics({f'{self.prefix} best MAE': self.best_mae})


class QM9LRSchedulerCallback(LRSchedulerCallback):
    def __init__(self, logger, epochs):
        super().__init__(logger)
        self.epochs = epochs

    def get_scheduler(self, optimizer, args):
        min_lr = args.min_learning_rate if args.min_learning_rate else args.learning_rate / 10.0
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.epochs, eta_min=min_lr)


class PerformanceCallback(BaseCallback):
    def __init__(self, logger, batch_size: int, warmup_epochs: int = 1, mode: str = 'train'):
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.epoch = 0
        self.timestamps = []
        self.mode = mode
        self.logger = logger

    def on_batch_start(self):
        if self.epoch >= self.warmup_epochs:
            self.timestamps.append(time.time() * 1000.0)

    def _log_perf(self):
        stats = self.process_performance_stats()
        for k, v in stats.items():
            logging.info(f'performance {k}: {v}')

        self.logger.log_metrics(stats)

    def on_epoch_end(self):
        self.epoch += 1

    def on_fit_end(self):
        if self.epoch > self.warmup_epochs:
            self._log_perf()
            self.timestamps = []

    def process_performance_stats(self):
        timestamps = np.asarray(self.timestamps)
        deltas = np.diff(timestamps)
        throughput = (self.batch_size / deltas).mean()
        stats = {
            f"throughput_{self.mode}": throughput,
            f"latency_{self.mode}_mean": deltas.mean(),
            f"total_time_{self.mode}": timestamps[-1] - timestamps[0],
        }
        for level in [90, 95, 99]:
            stats.update({f"latency_{self.mode}_{level}": np.percentile(deltas, level)})

        return stats

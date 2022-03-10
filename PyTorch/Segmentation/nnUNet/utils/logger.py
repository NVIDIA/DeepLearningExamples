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

import os
import time

import dllogger as logger
import numpy as np
from dllogger import JSONStreamBackend, StdOutBackend, Verbosity
from pytorch_lightning import Callback

from utils.utils import rank_zero


class DLLogger:
    def __init__(self, log_dir, filename, append=True):
        super().__init__()
        self._initialize_dllogger(log_dir, filename, append)

    @rank_zero
    def _initialize_dllogger(self, log_dir, filename, append):
        backends = [
            JSONStreamBackend(Verbosity.VERBOSE, os.path.join(log_dir, filename), append=append),
            StdOutBackend(Verbosity.VERBOSE),
        ]
        logger.init(backends=backends)

    @rank_zero
    def log_metrics(self, metrics, step=None):
        if step is None:
            step = ()
        logger.log(step=step, data=metrics)

    @rank_zero
    def flush(self):
        logger.flush()


class LoggingCallback(Callback):
    def __init__(self, log_dir, filnename, global_batch_size, mode, warmup, dim):
        self.dllogger = DLLogger(log_dir, filnename)
        self.warmup_steps = warmup
        self.global_batch_size = global_batch_size
        self.step = 0
        self.dim = dim
        self.mode = mode
        self.timestamps = []

    def do_step(self):
        self.step += 1
        if self.step > self.warmup_steps:
            self.timestamps.append(time.time())

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if trainer.current_epoch == 1:
            self.do_step()

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if trainer.current_epoch == 1:
            self.do_step()

    def process_performance_stats(self, deltas):
        def _round3(val):
            return round(val, 3)

        throughput_imgps = _round3(self.global_batch_size / np.mean(deltas))
        timestamps_ms = 1000 * deltas
        stats = {
            f"throughput_{self.mode}": throughput_imgps,
            f"latency_{self.mode}_mean": _round3(timestamps_ms.mean()),
        }
        for level in [90, 95, 99]:
            stats.update({f"latency_{self.mode}_{level}": _round3(np.percentile(timestamps_ms, level))})

        return stats

    @rank_zero
    def _log(self):
        stats = self.process_performance_stats(np.diff(self.timestamps))
        self.dllogger.log_metrics(metrics=stats)
        self.dllogger.flush()

    def on_train_end(self, trainer, pl_module):
        self._log()

    def on_test_end(self, trainer, pl_module):
        if trainer.current_epoch == 1:
            self._log()

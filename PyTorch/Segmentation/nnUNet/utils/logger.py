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

import operator
import time

import dllogger as logger
import numpy as np
import torch.cuda.profiler as profiler
from dllogger import JSONStreamBackend, StdOutBackend, Verbosity
from pytorch_lightning import Callback

from utils.utils import is_main_process


class LoggingCallback(Callback):
    def __init__(self, log_dir, global_batch_size, mode, warmup, dim, profile):
        logger.init(backends=[JSONStreamBackend(Verbosity.VERBOSE, log_dir), StdOutBackend(Verbosity.VERBOSE)])
        self.warmup_steps = warmup
        self.global_batch_size = global_batch_size
        self.step = 0
        self.dim = dim
        self.mode = mode
        self.profile = profile
        self.timestamps = []

    def do_step(self):
        self.step += 1
        if self.profile and self.step == self.warmup_steps:
            profiler.start()
        if self.step >= self.warmup_steps:
            self.timestamps.append(time.time())

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.do_step()

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
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

    def log(self):
        if is_main_process():
            diffs = list(map(operator.sub, self.timestamps[1:], self.timestamps[:-1]))
            deltas = np.array(diffs)
            stats = self.process_performance_stats(deltas)
            logger.log(step=(), data=stats)
            logger.flush()

    def on_train_end(self, trainer, pl_module):
        if self.profile:
            profiler.stop()
        self.log()

    def on_test_end(self, trainer, pl_module):
        self.log()

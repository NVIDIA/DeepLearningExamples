# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import time

import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

from utils.parse_results import process_performance_stats


class ProfilingHook(tf.estimator.SessionRunHook):

    def __init__(self, logger, batch_size, log_every, warmup_steps, mode):
        self._log_every = log_every
        self._warmup_steps = warmup_steps
        self._current_step = 0
        self._global_batch_size = batch_size * hvd.size()
        self._t0 = 0
        self._timestamps = []
        self.logger = logger
        self.mode = mode

    def before_run(self, run_context):
        if self._current_step > self._warmup_steps:
            self._t0 = time.time()

    def after_run(self,
                  run_context,
                  run_values):
        if self._current_step > self._warmup_steps:
            self._timestamps.append(time.time() - self._t0)
        self._current_step += 1

    def begin(self):
        pass

    def end(self, session):
        if hvd.rank() == 0:
            throughput_imgps, latency_ms = process_performance_stats(np.array(self._timestamps),
                                                                     self._global_batch_size)
            self.logger.log(step=(),
                            data={'throughput_{}'.format(self.mode): throughput_imgps,
                                  'latency_{}'.format(self.mode): latency_ms})

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

import tensorflow as tf
import horovod.tensorflow as hvd

from dllogger import LOGGER, tags, AverageMeter


class ProfilingHook(tf.train.SessionRunHook):

    def __init__(self, batch_size, log_every, warmup_steps):
        self._log_every = log_every
        self._warmup_steps = warmup_steps
        self._current_step = 0
        self._global_batch_size = batch_size * hvd.size()
        self._meter = AverageMeter()
        self._t0 = 0

    def before_run(self, run_context):
        if self._current_step % self._log_every == 0:
            LOGGER.log('iter_start', self._current_step)

        if self._current_step > self._warmup_steps:
            self._t0 = time.time()

    def after_run(self,
                  run_context,
                  run_values):
        if self._current_step > self._warmup_steps:
            batch_time = time.time() - self._t0
            ips = self._global_batch_size / batch_time
            self._meter.record(ips)

        self._current_step += 1

    def begin(self):
        pass

    def end(self, session):
        LOGGER.log('average_images_per_second', self._meter.get_value())

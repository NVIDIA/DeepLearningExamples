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

import dllogger as DLLogger


class ProfilingHook(tf.estimator.SessionRunHook):
    def __init__(self, warmup_steps, global_batch_size, logger, training=True):
        self._warmup_steps = warmup_steps
        self._global_batch_size = global_batch_size
        self._step = 0
        self._timestamps = []
        self._logger = logger
        self._training = training

    def before_run(self, run_context):
        self._step += 1
        if self._step >= self._warmup_steps:
            self._timestamps.append(time.time())

    def end(self, session):
        deltas = [self._timestamps[i + 1] - self._timestamps[i] for i in range(len(self._timestamps) - 1)]
        self._logger.log(step=(), data={
            'average_throughput_train' if self._training else 'average_throughput_test': self._global_batch_size / np.mean(deltas)})
        self._logger.flush()

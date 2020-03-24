#! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import dllogger

from .training_hooks import MeanAccumulator


__all__ = ['BenchmarkLoggingHook']


class BenchmarkLoggingHook(tf.train.SessionRunHook):

    def __init__(self, global_batch_size, warmup_steps=100):
        self.warmup_steps = warmup_steps
        self.global_batch_size = global_batch_size
        self.current_step = 0
        self.t0 = None
        self.mean_throughput = MeanAccumulator()

    def before_run(self, run_context):
        self.t0 = time.time()

    def after_run(self, run_context, run_values):
        batch_time = time.time() - self.t0
        samplesps = self.global_batch_size / batch_time
        if self.current_step >= self.warmup_steps:
            self.mean_throughput.consume(samplesps)
            dllogger.log(data={"samplesps" : samplesps}, step=(0, self.current_step))

        self.current_step += 1


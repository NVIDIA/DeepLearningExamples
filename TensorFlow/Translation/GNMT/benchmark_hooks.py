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


__all__ = ['BenchmarkHook']


class BenchmarkHook(tf.train.SessionRunHook):

  def __init__(self, global_batch_size, warmup_steps=5):
    self.warmup_steps = warmup_steps
    self.global_batch_size = global_batch_size
    self.iter_times = []

  def before_run(self, run_context):
    self.t0 = time.time()

  def after_run(self, run_context, run_values):
    batch_time = time.time() - self.t0
    self.iter_times.append(batch_time)

  def get_average_speed(self):
    avg_time = np.mean(self.iter_times[self.warmup_steps:])
    speed = self.global_batch_size / avg_time
    return speed

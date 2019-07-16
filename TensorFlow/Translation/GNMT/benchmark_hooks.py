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
  latencies = ['avg', 50, 90, 95, 99, 100]

  def __init__(self, global_batch_size, warmup_steps=10):
    self.warmup_steps = warmup_steps
    self.global_batch_size = global_batch_size
    self.iter_times = []

  def before_run(self, run_context):
    self.t0 = time.time()

  def after_run(self, run_context, run_values):
    batch_time = time.time() - self.t0
    self.iter_times.append(batch_time)

  def get_average_speed_and_latencies(self):
    if len(self.iter_times) > self.warmup_steps + 5:
      warmup_steps = self.warmup_steps
    elif len(self.iter_times) > 15:
      warmup_steps = 10
    elif len(self.iter_times) > 10:
      warmup_steps = 5
    elif len(self.iter_times) > 4:
      warmup_steps = 2
    elif len(self.iter_times) > 1:
      warmup_steps = 1
    else:
      warmup_steps = 0

    times = self.iter_times[warmup_steps:]
    avg_time = np.mean(times)
    speed = self.global_batch_size / avg_time

    latencies = {}
    for lat in self.latencies:
      if lat == 'avg':
        val = avg_time
      else:
        val = np.percentile(times, lat)
      latencies[str(lat)] = val

    return speed, latencies

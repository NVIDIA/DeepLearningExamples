# Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from time import perf_counter

import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
from horovod.tensorflow.mpi_ops import Sum


class PerformanceCalculator:

    """
    PerformanceCalculator for throughput and latency statistics.

    Computes the statistics over a given number of steps. Timers should be initialized by the user by
    calling init() at the right moment -- just before running consecutive iterations of training.

    Attributes:
        warmup_steps (int): Number of initial steps to ignore for computing results.
        total_steps (int): Number of steps to collect data for (excluding warmup_steps); use <= 0 for unbounded horizon.
    """

    def __init__(self, warmup_steps=0, total_steps=0):
        self.warmup_steps = max(warmup_steps, 0)
        self.total_steps = self.warmup_steps + max(total_steps, 0)
        self.step = 0
        self.step_start_time = None
        self.benchmark_start_time = None
        self.benchmark_after_warmup_start_time = None
        self.step_latencies = []
        self.latency_percentiles = (90, 95, 99)
        self._results = {}
        with tf.device("/CPU:0"):
            self.samples = tf.Variable(0, trainable=False, dtype=tf.int64)

    def init(self):
        self.samples.assign(0)
        self.step_latencies = []
        self._results = {}
        # used to represent duration of entire training
        self.benchmark_start_time = perf_counter()
        # used to represent a time interval from post-warmup until the end
        self.benchmark_after_warmup_start_time = perf_counter()
        self.step_start_time = perf_counter()

    @property
    def results(self):
        return self._results.copy()

    @property
    def completed(self):
        return bool(self._results)

    def get_current_benchmark_results(self):
        if self.benchmark_start_time is None:
            raise RuntimeError(f"{self.__class__.__name__} has not been initialized")
        if self.step <= self.warmup_steps:
            raise RuntimeError(f"{self.__class__.__name__} is in warmup phase")
        results = self._calculate_throughput()
        results.update(self._calculate_latency())
        return results

    def _calculate_latency(self):
        latency_stats = {"latency_mean": 1000 * np.mean(self.step_latencies)}  # in milliseconds
        for p in self.latency_percentiles:
            latency_stats[f"latency_p{p}"] = 1000 * np.percentile(self.step_latencies, p)
        return latency_stats

    def _calculate_throughput(self):
        time_elapsed = perf_counter() - self.benchmark_start_time
        time_elapsed_after_warmup = perf_counter() - self.benchmark_after_warmup_start_time
        all_samples = hvd.allreduce(self.samples, op=Sum)
        benchmark_throughput = all_samples.numpy() / time_elapsed_after_warmup
        return {"throughput": benchmark_throughput, "time": time_elapsed}

    def __call__(self, n_samples):
        self.samples.assign_add(n_samples)
        step_latency = perf_counter() - self.step_start_time
        step_throughput = n_samples * hvd.size() / step_latency
        self.step_latencies.append(step_latency)
        self.step += 1
        if self.step == self.warmup_steps:
            self.samples.assign(0)
            self.step_latencies = []
            self.benchmark_after_warmup_start_time = perf_counter()
        elif self.step == self.total_steps:
            self._results = self.get_current_benchmark_results()
        self.step_start_time = perf_counter()
        return step_throughput

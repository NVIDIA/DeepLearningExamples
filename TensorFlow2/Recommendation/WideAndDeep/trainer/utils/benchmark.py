# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

import dllogger
import horovod.tensorflow as hvd
import tensorflow as tf
from horovod.tensorflow.mpi_ops import Sum


class ThroughputCalculator:
    def __init__(self, args):
        self.args = args
        self.boundary = max(self.args.benchmark_warmup_steps, 1)
        self.step = 0
        self.t0 = None
        self.start_batch_time = None
        with tf.device("/CPU:0"):
            self.samples = tf.Variable(0, trainable=False, dtype=tf.int64)

    def _init_benchmark(self):
        self.t0 = time.perf_counter()

    def on_epoch_end_log(self, step, shape):
        batch_time = time.perf_counter() - self.start_batch_time
        self.samples.assign_add(shape)
        workers = hvd.size() if not self.args.cpu else 1
        samplesps = shape * workers / batch_time
        if self.args.cpu or hvd.rank() == 0:
            dllogger.log(data={"batch_samplesps": samplesps}, step=(1, step))

    def on_benchmark_end_log(self, eval_benchmark=False):
        train_time = time.perf_counter() - self.t0
        hvd.join()
        if not self.args.cpu:
            all_samples = hvd.allreduce(self.samples, op=Sum)
        else:
            all_samples = self.samples

        all_samples = all_samples.numpy()

        if self.args.cpu or hvd.rank() == 0:
            key = "train_throughput" if not eval_benchmark else "validation_throughput"
            throughput = all_samples / train_time
            dllogger.log(data={key: throughput}, step=tuple())

    def __call__(self, shape, eval_benchmark=False):
        if self.args.benchmark:
            if self.step == self.boundary:
                self._init_benchmark()
            if self.step > self.boundary:
                self.on_epoch_end_log(self.step, shape)
                if self.args.benchmark_steps <= self.step:
                    self.on_benchmark_end_log(eval_benchmark=eval_benchmark)
                    exit(0)

            self.step += 1
            self.start_batch_time = time.perf_counter()

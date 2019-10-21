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

from mxnet.io import DataIter
import time

class BenchmarkingDataIter:
    def __init__(self, data_iter, benchmark_iters=None):
        self.data_iter = data_iter
        self.benchmark_iters = benchmark_iters
        self.overall_time = 0
        self.num = 0

    def __iter__(self):
        iter(self.data_iter)
        return self

    def next(self):
        if self.benchmark_iters is not None and self.num >= self.benchmark_iters:
            raise StopIteration
        try:
            start_time = time.time()
            ret = self.data_iter.next()
            end_time = time.time()
        except StopIteration:
            if self.benchmark_iters is None:
                raise
            self.data_iter.reset()
            start_time = time.time()
            ret = self.data_iter.next()
            end_time = time.time()

        if self.num != 0:
            self.overall_time += end_time - start_time
        self.num += 1
        return ret

    def __next__(self):
        return self.next()

    def __getattr__(self, attr):
        return getattr(self.data_iter, attr)

    def get_avg_time(self):
        if self.num <= 1:
            avg = float('nan')
        else:
            avg = self.overall_time / (self.num - 1)
        return avg

    def reset(self):
        self.overall_time = 0
        self.num = 0
        self.data_iter.reset()

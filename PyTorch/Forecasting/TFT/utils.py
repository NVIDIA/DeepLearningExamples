# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import torch.distributed as dist
import torch

class PerformanceMeter():
    def __init__(self, benchmark_mode=True):
        self.benchmark_mode = benchmark_mode
        self.reset()

    def reset(self):
        if self.benchmark_mode:
            torch.cuda.synchronize()
        self.avg = 0
        self.count = 0
        self.total_time = 0
        self.last_update_time = time.time()
        self.intervals = []

    def update(self, n, exclude_from_total=False):
        if self.benchmark_mode:
            torch.cuda.synchronize()
        delta = time.time() - self.last_update_time
        self.intervals.append(delta)
        if not exclude_from_total:
            self.total_time += delta
            self.count += n
            self.avg = self.count / self.total_time
        self.last_update_time = time.time()

        return n/delta

    def reset_current_lap(self):
        if self.benchmark_mode:
            torch.cuda.synchronize()
        self.last_update_time = time.time()

    def p(self, i):
        assert i <= 100
        idx = int(len(self.intervals) * i / 100)
        return sorted(self.intervals)[idx]

def print_once(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


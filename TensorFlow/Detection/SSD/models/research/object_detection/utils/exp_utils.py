# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
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

import dllogger
import os

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self, warmup=0, keep=False):
        self.reset()
        self.warmup = warmup
        self.keep = keep

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.iters = 0
        self.vals = []

    def update(self, val, n=1):
        self.iters += 1
        self.val = val

        if self.iters > self.warmup:
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
            if self.keep:
                self.vals.append(val)

def setup_dllogger(enabled=True, filename=os.devnull, rank=0):
    if enabled and rank == 0:
        backends = [
            dllogger.JSONStreamBackend(
                dllogger.Verbosity.VERBOSE,
                filename,
                ),
            ]
        dllogger.init(backends)
    else:
        dllogger.init([])

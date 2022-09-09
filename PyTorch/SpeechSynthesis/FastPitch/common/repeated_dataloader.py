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

"""Data pipeline elements which wrap the data N times

A RepeatedDataLoader resets its iterator less frequently. This saves time
on multi-GPU platforms and is invisible to the training loop.

NOTE: Repeating puts a block of (len(dataset) * repeats) int64s into RAM.
Do not use more repeats than necessary (e.g., 10**6 to simulate infinity).
"""

import itertools

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class RepeatedDataLoader(DataLoader):
    def __init__(self, repeats, *args, **kwargs):
        self.repeats = repeats
        super().__init__(*args, **kwargs)

    def __iter__(self):
        if self._iterator is None or self.repeats_done >= self.repeats:
            self.repeats_done = 1
            return super().__iter__()
        else:
            self.repeats_done += 1
            return self._iterator


class RepeatedDistributedSampler(DistributedSampler):
    def __init__(self, repeats, *args, **kwargs):
        self.repeats = repeats
        assert self.repeats <= 10000, "Too many repeats overload RAM."
        super().__init__(*args, **kwargs)

    def __iter__(self):
        # Draw indices for `self.repeats` epochs forward
        start_epoch = self.epoch
        iters = []
        for r in range(self.repeats):
            self.set_epoch(start_epoch + r)
            iters.append(super().__iter__())
        self.set_epoch(start_epoch)

        return itertools.chain.from_iterable(iters)

# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
from torch import Tensor


class Metric(ABC):
    """ Metric class with synchronization capabilities similar to TorchMetrics """

    def __init__(self):
        self.states = {}

    def add_state(self, name: str, default: Tensor):
        assert name not in self.states
        self.states[name] = default.clone()
        setattr(self, name, default)

    def synchronize(self):
        if dist.is_initialized():
            for state in self.states:
                dist.all_reduce(getattr(self, state), op=dist.ReduceOp.SUM, group=dist.group.WORLD)

    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def reset(self):
        for name, default in self.states.items():
            setattr(self, name, default.clone())

    def compute(self):
        self.synchronize()
        value = self._compute().item()
        self.reset()
        return value

    @abstractmethod
    def _compute(self):
        pass

    @abstractmethod
    def update(self, preds: Tensor, targets: Tensor):
        pass


class MeanAbsoluteError(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('error', torch.tensor(0, dtype=torch.float32, device='cuda'))
        self.add_state('total', torch.tensor(0, dtype=torch.int32, device='cuda'))

    def update(self, preds: Tensor, targets: Tensor):
        preds = preds.detach()
        n = preds.shape[0]
        error = torch.abs(preds.view(n, -1) - targets.view(n, -1)).sum()
        self.total += n
        self.error += error

    def _compute(self):
        return self.error / self.total

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


from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from se3_transformer.model.fiber import Fiber


class LinearSE3(nn.Module):
    """
    Graph Linear SE(3)-equivariant layer, equivalent to a 1x1 convolution.
    Maps a fiber to a fiber with the same degrees (channels may be different).
    No interaction between degrees, but interaction between channels.

    type-0 features (C_0 channels) ────> Linear(bias=False) ────> type-0 features (C'_0 channels)
    type-1 features (C_1 channels) ────> Linear(bias=False) ────> type-1 features (C'_1 channels)
                                                 :
    type-k features (C_k channels) ────> Linear(bias=False) ────> type-k features (C'_k channels)
    """

    def __init__(self, fiber_in: Fiber, fiber_out: Fiber):
        super().__init__()
        self.weights = nn.ParameterDict({
            str(degree_out): nn.Parameter(
                torch.randn(channels_out, fiber_in[degree_out]) / np.sqrt(fiber_in[degree_out]))
            for degree_out, channels_out in fiber_out
        })

    def forward(self, features: Dict[str, Tensor], *args, **kwargs) -> Dict[str, Tensor]:
        return {
            degree: self.weights[degree] @ features[degree]
            for degree, weight in self.weights.items()
        }

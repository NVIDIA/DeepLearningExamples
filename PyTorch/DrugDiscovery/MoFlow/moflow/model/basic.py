# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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


# Copyright 2020 Chengxi Zang
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.


import math
from typing import Tuple
import numpy as np
from scipy import linalg as la
import torch
from torch import nn
from torch.nn import functional as F

from moflow.runtime.distributed_utils import get_world_size, reduce_tensor


class ActNorm(nn.Module):
    def __init__(self, num_channels, num_dims, channels_dim=1):
        super().__init__()
        self.num_channels = num_channels
        self.num_dims = num_dims
        self.channels_dim = channels_dim
        self.shape = [1] * num_dims
        self.shape[channels_dim] = num_channels
        self.loc = nn.Parameter(torch.zeros(*self.shape))
        self.scale = nn.Parameter(torch.ones(*self.shape))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.register_buffer('num_elements', torch.tensor(0, dtype=torch.uint8))

    @torch.jit.ignore
    def initialize(self, input):
        if self.initialized.item() == 1:
            return

        dims = list(input.shape[1:])
        del dims[self.channels_dim -1]

        num_elems = math.prod(dims)
        permutation = [self.channels_dim] + [i for i in range(self.num_dims) if i != self.channels_dim]
        with torch.no_grad():

            flatten = input.permute(*permutation).contiguous().view(self.num_channels, -1)
            mean = flatten.mean(1).view(self.shape)
            std = flatten.std(1).view(self.shape)

            num_gpus = get_world_size()
            mean = reduce_tensor(mean, num_gpus)
            std = reduce_tensor(std, num_gpus)
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))
            self.initialized.fill_(1)
            self.num_elements.fill_(num_elems)

    def forward(self, input):
        log_abs = torch.log(torch.abs(self.scale))
        logdet = self.num_elements * torch.sum(log_abs)
        return self.scale * (input + self.loc), logdet

    @torch.jit.export
    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)
        logdet = (
            height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l).contiguous()
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(torch.log(torch.abs(w_s)))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()
        dtype = weight.dtype
        weight = weight.float()
        weight_inv = weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        weight_inv = weight_inv.to(dtype=dtype)

        return F.conv2d(output, weight_inv)


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_atoms, num_edge_type=4):
        super(GraphConv, self).__init__()

        self.graph_linear_self = nn.Linear(in_channels, out_channels)
        self.graph_linear_edge = nn.Linear(in_channels, out_channels * num_edge_type)
        self.num_edge_type = num_edge_type
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.num_atoms = num_atoms

    def forward(self, graph: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        adj, nodes = graph
        hs = self.graph_linear_self(nodes)
        m = self.graph_linear_edge(nodes)
        m = m.view(-1, self.num_atoms, self.out_ch, self.num_edge_type)
        hr = torch.einsum('bemn,bnce->bmc', adj, m)
        hr = hr.unsqueeze(2)
        return hs + hr

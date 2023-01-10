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


from typing import Tuple
import torch
import torch.nn as nn
from torch.nn.functional import logsigmoid

from moflow.model.basic import GraphConv


def sigmoid_inverse(x):
    """Calculates 1/sigmoid(x) in a more numerically stable way"""
    return 1 + torch.exp(-x)


class AffineCoupling(nn.Module):  # delete
    def __init__(self, in_channel, hidden_channels, mask_swap=False):  # filter_size=512,  --> hidden_channels =(512, 512)
        super(AffineCoupling, self).__init__()

        self.mask_swap=mask_swap
        # self.norms_in = nn.ModuleList()
        last_h = in_channel // 2
        vh = tuple(hidden_channels)
        layers = []
        for h in vh:
            layers.append(nn.Conv2d(last_h, h, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(h))
            layers.append(nn.ReLU(inplace=True))
            last_h = h
        layers.append(nn.Conv2d(last_h, in_channel, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        in_a, in_b = input.chunk(2, 1)  # (2,12,32,32) --> (2,6,32,32), (2,6,32,32)

        if self.mask_swap:
            in_a, in_b = in_b, in_a

        s_logits, t = self._s_t_function(in_a)
        s = torch.sigmoid(s_logits)
        out_b = (in_b + t) * s
        logdet = torch.sum(logsigmoid(s_logits).reshape(input.shape[0], -1), 1)
        
        if self.mask_swap:
            result = torch.cat([out_b, in_a], 1)
        else:
            result = torch.cat([in_a, out_b], 1)

        return result, logdet

    @torch.jit.export
    def reverse(self, output: torch.Tensor) -> torch.Tensor:
        out_a, out_b = output.chunk(2, 1)
        if self.mask_swap:
            out_a, out_b = out_b, out_a

        s_logits, t = self._s_t_function(out_a)
        s_inverse = sigmoid_inverse(s_logits)
        in_b = out_b * s_inverse - t

        if self.mask_swap:
            result = torch.cat([in_b, out_a], 1)
        else:
            result = torch.cat([out_a, in_b], 1)

        return result

    def _s_t_function(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.layers(x)
        s_logits, t = h.chunk(2, 1)
        return s_logits, t


class ConvCouplingBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_node: int) -> None:
        super().__init__()
        self.graph_conv = GraphConv(in_dim, out_dim, n_node)
        self.bn = nn.BatchNorm2d(n_node)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, graph: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        adj, nodes = graph
        h = self.graph_conv(graph)
        h = h.to(memory_format=torch.channels_last)
        h = self.bn(h)
        h = self.relu(h)
        return adj, h


class LinCouplingBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_node: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm2d(n_node)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        h = h.to(memory_format=torch.channels_last)
        h = self.bn(h)
        h = self.relu(h)
        return h


class GraphAffineCoupling(nn.Module):
    def __init__(self, n_node, in_dim, hidden_dim_dict, masked_row):
        super(GraphAffineCoupling, self).__init__()
        self.n_node = n_node
        self.in_dim = in_dim
        self.hidden_dim_dict = hidden_dim_dict
        self.masked_row = masked_row

        self.hidden_dim_gnn = hidden_dim_dict['gnn']
        self.hidden_dim_linear = hidden_dim_dict['linear']

        conv_layers = []
        last_dim = in_dim
        for out_dim in self.hidden_dim_gnn:
            conv_layers.append(ConvCouplingBlock(last_dim, out_dim, n_node))
            last_dim = out_dim
        self.net_conv = nn.ModuleList(conv_layers)

        lin_layers = []
        for out_dim in self.hidden_dim_linear:
            lin_layers.append(LinCouplingBlock(last_dim, out_dim, n_node))
            last_dim = out_dim
        lin_layers.append(nn.Linear(last_dim, in_dim*2))
        self.net_lin = nn.Sequential(*lin_layers)

        mask = torch.ones(n_node, in_dim)
        mask[masked_row, :] = 0  # masked_row are kept same, and used for _s_t for updating the left rows
        self.register_buffer('mask', mask)

    def forward(self, graph: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        adj, input = graph
        masked_x = self.mask * input
        masked_x_sq = masked_x.unsqueeze(2)
        s_logits, t = self._s_t_function((adj, masked_x_sq))
        s = torch.sigmoid(s_logits)
        out = masked_x + (1-self.mask) * (input + t) * s
        logdet = torch.sum(logsigmoid(s_logits).reshape(input.shape[0], -1), 1)
        return out, logdet

    @torch.jit.export
    def reverse(self, graph: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        adj, output = graph
        masked_y = self.mask * output
        masked_y_sq = masked_y.unsqueeze(2)
        s_logits, t = self._s_t_function((adj, masked_y_sq))
        s_inverse = sigmoid_inverse(s_logits)
        input = masked_y + (1 - self.mask) * (output * s_inverse - t)
        return input

    def _s_t_function(self, graph: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        for l in self.net_conv:
            graph = l(graph)
        adj, h = graph
        h = self.net_lin(h)
        h = h.squeeze(2)
        s_logits, t = h.chunk(2, dim=-1)

        return s_logits, t

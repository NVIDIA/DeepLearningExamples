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

from moflow.model.basic import ActNorm, InvConv2dLU, InvConv2d
from moflow.model.coupling import AffineCoupling, GraphAffineCoupling


class Flow(nn.Module):
    def __init__(self, in_channel, hidden_channels, conv_lu=2, mask_swap=False):
        super(Flow, self).__init__()

        # More stable to support more flows
        self.actnorm = ActNorm(num_channels=in_channel, num_dims=4)

        if conv_lu == 0:
            self.invconv = InvConv2d(in_channel)
        elif conv_lu == 1:
            self.invconv = InvConv2dLU(in_channel)
        elif conv_lu == 2:
            self.invconv = None
        else:
            raise ValueError("conv_lu in {0,1,2}, 0:InvConv2d, 1:InvConv2dLU, 2:none-just swap to update in coupling")

        self.coupling = AffineCoupling(in_channel, hidden_channels, mask_swap=mask_swap)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet = self.actnorm(input)
        if self.invconv is not None:
            out, det1 = self.invconv(out)
        else:
            det1 = 0
        out, det2 = self.coupling(out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    @torch.jit.export
    def reverse(self, output: torch.Tensor) -> torch.Tensor:
        input = self.coupling.reverse(output)
        if self.invconv is not None:
            input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


class FlowOnGraph(nn.Module):
    def __init__(self, n_node, in_dim, hidden_dim_dict, masked_row):
        super(FlowOnGraph, self).__init__()
        self.n_node = n_node
        self.in_dim = in_dim
        self.hidden_dim_dict = hidden_dim_dict
        self.masked_row = masked_row
        self.actnorm = ActNorm(num_channels=n_node, num_dims=3)
        self.coupling = GraphAffineCoupling(n_node, in_dim, hidden_dim_dict, masked_row)

    def forward(self, graph: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        adj, input = graph
        out, logdet = self.actnorm(input)
        det1 = 0
        out, det2 = self.coupling((adj, out))

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2
        return out, logdet

    @torch.jit.export
    def reverse(self, graph: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        adj, output = graph
        input = self.coupling.reverse((adj, output))
        input = self.actnorm.reverse(input)
        return input


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, squeeze_fold, hidden_channels, conv_lu=2):
        super(Block, self).__init__()
        self.squeeze_fold = squeeze_fold
        squeeze_dim = in_channel * self.squeeze_fold * self.squeeze_fold

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            if conv_lu in (0, 1):
                self.flows.append(Flow(squeeze_dim, hidden_channels,
                                       conv_lu=conv_lu, mask_swap=False))
            else:
                self.flows.append(Flow(squeeze_dim, hidden_channels,
                                       conv_lu=2, mask_swap=bool(i % 2)))

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self._squeeze(input)
        logdet = 0

        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        out = self._unsqueeze(out)
        return out, logdet

    @torch.jit.export
    def reverse(self, output: torch.Tensor) -> torch.Tensor:
        input = self._squeeze(output)

        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        unsqueezed = self._unsqueeze(input)
        return unsqueezed

    def _squeeze(self, x: torch.Tensor) -> torch.Tensor:
        """Trade spatial extent for channels. In forward direction, convert each
        1x4x4 volume of input into a 4x1x1 volume of output.

        Args:
            x (torch.Tensor): Input to squeeze or unsqueeze.
            reverse (bool): Reverse the operation, i.e., unsqueeze.

        Returns:
            x (torch.Tensor): Squeezed or unsqueezed tensor.
        """
        assert len(x.shape) == 4
        b_size, n_channel, height, width = x.shape
        fold = self.squeeze_fold

        squeezed = x.view(b_size, n_channel, height // fold,  fold,  width // fold,  fold)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4).contiguous()
        out = squeezed.view(b_size, n_channel * fold * fold, height // fold, width // fold)
        return out

    def _unsqueeze(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 4
        b_size, n_channel, height, width = x.shape
        fold = self.squeeze_fold
        unsqueezed = x.view(b_size, n_channel // (fold * fold), fold, fold, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3).contiguous()
        out = unsqueezed.view(b_size, n_channel // (fold * fold), height * fold, width * fold)
        return out


class BlockOnGraph(nn.Module):
    def __init__(self, n_node, in_dim, hidden_dim_dict, n_flow, mask_row_size=1, mask_row_stride=1):
        super(BlockOnGraph, self).__init__()
        assert 0 < mask_row_size < n_node
        self.flows = nn.ModuleList()
        for i in range(n_flow):
            start = i * mask_row_stride
            masked_row =[r % n_node for r in range(start, start+mask_row_size)]
            self.flows.append(FlowOnGraph(n_node, in_dim, hidden_dim_dict, masked_row=masked_row))

    def forward(self, graph: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        adj, input = graph
        out = input
        logdet = 0
        for flow in self.flows:
            out, det = flow((adj, out))
            logdet = logdet + det
        return out, logdet

    @torch.jit.export
    def reverse(self, graph: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        adj, output = graph
        input = output
        for flow in self.flows[::-1]:
            input = flow.reverse((adj, input))
        return input


class Glow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, squeeze_fold, hidden_channel, conv_lu=2):
        super(Glow, self).__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block):
            self.blocks.append(Block(n_channel, n_flow, squeeze_fold, hidden_channel, conv_lu=conv_lu))

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet = 0
        out = input

        for block in self.blocks:
            out, det = block(out)
            logdet = logdet + det

        return out, logdet

    @torch.jit.export
    def reverse(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        for i, block in enumerate(self.blocks[::-1]):
            h = block.reverse(h)

        return h


class GlowOnGraph(nn.Module):
    def __init__(self, n_node, in_dim, hidden_dim_dict, n_flow, n_block,
                 mask_row_size_list=(2,), mask_row_stride_list=(1,)):
        super(GlowOnGraph, self).__init__()

        assert len(mask_row_size_list) == n_block or len(mask_row_size_list) == 1
        assert len(mask_row_stride_list) == n_block or len(mask_row_stride_list) == 1
        if len(mask_row_size_list) == 1:
            mask_row_size_list = mask_row_size_list * n_block
        if len(mask_row_stride_list) == 1:
            mask_row_stride_list = mask_row_stride_list * n_block
        self.blocks = nn.ModuleList()
        for i in range(n_block):
            mask_row_size = mask_row_size_list[i]
            mask_row_stride = mask_row_stride_list[i]
            self.blocks.append(BlockOnGraph(n_node, in_dim, hidden_dim_dict, n_flow, mask_row_size, mask_row_stride))

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet = 0
        out = x
        for block in self.blocks:
            out, det = block((adj, out))
            logdet = logdet + det
        return out, logdet

    @torch.jit.export
    def reverse(self, graph: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        adj, z = graph
        input = z
        for i, block in enumerate(self.blocks[::-1]):
            input = block.reverse((adj, input))

        return input

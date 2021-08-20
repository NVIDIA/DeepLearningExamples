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

import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl import DGLGraph
from dgl.ops import edge_softmax
from torch import Tensor
from typing import Dict, Optional, Union

from se3_transformer.model.fiber import Fiber
from se3_transformer.model.layers.convolution import ConvSE3, ConvSE3FuseLevel
from se3_transformer.model.layers.linear import LinearSE3
from se3_transformer.runtime.utils import degree_to_dim, aggregate_residual, unfuse_features
from torch.cuda.nvtx import range as nvtx_range


class AttentionSE3(nn.Module):
    """ Multi-headed sparse graph self-attention (SE(3)-equivariant) """

    def __init__(
            self,
            num_heads: int,
            key_fiber: Fiber,
            value_fiber: Fiber
    ):
        """
        :param num_heads:     Number of attention heads
        :param key_fiber:     Fiber for the keys (and also for the queries)
        :param value_fiber:   Fiber for the values
        """
        super().__init__()
        self.num_heads = num_heads
        self.key_fiber = key_fiber
        self.value_fiber = value_fiber

    def forward(
            self,
            value: Union[Tensor, Dict[str, Tensor]],  # edge features (may be fused)
            key: Union[Tensor, Dict[str, Tensor]],  # edge features (may be fused)
            query: Dict[str, Tensor],  # node features
            graph: DGLGraph
    ):
        with nvtx_range('AttentionSE3'):
            with nvtx_range('reshape keys and queries'):
                if isinstance(key, Tensor):
                    # case where features of all types are fused
                    key = key.reshape(key.shape[0], self.num_heads, -1)
                    # need to reshape queries that way to keep the same layout as keys
                    out = torch.cat([query[str(d)] for d in self.key_fiber.degrees], dim=-1)
                    query = out.reshape(list(query.values())[0].shape[0], self.num_heads, -1)
                else:
                    # features are not fused, need to fuse and reshape them
                    key = self.key_fiber.to_attention_heads(key, self.num_heads)
                    query = self.key_fiber.to_attention_heads(query, self.num_heads)

            with nvtx_range('attention dot product + softmax'):
                # Compute attention weights (softmax of inner product between key and query)
                edge_weights = dgl.ops.e_dot_v(graph, key, query).squeeze(-1)
                edge_weights /= np.sqrt(self.key_fiber.num_features)
                edge_weights = edge_softmax(graph, edge_weights)
                edge_weights = edge_weights[..., None, None]

            with nvtx_range('weighted sum'):
                if isinstance(value, Tensor):
                    # features of all types are fused
                    v = value.view(value.shape[0], self.num_heads, -1, value.shape[-1])
                    weights = edge_weights * v
                    feat_out = dgl.ops.copy_e_sum(graph, weights)
                    feat_out = feat_out.view(feat_out.shape[0], -1, feat_out.shape[-1])  # merge heads
                    out = unfuse_features(feat_out, self.value_fiber.degrees)
                else:
                    out = {}
                    for degree, channels in self.value_fiber:
                        v = value[str(degree)].view(-1, self.num_heads, channels // self.num_heads,
                                                    degree_to_dim(degree))
                        weights = edge_weights * v
                        res = dgl.ops.copy_e_sum(graph, weights)
                        out[str(degree)] = res.view(-1, channels, degree_to_dim(degree))  # merge heads

                return out


class AttentionBlockSE3(nn.Module):
    """ Multi-headed sparse graph self-attention block with skip connection, linear projection (SE(3)-equivariant) """

    def __init__(
            self,
            fiber_in: Fiber,
            fiber_out: Fiber,
            fiber_edge: Optional[Fiber] = None,
            num_heads: int = 4,
            channels_div: int = 2,
            use_layer_norm: bool = False,
            max_degree: bool = 4,
            fuse_level: ConvSE3FuseLevel = ConvSE3FuseLevel.FULL,
            **kwargs
    ):
        """
        :param fiber_in:         Fiber describing the input features
        :param fiber_out:        Fiber describing the output features
        :param fiber_edge:       Fiber describing the edge features (node distances excluded)
        :param num_heads:        Number of attention heads
        :param channels_div:     Divide the channels by this integer for computing values
        :param use_layer_norm:   Apply layer normalization between MLP layers
        :param max_degree:       Maximum degree used in the bases computation
        :param fuse_level:       Maximum fuse level to use in TFN convolutions
        """
        super().__init__()
        if fiber_edge is None:
            fiber_edge = Fiber({})
        self.fiber_in = fiber_in
        # value_fiber has same structure as fiber_out but #channels divided by 'channels_div'
        value_fiber = Fiber([(degree, channels // channels_div) for degree, channels in fiber_out])
        # key_query_fiber has the same structure as fiber_out, but only degrees which are in in_fiber
        # (queries are merely projected, hence degrees have to match input)
        key_query_fiber = Fiber([(fe.degree, fe.channels) for fe in value_fiber if fe.degree in fiber_in.degrees])

        self.to_key_value = ConvSE3(fiber_in, value_fiber + key_query_fiber, pool=False, fiber_edge=fiber_edge,
                                    use_layer_norm=use_layer_norm, max_degree=max_degree, fuse_level=fuse_level,
                                    allow_fused_output=True)
        self.to_query = LinearSE3(fiber_in, key_query_fiber)
        self.attention = AttentionSE3(num_heads, key_query_fiber, value_fiber)
        self.project = LinearSE3(value_fiber + fiber_in, fiber_out)

    def forward(
            self,
            node_features: Dict[str, Tensor],
            edge_features: Dict[str, Tensor],
            graph: DGLGraph,
            basis: Dict[str, Tensor]
    ):
        with nvtx_range('AttentionBlockSE3'):
            with nvtx_range('keys / values'):
                fused_key_value = self.to_key_value(node_features, edge_features, graph, basis)
                key, value = self._get_key_value_from_fused(fused_key_value)

            with nvtx_range('queries'):
                query = self.to_query(node_features)

            z = self.attention(value, key, query, graph)
            z_concat = aggregate_residual(node_features, z, 'cat')
            return self.project(z_concat)

    def _get_key_value_from_fused(self, fused_key_value):
        # Extract keys and queries features from fused features
        if isinstance(fused_key_value, Tensor):
            # Previous layer was a fully fused convolution
            value, key = torch.chunk(fused_key_value, chunks=2, dim=-2)
        else:
            key, value = {}, {}
            for degree, feat in fused_key_value.items():
                if int(degree) in self.fiber_in.degrees:
                    value[degree], key[degree] = torch.chunk(feat, chunks=2, dim=-2)
                else:
                    value[degree] = feat

        return key, value

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

from enum import Enum
from itertools import product
from typing import Dict

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from dgl import DGLGraph
from torch import Tensor
from torch.cuda.nvtx import range as nvtx_range

from se3_transformer.model.fiber import Fiber
from se3_transformer.runtime.utils import degree_to_dim, unfuse_features


class ConvSE3FuseLevel(Enum):
    """
    Enum to select a maximum level of fusing optimizations that will be applied when certain conditions are met.
    If a desired level L is picked and the level L cannot be applied to a level, other fused ops < L are considered.
    A higher level means faster training, but also more memory usage.
    If you are tight on memory and want to feed large inputs to the network, choose a low value.
    If you want to train fast, choose a high value.
    Recommended value is FULL with AMP.

    Fully fused TFN convolutions requirements:
    - all input channels are the same
    - all output channels are the same
    - input degrees span the range [0, ..., max_degree]
    - output degrees span the range [0, ..., max_degree]

    Partially fused TFN convolutions requirements:
    * For fusing by output degree:
    - all input channels are the same
    - input degrees span the range [0, ..., max_degree]
    * For fusing by input degree:
    - all output channels are the same
    - output degrees span the range [0, ..., max_degree]

    Original TFN pairwise convolutions: no requirements
    """

    FULL = 2
    PARTIAL = 1
    NONE = 0


class RadialProfile(nn.Module):
    """
    Radial profile function.
    Outputs weights used to weigh basis matrices in order to get convolution kernels.
    In TFN notation: $R^{l,k}$
    In SE(3)-Transformer notation: $\phi^{l,k}$

    Note:
        In the original papers, this function only depends on relative node distances ||x||.
        Here, we allow this function to also take as input additional invariant edge features.
        This does not break equivariance and adds expressive power to the model.

    Diagram:
        invariant edge features (node distances included) ───> MLP layer (shared across edges) ───> radial weights
    """

    def __init__(
            self,
            num_freq: int,
            channels_in: int,
            channels_out: int,
            edge_dim: int = 1,
            mid_dim: int = 32,
            use_layer_norm: bool = False
    ):
        """
        :param num_freq:         Number of frequencies
        :param channels_in:      Number of input channels
        :param channels_out:     Number of output channels
        :param edge_dim:         Number of invariant edge features (input to the radial function)
        :param mid_dim:          Size of the hidden MLP layers
        :param use_layer_norm:   Apply layer normalization between MLP layers
        """
        super().__init__()
        modules = [
            nn.Linear(edge_dim, mid_dim),
            nn.LayerNorm(mid_dim) if use_layer_norm else None,
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.LayerNorm(mid_dim) if use_layer_norm else None,
            nn.ReLU(),
            nn.Linear(mid_dim, num_freq * channels_in * channels_out, bias=False)
        ]

        self.net = nn.Sequential(*[m for m in modules if m is not None])

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class VersatileConvSE3(nn.Module):
    """
    Building block for TFN convolutions.
    This single module can be used for fully fused convolutions, partially fused convolutions, or pairwise convolutions.
    """

    def __init__(self,
                 freq_sum: int,
                 channels_in: int,
                 channels_out: int,
                 edge_dim: int,
                 use_layer_norm: bool,
                 fuse_level: ConvSE3FuseLevel):
        super().__init__()
        self.freq_sum = freq_sum
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.fuse_level = fuse_level
        self.radial_func = RadialProfile(num_freq=freq_sum,
                                         channels_in=channels_in,
                                         channels_out=channels_out,
                                         edge_dim=edge_dim,
                                         use_layer_norm=use_layer_norm)

    def forward(self, features: Tensor, invariant_edge_feats: Tensor, basis: Tensor):
        with nvtx_range(f'VersatileConvSE3'):
            num_edges = features.shape[0]
            in_dim = features.shape[2]
            with nvtx_range(f'RadialProfile'):
                radial_weights = self.radial_func(invariant_edge_feats) \
                    .view(-1, self.channels_out, self.channels_in * self.freq_sum)

            if basis is not None:
                # This block performs the einsum n i l, n o i f, n l f k -> n o k
                basis_view = basis.view(num_edges, in_dim, -1)
                tmp = (features @ basis_view).view(num_edges, -1, basis.shape[-1])
                return radial_weights @ tmp
            else:
                # k = l = 0 non-fused case
                return radial_weights @ features


class ConvSE3(nn.Module):
    """
    SE(3)-equivariant graph convolution (Tensor Field Network convolution).
    This convolution can map an arbitrary input Fiber to an arbitrary output Fiber, while preserving equivariance.
    Features of different degrees interact together to produce output features.

    Note 1:
        The option is given to not pool the output. This means that the convolution sum over neighbors will not be
        done, and the returned features will be edge features instead of node features.

    Note 2:
        Unlike the original paper and implementation, this convolution can handle edge feature of degree greater than 0.
        Input edge features are concatenated with input source node features before the kernel is applied.
     """

    def __init__(
            self,
            fiber_in: Fiber,
            fiber_out: Fiber,
            fiber_edge: Fiber,
            pool: bool = True,
            use_layer_norm: bool = False,
            self_interaction: bool = False,
            max_degree: int = 4,
            fuse_level: ConvSE3FuseLevel = ConvSE3FuseLevel.FULL,
            allow_fused_output: bool = False,
            low_memory: bool = False
    ):
        """
        :param fiber_in:           Fiber describing the input features
        :param fiber_out:          Fiber describing the output features
        :param fiber_edge:         Fiber describing the edge features (node distances excluded)
        :param pool:               If True, compute final node features by averaging incoming edge features
        :param use_layer_norm:     Apply layer normalization between MLP layers
        :param self_interaction:   Apply self-interaction of nodes
        :param max_degree:         Maximum degree used in the bases computation
        :param fuse_level:         Maximum fuse level to use in TFN convolutions
        :param allow_fused_output: Allow the module to output a fused representation of features
        """
        super().__init__()
        self.pool = pool
        self.fiber_in = fiber_in
        self.fiber_out = fiber_out
        self.self_interaction = self_interaction
        self.max_degree = max_degree
        self.allow_fused_output = allow_fused_output
        self.conv_checkpoint = torch.utils.checkpoint.checkpoint if low_memory else lambda m, *x: m(*x)

        # channels_in: account for the concatenation of edge features
        channels_in_set = set([f.channels + fiber_edge[f.degree] * (f.degree > 0) for f in self.fiber_in])
        channels_out_set = set([f.channels for f in self.fiber_out])
        unique_channels_in = (len(channels_in_set) == 1)
        unique_channels_out = (len(channels_out_set) == 1)
        degrees_up_to_max = list(range(max_degree + 1))
        common_args = dict(edge_dim=fiber_edge[0] + 1, use_layer_norm=use_layer_norm)

        if fuse_level.value >= ConvSE3FuseLevel.FULL.value and \
                unique_channels_in and fiber_in.degrees == degrees_up_to_max and \
                unique_channels_out and fiber_out.degrees == degrees_up_to_max:
            # Single fused convolution
            self.used_fuse_level = ConvSE3FuseLevel.FULL

            sum_freq = sum([
                degree_to_dim(min(d_in, d_out))
                for d_in, d_out in product(degrees_up_to_max, degrees_up_to_max)
            ])

            self.conv = VersatileConvSE3(sum_freq, list(channels_in_set)[0], list(channels_out_set)[0],
                                         fuse_level=self.used_fuse_level, **common_args)

        elif fuse_level.value >= ConvSE3FuseLevel.PARTIAL.value and \
                unique_channels_in and fiber_in.degrees == degrees_up_to_max:
            # Convolutions fused per output degree
            self.used_fuse_level = ConvSE3FuseLevel.PARTIAL
            self.conv_out = nn.ModuleDict()
            for d_out, c_out in fiber_out:
                sum_freq = sum([degree_to_dim(min(d_out, d)) for d in fiber_in.degrees])
                self.conv_out[str(d_out)] = VersatileConvSE3(sum_freq, list(channels_in_set)[0], c_out,
                                                             fuse_level=self.used_fuse_level, **common_args)

        elif fuse_level.value >= ConvSE3FuseLevel.PARTIAL.value and \
                unique_channels_out and fiber_out.degrees == degrees_up_to_max:
            # Convolutions fused per input degree
            self.used_fuse_level = ConvSE3FuseLevel.PARTIAL
            self.conv_in = nn.ModuleDict()
            for d_in, c_in in fiber_in:
                channels_in_new = c_in + fiber_edge[d_in] * (d_in > 0)
                sum_freq = sum([degree_to_dim(min(d_in, d)) for d in fiber_out.degrees])
                self.conv_in[str(d_in)] = VersatileConvSE3(sum_freq, channels_in_new, list(channels_out_set)[0],
                                                           fuse_level=self.used_fuse_level, **common_args)
        else:
            # Use pairwise TFN convolutions
            self.used_fuse_level = ConvSE3FuseLevel.NONE
            self.conv = nn.ModuleDict()
            for (degree_in, channels_in), (degree_out, channels_out) in (self.fiber_in * self.fiber_out):
                dict_key = f'{degree_in},{degree_out}'
                channels_in_new = channels_in + fiber_edge[degree_in] * (degree_in > 0)
                sum_freq = degree_to_dim(min(degree_in, degree_out))
                self.conv[dict_key] = VersatileConvSE3(sum_freq, channels_in_new, channels_out,
                                                       fuse_level=self.used_fuse_level, **common_args)

        if self_interaction:
            self.to_kernel_self = nn.ParameterDict()
            for degree_out, channels_out in fiber_out:
                if fiber_in[degree_out]:
                    self.to_kernel_self[str(degree_out)] = nn.Parameter(
                        torch.randn(channels_out, fiber_in[degree_out]) / np.sqrt(fiber_in[degree_out]))

    def _try_unpad(self, feature, basis):
        # Account for padded basis
        if basis is not None:
            out_dim = basis.shape[-1]
            out_dim += out_dim % 2 - 1
            return feature[..., :out_dim]
        else:
            return feature

    def forward(
            self,
            node_feats: Dict[str, Tensor],
            edge_feats: Dict[str, Tensor],
            graph: DGLGraph,
            basis: Dict[str, Tensor]
    ):
        with nvtx_range(f'ConvSE3'):
            invariant_edge_feats = edge_feats['0'].squeeze(-1)
            src, dst = graph.edges()
            out = {}
            in_features = []

            # Fetch all input features from edge and node features
            for degree_in in self.fiber_in.degrees:
                src_node_features = node_feats[str(degree_in)][src]
                if degree_in > 0 and str(degree_in) in edge_feats:
                    # Handle edge features of any type by concatenating them to node features
                    src_node_features = torch.cat([src_node_features, edge_feats[str(degree_in)]], dim=1)
                in_features.append(src_node_features)

            if self.used_fuse_level == ConvSE3FuseLevel.FULL:
                in_features_fused = torch.cat(in_features, dim=-1)
                out = self.conv_checkpoint(
                    self.conv, in_features_fused, invariant_edge_feats, basis['fully_fused']
                )

                if not self.allow_fused_output or self.self_interaction or self.pool:
                    out = unfuse_features(out, self.fiber_out.degrees)

            elif self.used_fuse_level == ConvSE3FuseLevel.PARTIAL and hasattr(self, 'conv_out'):
                in_features_fused = torch.cat(in_features, dim=-1)
                for degree_out in self.fiber_out.degrees:
                    basis_used = basis[f'out{degree_out}_fused']
                    out[str(degree_out)] = self._try_unpad(
                        self.conv_checkpoint(
                            self.conv_out[str(degree_out)], in_features_fused, invariant_edge_feats, basis_used
                        ), basis_used)

            elif self.used_fuse_level == ConvSE3FuseLevel.PARTIAL and hasattr(self, 'conv_in'):
                out = 0
                for degree_in, feature in zip(self.fiber_in.degrees, in_features):
                    out = out + self.conv_checkpoint(
                        self.conv_in[str(degree_in)], feature, invariant_edge_feats, basis[f'in{degree_in}_fused']
                    )
                if not self.allow_fused_output or self.self_interaction or self.pool:
                    out = unfuse_features(out, self.fiber_out.degrees)
            else:
                # Fallback to pairwise TFN convolutions
                for degree_out in self.fiber_out.degrees:
                    out_feature = 0
                    for degree_in, feature in zip(self.fiber_in.degrees, in_features):
                        dict_key = f'{degree_in},{degree_out}'
                        basis_used = basis.get(dict_key, None)
                        out_feature = out_feature + self._try_unpad(
                            self.conv_checkpoint(
                                self.conv[dict_key], feature, invariant_edge_feats, basis_used
                            ), basis_used)
                    out[str(degree_out)] = out_feature

            for degree_out in self.fiber_out.degrees:
                if self.self_interaction and str(degree_out) in self.to_kernel_self:
                    with nvtx_range(f'self interaction'):
                        dst_features = node_feats[str(degree_out)][dst]
                        kernel_self = self.to_kernel_self[str(degree_out)]
                        out[str(degree_out)] = out[str(degree_out)] + kernel_self @ dst_features

                if self.pool:
                    with nvtx_range(f'pooling'):
                        if isinstance(out, dict):
                            out[str(degree_out)] = dgl.ops.copy_e_sum(graph, out[str(degree_out)])
                        else:
                            out = dgl.ops.copy_e_sum(graph, out)
            return out

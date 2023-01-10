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
import torch
import torch.nn as nn

from moflow.config import Config
from moflow.model.glow import Glow, GlowOnGraph

def gaussian_nll(x, mean, ln_var):
    """Computes the negative log-likelihood of a Gaussian distribution.

    Given two variable ``mean`` representing :math:`\\mu` and ``ln_var``
    representing :math:`\\log(\\sigma^2)`, this function computes in
    elementwise manner the negative log-likelihood of :math:`x` on a
    Gaussian distribution :math:`N(\\mu, S)`,

    .. math::

        -\\log N(x; \\mu, \\sigma^2) =
        \\log\\left(\\sqrt{(2\\pi)^D |S|}\\right) +
        \\frac{1}{2}(x - \\mu)^\\top S^{-1}(x - \\mu),

    where :math:`D` is a dimension of :math:`x` and :math:`S` is a diagonal
    matrix where :math:`S_{ii} = \\sigma_i^2`.

    Args:
        x: Input variable.
        mean: Mean of a Gaussian distribution, :math:`\\mu`.
        ln_var: Logarithm of variance of a Gaussian distribution,
            :math:`\\log(\\sigma^2)`.

    Returns:
        torch.Tensor:
            Negative log-likelihood.
    """

    x_prec = torch.exp(-ln_var)
    x_diff = x - mean
    x_power = (x_diff * x_diff) * x_prec * -0.5
    loss = (ln_var + math.log(2 * (math.pi))) / 2 - x_power
    return loss


class MoFlowLoss(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.b_n_type = config.num_edge_features
        self.a_n_node = config.max_num_nodes
        self.a_n_type = config.num_node_features
        self.b_size = self.a_n_node * self.a_n_node * self.b_n_type
        self.a_size = self.a_n_node * self.a_n_type

        if config.model_config.learn_dist:
            self.ln_var = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('ln_var', torch.zeros(1))

    def forward(self, h, adj_h, sum_log_det_jacs_x, sum_log_det_jacs_adj):
        z = [h, adj_h]
        logdet = [sum_log_det_jacs_x, sum_log_det_jacs_adj]

        device = z[0].device
        dtype = z[0].dtype
        z[0] = z[0].reshape(z[0].shape[0],-1)
        z[1] = z[1].reshape(z[1].shape[0], -1)

        logdet[0] = logdet[0] - self.a_size * math.log(2.)
        logdet[1] = logdet[1] - self.b_size * math.log(2.)
        ln_var_adj = self.ln_var * torch.ones([self.b_size], device=device, dtype=dtype)
        ln_var_x = self.ln_var * torch.ones([self.a_size], device=device, dtype=dtype)
        nll_adj = torch.mean(
            torch.sum(gaussian_nll(z[1], torch.zeros(self.b_size, device=device, dtype=dtype), ln_var_adj), dim=1)
            - logdet[1])
        nll_adj = nll_adj / (self.b_size * math.log(2.))  # the negative log likelihood per dim with log base 2

        nll_x = torch.mean(torch.sum(
            gaussian_nll(z[0], torch.zeros(self.a_size, device=device, dtype=dtype), ln_var_x),
            dim=1) - logdet[0])
        nll_x = nll_x / (self.a_size * math.log(2.))  # the negative log likelihood per dim with log base 2

        return nll_x, nll_adj


class MoFlow(nn.Module):
    def __init__(self, config: Config):
        super(MoFlow, self).__init__()
        self.config = config
        self.b_n_type = config.num_edge_features
        self.a_n_node = config.max_num_nodes
        self.a_n_type = config.num_node_features
        self.b_size = self.a_n_node * self.a_n_node * self.b_n_type
        self.a_size = self.a_n_node * self.a_n_type
        self.noise_scale = config.model_config.noise_scale

        self.bond_model = Glow(
            in_channel=self.b_n_type,
            n_flow=config.model_config.bond_config.n_flow,
            n_block=config.model_config.bond_config.n_block,
            squeeze_fold=config.model_config.bond_config.n_squeeze,
            hidden_channel=config.model_config.bond_config.hidden_ch,
            conv_lu=config.model_config.bond_config.conv_lu
        )

        self.atom_model = GlowOnGraph(
            n_node=self.a_n_node,
            in_dim=self.a_n_type,
            hidden_dim_dict={
                'gnn': config.model_config.atom_config.hidden_gnn,
                'linear': config.model_config.atom_config.hidden_lin
            },
            n_flow=config.model_config.atom_config.n_flow,
            n_block=config.model_config.atom_config.n_block,
            mask_row_size_list=config.model_config.atom_config.mask_row_size_list,
            mask_row_stride_list=config.model_config.atom_config.mask_row_stride_list,
        )

        self._cuda_graphs = dict()
        self.atom_stream = None
        self.bond_stream = None

    @torch.jit.ignore
    def forward(self, adj: torch.Tensor, x: torch.Tensor, with_cuda_graph: bool = False):
        """
        :param adj:  (256,4,9,9)
        :param x: (256,9,5)
        :return:
        """
        if with_cuda_graph and self.atom_stream is None:
            self.atom_stream = torch.cuda.Stream()
            self.bond_stream = torch.cuda.Stream()
        h = x
        # add uniform noise to node feature matrices
        if self.training:
            if self.noise_scale == 0:
                h = h/2.0 - 0.5 + torch.rand_like(x) * 0.4
            else:
                h = h + torch.rand_like(x) * self.noise_scale
        if with_cuda_graph:
            if self.atom_model not in self._cuda_graphs:
                h, sum_log_det_jacs_x = self._forward_graph(self.atom_model, adj, h)
            else:
                self.atom_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.atom_stream):
                    h, sum_log_det_jacs_x = self._forward_graph(self.atom_model, adj, h)
        else:
            h, sum_log_det_jacs_x = self.atom_model(adj, h)

        # add uniform noise to adjacency tensors
        if self.training:
            if self.noise_scale == 0:
                adj_bond = adj/2.0 - 0.5 + torch.rand_like(adj) * 0.4
            else:
                adj_bond = adj + torch.rand_like(adj) * self.noise_scale
        else:
            adj_bond = adj
        if with_cuda_graph:
            if self.bond_model not in self._cuda_graphs:
                adj_h, sum_log_det_jacs_adj = self._forward_graph(self.bond_model, adj_bond)
            else:
                self.bond_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.bond_stream):
                    adj_h, sum_log_det_jacs_adj = self._forward_graph(self.bond_model, adj_bond)
        else:
            adj_h, sum_log_det_jacs_adj = self.bond_model(adj_bond)
        if with_cuda_graph:
            torch.cuda.current_stream().wait_stream(self.atom_stream)
            torch.cuda.current_stream().wait_stream(self.bond_stream)
        return h, adj_h, sum_log_det_jacs_x, sum_log_det_jacs_adj

    @torch.jit.export
    def reverse(self, z):
        """
        Returns a molecule, given its latent vector.
        :param z: latent vector. Shape: [B, N*N*M + N*T]
            B = Batch size, N = number of atoms, M = number of bond types,
            T = number of atom types (Carbon, Oxygen etc.)
        :return: adjacency matrix and feature matrix of a molecule
        """
        batch_size = z.shape[0]
        z_x = z[:, :self.a_size]
        z_adj = z[:, self.a_size:]

        h_adj = z_adj.reshape(batch_size, self.b_n_type, self.a_n_node, self.a_n_node)
        h_adj = h_adj.to(memory_format=torch.channels_last)
        h_adj = self.bond_model.reverse(h_adj)

        if self.noise_scale == 0:
            h_adj = (h_adj + 0.5) * 2
        adj = h_adj
        adj = adj + adj.permute(0, 1, 3, 2)
        adj = adj / 2
        adj = adj.softmax(dim=1)
        max_bond = adj.max(dim=1).values.reshape(batch_size, -1, self.a_n_node, self.a_n_node)
        adj = torch.floor(adj / max_bond)

        adj = adj.to(memory_format=torch.channels_last)
        h_x = z_x.reshape(batch_size, self.a_n_node, self.a_n_type)
        h_x = self.atom_model.reverse((adj, h_x))
        if self.noise_scale == 0:
            h_x = (h_x + 0.5) * 2
        return adj, h_x

    @torch.jit.ignore
    def _forward_graph(self, model, *args):
        if model not in self._cuda_graphs:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            torch.cuda.synchronize()
            self._cuda_graphs[model] = torch.cuda.make_graphed_callables(
                model,
                args,
            )
            torch.cuda.synchronize()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        return self._cuda_graphs[model](*args)

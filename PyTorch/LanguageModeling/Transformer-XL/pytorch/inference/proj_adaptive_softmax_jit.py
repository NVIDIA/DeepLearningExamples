# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectedAdaptiveLogSoftmax(torch.jit.ScriptModule):
    __constants__ = ['n_clusters', 'cutoffs', 'cutoff_ends', 'keep_order']

    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                 keep_order=False):
        super().__init__()

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj

        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = type(self.cutoffs)([0]) + self.cutoffs
        self.div_val = div_val

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))

        self.out_layers = nn.ModuleList()

        if d_proj != d_embed:
            raise RuntimeError('TorchScripted module requires d_proj == d_embed')
        if div_val != 1:
            raise RuntimeError('TorchScripted module requires div_val == 1')

        self.out_layers.append(nn.Linear(d_embed, n_token))

        self.keep_order = keep_order

    @torch.jit.script_method
    def _compute_logit(self, hidden, weight, bias, proj: Optional[torch.Tensor]):
        if proj is not None:
            raise RuntimeError('TorchScripted module requires proj == None')
        logit = F.linear(hidden, weight, bias=bias)
        return logit

    @torch.jit.script_method
    def forward(self, hidden, target, keep_order: bool = False):
        '''
            hidden :: [len*bsz x d_proj]
            target :: [len*bsz]
        '''

        if hidden.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        if self.n_clusters == 0:
            for out_layer in self.out_layers:
                hidden = self._compute_logit(hidden, out_layer.weight,
                                             out_layer.bias, None)
            nll = -F.log_softmax(hidden, dim=-1) \
                    .gather(1, target.unsqueeze(1)).squeeze(1)
        else:
            # construct weights and biases
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                for out_layer in self.out_layers:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = out_layer.weight[l_idx:r_idx]
                    bias_i = out_layer.bias[l_idx:r_idx]

                    if i == 0:
                        weight_i = torch.cat(
                            [weight_i, self.cluster_weight], dim=0)
                        bias_i = torch.cat(
                            [bias_i, self.cluster_bias], dim=0)

                    weights.append(weight_i)
                    biases.append(bias_i)

            head_weight, head_bias, head_proj = weights[0], biases[0], None

            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            head_logprob = F.log_softmax(head_logit, dim=1)

            nll = torch.zeros_like(target, layout=torch.strided,
                                   dtype=hidden.dtype, device=hidden.device)

            offset = 0
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

                mask_i = (target >= l_idx) & (target < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                target_i = target.index_select(0, indices_i) - l_idx
                head_logprob_i = head_logprob.index_select(0, indices_i)

                if i == 0:
                    logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], None

                    hidden_i = hidden.index_select(0, indices_i)

                    tail_logit_i = self._compute_logit(hidden_i, weight_i,
                                                       bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)

                    logprob_i = head_logprob_i[:, -i] \
                        + tail_logprob_i.gather(1, target_i[:, None]).squeeze(1)

                if self.keep_order or keep_order:
                    nll.index_copy_(0, indices_i, -logprob_i)
                else:
                    nll[offset:offset+logprob_i.size(0)].copy_(-logprob_i)

                offset += logprob_i.size(0)

        return nll

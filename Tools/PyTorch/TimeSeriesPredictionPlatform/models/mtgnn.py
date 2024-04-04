# Copyright (c) 2024 NVIDIA CORPORATION. All rights reserved.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import pickle

from .tft_pyt.modeling import LazyEmbedding

# This is copied from torch source and adjusted to take in indices of nodes as well
class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class GraphConstructor(nn.Module):

    def __init__(self, nnodes, k, dim, alpha=3, static_feat=None):
        super().__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        #a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        # This comes from (AB^T)^T = BA^T
        m = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        a = m - m.transpose(1,0)
        #####
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = adj.new_zeros((idx.size(0), idx.size(0)))
        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
        mask.scatter_(1, t1, 1)
        adj = adj*mask
        return adj

class MixProp(nn.Module):
    def __init__(self, c_in, c_out, gdep, alpha):
        super().__init__()
        self.linear = torch.nn.Conv2d((gdep+1)*c_in, c_out, kernel_size=(1, 1))
        self.gdep = gdep
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        d = adj.sum(1)
        a = adj / d.unsqueeze(-1)
        h = x
        out = [h]
        for i in range(self.gdep):
            h = torch.einsum('ncwl,vw->ncvl', h, a)
            h = self.alpha * x + (1 - self.alpha) * h
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.linear(ho)
        return ho

class GCModule(nn.Module):

    def __init__(self, conv_channels, residual_channels, gcn_depth, propalpha):
        super().__init__()
        self.gc1 = MixProp(conv_channels, residual_channels, gcn_depth, propalpha)
        self.gc2 = MixProp(conv_channels, residual_channels, gcn_depth, propalpha)

    def forward(self, x, adj):
        x1 = self.gc1(x, adj)
        x2 = self.gc2(x, adj.transpose(1, 0))
        return x1 + x2

class DilatedInception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super().__init__()
        self.kernel_set = [2,3,6,7]
        cout = int(cout / len(self.kernel_set))
        self.tconv = nn.ModuleList([nn.Conv2d(cin, cout, (1, k), dilation=(1, dilation_factor)) for k in self.kernel_set])

    def forward(self,input):
        x = []
        for conv in self.tconv:
            x.append(conv(input))

        # This truncation is described in the paper and seemingly drops some information
        # Information drop is counteracted by padding time dimension with 0.
        # Ex: for the largest filter of size 7 input is paddded by 7 zeros.
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x,dim=1)
        return x

class TCModule(nn.Module):
    def __init__(self, residual_channels, conv_channels, dilation_factor):
        super().__init__()
        self.filter = DilatedInception(residual_channels, conv_channels, dilation_factor)
        self.gate = DilatedInception(residual_channels, conv_channels, dilation_factor)

    def forward(self, x):
        f = self.filter(x)
        f = torch.tanh(f)
        g = self.gate(x)
        g = torch.sigmoid(g)
        x = f * g
        return x


class MTGNNLayer(nn.Module):
    def __init__(self,
                 r_channels,
                 c_channels,
                 s_channels,
                 kernel_size,
                 dilation_factor,
                 dropout,
                 num_nodes,
                 use_gcn,
                 gcn_depth,
                 propalpha):
        super().__init__()
        self.use_gcn = use_gcn
        self.tc_module = TCModule(r_channels, c_channels, dilation_factor)
        self.skip_conv = nn.Conv2d(in_channels=c_channels,
                                   out_channels=s_channels,
                                   kernel_size=(1, kernel_size))
        self.dropout = nn.Dropout(dropout)
        self.ln = LayerNorm((r_channels, num_nodes, kernel_size),elementwise_affine=True)

        if use_gcn:
            self.out_module = GCModule(c_channels, r_channels, gcn_depth, propalpha)
        else:
            self.out_module = nn.Conv2d(in_channels=c_channels,
                                        out_channels=r_channels,
                                        kernel_size=(1, 1))
    def forward(self, x, idx, adp):
        residual = x
        x = self.tc_module(x)
        x = self.dropout(x)
        s = x
        s = self.skip_conv(s)
        if self.use_gcn:
            x = self.out_module(x, adp)
        else:
            x = self.out_module(x)

        x = x + residual[:, :, :, -x.size(3):]
        x = self.ln(x,idx)

        return x, s


class MTGNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_gcn = config.use_gcn
        self.gcn_depth = config.gcn_depth
        self.predefined_adj = config.get('predefined_adj')
        if self.predefined_adj is not None:
            A = pickle.load(open(self.predefined_adj, 'rb'))
            self.register_buffer('predefined_adj', A)
        self.propalpha = config.propalpha
        self.tanhalpha = config.tanhalpha

        self.num_nodes = config.num_nodes
        self.dropout = config.dropout
        self.in_dim = config.in_dim
        self.out_dim = config.example_length - config.encoder_length
        self.residual_channels = config.residual_channels
        self.conv_channels = config.conv_channels
        self.skip_channels = config.skip_channels
        self.end_channels = config.end_channels
        self.subgraph_size = config.subgraph_size
        self.node_dim = config.node_dim
        self.dilation_exponential = config.dilation_exponential
        self.seq_length = config.encoder_length
        self.num_layers = config.num_layers
        self.use_embedding = config.use_embedding

        ### New embedding
        if self.use_embedding:
            self.config.hidden_size = self.config.in_dim
            self.embedding = LazyEmbedding(self.config)

        self.include_static_data = config.include_static_data
        ####


        self.layers = nn.ModuleList()
        self.start_conv = nn.LazyConv2d(out_channels=self.residual_channels, kernel_size=(1, 1))
        self.gc = GraphConstructor(self.num_nodes, self.subgraph_size, self.node_dim, alpha=self.tanhalpha)

        kernel_size = 7

        def rf_size(c,q,m):
            assert q >= 1
            if q > 1:
                return int(1 + (c-1)*(q**m - 1)/(q-1))
            return m*(c-1) + 1

        self.receptive_field = rf_size(kernel_size, self.dilation_exponential, self.num_layers)
        new_dilation = 1
        for j in range(self.num_layers):
            rfs = rf_size(kernel_size, self.dilation_exponential, j+1) 
            kernel_len = max(self.seq_length, self.receptive_field) - rfs + 1

            self.layers.append(MTGNNLayer(self.residual_channels, self.conv_channels, self.skip_channels,
                                          kernel_len, new_dilation, self.dropout, self.num_nodes, self.use_gcn,
                                          self.gcn_depth, self.propalpha
                                          )
                               )

            new_dilation *= self.dilation_exponential

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=self.end_channels,
                                    kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.out_dim,
                                    kernel_size=(1, 1))

        if self.seq_length > self.receptive_field:
            self.skip0 = nn.LazyConv2d(out_channels=self.skip_channels, kernel_size=(1, self.seq_length))
            self.skipE = nn.Conv2d(in_channels=self.residual_channels,
                                   out_channels=self.skip_channels,
                                   kernel_size=(1, self.seq_length - self.receptive_field + 1)
                                   )

        else:
            self.skip0 = nn.LazyConv2d(out_channels=self.skip_channels, kernel_size=(1, self.receptive_field))
            self.skipE = nn.Conv2d(in_channels=self.residual_channels, out_channels=self.skip_channels, kernel_size=(1, 1))

        idx = torch.arange(self.num_nodes)
        self.register_buffer('idx', idx)

    def forward(self, batch, idx=None):
        if self.use_embedding:
            batch = {k: v[:, :self.seq_length] if v is not None else None for k, v in batch.items()}
            emb = self.embedding(batch)
            emb = [e.view(*e.shape[:-2], -1) for e in emb if e is not None]
            emb[0] = emb[0].unsqueeze(1).expand(emb[0].shape[0], self.seq_length, *emb[0].shape[1:])
            if not self.include_static_data:
                emb = emb[1:]
            input = torch.cat(emb, dim=-1).transpose(1, 3)
        else:

            # TSPP compatibility code
            t = batch['k_cont'][:, :self.seq_length, 0, 2:]
            t = torch.einsum('btk,k->bt', t, t.new([1, 0.16]))
            t = t.unsqueeze(-1).expand(*t.shape, self.num_nodes)
            target = batch['target'][:, :self.seq_length].squeeze(-1)
            input = torch.stack((target, t), dim=1).transpose(2, 3)
            ####

        seq_len = input.size(3)
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'
        if idx is None:
            idx = self.idx

        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))

        if self.use_gcn:
            if not self.predefined_adj:
                adp = self.gc(idx)
            else:
                adp = self.predefined_adj

        x = self.start_conv(input)  # 1x1 conv for upscaling. Acts like a linear procection on 1 dim
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for layer in self.layers:
            x, s = layer(x, idx, adp)
            skip = skip + s

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

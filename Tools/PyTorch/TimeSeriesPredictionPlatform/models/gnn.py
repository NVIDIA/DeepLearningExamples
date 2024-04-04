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
from typing import Optional, Tuple, Dict
from torch import Tensor
import dgl
from dgl.nn.pytorch.conv import GraphConv

import networkx as nx
import numpy as np
from copy import copy

def list_contract_nodes_(graph, l_nodes):
    """
    l_nodes: List[List[Int]]: nodes to merge
    Returns node mapping
    """
    pooled_feat = []
    _nodes_flat = [x for y in l_nodes for x in y]

    _unmerged_nodes = list(range(graph.num_nodes()))
    for n in _nodes_flat:
        _unmerged_nodes.remove(n)

    node_mapping = {i:[n] for i,n in enumerate(_unmerged_nodes)}
    num_nodes = graph.num_nodes()
    i = 0
    while l_nodes:
        nodes = l_nodes.pop()
        # Add features
        ndata = {k:v[nodes].mean() for k, v in graph.ndata.items()}
        pooled_feat.append({k: v[nodes].mean(dim=0) for k,v in graph.ndata.items()})
        # Add edges
        predecessors = torch.cat([graph.predecessors(n) for n in nodes])
        successors = torch.cat([graph.successors(n) for n in nodes])
        nidx = graph.num_nodes()
        graph.add_edges(torch.full_like(predecessors, nidx), predecessors)
        graph.add_edges(torch.full_like(successors, nidx), successors)
        # Add key to super node mapping
        node_mapping[num_nodes - len(_nodes_flat) + i] = nodes
        i += 1

    graph.remove_nodes(_nodes_flat)

    # Insert pooled features
    pooled_feat = {k: torch.stack([d[k] for d in pooled_feat], dim=0) for k in graph.ndata.keys()}
    for k, v in pooled_feat.items():
        graph.ndata[k][-v.shape[0]:] = v

    return graph, node_mapping

def coarsen(graph):
    g_nx = graph.cpu().to_networkx().to_undirected()
    g_nx = nx.Graph(g_nx)
    matching = nx.algorithms.matching.max_weight_matching(g_nx)
    matching = [list(x) for x in matching]
    g, s_node_map = list_contract_nodes_(graph, matching)
    return g, s_node_map

class SpatialPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.s_node_map = None
        self.cached_graph = None
        self.ukey = f'feat_{id(self)}'

    def forward(self, graph, feat):
        self.cached_graph = graph
        _graph = copy(graph)
        _graph.ndata[self.ukey] = feat
        g, s_node_map = coarsen(_graph)
        self.s_node_map = s_node_map
        return g, g.ndata[self.ukey]

    def unpool(self, feat):
        """ Unpools by copying values"""
        _feat = []
        for k,v in self.s_node_map.items():
            for node in v:
                _feat.append((node, feat[k]))
        u_feat = torch.stack([t[1] for t in sorted(_feat, key=lambda x: x[0])])
        return self.cached_graph, u_feat

class TFTEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.s_cat_inp_lens    = config.static_categorical_inp_lens
        self.t_cat_k_inp_lens  = config.temporal_known_categorical_inp_lens
        self.t_cat_o_inp_lens  = config.temporal_observed_categorical_inp_lens
        self.s_cont_inp_size   = config.static_continuous_inp_size
        self.t_cont_k_inp_size = config.temporal_known_continuous_inp_size
        self.t_cont_o_inp_size = config.temporal_observed_continuous_inp_size
        self.t_tgt_size        = config.temporal_target_size

        self.hidden_size = config.hidden_size

        # There are 7 types of input:
        # 1. Static categorical
        # 2. Static continuous
        # 3. Temporal known a priori categorical
        # 4. Temporal known a priori continuous
        # 5. Temporal observed categorical
        # 6. Temporal observed continuous
        # 7. Temporal observed targets (time series obseved so far)

        self.s_cat_embed = nn.ModuleList([
            nn.Embedding(n, self.hidden_size) for n in self.s_cat_inp_lens]) if self.s_cat_inp_lens else None
        self.t_cat_k_embed = nn.ModuleList([
            nn.Embedding(n, self.hidden_size) for n in self.t_cat_k_inp_lens]) if self.t_cat_k_inp_lens else None
        self.t_cat_o_embed = nn.ModuleList([
            nn.Embedding(n, self.hidden_size) for n in self.t_cat_o_inp_lens]) if self.t_cat_o_inp_lens else None

        self.s_cont_embedding_vectors = nn.Parameter(torch.Tensor(self.s_cont_inp_size, self.hidden_size)) if self.s_cont_inp_size else None
        self.t_cont_k_embedding_vectors = nn.Parameter(torch.Tensor(self.t_cont_k_inp_size, self.hidden_size)) if self.t_cont_k_inp_size else None
        self.t_cont_o_embedding_vectors = nn.Parameter(torch.Tensor(self.t_cont_o_inp_size, self.hidden_size)) if self.t_cont_o_inp_size else None
        self.t_tgt_embedding_vectors = nn.Parameter(torch.Tensor(self.t_tgt_size, self.hidden_size))

        self.s_cont_embedding_bias = nn.Parameter(torch.zeros(self.s_cont_inp_size, self.hidden_size)) if self.s_cont_inp_size else None
        self.t_cont_k_embedding_bias = nn.Parameter(torch.zeros(self.t_cont_k_inp_size, self.hidden_size)) if self.t_cont_k_inp_size else None
        self.t_cont_o_embedding_bias = nn.Parameter(torch.zeros(self.t_cont_o_inp_size, self.hidden_size)) if self.t_cont_o_inp_size else None
        self.t_tgt_embedding_bias = nn.Parameter(torch.zeros(self.t_tgt_size, self.hidden_size))

        if self.s_cont_embedding_vectors is not None:
            torch.nn.init.xavier_normal_(self.s_cont_embedding_vectors)
        if self.t_cont_k_embedding_vectors is not None:
            torch.nn.init.xavier_normal_(self.t_cont_k_embedding_vectors)
        if self.t_cont_o_embedding_vectors is not None:
            torch.nn.init.xavier_normal_(self.t_cont_o_embedding_vectors)
        torch.nn.init.xavier_normal_(self.t_tgt_embedding_vectors)

    def _apply_embedding(self,
            cat: Optional[Tensor],
            cont: Optional[Tensor],
            cat_emb: Optional[nn.ModuleList], 
            cont_emb: Tensor,
            cont_bias: Tensor,
            ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        e_cat = torch.stack([embed(cat[...,i]) for i, embed in enumerate(cat_emb)], dim=-2) if cat is not None else None
        if cont is not None:
            #the line below is equivalent to following einsums
            #e_cont = torch.einsum('btf,fh->bthf', cont, cont_emb)
            #e_cont = torch.einsum('bf,fh->bhf', cont, cont_emb)
            e_cont = torch.mul(cont.unsqueeze(-1), cont_emb)
            e_cont = e_cont + cont_bias
        else:
            e_cont = None

        if e_cat is not None and e_cont is not None:
            return torch.cat([e_cat, e_cont], dim=-2)
        elif e_cat is not None:
            return e_cat
        elif e_cont is not None:
            return e_cont
        else:
            return None

    def forward(self, x: Dict[str, Tensor]):
        # temporal/static categorical/continuous known/observed input 
        x = {k:v for k,v in x.items() if v.numel()}
        s_cat_inp = x.get('s_cat', None)
        s_cont_inp = x.get('s_cont', None)
        t_cat_k_inp = x.get('k_cat', None)
        t_cont_k_inp = x.get('k_cont', None)
        t_cat_o_inp = x.get('o_cat', None)
        t_cont_o_inp = x.get('o_cont', None)
        t_tgt_obs = x['target'] # Has to be present

        # Static inputs are expected to be equal for all timesteps
        # For memory efficiency there is no assert statement
        s_cat_inp = s_cat_inp[:,0,:] if s_cat_inp is not None else None
        s_cont_inp = s_cont_inp[:,0,:] if s_cont_inp is not None else None

        s_inp = self._apply_embedding(s_cat_inp,
                                      s_cont_inp,
                                      self.s_cat_embed,
                                      self.s_cont_embedding_vectors,
                                      self.s_cont_embedding_bias)
        t_known_inp = self._apply_embedding(t_cat_k_inp,
                                            t_cont_k_inp,
                                            self.t_cat_k_embed,
                                            self.t_cont_k_embedding_vectors,
                                            self.t_cont_k_embedding_bias)
        t_observed_inp = self._apply_embedding(t_cat_o_inp,
                                               t_cont_o_inp,
                                               self.t_cat_o_embed,
                                               self.t_cont_o_embedding_vectors,
                                               self.t_cont_o_embedding_bias)

        # Temporal observed targets
        # t_observed_tgt = torch.einsum('btf,fh->btfh', t_tgt_obs, self.t_tgt_embedding_vectors)
        t_observed_tgt = torch.matmul(t_tgt_obs.unsqueeze(3).unsqueeze(4), self.t_tgt_embedding_vectors.unsqueeze(1)).squeeze(3)
        t_observed_tgt = t_observed_tgt + self.t_tgt_embedding_bias

        return s_inp, t_known_inp, t_observed_inp, t_observed_tgt

class GCGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.conv_i = GraphConv(input_size, 3 * hidden_size) #According to https://arxiv.org/pdf/1903.05631.pdf
        self.conv_h = GraphConv(hidden_size, 3 * hidden_size) # this should be ChebConv
        self.hidden_size = hidden_size
        self.state = None
    def forward(self, graph, feat, hx):
        i = self.conv_i(graph, feat)
        h = self.conv_h(graph, hx)
        i_r, i_z, i_n = torch.chunk(i, 3, dim=-1)
        h_r, h_z, h_n = torch.chunk(h, 3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h = (1-z) * n + z * hx
        
        return h
    
class GCGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        cells = [GCGRUCell(input_size, hidden_size)]
        cells += [GCGRUCell(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        self.cells = nn.ModuleList(cells)
        
    def forward(self, graph, input, hx=None):
        if hx is None:
            hx = [torch.zeros(graph.num_nodes(), self.hidden_size,
                    dtype=input.dtype, device=input.device)] * self.num_layers
        
 
        out = []
        states = []
        intermediate = [input[:,t,...] for t in range(input.shape[1])]

        for i, cell in enumerate(self.cells):
            inner_out = []
            h = hx[i]
            for x in intermediate:
                h = cell(graph, x, h)
                inner_out.append(h)
            out.append(inner_out)
            intermediate = inner_out

        output = torch.stack(out[-1], dim=1)

        return output, out[-1]
                

class ToyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_steps = config.encoder_length
        self.num_future_vars = config.num_future_vars
        self.num_historic_vars = config.num_historic_vars
        self.num_static_vars = config.num_static_vars
        self.hidden_size = config.hidden_size
        self.embedding = TFTEmbedding(config)

        self.static_proj = nn.Linear(config.hidden_size * self.num_static_vars, config.num_layers * config.hidden_size)
        self.history_recurrent = GCGRU(config.hidden_size, config.hidden_size, config.num_layers)
        self.future_recurrent = GCGRU(config.hidden_size, config.hidden_size, config.num_layers)
        self.history_down_proj = nn.Linear(self.num_historic_vars * config.hidden_size, config.hidden_size)
        self.future_down_proj = nn.Linear(self.num_future_vars * config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, graph):
        s_inp, t_known_inp, t_observed_inp, t_observed_tgt = self.embedding(graph.ndata)
        s_inp = s_inp.view(s_inp.shape[0], -1)
        init_state = self.static_proj(s_inp)
        init_state = init_state.view(init_state.shape[0], -1, self.hidden_size).transpose(0,1)

        feat = torch.cat([t_known_inp, t_observed_inp, t_observed_tgt], dim=2)
        historic_feat = feat[:,:self.encoder_steps,:]
        historic_feat = historic_feat.view(historic_feat.shape[0], historic_feat.shape[1], -1)
        historic_feat = self.history_down_proj(historic_feat)
        history, state = self.history_recurrent(graph, historic_feat, hx=init_state)
        
        future_feat = t_known_inp[:,self.encoder_steps:, :]
        future_feat = future_feat.view(future_feat.shape[0], future_feat.shape[1], -1)
        future_feat = self.future_down_proj(future_feat)
        future, _ = self.future_recurrent(graph, future_feat, hx=state)
        out = self.out_proj(future)

        return out

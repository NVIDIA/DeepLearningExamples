# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNEC(nn.Module):

    @staticmethod
    def add_args(parser):
        return parser

    def __init__(
        self, in_dim, hidden_dim, out_dim, num_classes, n_layers, **kwargs
    ):
        super().__init__()
        self.gcn = StochasticLayerGCN(in_dim, hidden_dim, out_dim, n_layers)
        self.predictor = ScorePredictor(num_classes, out_dim)

    def forward(self, blocks, edge_subgraph, input_features, *args, **kwargs):
        x = self.gcn(blocks, input_features)
        return self.predictor(edge_subgraph, x)

    def loss(self, pred, label):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred, label
        )
        return loss


class StochasticLayerGCN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, n_layers):
        super().__init__()
        self.layers = []

        if n_layers <= 1:
            self.layers.append(dglnn.GraphConv(in_feats, out_feats))
        else:
            self.layers.append(dglnn.GraphConv(in_feats, h_feats))
            for _ in range(n_layers - 2):
                self.layers.append(dglnn.GraphConv(h_feats, h_feats))
            self.layers.append(dglnn.GraphConv(h_feats, out_feats))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, blocks, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(blocks[i], x))
        return x


class ScorePredictor(nn.Module):
    def __init__(self, num_classes, in_feats):
        super().__init__()
        self.W = nn.Linear(2 * in_feats, num_classes)

    def apply_edges(self, edges):
        data = torch.cat([edges.src["x"], edges.dst["x"]], dim=1)
        return {"score": self.W(data)}

    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata["x"] = x
            edge_subgraph.apply_edges(self.apply_edges)
            return edge_subgraph.edata["score"]

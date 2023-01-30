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
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
from syngen.benchmark.models.layers.gat_layers import (
    CustomGATLayer,
    CustomGATLayerEdgeReprFeat,
    CustomGATLayerIsotropic,
    GATLayer,
)
from syngen.benchmark.models.layers.score_predictor import ScorePredictor


class GATEC(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--in-feat-dropout",
            type=float,
            default=0.1,
            help="input feature dropout (default: 0.1)",
        )
        parser.add_argument(
            "--dropout", type=float, default=0.1, help="dropout (default: 0.1)"
        )
        parser.add_argument("--batch_norm", action="store_true", default=False)
        parser.add_argument("--n-heads", type=int, default=2)
        parser.add_argument("--layer-type", type=str, default="dgl")
        parser.add_argument("--residual", action="store_true", default=False)
        parser.add_argument("--edge_feat", action="store_true", default=False)

    def __init__(
        self,
        in_dim,
        in_dim_edge,
        hidden_dim,
        out_dim,
        num_classes,
        n_heads,
        in_feat_dropout,
        dropout,
        n_layers,
        readout=False,
        edge_feat=False,
        batch_norm=False,
        residual=False,
        layer_type="dgl",
        device="cuda",
        **kwargs,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.in_dim_edge = in_dim_edge
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.n_heads = n_heads
        self.dropout = dropout
        self.n_layers = n_layers
        self.readout = readout
        self.batch_norm = batch_norm
        self.residual = residual
        self.device = device

        self.layer_type = {
            "dgl": GATLayer,
            "edgereprfeat": CustomGATLayerEdgeReprFeat,
            "edgefeat": CustomGATLayer,
            "isotropic": CustomGATLayerIsotropic,
        }.get(layer_type, GATLayer)

        self.embedding_h = nn.Linear(
            self.in_dim, self.hidden_dim * self.n_heads
        )

        if self.layer_type != GATLayer:
            self.edge_feat = edge_feat
            self.embedding_e = nn.Linear(
                self.in_dim_edge, self.hidden_dim * self.n_heads
            )

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList(
            [
                self.layer_type(
                    self.hidden_dim * self.n_heads,
                    self.hidden_dim,
                    self.n_heads,
                    self.dropout,
                    self.batch_norm,
                    self.residual,
                )
                for _ in range(n_layers - 1)
            ]
        )
        self.layers.append(
            self.layer_type(
                self.hidden_dim * self.n_heads,
                self.out_dim,
                1,
                self.dropout,
                self.batch_norm,
                self.residual,
            )
        )
        self.edge_score = ScorePredictor(2 * out_dim, num_classes)

    def forward(
        self,
        blocks,
        edge_subgraph,
        input_features,
        edge_features,
        *args,
        **kwargs,
    ):
        h = self.embedding_h(input_features.float())
        h = self.in_feat_dropout(h)
        if self.layer_type == GATLayer:
            for idx, conv in enumerate(self.layers):
                h = conv(blocks[idx], h)
        else:
            if not self.edge_feat:
                e = torch.ones_like(edge_features).to(self.device)
            e = self.embedding_e(edge_features.float())

            for idx, conv in enumerate(self.layers):
                h, e = conv(blocks[idx], h, e)
        edge_subgraph.ndata["h"] = h

        def _edge_feat(edges):
            e = torch.cat([edges.src["h"], edges.dst["h"]], dim=1)
            e = self.edge_score(e)
            return {"e": e}

        edge_subgraph.apply_edges(_edge_feat)

        return edge_subgraph.edata["e"]

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss(weight=None)
        loss = criterion(pred, label)

        return loss

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

from typing import Optional

import dgl
import numpy as np
import torch

from syngen.utils.types import DataFrameType, MetaData

from .base_dataset import BaseDataset


class EdgeDS(BaseDataset):
    """
    lean DGL graph builder for edge classification,
    """

    def __init__(
        self,
        target_col: str = "label",
        add_reverse: bool = True,
        train_ratio: float = 0.8,
        test_ratio: float = 0.1,
        val_ratio: float = 0.1,
        **kwargs,
    ):

        self.target_col = target_col
        self.add_reverse = add_reverse
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio

    def get_graph(
        self,
        edge_data: DataFrameType,
        graph_info: dict,
        *,
        node_data: Optional[DataFrameType] = None,
    ):

        src_name = graph_info[MetaData.EDGE_DATA][MetaData.SRC_NAME]
        dst_name = graph_info[MetaData.EDGE_DATA][MetaData.DST_NAME]
        edge_data = edge_data.fillna(0)
        src_nodes = edge_data[src_name]
        dst_nodes = edge_data[dst_name]

        # - construct dgl graph
        g = dgl.graph((src_nodes, dst_nodes))

        if node_data is None:
            # - random features
            g.ndata["feat"] = torch.rand((g.num_nodes(), 32))
        else:
            node_feature_columns = graph_info[MetaData.NODE_DATA].get(
                MetaData.CONTINUOUS_COLUMNS, []
            ) + graph_info[MetaData.NODE_DATA].get(
                MetaData.CATEGORICAL_COLUMNS, []
            )
            node_feature_columns = list(set(node_feature_columns) - set(
                [graph_info[MetaData.NODE_ID]]))
            node_features = node_data[node_feature_columns].values
            node_features = torch.Tensor(node_features)
            g.ndata["feat"] = node_features

        edges = edge_data[[src_name, dst_name]].values

        if self.add_reverse:
            # - add reverse edges
            edge_reverse = np.zeros_like(edges)
            edge_reverse[:, 0] = edges[:, 1]
            edge_reverse[:, 1] = edges[:, 0]
            g.add_edges(list(edge_reverse[:, 0]), list(edge_reverse[:, 1]))

        num_rows = len(edge_data)
        num_edges = g.num_edges()
        # - extract edge features + labels
        feature_cols = list(set(edge_data.columns) - set(
            [src_name, dst_name, self.target_col]))
        feature_cols = list(feature_cols)
        features = edge_data[feature_cols].values
        labels = edge_data[self.target_col].values
        if num_rows == num_edges // 2:
            # - add reverse features
            features = np.concatenate([features, features], axis=0)
            # - add reverse labels
            labels = np.concatenate([labels, labels], axis=0)

        # - add edge data
        g.edata["feat"] = torch.Tensor(features)
        g.edata["labels"] = torch.Tensor(labels)

        # - dataset split
        num_train = int(self.train_ratio * num_edges)
        num_val = int(self.val_ratio * num_edges)
        num_test = int(self.test_ratio * num_edges)

        masks = torch.randperm(len(features))
        train_idx = masks[:num_train]
        val_idx = masks[num_train : num_train + num_val]
        test_idx = masks[num_train + num_val : num_train + num_val + num_test]

        train_mask = torch.zeros(len(features), dtype=torch.bool)
        train_mask[train_idx] = True

        val_mask = torch.zeros(len(features), dtype=torch.bool)
        val_mask[val_idx] = True

        test_mask = torch.zeros(len(features), dtype=torch.bool)
        test_mask[test_idx] = True

        g.edata["train_mask"] = train_mask
        g.edata["val_mask"] = val_mask
        g.edata["test_mask"] = test_mask

        edge_eids = np.arange(0, len(edges))
        return g, edge_eids

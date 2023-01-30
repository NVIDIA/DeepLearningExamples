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

from __future__ import annotations

from .str_enum import StrEnum


class MetaData(StrEnum):
    DATA = "data"
    EDGE = "edge"
    GEN_EDGE = "gen_edge"
    GEN_NODE = "gen_node"
    NODE = "node"
    NODE_DATA = "node_data"
    NODE_ID = "nid"
    NODE_IDS = "nids"
    EDGE_DATA = "edge_data"
    EDGE_LIST = "edge_list"
    EDGE_ID = "eid"
    SRC = "src"
    SRC_NAME = "src_name"
    SRC_COLUMNS = "src_columns"
    DST = "dst"
    DST_NAME = "dst_name"
    DST_COLUMNS = "dst_columns"
    NODE_NAME = "node_name"
    NODE_COLUMNS = "node_columns"
    LABELS = "labels"
    NODE_FEAT = "node_feat"
    EDGE_FEAT = "edge_feat"
    TRAIN_MASK = "train_mask"
    VAL_MASK = "val_mask"
    TEST_MASK = "test_mask"
    CONTINUOUS_COLUMNS = "continuous_columns"
    CATEGORICAL_COLUMNS = "categorical_columns"
    UNDIRECTED = "undirected"
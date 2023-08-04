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
    PATH = "path"
    EDGES = "edges"
    NODES = "nodes"
    ALIGNERS = "[gen]aligners"

    GRAPHS = "graphs"

    NAME = "name"
    COUNT = "count"

    NODE_DATA = "node_data"
    EDGE_DATA = "edge_data"
    TYPE = "type"
    DTYPE = "dtype"
    SRC = "src"
    SRC_NAME = "src_name"
    SRC_NODE_TYPE = "src_node_type"

    DST = "dst"
    DST_NAME = "dst_name"
    DST_NODE_TYPE = "dst_node_type"

    NODE_NAME = "node_name"
    NODE_COLUMNS = "node_columns"
    EDGE_NAME = "edge_name"
    LABELS = "labels"
    FEATURES = "features"
    FEATURES_PATH = "features_path"
    FEATURES_DATA = "features_data"
    FEATURE_TYPE = "feature_type"
    FEATURE_FILE = "feature_file"
    FILENAME_PREFIX = "filename_prefix"
    STRUCTURE_PATH = "structure_path"
    STRUCTURE_DATA = "structure_data"

    NODE_FEAT = "node_feat"
    EDGE_FEAT = "edge_feat"
    TRAIN_MASK = "train_mask"
    VAL_MASK = "val_mask"
    TEST_MASK = "test_mask"

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"

    CONTINUOUS_COLUMNS = "continuous_columns"
    CATEGORICAL_COLUMNS = "categorical_columns"
    UNDIRECTED = "undirected"
    DIRECTED = "directed"

    # generation related keys
    STRUCTURE_GENERATOR = "[gen]structure_generator"
    TABULAR_GENERATORS = "[gen]tabular_generators"
    DATA_SOURCE = "data_source"
    FEATURES_LIST = "features_list"
    PARAMS = "params"
    DUMP_PATH = "dump_path"


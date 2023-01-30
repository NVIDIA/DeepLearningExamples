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
import pickle
from typing import Dict

import cupy
import dask_cudf
import numpy as np
import pandas as pd

from syngen.graph_aligner.base_graph_aligner import BaseGraphAligner
from syngen.utils.types import DataFrameType, MetaData
from syngen.utils.utils import df_to_cudf, df_to_dask_cudf, df_to_pandas


class RandomAligner(BaseGraphAligner):
    def __init__(self, **kwargs):
        super(BaseGraphAligner).__init__()
        self.edge_columns = []
        self.node_columns = []
        self.node_id_col = None

    def fit(
        self,
        data: Dict[str, DataFrameType],
        src_col: str,
        dst_col: str,
        node_id_col: str = "id",
        *args,
        **kwargs,
    ):
        self.node_id_col = node_id_col
        if data.get(MetaData.EDGE_DATA, None) is not None:
            self.edge_columns = list(
                set(data[MetaData.EDGE_DATA].columns) - {src_col, dst_col}
            )
        if data.get(MetaData.NODE_DATA, None) is not None:
            self.node_columns = list(
                set(data[MetaData.NODE_DATA].columns) - {node_id_col}
            )

    def align(
        self,
        data: Dict[str, DataFrameType],
        src_col: str,
        dst_col: str,
        node_id_col: str = "id",
        batch_size: int = 1000,
    ) -> Dict[str, DataFrameType]:
        """ Align given features onto graph defined in `data[MetaData.EDGE_LIST]`

        Args:
            data (Dict[str, DataFrameType]): dictionary containing graph edge list and edge/node
            features to align. Each stored in `MetaData.EDGE_LIST`, `MetaData.EDGE_DATA`,
            `MetaData.NODE_DATA` correspondingly.
            src_col (str): source column in `MetaData.EDGE_LIST`
            dst_col (str): destination column in `MetaData.EDGE_LIST`

        """

        edge_list = data[MetaData.EDGE_LIST]
        edge_data = edge_list
        node_data = None
        if data.get(MetaData.EDGE_DATA, None) is not None:
            edge_data = data[MetaData.EDGE_DATA]
            edge_data = self.align_edge(
                edge_list, edge_data, src_col, dst_col, batch_size
            )

        if data.get(MetaData.NODE_DATA, None) is not None:
            node_data = data[MetaData.NODE_DATA]
            node_data = self.align_node(
                edge_list, node_data, src_col, dst_col, batch_size
            )
        return {MetaData.EDGE_DATA: edge_data, MetaData.NODE_DATA: node_data}

    def align_edge(
        self,
        edges: DataFrameType,
        features: DataFrameType,
        src_col: str,
        dst_col: str,
        batch_size: int = 1000,
    ) -> pd.DataFrame:
        assert len(features) >= len(
            edges
        ), "generated tabular data must be greater or \
                equal to the number of edges in dest_data"

        # - load generated el into cudf
        generated_el_df = df_to_pandas(edges)

        rand_idxs = np.random.choice(
            np.arange(len(features)), size=len(generated_el_df)
        )

        features = df_to_pandas(features)
        columns = list(self.edge_columns)
        features = features.iloc[rand_idxs][columns].reset_index(drop=True)
        overlayed_df = pd.concat([generated_el_df, features], axis=1)
        return overlayed_df

    def align_node(
        self,
        edges: dask_cudf.DataFrame,
        features: dask_cudf.DataFrame,
        src_col: str,
        dst_col: str,
        batch_size: int = 1000,
    ) -> pd.DataFrame:
        # - load generated el into cudf
        generated_el_df = df_to_pandas(edges)
        vertices = set(
            generated_el_df[src_col].values.tolist()
            + generated_el_df[dst_col].values.tolist()
        )
        vertices = list(vertices)
        assert len(vertices) <= len(
            features
        ), "generated tabular data must be greater or \
                equal to the number of nodes in dest_data"
        rand_idxs = np.random.choice(
            np.arange(len(features)), size=len(vertices)
        )
        features = df_to_pandas(features)
        columns = list(self.node_columns)
        features = features.iloc[rand_idxs][columns].reset_index(drop=True)
        features[self.node_id_col] = vertices
        data = features
        return data

    def save(self, path):
        with open(path, 'wb') as file_handler:
            pickle.dump(self, file_handler, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as file_handler:
            model = pickle.load(file_handler)
        return model

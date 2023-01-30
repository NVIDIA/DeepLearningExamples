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

import logging

import cudf
import dask_cudf
import pandas as pd

from syngen.preprocessing.base_preprocessing import BasePreprocessing
from syngen.utils.types import MetaData

logger = logging.getLogger(__name__)
log = logger


class PaysimPreprocessing(BasePreprocessing):
    """Preprocessing for https://www.kaggle.com/datasets/ealaxi/paysim1

        Args:
            cached (bool): skip preprocessing and use cached files
            nrows (int): number of rows to load from dataframe
            drop_cols (list): columns to drop from dataframe
    """

    def __init__(
        self,
        cached: bool = False,
        nrows: int = None,
        drop_cols: list = ["step"],
        **kwargs,
    ):
        super(PaysimPreprocessing, self).__init__(cached, nrows, drop_cols)

        self.graph_info = {
            MetaData.EDGE_DATA: {
                MetaData.SRC_NAME: "nameOrig",
                MetaData.SRC_COLUMNS: ["nameOrig"],
                MetaData.DST_NAME: "nameDest",
                MetaData.DST_COLUMNS: ["nameDest"],
            },
            MetaData.UNDIRECTED: True,
        }

        self.graph_info[MetaData.EDGE_DATA][MetaData.CATEGORICAL_COLUMNS] = [
            "type",
            "nameOrig",
            "nameDest",
            "isFraud",
            "isFlaggedFraud",
        ]

        self.graph_info[MetaData.EDGE_DATA][MetaData.CONTINUOUS_COLUMNS] = [
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
        ]

    def transform_graph(self, data) -> pd.DataFrame:
        """ Preprocess data into graph
        """
        data = self.add_graph_edge_cols(data)
        continuous_columns = [
            c
            for c in data.columns
            if c
            in self.graph_info[MetaData.EDGE_DATA][MetaData.CONTINUOUS_COLUMNS]
        ]
        categorical_columns = [
            c
            for c in data.columns
            if c
            in self.graph_info[MetaData.EDGE_DATA][
                MetaData.CATEGORICAL_COLUMNS
            ]
        ]

        columns_to_select = categorical_columns + continuous_columns
        for col in categorical_columns:
            data[col] = data[col].astype("category").cat.codes
            data[col] = data[col].astype(int)

        # - bipartite
        src_name = self.graph_info[MetaData.EDGE_DATA][MetaData.SRC_NAME]
        dst_name = self.graph_info[MetaData.EDGE_DATA][MetaData.DST_NAME]
        src_ids = data[src_name].unique()
        data[dst_name] = data[dst_name].astype(int) + len(src_ids)
        data = data[columns_to_select]
        return {MetaData.EDGE_DATA: data}

    def inverse_transform(self, data):
        return data

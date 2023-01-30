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
import math
import os
from os import path
from pathlib import Path, PosixPath
from typing import List, Union

import cudf
import dask_cudf
import pandas as pd

from syngen.preprocessing.base_preprocessing import BasePreprocessing
from syngen.utils import write_csv
from syngen.utils.types import MetaData

logger = logging.getLogger(__name__)
log = logger


class RatingsPreprocessing(BasePreprocessing):
    def __init__(
        self, cached: bool = True, nrows: int = None, drop_cols: list = [],
    ):
        """
        preprocessing for http://www.trustlet.org/downloaded_epinions.html

        Args:
            user_ids: (List[int])
                 list of users to filter dataframe
            cached: (bool)
                 skip preprocessing and use cached files
            data_path: (str)
                 path to file containing data
            nrows: (int)
                 number of rows to load from dataframe
        """
        super().__init__(cached, nrows, drop_cols)
        self.graph_info = {
            MetaData.EDGE_DATA: {
                MetaData.SRC_NAME: "user_id",
                MetaData.SRC_COLUMNS: ["user_id"],
                MetaData.DST_NAME: "item_id",
                MetaData.DST_COLUMNS: ["item_id"],
            },
            MetaData.UNDIRECTED: True,
        }

        self.graph_info[MetaData.EDGE_DATA][MetaData.CONTINUOUS_COLUMNS] = []
        self.graph_info[MetaData.EDGE_DATA][MetaData.CATEGORICAL_COLUMNS] = [
            "rating",
            "user_id",
            "item_id",
        ]

    def transform_graph(self, data) -> pd.DataFrame:
        """ Preprocess data into graph
        """
        data = self.add_graph_edge_cols(data)
        data = data.fillna(0)
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

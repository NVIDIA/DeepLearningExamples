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


class CreditPreprocessing(BasePreprocessing):
    def __init__(
        self,
        cached: bool = False,
        nrows: int = None,
        drop_cols: list = [
            "Unnamed: 0",
            "trans_date_trans_time",
            "trans_num",
            "dob",
        ],
        **kwargs,
    ):
        """
        preprocessing for https://www.kaggle.com/datasets/kartik2112/fraud-detection

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
                MetaData.SRC_NAME: "user",
                MetaData.SRC_COLUMNS: ["first", "last"],
                MetaData.DST_NAME: "merchant",
                MetaData.DST_COLUMNS: ["merchant"],
            },
            MetaData.UNDIRECTED: True,
        }

        # - tabular data info
        self.graph_info[MetaData.EDGE_DATA][MetaData.CONTINUOUS_COLUMNS] = [
            "amt",
            "city_pop",
            "dob",
            "lat",
            "long",
            "merch_lat",
            "merch_long",
            "trans_date_trans_time",
            "trans_num",
            "unix_time",
        ]
        self.graph_info[MetaData.EDGE_DATA][MetaData.CATEGORICAL_COLUMNS] = [
            "gender",
            "merchant",
            "first",
            "last",
            "category",
            "job",
            "street",
            "cc_num",
            "state",
            "zip",
        ]

    def transform_graph(self, data) -> pd.DataFrame:
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

        data = data[columns_to_select]
        return {MetaData.EDGE_DATA: data}

    def inverse_transform(self, data):
        return data

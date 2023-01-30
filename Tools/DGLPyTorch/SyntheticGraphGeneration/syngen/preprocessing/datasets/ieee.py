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

import pandas as pd
from sklearn.utils import shuffle

from syngen.preprocessing.base_preprocessing import BasePreprocessing
from syngen.utils.types import MetaData

logger = logging.getLogger(__name__)
log = logger


class IEEEPreprocessing(BasePreprocessing):
    """Graph preprocessing for https://www.kaggle.com/competitions/ieee-fraud-detection

        Args:
            cached (bool): skip preprocessing and use cached files
            nrows (int): number of rows to load from dataframe
            drop_cols (list): columns to drop from dataframe
    """

    def __init__(
        self,
        cached: bool = False,
        nrows: int = None,
        drop_cols: list = [],
        **kwargs,
    ):
        super().__init__(cached, nrows, drop_cols)

        # - graph metadata
        self.graph_info = {
            MetaData.EDGE_DATA: {
                MetaData.SRC_NAME: "user_id",
                MetaData.SRC_COLUMNS: ["user_id"],
                MetaData.DST_NAME: "product_id",
                MetaData.DST_COLUMNS: ["product_id"],
            },
            MetaData.UNDIRECTED: True,
        }
        self.graph_info[MetaData.EDGE_DATA][MetaData.CONTINUOUS_COLUMNS] = [
            'TransactionDT', 'TransactionAmt', 'C1', 'C2', 'C3', 'C4',
            'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C14', 'V279',
            'V280', 'V284', 'V285', 'V286', 'V287', 'V290', 'V291', 'V292', 'V293',
            'V294', 'V295', 'V297', 'V298', 'V299', 'V302', 'V303', 'V304', 'V305',
            'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V316', 'V317',
            'V318', 'V319', 'V320', 'V321',
        ]
        self.graph_info[MetaData.EDGE_DATA][MetaData.CATEGORICAL_COLUMNS] = [
            "user_id",
            "isFraud",
            "product_id",
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

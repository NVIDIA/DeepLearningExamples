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

import cudf
import dask_cudf
import numpy as np
import pandas as pd

from syngen.preprocessing.base_preprocessing import BasePreprocessing
from syngen.utils.types import DataFrameType, MetaData

logger = logging.getLogger(__name__)
log = logger


class TabFormerPreprocessing(BasePreprocessing):
    def __init__(
        self,
        cached: bool = False,
        nrows: int = None,
        drop_cols: list = [],
        **kwargs,
    ):
        """
        preprocessing for https://github.com/IBM/TabFormer
        Dataframe `card_transaction.v1.csv`
        Args:
            cached (bool): skip preprocessing and use cached files
            nrows (int): number of rows to load from dataframe
            drop_cols (list): list of columns to drop (default: [])
        """
        super().__init__(cached, nrows, drop_cols)

        self.graph_info = {
            MetaData.EDGE_DATA: {
                MetaData.SRC_NAME: "card_id",
                MetaData.SRC_COLUMNS: ["user", "card"],
                MetaData.DST_NAME: "merchant_id",
                MetaData.DST_COLUMNS: ["merchant_name"],
            },
            MetaData.UNDIRECTED: True,
        }

        self.graph_info[MetaData.EDGE_DATA][MetaData.CONTINUOUS_COLUMNS] = [
            "amount"
        ]
        self.graph_info[MetaData.EDGE_DATA][MetaData.CATEGORICAL_COLUMNS] = [
            "card_id",
            "merchant_id",
            "use_chip",
            "errors",
            "is_fraud",
        ]

    @staticmethod
    def nanZero(X: DataFrameType) -> DataFrameType:
        return X.where(X.notnull(), 0)

    @staticmethod
    def nanNone(X: DataFrameType) -> DataFrameType:
        return X.where(X.notnull(), "None")

    @staticmethod
    def amountEncoder(X: DataFrameType) -> DataFrameType:
        return (
            X.str.slice(start=1)
            .astype(float)
            .clip(lower=1.0)
            .map(lambda x: math.log(x))
        )

    def transform_graph(self, data: DataFrameType) -> DataFrameType:
        data.columns = [
            i.lower().replace(" ", "_") for i in data.columns.tolist()
        ]
        data = data.rename(
            columns={"is_fraud?": "is_fraud", "errors?": "errors"}
        )
        data["errors"] = data["errors"].fillna(0)
        data["use_chip"] = self.nanNone(data["use_chip"])
        data["amount"] = self.amountEncoder(data["amount"])

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
        for col in categorical_columns:
            data[col] = data[col].astype("category").cat.codes
            data[col] = data[col].astype(int)

        columns_to_select = categorical_columns + continuous_columns
        src_name = self.graph_info[MetaData.EDGE_DATA][MetaData.SRC_NAME]
        dst_name = self.graph_info[MetaData.EDGE_DATA][MetaData.DST_NAME]
        src_ids = data[src_name].unique()
        data[dst_name] = data[dst_name].astype(int) + len(src_ids)
        data = data[columns_to_select]
        return {MetaData.EDGE_DATA: data}

    def inverse_transform(self, data):
        data["amount"] = data["amount"].map(lambda x: "$" + str(math.exp(x)))
        return data

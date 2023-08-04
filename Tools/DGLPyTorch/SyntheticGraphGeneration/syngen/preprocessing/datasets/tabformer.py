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

import math
import os
import json
import logging
import shutil
from typing import Optional

import cudf
import cupy as cp
import numpy as np
import pandas as pd

from syngen.utils.types import DataFrameType
from syngen.configuration import SynGenDatasetFeatureSpec
from syngen.preprocessing.base_preprocessing import BasePreprocessing
from syngen.utils.types import MetaData


class TabFormerPreprocessing(BasePreprocessing):
    """
        preprocessing for https://github.com/IBM/TabFormer

    """

    def __init__(
            self,
            source_path: str,
            destination_path: Optional[str] = None,
            download: bool = False,
            **kwargs,
    ):
        super().__init__(source_path, destination_path, download, **kwargs)

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

    def transform(self, gpu=False, use_cache=False) -> SynGenDatasetFeatureSpec:

        if use_cache and os.path.exists(self.destination_path):
            return SynGenDatasetFeatureSpec.instantiate_from_preprocessed(self.destination_path)

        operator = cp if gpu else np
        tabular_operator = cudf if gpu else pd

        data = tabular_operator.read_csv(os.path.join(self.source_path, 'card_transaction.v2.csv'))
        data.columns = [
            i.lower().replace(" ", "_") for i in data.columns.tolist()
        ]
        data = data.rename(
            columns={"is_fraud?": "is_fraud", "errors?": "errors", "merchant_name": "merchant_id"}
        )

        data['card_id'] = data['user'] + data['card']
        data.drop(columns=['user', 'card'], inplace=True)

        data["errors"] = data["errors"].fillna(0)
        data["use_chip"] = self.nanNone(data["use_chip"])
        data["amount"] = self.amountEncoder(data["amount"])

        cont_columns = ["amount"]

        cat_columns = ["use_chip", "errors", "is_fraud"]

        for col in ("card_id", "merchant_id", *cat_columns):
            data[col] = data[col].astype("category").cat.codes
            data[col] = data[col].astype(int)

        structural_data = data[['card_id', 'merchant_id']]
        tabular_data = data[[*cat_columns, *cont_columns]]

        edge_features = self._prepare_feature_list(tabular_data, cat_columns, cont_columns)

        graph_metadata = {
            MetaData.NODES: [
                {
                    MetaData.NAME: "card",
                    MetaData.COUNT: int(structural_data['card_id'].max()),
                    MetaData.FEATURES: [],
                    MetaData.FEATURES_PATH: None,
                },
                {
                    MetaData.NAME: "merchant",
                    MetaData.COUNT: int(structural_data['merchant_id'].max()),
                    MetaData.FEATURES: [],
                    MetaData.FEATURES_PATH: None,
                }
            ],
            MetaData.EDGES: [
                {
                    MetaData.NAME: "transaction",
                    MetaData.COUNT: len(structural_data),
                    MetaData.SRC_NODE_TYPE: "card",
                    MetaData.DST_NODE_TYPE: "merchant",
                    MetaData.DIRECTED: False,
                    MetaData.FEATURES: edge_features,
                    MetaData.FEATURES_PATH: "transaction.parquet",
                    MetaData.STRUCTURE_PATH: "transaction_edge_list.parquet",
                }
            ]
        }

        shutil.rmtree(self.destination_path, ignore_errors=True)
        os.makedirs(self.destination_path)

        tabular_data.to_parquet(os.path.join(self.destination_path, "transaction.parquet"))
        structural_data.to_parquet(os.path.join(self.destination_path, "transaction_edge_list.parquet"))

        with open(os.path.join(self.destination_path, 'graph_metadata.json'), 'w') as f:
            json.dump(graph_metadata, f, indent=4)

        graph_metadata[MetaData.PATH] = self.destination_path
        return SynGenDatasetFeatureSpec(graph_metadata)

    def download(self):
        raise NotImplementedError(
            "TabFormer dataset does not support automatic downloading. Please run /workspace/scripts/get_datasets.sh"
        )

    def _check_files(self) -> bool:
        files = ['card_transaction.v2.csv']
        return all(os.path.exists(os.path.join(self.source_path, file)) for file in files)


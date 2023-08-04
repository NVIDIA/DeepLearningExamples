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

import os
import json
import logging
import shutil
from typing import Optional

import cudf
import cupy as cp
import numpy as np
import pandas as pd

from syngen.configuration import SynGenDatasetFeatureSpec
from syngen.preprocessing.base_preprocessing import BasePreprocessing
from syngen.utils.types import MetaData

logger = logging.getLogger(__name__)
log = logger


class IEEEPreprocessing(BasePreprocessing):
    """
        preprocessing for https://www.kaggle.com/competitions/ieee-fraud-detection
    """

    def __init__(
            self,
            source_path: str,
            destination_path: Optional[str] = None,
            download: bool = False,
            **kwargs,
    ):
        super().__init__(source_path, destination_path, download, **kwargs)

    def transform(self, gpu=False, use_cache=False):

        if use_cache and os.path.exists(self.destination_path):
            return SynGenDatasetFeatureSpec.instantiate_from_preprocessed(self.destination_path)

        operator = cp if gpu else np
        tabular_operator = cudf if gpu else pd

        data = tabular_operator.read_csv(os.path.join(self.source_path, 'data.csv'))
        data = data.fillna(0)

        cont_columns = [
            'TransactionDT', 'TransactionAmt', 'C1', 'C2', 'C3', 'C4',
            'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C14', 'V279',
            'V280', 'V284', 'V285', 'V286', 'V287', 'V290', 'V291', 'V292', 'V293',
            'V294', 'V295', 'V297', 'V298', 'V299', 'V302', 'V303', 'V304', 'V305',
            'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V316', 'V317',
            'V318', 'V319', 'V320', 'V321',
        ]

        cat_columns = ["isFraud"]

        for col in ('user_id', 'product_id', *cat_columns):
            data[col] = data[col].astype("category").cat.codes
            data[col] = data[col].astype(int)

        structural_data = data[['user_id', 'product_id']]

        tabular_data = data[[*cat_columns, *cont_columns]]

        edge_features = self._prepare_feature_list(tabular_data, cat_columns, cont_columns)

        graph_metadata = {
            MetaData.NODES: [
                {
                    MetaData.NAME: "user",
                    MetaData.COUNT: int(structural_data['user_id'].max()),
                    MetaData.FEATURES: [],
                    MetaData.FEATURES_PATH: None,
                },
                {
                    MetaData.NAME: "product",
                    MetaData.COUNT: int(structural_data['product_id'].max()),
                    MetaData.FEATURES: [],
                    MetaData.FEATURES_PATH: None,
                }
            ],
            MetaData.EDGES: [
                {
                    MetaData.NAME: "user-product",
                    MetaData.COUNT: len(structural_data),
                    MetaData.SRC_NODE_TYPE: "user",
                    MetaData.DST_NODE_TYPE: "product",
                    MetaData.DIRECTED: False,
                    MetaData.FEATURES: edge_features,
                    MetaData.FEATURES_PATH: "user-product.parquet",
                    MetaData.STRUCTURE_PATH: "user-product_edge_list.parquet",
                }
            ]
        }

        shutil.rmtree(self.destination_path, ignore_errors=True)
        os.makedirs(self.destination_path)

        tabular_data.to_parquet(os.path.join(self.destination_path, "user-product.parquet"))
        structural_data.to_parquet(os.path.join(self.destination_path, "user-product_edge_list.parquet"))

        with open(os.path.join(self.destination_path, 'graph_metadata.json'), 'w') as f:
            json.dump(graph_metadata, f, indent=4)

        graph_metadata[MetaData.PATH] = self.destination_path
        return SynGenDatasetFeatureSpec(graph_metadata)

    def download(self):
        raise NotImplementedError(
            "IEEE dataset does not support automatic downloading. Please run /workspace/scripts/get_datasets.sh"
        )

    def _check_files(self) -> bool:
        files = ['data.csv']
        return all(os.path.exists(os.path.join(self.source_path, file)) for file in files)
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

import json
import logging
import os
import shutil
import tarfile
from typing import Optional
from urllib.request import urlopen

import cudf
import cupy as cp
import numpy as np
import pandas as pd

from syngen.configuration import SynGenDatasetFeatureSpec
from syngen.preprocessing.base_preprocessing import BasePreprocessing
from syngen.utils.types import MetaData

logger = logging.getLogger(__name__)
log = logger


class EpinionsPreprocessing(BasePreprocessing):
    ITEM_SPACE_ARCHIVE_URL = (
        "http://konect.cc/files/download.tsv.epinions-rating.tar.bz2"
    )

    SOCIAL_SPACE_ARCHIVE_URL = (
        "http://konect.cc/files/download.tsv.epinions.tar.bz2"
    )

    def __init__(
            self,
            source_path: str,
            destination_path: Optional[str] = None,
            download: bool = False,
            **kwargs,
    ):
        """
        preprocessing for http://www.trustlet.org/wiki/Extended_Epinions_dataset

        Args:

        """
        self.ratings_file = os.path.join(source_path, 'epinions-rating', 'out.epinions-rating')
        self.trust_file = os.path.join(source_path, 'epinions', 'out.epinions')
        super().__init__(source_path, destination_path, download, **kwargs)

    def transform(self, gpu=False, use_cache=False):

        if use_cache and os.path.exists(self.destination_path):
            return SynGenDatasetFeatureSpec.instantiate_from_preprocessed(self.destination_path)

        operator = cp if gpu else np
        tabular_operator = cudf if gpu else pd

        item_space_data = tabular_operator.read_csv(
            self.ratings_file,
            sep=" ",
            names=["userId", "itemId", "rating",  "timestamp"],
            skiprows=1,
        )
        social_space_data = tabular_operator.read_csv(
            self.trust_file,
            sep=" ",
            names=["userId", "friendId", "trust", "timestamp"],
            skiprows=1,
        )
        social_space_data = social_space_data[social_space_data["trust"] == 1]

        min_item_id = int(item_space_data['itemId'].min())

        item_space_data['itemId'] = item_space_data['itemId'] - min_item_id

        min_user_id = min(
            int(item_space_data['userId'].min()),
            int(social_space_data['userId'].min()),
            int(social_space_data['friendId'].min())
        )

        item_space_data['userId'] = item_space_data['userId'] - min_user_id
        social_space_data['userId'] = social_space_data['userId'] - min_user_id
        social_space_data['friendId'] = social_space_data['friendId'] - min_user_id

        graph_metadata = {
            MetaData.NODES: [
                {
                    MetaData.NAME: "user",
                    MetaData.COUNT: int(item_space_data['userId'].max()),
                    MetaData.FEATURES: [],
                    MetaData.FEATURES_PATH: None,
                },
                {
                    MetaData.NAME: "item",
                    MetaData.COUNT: int(item_space_data['itemId'].max()),
                    MetaData.FEATURES: [],
                    MetaData.FEATURES_PATH: None,
                }
            ],
            MetaData.EDGES: [
                {
                    MetaData.NAME: "user-item",
                    MetaData.COUNT: len(item_space_data),
                    MetaData.SRC_NODE_TYPE: "user",
                    MetaData.DST_NODE_TYPE: "item",
                    MetaData.DIRECTED: False,
                    MetaData.FEATURES: [
                        {
                            MetaData.NAME: "rating",
                            MetaData.DTYPE: str(item_space_data["rating"].dtype),
                            MetaData.FEATURE_TYPE: MetaData.CATEGORICAL,
                        }
                    ],
                    MetaData.FEATURES_PATH: "user-item.parquet",
                    MetaData.STRUCTURE_PATH: "user-item_edge_list.parquet",
                },
                {
                    MetaData.NAME: "user-user",
                    MetaData.COUNT: len(social_space_data),
                    MetaData.SRC_NODE_TYPE: "user",
                    MetaData.DST_NODE_TYPE: "item",
                    MetaData.DIRECTED: False,
                    MetaData.FEATURES: [],
                    MetaData.FEATURES_PATH: None,
                    MetaData.STRUCTURE_PATH: "user-user_edge_list.parquet",
                }
            ]
        }
        shutil.rmtree(self.destination_path, ignore_errors=True)
        os.makedirs(self.destination_path)

        item_space_data[['rating']] \
            .to_parquet(os.path.join(self.destination_path, "user-item.parquet"))

        item_space_data[['userId', 'itemId']] \
            .rename(columns={'userId': MetaData.SRC, 'itemId': MetaData.DST}) \
            .to_parquet(os.path.join(self.destination_path, "user-item_edge_list.parquet"))

        social_space_data[['userId', 'friendId']] \
            .rename(columns={'userId': MetaData.SRC, 'friendId': MetaData.DST}) \
            .to_parquet(os.path.join(self.destination_path, "user-user_edge_list.parquet"))

        with open(os.path.join(self.destination_path, 'graph_metadata.json'), 'w') as f:
            json.dump(graph_metadata, f, indent=4)

        graph_metadata[MetaData.PATH] = self.destination_path
        return SynGenDatasetFeatureSpec(graph_metadata)

    def download(self):
        if not os.path.exists(self.source_path):
            os.makedirs(self.source_path)

        if not os.path.exists(self.ratings_file):
            with tarfile.open(fileobj=urlopen(self.ITEM_SPACE_ARCHIVE_URL), mode="r|bz2") as tar:
                tar.extractall(self.source_path)

        if not os.path.exists(self.trust_file):
            with tarfile.open(fileobj=urlopen(self.SOCIAL_SPACE_ARCHIVE_URL), mode="r|bz2") as tar:
                tar.extractall(self.source_path)

    def _check_files(self):
        files = [self.ratings_file, self.trust_file]
        return all(os.path.exists(file) for file in files)

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
import os
from abc import ABC, abstractmethod
from typing import Optional

from syngen.utils.types import MetaData
from syngen.configuration import SynGenDatasetFeatureSpec

logger = logging.getLogger(__name__)
log = logger


class BasePreprocessing(ABC):
    """Base class for all preprocessing transforms.

       Args:
            source_path: path to the raw dataset
            destination_path: path to store the dataset in SynGen format
            download: tries automatically download the dataset if True
    """

    def __init__(
            self,
            source_path: str,
            destination_path: Optional[str] = None,
            download: bool = False,
            **kwargs,
    ):
        self.source_path = source_path
        self.destination_path = destination_path or os.path.join(source_path, 'syngen_preprocessed')

        if download:
            self.download()
        assert self._check_files()

    def _prepare_feature_list(self, tabular_data, cat_columns, cont_columns):
        feature_list = [
            {
                MetaData.NAME: feat_name,
                MetaData.DTYPE: str(tabular_data[feat_name].dtype),
                MetaData.FEATURE_TYPE: MetaData.CONTINUOUS,
            }
            for feat_name in cont_columns
        ]
        feature_list.extend([
            {
                MetaData.NAME: feat_name,
                MetaData.DTYPE: str(tabular_data[feat_name].dtype),
                MetaData.FEATURE_TYPE: MetaData.CATEGORICAL,
            }
            for feat_name in cat_columns
        ])
        return feature_list


    @abstractmethod
    def transform(self, gpu=False, use_cache=False) -> SynGenDatasetFeatureSpec:
        raise NotImplementedError()

    @abstractmethod
    def download(self):
        raise NotImplementedError()

    @abstractmethod
    def _check_files(self) -> bool:
        raise NotImplementedError()

    @classmethod
    def add_cli_args(cls, parser):
        return parser

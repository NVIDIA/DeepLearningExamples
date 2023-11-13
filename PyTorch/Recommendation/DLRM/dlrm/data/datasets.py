# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import concurrent
import math
import os
import queue

import torch

import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Sequence, Tuple, List

from dlrm.data.defaults import CATEGORICAL_CHANNEL, NUMERICAL_CHANNEL, LABEL_CHANNEL, \
    DTYPE_SELECTOR, FEATURES_SELECTOR, FILES_SELECTOR
from dlrm.data.feature_spec import FeatureSpec


class SyntheticDataset(Dataset):
    """Synthetic dataset version of criteo dataset."""

    def __init__(
            self,
            num_entries: int,
            device: str = 'cuda',
            batch_size: int = 32768,
            numerical_features: Optional[int] = None,
            categorical_feature_sizes: Optional[Sequence[int]] = None  # features are returned in this order
    ):
        cat_features_count = len(categorical_feature_sizes) if categorical_feature_sizes is not None else 0
        num_features_count = numerical_features if numerical_features is not None else 0

        self._batches_per_epoch = math.ceil(num_entries / batch_size)
        self._num_tensor = torch.rand(size=(batch_size, num_features_count), device=device, dtype=torch.float32) \
            if num_features_count > 0 else None
        self._label_tensor = torch.randint(low=0, high=2, size=(batch_size,), device=device, dtype=torch.float32)
        self._cat_tensor = torch.cat(
            [torch.randint(low=0, high=cardinality, size=(batch_size, 1), device=device, dtype=torch.long)
             for cardinality in categorical_feature_sizes], dim=1) if cat_features_count > 0 else None

    def __len__(self):
        return self._batches_per_epoch

    def __getitem__(self, idx: int):
        if idx >= self._batches_per_epoch:
            raise IndexError()

        return self._num_tensor, self._cat_tensor, self._label_tensor


class ParametricDataset(Dataset):
    def __init__(
            self,
            feature_spec: FeatureSpec,
            mapping: str,
            batch_size: int = 1,
            numerical_features_enabled: bool = False,
            categorical_features_to_read: List[str] = None,  # This parameter dictates order of returned features
            prefetch_depth: int = 10,
            drop_last_batch: bool = False,
            **kwargs
    ):
        self._feature_spec = feature_spec
        self._batch_size = batch_size
        self._mapping = mapping
        feature_spec.check_feature_spec()
        categorical_features = feature_spec.channel_spec[CATEGORICAL_CHANNEL]
        numerical_features = feature_spec.channel_spec[NUMERICAL_CHANNEL]
        label_features = feature_spec.channel_spec[LABEL_CHANNEL]

        set_of_categorical_features = set(categorical_features)
        set_of_numerical_features = set(numerical_features)
        set_of_label_features = set(label_features)

        set_of_categoricals_to_read = set(categorical_features_to_read)
        bytes_per_feature = {feature_name: np.dtype(feature_spec.feature_spec[feature_name][DTYPE_SELECTOR]).itemsize
                             for feature_name in feature_spec.feature_spec.keys()}

        self._numerical_features_file = None
        self._label_file = None
        self._numerical_bytes_per_batch = bytes_per_feature[numerical_features[0]] * \
                                          len(numerical_features) * batch_size
        self._label_bytes_per_batch = np.dtype(bool).itemsize * batch_size
        self._number_of_numerical_features = len(numerical_features)

        chosen_mapping = feature_spec.source_spec[mapping]
        categorical_feature_files = {}
        root_path = feature_spec.base_directory
        number_of_batches = None
        for chunk in chosen_mapping:
            contained_features = chunk[FEATURES_SELECTOR]
            containing_file = chunk[FILES_SELECTOR][0]
            first_feature = contained_features[0]

            if first_feature in set_of_categorical_features:
                # Load categorical
                if first_feature not in set_of_categoricals_to_read:
                    continue  # skip chunk

                path_to_open = os.path.join(root_path, containing_file)
                cat_file = os.open(path_to_open, os.O_RDONLY)
                bytes_per_batch = bytes_per_feature[first_feature] * self._batch_size
                batch_num_float = os.fstat(cat_file).st_size / bytes_per_batch
                categorical_feature_files[first_feature] = cat_file

            elif first_feature in set_of_numerical_features:
                # Load numerical
                if not numerical_features_enabled:
                    continue  # skip chunk

                path_to_open = os.path.join(root_path, containing_file)
                self._numerical_features_file = os.open(path_to_open, os.O_RDONLY)
                batch_num_float = os.fstat(self._numerical_features_file).st_size / self._numerical_bytes_per_batch

            elif first_feature in set_of_label_features:
                # Load label
                path_to_open = os.path.join(root_path, containing_file)
                self._label_file = os.open(path_to_open, os.O_RDONLY)
                batch_num_float = os.fstat(self._label_file).st_size / self._label_bytes_per_batch

            else:
                raise ValueError("Unknown chunk type")

            local_number_of_batches = math.ceil(batch_num_float) if not drop_last_batch else math.floor(batch_num_float)
            if number_of_batches is not None:
                if local_number_of_batches != number_of_batches:
                    raise ValueError("Size mismatch in data files")
            else:
                number_of_batches = local_number_of_batches

        self._categorical_features_files = None
        if len(categorical_features_to_read) > 0:
            self._categorical_features_files = [categorical_feature_files[feature] for feature in
                                                categorical_features_to_read]
            self._categorical_bytes_per_batch = [bytes_per_feature[feature] * self._batch_size for feature in
                                                 categorical_features_to_read]
            self._categorical_types = [feature_spec.feature_spec[feature][DTYPE_SELECTOR] for feature in
                                       categorical_features_to_read]
        self._num_entries = number_of_batches
        self._prefetch_depth = min(prefetch_depth, self._num_entries)
        self._prefetch_queue = queue.Queue()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def __len__(self):
        return self._num_entries

    def __getitem__(self, idx: int):
        """ Numerical features are returned in the order they appear in the channel spec section
        For performance reasons, this is required to be the order they are saved in, as specified
        by the relevant chunk in source spec.

        Categorical features are returned in the order they appear in the channel spec section """
        if idx >= self._num_entries:
            raise IndexError()

        if self._prefetch_depth <= 1:
            return self._get_item(idx)

        # At the start, fill up the prefetching queue
        if idx == 0:
            for i in range(self._prefetch_depth):
                self._prefetch_queue.put(self._executor.submit(self._get_item, (i)))
        # Extend the prefetching window by one if not at the end of the dataset
        if idx < self._num_entries - self._prefetch_depth:
            self._prefetch_queue.put(self._executor.submit(self._get_item, (idx + self._prefetch_depth)))
        return self._prefetch_queue.get().result()

    def _get_item(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        click = self._get_label(idx)
        numerical_features = self._get_numerical_features(idx)
        categorical_features = self._get_categorical_features(idx)
        return numerical_features, categorical_features, click

    def _get_label(self, idx: int) -> torch.Tensor:
        raw_label_data = os.pread(self._label_file, self._label_bytes_per_batch,
                                  idx * self._label_bytes_per_batch)
        array = np.frombuffer(raw_label_data, dtype=bool)
        return torch.from_numpy(array).to(torch.float32)

    def _get_numerical_features(self, idx: int) -> Optional[torch.Tensor]:
        if self._numerical_features_file is None:
            return None

        raw_numerical_data = os.pread(self._numerical_features_file, self._numerical_bytes_per_batch,
                                      idx * self._numerical_bytes_per_batch)
        array = np.frombuffer(raw_numerical_data, dtype=np.float16)
        return torch.from_numpy(array).view(-1, self._number_of_numerical_features)

    def _get_categorical_features(self, idx: int) -> Optional[torch.Tensor]:
        if self._categorical_features_files is None:
            return None

        categorical_features = []
        for cat_bytes, cat_type, cat_file in zip(self._categorical_bytes_per_batch,
                                                 self._categorical_types,
                                                 self._categorical_features_files):
            raw_cat_data = os.pread(cat_file, cat_bytes, idx * cat_bytes)
            array = np.frombuffer(raw_cat_data, dtype=cat_type)
            tensor = torch.from_numpy(array).unsqueeze(1).to(torch.long)
            categorical_features.append(tensor)
        return torch.cat(categorical_features, dim=1)

    def __del__(self):
        data_files = [self._label_file, self._numerical_features_file]
        if self._categorical_features_files is not None:
            data_files += self._categorical_features_files

        for data_file in data_files:
            if data_file is not None:
                os.close(data_file)

# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
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
from typing import Optional, Sequence, Tuple, Any, Dict

from dlrm.data.utils import get_categorical_feature_type
from dlrm.utils.distributed import get_rank


class SyntheticDataset(Dataset):
    """Synthetic dataset version of criteo dataset."""

    def __init__(
        self,
        num_entries: int,
        device: str = 'cuda',
        batch_size: int = 1,
        numerical_features: Optional[int] = None,
        categorical_feature_sizes: Optional[Sequence[int]] = None,
        device_mapping: Optional[Dict[str, Any]] = None
    ):
        if device_mapping:
            # distributed setting
            rank = get_rank()
            numerical_features = numerical_features if device_mapping["bottom_mlp"] == rank else None
            categorical_feature_sizes = device_mapping["embedding"][rank]

        self.cat_features_count = len(categorical_feature_sizes) if categorical_feature_sizes is not None else 0
        self.num_features_count = numerical_features if numerical_features is not None else 0

        self.tot_fea = 1 + self.num_features_count + self.cat_features_count
        self.batch_size = batch_size
        self.batches_per_epoch = math.ceil(num_entries / batch_size)
        self.categorical_feature_sizes = categorical_feature_sizes
        self.device = device

        self.tensor = torch.randint(low=0, high=2, size=(self.batch_size, self.tot_fea), device=self.device)
        self.tensor = self.tensor.float()

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, idx: int):
        if idx >= self.batches_per_epoch:
            raise IndexError()

        numerical_features = (self.tensor[:, 1: 1 + self.num_features_count].to(torch.float32)
                              if self.num_features_count > 0 else None)
        categorical_features = (self.tensor[:, 1 + self.num_features_count:].to(torch.long)
                                if self.cat_features_count > 0 else None)
        target = self.tensor[:, 0].to(torch.float32)

        return numerical_features, categorical_features, target


class CriteoBinDataset(Dataset):
    """Simple dataloader for a recommender system. Designed to work with a single binary file."""

    def __init__(
        self,
        data_file: str,
        batch_size: int = 1,
        subset: float = None,
        numerical_features: int = 13,
        categorical_features: int = 26,
        data_type: str = 'int32'
    ):
        self.data_type = np.__dict__[data_type]
        bytes_per_feature = self.data_type().nbytes

        self.tad_fea = 1 + numerical_features
        self.tot_fea = 1 + numerical_features + categorical_features

        self.batch_size = batch_size
        self.bytes_per_entry = (bytes_per_feature * self.tot_fea * batch_size)
        self.num_entries = math.ceil(os.path.getsize(data_file) / self.bytes_per_entry)

        if subset is not None:
            if subset <= 0 or subset > 1:
                raise ValueError('Subset parameter must be in (0,1) range')
            self.num_entries = math.ceil(self.num_entries * subset)

        self.file = open(data_file, 'rb')
        self._last_read_idx = -1

    def __len__(self):
        return self.num_entries

    def __getitem__(self, idx):
        if idx >= self.num_entries:
            raise IndexError()

        if idx == 0:
            self.file.seek(0, 0)
        elif self._last_read_idx != (idx - 1):
            self.file.seek(idx * self.bytes_per_entry, 0)

        raw_data = self.file.read(self.bytes_per_entry)
        self._last_read_idx = idx

        array = np.frombuffer(raw_data, dtype=self.data_type).reshape(-1, self.tot_fea)
        return array

    def __del__(self):
        self.file.close()


class SplitCriteoDataset(Dataset):
    """Split version of Criteo dataset

    Args:
        data_path (str): Full path to split binary file of dataset. It must contain numerical.bin, label.bin and
            cat_0 ~ cat_25.bin
        batch_size (int):
        numerical_features(boolean): If True, load numerical features for bottom_mlp. Default False
        categorical_features (list or None): categorical features used by the rank
        prefetch_depth (int): How many samples to prefetch. Default 10.
    """
    def __init__(
        self,
        data_path: str,
        batch_size: int = 1,
        numerical_features: bool = False,
        categorical_features: Optional[Sequence[int]] = None,
        categorical_feature_sizes: Optional[Sequence[int]] = None,
        prefetch_depth: int = 10
    ):
        self._label_bytes_per_batch = np.dtype(np.bool).itemsize * batch_size
        self._numerical_bytes_per_batch = 13 * np.dtype(np.float16).itemsize * batch_size if numerical_features else 0
        self._categorical_feature_types = [
            get_categorical_feature_type(size) for size in categorical_feature_sizes
        ] if categorical_feature_sizes else []
        self._categorical_bytes_per_batch = [
            np.dtype(cat_type).itemsize * batch_size for cat_type in self._categorical_feature_types
        ]
        self._categorical_features = categorical_features
        self._batch_size = batch_size
        self._label_file = os.open(os.path.join(data_path, F"label.bin"), os.O_RDONLY)
        self._num_entries = int(math.ceil(os.fstat(self._label_file).st_size / self._label_bytes_per_batch))

        if numerical_features:
            self._numerical_features_file = os.open(os.path.join(data_path, "numerical.bin"), os.O_RDONLY)
            if math.ceil(os.fstat(self._numerical_features_file).st_size /
                         self._numerical_bytes_per_batch) != self._num_entries:
                raise ValueError("Size miss match in data files")
        else:
            self._numerical_features_file = None

        if categorical_features:
            self._categorical_features_files = []
            for cat_id in categorical_features:
                cat_file = os.open(os.path.join(data_path, F"cat_{cat_id}.bin"), os.O_RDONLY)
                cat_bytes = self._categorical_bytes_per_batch[cat_id]
                if math.ceil(
                        os.fstat(cat_file).st_size / cat_bytes) != self._num_entries:
                    raise ValueError("Size miss match in data files")
                self._categorical_features_files.append(cat_file)
        else:
            self._categorical_features_files = None

        self._prefetch_depth = min(prefetch_depth, self._num_entries)
        self._prefetch_queue = queue.Queue()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def __len__(self):
        return self._num_entries

    def __getitem__(self, idx: int):
        if idx >= self._num_entries:
            raise IndexError()

        if self._prefetch_depth <= 1:
            return self._get_item(idx)

        if idx == 0:
            for i in range(self._prefetch_depth):
                self._prefetch_queue.put(self._executor.submit(self._get_item, (i)))
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
        array = np.frombuffer(raw_label_data, dtype=np.bool)
        return torch.from_numpy(array).to(torch.float32)

    def _get_numerical_features(self, idx: int) -> Optional[torch.Tensor]:
        if self._numerical_features_file is None:
            return None

        raw_numerical_data = os.pread(self._numerical_features_file, self._numerical_bytes_per_batch,
                                      idx * self._numerical_bytes_per_batch)
        array = np.frombuffer(raw_numerical_data, dtype=np.float16)
        return torch.from_numpy(array).view(-1, 13)

    def _get_categorical_features(self, idx: int) -> Optional[torch.Tensor]:
        if self._categorical_features_files is None:
            return None

        categorical_features = []
        for cat_id, cat_file in zip(self._categorical_features, self._categorical_features_files):
            cat_bytes = self._categorical_bytes_per_batch[cat_id]
            cat_type = self._categorical_feature_types[cat_id]
            raw_cat_data = os.pread(cat_file, cat_bytes, idx * cat_bytes)
            array = np.frombuffer(raw_cat_data, dtype=cat_type)
            tensor = torch.from_numpy(array).unsqueeze(1).to(torch.long)
            categorical_features.append(tensor)
        return torch.cat(categorical_features, dim=1)

    def __del__(self):
        data_files = [self._label_file, self._numerical_features_file] + self._categorical_features_files
        for data_file in data_files:
            if data_file is not None:
                os.close(data_file)

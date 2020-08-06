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

import functools
import os
from typing import Tuple, Optional, Callable, Dict, Sequence

import torch
from torch.utils.data import Dataset, Sampler, RandomSampler

from dlrm.data.datasets import CriteoBinDataset, SyntheticDataset, SplitCriteoDataset
from dlrm.data.samplers import RandomDistributedSampler
from dlrm.data.utils import collate_array, write_dataset_to_disk, get_categorical_feature_sizes, collate_split_tensors
from dlrm.utils.distributed import is_distributed, is_main_process, get_rank


def create_synthetic_datasets(flags, device_mapping: Optional[Dict] = None):
    dataset_train = SyntheticDataset(num_entries=flags.synthetic_dataset_num_entries,
                                     batch_size=flags.batch_size,
                                     numerical_features=flags.num_numerical_features,
                                     categorical_feature_sizes=get_categorical_feature_sizes(flags),
                                     device_mapping=device_mapping)

    dataset_test = SyntheticDataset(num_entries=flags.synthetic_dataset_num_entries,
                                    batch_size=flags.test_batch_size,
                                    numerical_features=flags.num_numerical_features,
                                    categorical_feature_sizes=get_categorical_feature_sizes(flags),
                                    device_mapping=device_mapping)
    return dataset_train, dataset_test


def create_real_datasets(flags, path, dataset_class: type = CriteoBinDataset):
    train_dataset = os.path.join(path, "train_data.bin")
    test_dataset = os.path.join(path, "test_data.bin")
    categorical_sizes = get_categorical_feature_sizes(flags)

    dataset_train = dataset_class(
        data_file=train_dataset,
        batch_size=flags.batch_size,
        subset=flags.dataset_subset,
        numerical_features=flags.num_numerical_features,
        categorical_features=len(categorical_sizes),
    )

    dataset_test = dataset_class(
        data_file=test_dataset,
        batch_size=flags.test_batch_size,
        numerical_features=flags.num_numerical_features,
        categorical_features=len(categorical_sizes),
    )

    return dataset_train, dataset_test


class DatasetFactory:

    def __init__(self, flags, device_mapping: Optional[Dict] = None):
        self._flags = flags
        self._device_mapping = device_mapping

    def create_collate_fn(self) -> Optional[Callable]:
        if self._device_mapping is not None:
            # selection of categorical features assigned to this device
            device_cat_features = torch.tensor(
                self._device_mapping["embedding"][get_rank()], device=self._flags.base_device, dtype=torch.long)
        else:
            device_cat_features = None

        orig_stream = torch.cuda.current_stream() if self._flags.base_device == 'cuda' else None
        return functools.partial(
            collate_array,
            device=self._flags.base_device,
            orig_stream=orig_stream,
            num_numerical_features=self._flags.num_numerical_features,
            selected_categorical_features=device_cat_features
        )

    def create_sampler(self, dataset: Dataset) -> Optional[Sampler]:
        return RandomDistributedSampler(dataset) if is_distributed() else RandomSampler(dataset)

    def create_datasets(self) -> Tuple[Dataset, Dataset]:
        raise NotImplementedError()

    def create_data_loader(self, dataset, collate_fn: Optional[Callable] = None, sampler: Optional[Sampler] = None):
        return torch.utils.data.DataLoader(
            dataset, collate_fn=collate_fn, sampler=sampler, batch_size=None,
            num_workers=0, pin_memory=False
        )


class SyntheticDiskDatasetFactory(DatasetFactory):

    def create_sampler(self, dataset: Dataset) -> Optional[Sampler]:
        return None

    def create_datasets(self) -> Tuple[Dataset, Dataset]:
        synthetic_train, synthetic_test = create_synthetic_datasets(self._flags)

        if is_distributed():
            self._synchronized_write(synthetic_train, synthetic_test)
        else:
            self._write(synthetic_train, synthetic_test)

        return create_real_datasets(self._flags, self._flags.synthetic_dataset_dir)

    def _synchronized_write(self, train_dataset: Dataset, test_dataset: Dataset):
        if is_main_process():
            self._write(train_dataset, test_dataset)
        torch.distributed.barrier()

    def _write(self, train_dataset: Dataset, test_dataset: Dataset):
        write_dataset_to_disk(self._flags.synthetic_dataset_dir, train_dataset, test_dataset,
                              self._flags.synthetic_dataset_table_sizes)


class SyntheticGpuDatasetFactory(DatasetFactory):

    def create_collate_fn(self) -> Optional[Callable]:
        return None

    def create_sampler(self, dataset) -> Optional[Sampler]:
        return None

    def create_datasets(self) -> Tuple[Dataset, Dataset]:
        return create_synthetic_datasets(self._flags, self._device_mapping)


class BinaryDatasetFactory(DatasetFactory):

    def create_datasets(self) -> Tuple[Dataset, Dataset]:
        return create_real_datasets(self._flags, self._flags.dataset)


class SplitBinaryDatasetFactory(DatasetFactory):

    def __init__(self, flags, numerical_features: bool, categorical_features: Sequence[int]):
        super().__init__(flags)
        self._numerical_features = numerical_features
        self._categorical_features = categorical_features

    def create_collate_fn(self):
        orig_stream = torch.cuda.current_stream() if self._flags.base_device == 'cuda' else None
        return functools.partial(
            collate_split_tensors,
            device=self._flags.base_device,
            orig_stream=orig_stream,
            numerical_type=torch.float16 if self._flags.amp else torch.float32
        )

    def create_datasets(self) -> Tuple[Dataset, Dataset]:
        train_dataset_path = os.path.join(self._flags.dataset, "train")
        test_dataset_path = os.path.join(self._flags.dataset, "test")
        categorical_sizes = get_categorical_feature_sizes(self._flags)

        dataset_train = SplitCriteoDataset(
            data_path=train_dataset_path,
            batch_size=self._flags.batch_size,
            numerical_features=self._numerical_features,
            categorical_features=self._categorical_features,
            categorical_feature_sizes=categorical_sizes
        )
        dataset_test = SplitCriteoDataset(
            data_path=test_dataset_path,
            batch_size=self._flags.test_batch_size,
            numerical_features=self._numerical_features,
            categorical_features=self._categorical_features,
            categorical_feature_sizes=categorical_sizes
        )
        return dataset_train, dataset_test


def create_dataset_factory(flags, device_mapping: Optional[dict] = None) -> DatasetFactory:
    """
    By default each dataset can be used in single GPU or distributed setting - please keep that in mind when adding
    new datasets. Distributed case requires selection of categorical features provided in `device_mapping`
    (see `DatasetFactory#create_collate_fn`).

    :param flags:
    :param device_mapping: dict, information about model bottom mlp and embeddings devices assignment
    :return:
    """
    dataset_type = flags.dataset_type

    if dataset_type == "binary":
        return BinaryDatasetFactory(flags, device_mapping)

    if dataset_type == "split":
        if is_distributed():
            assert device_mapping is not None, "Distributed dataset requires information about model device mapping."
            rank = get_rank()
            return SplitBinaryDatasetFactory(
                flags=flags,
                numerical_features=device_mapping["bottom_mlp"] == rank,
                categorical_features=device_mapping["embedding"][rank]
            )
        return SplitBinaryDatasetFactory(
            flags=flags,
            numerical_features=True,
            categorical_features=range(len(get_categorical_feature_sizes(flags)))
        )

    if dataset_type == "synthetic_gpu":
        return SyntheticGpuDatasetFactory(flags, device_mapping)

    if dataset_type == "synthetic_disk":
        return SyntheticDiskDatasetFactory(flags, device_mapping)

    raise NotImplementedError(f"unknown dataset type: {dataset_type}")

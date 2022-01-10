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

import functools
from typing import Tuple, Optional, Callable, Dict

import torch
from torch.utils.data import Dataset, Sampler, RandomSampler

from dlrm.data.datasets import SyntheticDataset, ParametricDataset
from dlrm.data.defaults import TEST_MAPPING, TRAIN_MAPPING
from dlrm.data.feature_spec import FeatureSpec
from dlrm.data.samplers import RandomDistributedSampler
from dlrm.data.utils import collate_split_tensors
from dlrm.utils.distributed import is_distributed, get_rank


class DatasetFactory:

    def __init__(self, flags, device_mapping: Optional[Dict] = None):
        self._flags = flags
        self._device_mapping = device_mapping

    def create_collate_fn(self) -> Optional[Callable]:
        raise NotImplementedError()

    def create_datasets(self) -> Tuple[Dataset, Dataset]:
        raise NotImplementedError()

    def create_sampler(self, dataset: Dataset) -> Optional[Sampler]:
        return RandomDistributedSampler(dataset) if is_distributed() else RandomSampler(dataset)

    def create_data_loader(
            self,
            dataset,
            collate_fn: Optional[Callable] = None,
            sampler: Optional[Sampler] = None):
        return torch.utils.data.DataLoader(
            dataset, collate_fn=collate_fn, sampler=sampler, batch_size=None,
            num_workers=0, pin_memory=False
        )


class SyntheticGpuDatasetFactory(DatasetFactory):
    def __init__(self, flags, local_numerical_features_num, local_categorical_feature_sizes):
        self.local_numerical_features = local_numerical_features_num
        self.local_categorical_features = local_categorical_feature_sizes
        super().__init__(flags)

    def create_collate_fn(self) -> Optional[Callable]:
        return None

    def create_sampler(self, dataset) -> Optional[Sampler]:
        return None

    def create_datasets(self) -> Tuple[Dataset, Dataset]:
        flags = self._flags
        dataset_train = SyntheticDataset(num_entries=flags.synthetic_dataset_num_entries,
                                         batch_size=flags.batch_size,
                                         numerical_features=self.local_numerical_features,
                                         categorical_feature_sizes=self.local_categorical_features)

        dataset_test = SyntheticDataset(num_entries=flags.synthetic_dataset_num_entries,
                                        batch_size=flags.test_batch_size,
                                        numerical_features=self.local_numerical_features,
                                        categorical_feature_sizes=self.local_categorical_features)
        return dataset_train, dataset_test


class ParametricDatasetFactory(DatasetFactory):

    def __init__(self, flags, feature_spec: FeatureSpec, numerical_features_enabled, categorical_features_to_read):
        super().__init__(flags)
        self._base_device = flags.base_device
        self._train_batch_size = flags.batch_size
        self._test_batch_size = flags.test_batch_size
        self._feature_spec = feature_spec
        self._numerical_features_enabled = numerical_features_enabled
        self._categorical_features_to_read = categorical_features_to_read

    def create_collate_fn(self):
        orig_stream = torch.cuda.current_stream() if self._base_device == 'cuda' else None
        return functools.partial(
            collate_split_tensors,
            device=self._base_device,
            orig_stream=orig_stream,
            numerical_type=torch.float32
        )

    def create_datasets(self) -> Tuple[Dataset, Dataset]:
        # prefetching is currently unsupported if using the batch-wise shuffle
        prefetch_depth = 0 if self._flags.shuffle_batch_order else 10

        dataset_train = ParametricDataset(
            feature_spec=self._feature_spec,
            mapping=TRAIN_MAPPING,
            batch_size=self._train_batch_size,
            numerical_features_enabled=self._numerical_features_enabled,
            categorical_features_to_read=self._categorical_features_to_read,
            prefetch_depth=prefetch_depth
        )

        dataset_test = ParametricDataset(
            feature_spec=self._feature_spec,
            mapping=TEST_MAPPING,
            batch_size=self._test_batch_size,
            numerical_features_enabled=self._numerical_features_enabled,
            categorical_features_to_read=self._categorical_features_to_read,
            prefetch_depth=prefetch_depth
        )

        return dataset_train, dataset_test


def create_dataset_factory(flags, feature_spec: FeatureSpec, device_mapping: Optional[dict] = None) -> DatasetFactory:
    """
    By default each dataset can be used in single GPU or distributed setting - please keep that in mind when adding
    new datasets. Distributed case requires selection of categorical features provided in `device_mapping`
    (see `DatasetFactory#create_collate_fn`).

    :param flags:
    :param device_mapping: dict, information about model bottom mlp and embeddings devices assignment
    :return:
    """
    dataset_type = flags.dataset_type
    num_numerical_features = feature_spec.get_number_of_numerical_features()
    if is_distributed() or device_mapping:
        assert device_mapping is not None, "Distributed dataset requires information about model device mapping."
        rank = get_rank()
        local_categorical_positions = device_mapping["embedding"][rank]
        numerical_features_enabled = device_mapping["bottom_mlp"] == rank
    else:
        local_categorical_positions = list(range(len(feature_spec.get_categorical_feature_names())))
        numerical_features_enabled = True

    if dataset_type == "parametric":
        local_categorical_names = feature_spec.cat_positions_to_names(local_categorical_positions)
        return ParametricDatasetFactory(flags=flags, feature_spec=feature_spec,
                                        numerical_features_enabled=numerical_features_enabled,
                                        categorical_features_to_read=local_categorical_names
                                        )
    if dataset_type == "synthetic_gpu":
        local_numerical_features = num_numerical_features if numerical_features_enabled else 0
        world_categorical_sizes = feature_spec.get_categorical_sizes()
        local_categorical_sizes = [world_categorical_sizes[i] for i in local_categorical_positions]
        return SyntheticGpuDatasetFactory(flags, local_numerical_features_num=local_numerical_features,
                                          local_categorical_feature_sizes=local_categorical_sizes)

    raise NotImplementedError(f"unknown dataset type: {dataset_type}")

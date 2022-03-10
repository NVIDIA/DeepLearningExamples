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

import os
from typing import Tuple, Optional, List

import numpy as np
import torch
from torch import Tensor
from torch.cuda import Stream
from torch.utils.data import Dataset, DataLoader
import tqdm

from dlrm.data.defaults import TRAIN_MAPPING, TEST_MAPPING, DTYPE_SELECTOR
from dlrm.data.feature_spec import FeatureSpec


def collate_split_tensors(
        tensors: Tuple[Tensor, Tensor, Tensor],
        device: str,
        orig_stream: Stream,
        numerical_type: torch.dtype = torch.float32
):
    tensors = [tensor.to(device, non_blocking=True) if tensor is not None else None for tensor in
               tensors]
    if device == 'cuda':
        for tensor in tensors:
            if tensor is not None:
                tensor.record_stream(orig_stream)

    numerical_features, categorical_features, click = tensors

    if numerical_features is not None:
        numerical_features = numerical_features.to(numerical_type)

    return numerical_features, categorical_features, click


def collate_array(
        array: np.array,
        device: str,
        orig_stream: Stream,
        num_numerical_features: int,
        selected_categorical_features: Optional[Tensor] = None
):
    # numerical features are encoded as float32
    numerical_features = array[:, 1:1 + num_numerical_features].view(dtype=np.float32)
    numerical_features = torch.from_numpy(numerical_features)

    categorical_features = torch.from_numpy(array[:, 1 + num_numerical_features:])
    click = torch.from_numpy(array[:, 0])

    categorical_features = categorical_features.to(device, non_blocking=True).to(torch.long)
    numerical_features = numerical_features.to(device, non_blocking=True)
    click = click.to(torch.float32).to(device, non_blocking=True)

    if selected_categorical_features is not None:
        categorical_features = categorical_features[:, selected_categorical_features]

    if device == 'cuda':
        numerical_features.record_stream(orig_stream)
        categorical_features.record_stream(orig_stream)
        click.record_stream(orig_stream)

    return numerical_features, categorical_features, click


def write_dataset_to_disk(dataset_train: Dataset, dataset_test: Dataset, feature_spec: FeatureSpec,
                          saving_batch_size=512) -> None:
    feature_spec.check_feature_spec()  # We rely on the feature spec being properly formatted

    categorical_features_list = feature_spec.get_categorical_feature_names()
    categorical_features_types = [feature_spec.feature_spec[feature_name][DTYPE_SELECTOR]
                                  for feature_name in categorical_features_list]
    number_of_numerical_features = feature_spec.get_number_of_numerical_features()
    number_of_categorical_features = len(categorical_features_list)

    for mapping_name, dataset in zip((TRAIN_MAPPING, TEST_MAPPING),
                                     (dataset_train, dataset_test)):
        file_streams = []
        label_path, numerical_path, categorical_paths = feature_spec.get_mapping_paths(mapping_name)
        try:
            os.makedirs(os.path.dirname(numerical_path), exist_ok=True)
            numerical_f = open(numerical_path, "wb+")
            file_streams.append(numerical_f)

            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            label_f = open(label_path, 'wb+')
            file_streams.append(label_f)

            categorical_fs = []
            for feature_name in categorical_features_list:
                local_path = categorical_paths[feature_name]
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                fs = open(local_path, 'wb+')
                categorical_fs.append(fs)
                file_streams.append(fs)

            for numerical, categorical, label in tqdm.tqdm(
                    DataLoader(dataset, saving_batch_size),
                    desc=mapping_name + " dataset saving",
                    unit_scale=saving_batch_size
            ):
                assert (numerical.shape[-1] == number_of_numerical_features)
                assert (categorical.shape[-1] == number_of_categorical_features)

                numerical_f.write(numerical.to(torch.float16).cpu().numpy().tobytes())
                label_f.write(label.to(torch.bool).cpu().numpy().tobytes())
                for cat_idx, cat_feature_type in enumerate(categorical_features_types):
                    categorical_fs[cat_idx].write(
                        categorical[:, :, cat_idx].cpu().numpy().astype(cat_feature_type).tobytes())
        finally:
            for stream in file_streams:
                stream.close()
    feature_spec.to_yaml()


def prefetcher(load_iterator, prefetch_stream):
    def _prefetch():
        with torch.cuda.stream(prefetch_stream):
            try:
                data_batch = next(load_iterator)
            except StopIteration:
                return None

        return data_batch

    next_data_batch = _prefetch()

    while next_data_batch is not None:
        torch.cuda.current_stream().wait_stream(prefetch_stream)
        data_batch = next_data_batch
        next_data_batch = _prefetch()
        yield data_batch


def get_embedding_sizes(fspec: FeatureSpec, max_table_size: Optional[int]) -> List[int]:
    if max_table_size is not None:
        return [min(s, max_table_size) for s in fspec.get_categorical_sizes()]
    else:
        return fspec.get_categorical_sizes()

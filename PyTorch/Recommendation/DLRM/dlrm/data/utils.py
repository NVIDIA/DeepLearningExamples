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

import json
import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.cuda import Stream
from torch.utils.data import Dataset, DataLoader
import tqdm

DATASET_SAVING_BATCH_SIZE = 512


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


def get_categorical_feature_type(size: int):
    types = (np.int8, np.int16, np.int32)

    for numpy_type in types:
        if size < np.iinfo(numpy_type).max:
            return numpy_type

    raise RuntimeError(f"Categorical feature of size {size} is too big for defined types")


def write_dataset_to_disk(destination, dataset_train: Dataset, dataset_test, table_sizes):
    for filename, dataset in zip(('train', 'test'),
                                 (dataset_train, dataset_test)):

        dst_file = os.path.join(destination, filename)
        os.makedirs(dst_file, exist_ok=True)

        cat_feature_types = [get_categorical_feature_type(int(cat_size)) for cat_size in
                             table_sizes]

        file_streams = []

        try:
            numerical_f = open(os.path.join(dst_file, "numerical.bin"), "wb+")
            file_streams.append(numerical_f)

            label_f = open(os.path.join(dst_file, 'label.bin'), 'wb+')
            file_streams.append(label_f)

            categorical_fs = []
            for i in range(len(table_sizes)):
                fs = open(os.path.join(dst_file, f'cat_{i}.bin'), 'wb+')
                categorical_fs.append(fs)
                file_streams.append(fs)

            for numerical, categorical, label in tqdm.tqdm(
                DataLoader(dataset, DATASET_SAVING_BATCH_SIZE),
                desc=filename + " dataset saving",
                unit_scale=DATASET_SAVING_BATCH_SIZE
            ):
                number_of_numerical_variables = numerical.shape[-1]
                number_of_categorical_variables = categorical.shape[-1]
                numerical_f.write(numerical.to(torch.float16).cpu().numpy().tobytes())
                label_f.write(label.to(torch.bool).cpu().numpy().tobytes())

                for cat_idx, cat_feature_type in enumerate(cat_feature_types):
                    categorical_fs[cat_idx].write(
                        categorical[:, :, cat_idx].cpu().numpy().astype(cat_feature_type).tobytes())

        finally:
            for stream in file_streams:
                stream.close()

    model_size_dict = {
        '_c' + str(i): size
        for i, size in zip(
            range(
                1 + number_of_numerical_variables,
                1 + number_of_numerical_variables + number_of_categorical_variables
            ),
            table_sizes
        )
    }
    with open(os.path.join(destination, 'model_size.json'), 'w') as f:
        json.dump(model_size_dict, f, indent=4, sort_keys=True)


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


def get_categorical_feature_sizes(FLAGS):
    if FLAGS.dataset_type in ['synthetic_gpu']:
        feature_sizes = [int(s) for s in FLAGS.synthetic_dataset_table_sizes]
        print('feature sizes: ', feature_sizes)
        return feature_sizes

    categorical_sizes_file = os.path.join(FLAGS.dataset, "model_size.json")
    with open(categorical_sizes_file) as f:
        categorical_sizes = [int(v) for v in json.load(f).values()]

    categorical_sizes = list(categorical_sizes)

    # need to add 1 because the JSON file contains the max value not the count
    categorical_sizes = [s + 1 for s in categorical_sizes]

    print('feature sizes: ', categorical_sizes)

    if FLAGS.max_table_size is None:
        return categorical_sizes

    clipped_sizes = [min(s, FLAGS.max_table_size) for s in categorical_sizes]
    return clipped_sizes

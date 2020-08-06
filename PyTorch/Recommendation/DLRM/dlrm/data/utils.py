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

import json
import os

import numpy as np
import pandas as pd
import torch
import tqdm
from torch import Tensor
from torch.cuda import Stream
from typing import Tuple, Optional


def collate_split_tensors(
        tensors: Tuple[Tensor, Tensor, Tensor],
        device: str,
        orig_stream: Stream,
        numerical_type: torch.dtype = torch.float32
):
    tensors = [tensor.to(device, non_blocking=True) if tensor is not None else None for tensor in tensors]
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


def write_dataset_to_disk(destination, dataset_train, dataset_test, table_sizes):
    for filename, dataset in zip(('train_data.bin', 'test_data.bin'),
                                 (dataset_train, dataset_test)):

        os.makedirs(destination, exist_ok=True)
        dst_file = os.path.join(destination, filename)
        if os.path.exists(dst_file):
            print(f'File {dst_file} already exists, skipping')
            continue

        with open(dst_file, 'wb') as dst_fd:
            for numeric, categorical, label in tqdm.tqdm(dataset):
                # numeric, categorical, label = collate(batch, device='cpu',
                #                                       orig_stream=None,
                #                                       num_numerical_features=13)

                categorical = categorical.to(torch.int32)
                label = label.to(torch.int32)

                l = pd.DataFrame(label.cpu().numpy())
                l.columns = ['label']
                n = pd.DataFrame(numeric.cpu().numpy())
                n.columns = ['n' + str(i) for i in range(len(n.columns))]

                c = pd.DataFrame(categorical.cpu().numpy())
                c.columns = ['c' + str(i) for i in range(len(c.columns))]
                df = pd.concat([l, n, c], axis=1)

                records = df.to_records(index=False)
                raw_data = records.tobytes()

                dst_fd.write(raw_data)

    model_size_dict = {'_c' + str(i): size for i, size in zip(range(14, 40), table_sizes)}
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
    if FLAGS.dataset_type in ['synthetic_disk', 'synthetic_gpu']:
        feature_sizes = [int(s) for s in FLAGS.synthetic_dataset_table_sizes]
        print('feature sizes: ', feature_sizes)
        return feature_sizes

    categorical_sizes_file = os.path.join(FLAGS.dataset, "model_size.json")
    with open(categorical_sizes_file) as f:
        categorical_sizes = json.load(f).values()

    categorical_sizes = list(categorical_sizes)

    # need to add 1 because the JSON file contains the max value not the count
    categorical_sizes = [s + 1 for s in categorical_sizes]

    print('feature sizes: ', categorical_sizes)

    if FLAGS.max_table_size is None:
        return categorical_sizes

    clipped_sizes = [min(s, FLAGS.max_table_size) for s in categorical_sizes]
    return clipped_sizes


def get_categorical_feature_type(size: int):
    types = (np.int8, np.int16, np.int32)

    for numpy_type in types:
        if size < np.iinfo(numpy_type).max:
            return numpy_type

    raise RuntimeError(f"Categorical feature of size {size} is too big for defined types")

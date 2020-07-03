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

import argparse
import json
import os
import math
from shutil import copyfile

from tqdm import tqdm
import numpy as np
from typing import Sequence

from dlrm.data.utils import get_categorical_feature_type


def split_binary_file(
    binary_file_path: str,
    output_dir: str,
    categorical_feature_sizes: Sequence[int],
    num_numerical_features: int,
    batch_size: int,
    source_data_type: str = 'int32',
):
    record_width = 1 + num_numerical_features + len(categorical_feature_sizes)  # label + numerical + categorical
    bytes_per_feature = np.__dict__[source_data_type]().nbytes
    bytes_per_entry = record_width * bytes_per_feature

    total_size = os.path.getsize(binary_file_path)
    batches_num = int(math.ceil((total_size // bytes_per_entry) / batch_size))

    cat_feature_types = [get_categorical_feature_type(cat_size) for cat_size in categorical_feature_sizes]

    file_streams = []
    try:
        input_data_f = open(binary_file_path, "rb")
        file_streams.append(input_data_f)

        numerical_f = open(os.path.join(output_dir, "numerical.bin"), "wb+")
        file_streams.append(numerical_f)

        label_f = open(os.path.join(output_dir, 'label.bin'), 'wb+')
        file_streams.append(label_f)

        categorical_fs = []
        for i in range(len(categorical_feature_sizes)):
            fs = open(os.path.join(output_dir, F'cat_{i}.bin'), 'wb+')
            categorical_fs.append(fs)
            file_streams.append(fs)

        for _ in tqdm(range(batches_num)):
            raw_data = np.frombuffer(input_data_f.read(bytes_per_entry * batch_size), dtype=np.int32)
            batch_data = raw_data.reshape(-1, record_width)

            numerical_features = batch_data[:, 1:1 + num_numerical_features].view(dtype=np.float32)
            numerical_f.write(numerical_features.astype(np.float16).tobytes())

            label = batch_data[:, 0]
            label_f.write(label.astype(np.bool).tobytes())

            cat_offset = num_numerical_features + 1
            for cat_idx, cat_feature_type in enumerate(cat_feature_types):
                cat_data = batch_data[:, (cat_idx + cat_offset):(cat_idx + cat_offset + 1)].astype(cat_feature_type)
                categorical_fs[cat_idx].write(cat_data.tobytes())
    finally:
        for stream in file_streams:
            stream.close()


def split_dataset(dataset_dir: str, output_dir: str, batch_size: int, numerical_features: int):
    categorical_sizes_file = os.path.join(dataset_dir, "model_size.json")
    with open(categorical_sizes_file) as f:
        categorical_sizes = list(json.load(f).values())

    train_file = os.path.join(dataset_dir, "train_data.bin")
    test_file = os.path.join(dataset_dir, "test_data.bin")
    val_file = os.path.join(dataset_dir, "val_data.bin")

    target_train = os.path.join(output_dir, "train")
    target_test = os.path.join(output_dir, "test")
    target_val = os.path.join(output_dir, "val")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(target_train, exist_ok=True)
    os.makedirs(target_test, exist_ok=True)
    os.makedirs(target_val, exist_ok=True)

    copyfile(categorical_sizes_file, os.path.join(output_dir, "model_size.json"))
    split_binary_file(test_file, target_test, categorical_sizes, numerical_features, batch_size)
    split_binary_file(train_file, target_train, categorical_sizes, numerical_features, batch_size)
    split_binary_file(val_file, target_val, categorical_sizes, numerical_features, batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32768)
    parser.add_argument('--numerical_features', type=int, default=13)
    args = parser.parse_args()

    split_dataset(
        dataset_dir=args.dataset,
        output_dir=args.output,
        batch_size=args.batch_size,
        numerical_features=args.numerical_features
    )


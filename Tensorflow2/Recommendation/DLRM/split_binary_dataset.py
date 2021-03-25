# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#


import tensorflow as tf

import concurrent
import math
import os
import queue
import json
from collections import namedtuple

import numpy as np
from typing import Optional, Sequence, Tuple, Any, Dict


DatasetMetadata = namedtuple('DatasetMetadata', ['num_numerical_features',
                                                 'categorical_cardinalities'])

class DummyDataset:
    def __init__(self, batch_size, num_numerical_features, num_categorical_features, num_batches):
        self.numerical_features = tf.zeros(shape=[batch_size, num_numerical_features])
        self.categorical_features = [tf.zeros(shape=[batch_size, 1], dtype=tf.int32)] * num_categorical_features
        self.labels = tf.ones(shape=[batch_size, 1])
        self.num_batches = num_batches

    def __getitem__(self, idx):
        if idx >= self.num_batches:
            raise StopIteration

        return (self.numerical_features, self.categorical_features), self.labels

    def __len__(self):
        return self.num_batches

    @staticmethod
    def get_metadata(FLAGS):
        cardinalities = [int(d) for d in FLAGS.synthetic_dataset_cardinalities]
        metadata = DatasetMetadata(num_numerical_features=FLAGS.num_numerical_features,
                                   categorical_cardinalities=cardinalities)
        return metadata


def get_categorical_feature_type(size: int):
    types = (np.int8, np.int16, np.int32)

    for numpy_type in types:
        if size < np.iinfo(numpy_type).max:
            return numpy_type

    raise RuntimeError(f"Categorical feature of size {size} is too big for defined types")


class RawBinaryDataset:
    """Split version of Criteo dataset

    Args:
        data_path (str): Full path to split binary file of dataset. It must contain numerical.bin, label.bin and
            cat_0 ~ cat_25.bin
        batch_size (int):
        numerical_features(boolean): Number of numerical features to load, default=0 (don't load any)
        categorical_features (list or None): categorical features used by the rank (IDs of the features)
        categorical_feature_sizes (list of integers): max value of each of the categorical features
        prefetch_depth (int): How many samples to prefetch. Default 10.
    """

    _model_size_filename = 'model_size.json'

    def __init__(
        self,
        data_path: str,
        batch_size: int = 1,
        numerical_features: int = 0,
        categorical_features: Optional[Sequence[int]] = None,
        categorical_feature_sizes: Optional[Sequence[int]] = None,
        prefetch_depth: int = 10,
        drop_last_batch: bool = False,
        valid : bool = False,
    ):
        suffix = 'test' if valid else 'train'
        data_path = os.path.join(data_path, suffix)
        self._label_bytes_per_batch = np.dtype(np.bool).itemsize * batch_size
        self._numerical_bytes_per_batch = numerical_features * np.dtype(np.float16).itemsize * batch_size
        self._numerical_features = numerical_features

        self._categorical_feature_types = [
            get_categorical_feature_type(size) for size in categorical_feature_sizes
        ] if categorical_feature_sizes else []
        self._categorical_bytes_per_batch = [
            np.dtype(cat_type).itemsize * batch_size for cat_type in self._categorical_feature_types
        ]
        self._categorical_features = categorical_features
        self._batch_size = batch_size
        self._label_file = os.open(os.path.join(data_path, 'label.bin'), os.O_RDONLY)
        self._num_entries = int(math.ceil(os.fstat(self._label_file).st_size
                                          / self._label_bytes_per_batch)) if not drop_last_batch \
                            else int(math.floor(os.fstat(self._label_file).st_size / self._label_bytes_per_batch))

        if numerical_features > 0:
            self._numerical_features_file = os.open(os.path.join(data_path, "numerical.bin"), os.O_RDONLY)
            number_of_numerical_batches = math.ceil(os.fstat(self._numerical_features_file).st_size
                                                    / self._numerical_bytes_per_batch) if not drop_last_batch \
                                          else math.floor(os.fstat(self._numerical_features_file).st_size
                                                          / self._numerical_bytes_per_batch)
            if number_of_numerical_batches != self._num_entries:
                raise ValueError(f"Size mismatch in data files. Expected: {self._num_entries}, got: {number_of_numerical_batches}")
        else:
            self._numerical_features_file = None

        if categorical_features:
            self._categorical_features_files = []
            for cat_id in categorical_features:
                cat_file = os.open(os.path.join(data_path, f"cat_{cat_id}.bin"), os.O_RDONLY)
                cat_bytes = self._categorical_bytes_per_batch[cat_id]
                number_of_categorical_batches = math.ceil(os.fstat(cat_file).st_size / cat_bytes) if not drop_last_batch \
                                                else math.floor(os.fstat(cat_file).st_size / cat_bytes)
                if number_of_categorical_batches != self._num_entries:
                    raise ValueError(f"Size mismatch in data files. Expected: {self._num_entries}, got: {number_of_categorical_batches}")
                self._categorical_features_files.append(cat_file)
        else:
            self._categorical_features_files = None

        self._prefetch_depth = min(prefetch_depth, self._num_entries)
        self._prefetch_queue = queue.Queue()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    @classmethod
    def get_metadata(cls, path, num_numerical_features):
        with open(os.path.join(path, cls._model_size_filename), 'r') as f:
            global_table_sizes = json.load(f)

        global_table_sizes = list(global_table_sizes.values())
        global_table_sizes = [s + 1 for s in global_table_sizes]

        metadata = DatasetMetadata(num_numerical_features=num_numerical_features,
                                   categorical_cardinalities=global_table_sizes)
        return metadata

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

    def _get_item(self, idx: int) -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]]:
        click = self._get_label(idx)
        numerical_features = self._get_numerical_features(idx)
        categorical_features = self._get_categorical_features(idx)
        return (numerical_features, categorical_features), click

    def _get_label(self, idx: int) -> tf.Tensor:
        raw_label_data = os.pread(self._label_file, self._label_bytes_per_batch,
                                  idx * self._label_bytes_per_batch)
        array = np.frombuffer(raw_label_data, dtype=np.bool)
        array = tf.convert_to_tensor(array, dtype=tf.float32)
        array = tf.expand_dims(array, 1)
        return array

    def _get_numerical_features(self, idx: int) -> Optional[tf.Tensor]:
        if self._numerical_features_file is None:
            return -1

        raw_numerical_data = os.pread(self._numerical_features_file, self._numerical_bytes_per_batch,
                                      idx * self._numerical_bytes_per_batch)
        array = np.frombuffer(raw_numerical_data, dtype=np.float16)
        array = tf.convert_to_tensor(array)
        return tf.reshape(array, shape=[self._batch_size, self._numerical_features])

    def _get_categorical_features(self, idx: int) -> Optional[tf.Tensor]:
        if self._categorical_features_files is None:
            return -1

        categorical_features = []
        for cat_id, cat_file in zip(self._categorical_features, self._categorical_features_files):
            cat_bytes = self._categorical_bytes_per_batch[cat_id]
            cat_type = self._categorical_feature_types[cat_id]
            raw_cat_data = os.pread(cat_file, cat_bytes, idx * cat_bytes)
            array = np.frombuffer(raw_cat_data, dtype=cat_type)
            tensor = tf.convert_to_tensor(array)
            tensor = tf.expand_dims(tensor, axis=1)
            categorical_features.append(tensor)
        return categorical_features

    def __del__(self):
        data_files = [self._label_file, self._numerical_features_file]
        if self._categorical_features_files is not None:
            data_files += self._categorical_features_files

        for data_file in data_files:
            if data_file is not None:
                os.close(data_file)

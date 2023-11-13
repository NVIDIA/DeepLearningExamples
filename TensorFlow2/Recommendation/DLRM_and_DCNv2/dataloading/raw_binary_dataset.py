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
# author: Tomasz Grel (tgrel@nvidia.com), Tomasz Cheda (tcheda@nvidia.com)


import tensorflow as tf
import os
import numpy as np
from itertools import chain
from collections import namedtuple
from typing import Optional, Tuple, List

import tqdm

from .defaults import CATEGORICAL_CHANNEL, NUMERICAL_CHANNEL, LABEL_CHANNEL, DTYPE_SELECTOR
from .feature_spec import FeatureSpec, FEATURES_SELECTOR, FILES_SELECTOR, get_categorical_feature_type

DatasetMetadata = namedtuple('DatasetMetadata', ['num_numerical_features',
                                                 'categorical_cardinalities'])

fspec_type_to_tf_type = {
    'int8': tf.int8,
    'int16': tf.int16,
    'int32': tf.int32
}


def create_reader(filename, bytes_per_batch):
    fd = os.open(filename, os.O_RDONLY)
    file_len = os.fstat(fd).st_size
    os.close(fd)
    num_batches = int(file_len / bytes_per_batch)
    file_len_patched = num_batches * bytes_per_batch
    footer_bytes = file_len - file_len_patched

    reader = tf.data.FixedLengthRecordDataset(filenames=[filename],
                                              record_bytes=bytes_per_batch,
                                              footer_bytes=footer_bytes)
    return reader, num_batches


class TfRawBinaryDataset:
    """Dataset for reading labels, numerical and categorical features from
    a set of binary files. Internally uses TensorFlow's FixedLengthRecordDataset
    and decode_raw for best performance.

    """

    def __init__(
            self,
            feature_spec: FeatureSpec,
            instance: str,
            local_categorical_feature_names: List[str],
            batch_size: int = 1,
            numerical_features_enabled: bool = False,
            rank: int = 0,
            world_size: int = 1,
            concat_features: bool = False,
            data_parallel_categoricals = False,
    ):

        self._concat_features = concat_features
        self._feature_spec = feature_spec
        self._batch_size = batch_size
        self._data_parallel_categoricals = data_parallel_categoricals

        local_batch_size = int(batch_size / world_size)
        batch_sizes_per_gpu = [local_batch_size] * world_size
        indices = tuple(np.cumsum([0] + list(batch_sizes_per_gpu)))
        self.dp_begin_idx = indices[rank]
        self.dp_end_idx = indices[rank + 1]

        self._rank = rank
        self._world_size = world_size
        self._instance = instance
        feature_spec.check_feature_spec()
        self._create_readers(feature_spec, local_categorical_feature_names, numerical_features_enabled)
        self._categorical_types_tf = [fspec_type_to_tf_type[feature_spec.feature_spec[feature][DTYPE_SELECTOR]] for
                                      feature in
                                      local_categorical_feature_names]

    def _create_readers(self, feature_spec, local_categorical_feature_names, numerical_features_enabled):
        categorical_features = feature_spec.channel_spec[CATEGORICAL_CHANNEL]
        numerical_features = feature_spec.channel_spec[NUMERICAL_CHANNEL]
        label_features = feature_spec.channel_spec[LABEL_CHANNEL]
        self._number_of_numerical_features = len(numerical_features) if numerical_features_enabled else 0

        set_of_categorical_features = set(categorical_features)
        set_of_numerical_features = set(numerical_features)
        set_of_label_features = set(label_features)

        set_of_categoricals_to_read = set(local_categorical_feature_names)
        bytes_per_feature = {feature_name: np.dtype(feature_spec.feature_spec[feature_name][DTYPE_SELECTOR]).itemsize
                             for feature_name in feature_spec.feature_spec.keys()}
        chosen_instance = feature_spec.source_spec[self._instance]
        categorical_feature_readers = {}
        root_path = feature_spec.base_directory
        number_of_batches = None
        for chunk in chosen_instance:
            contained_features = chunk[FEATURES_SELECTOR]
            containing_file = chunk[FILES_SELECTOR][0]
            path_to_open = os.path.join(root_path, containing_file)
            first_feature = contained_features[0]

            if first_feature in set_of_categorical_features:
                # Load categorical
                # We verified earlier that only one feature is present per chunk
                if first_feature not in set_of_categoricals_to_read:
                    continue  # skip chunk

                bytes_per_batch = bytes_per_feature[first_feature] * self._batch_size
                reader, batches = create_reader(path_to_open, bytes_per_batch)
                categorical_feature_readers[first_feature] = reader

            elif first_feature in set_of_numerical_features:
                # Load numerical
                # We verified earlier that all numerical features are in one chunk
                if not numerical_features_enabled:
                    self._numerical = tuple()
                    continue  # skip chunk
                numerical_bytes_per_batch = bytes_per_feature[numerical_features[0]] * \
                                            len(numerical_features) * self._batch_size
                self._numerical, batches = create_reader(path_to_open, numerical_bytes_per_batch)

            elif first_feature in set_of_label_features:
                # Load label
                # We verified earlier that there is only one label feature
                label_bytes_per_batch = np.dtype(bool).itemsize * self._batch_size
                self._label, batches = create_reader(path_to_open, label_bytes_per_batch)
            else:
                raise ValueError("Unknown chunk type")

            if number_of_batches is not None:
                if batches != number_of_batches:
                    raise ValueError(f'Size mismatch. Expected: {number_of_batches}, got: {batches}')
            else:
                number_of_batches = batches

        self._categorical = tuple(categorical_feature_readers[feature] for feature in local_categorical_feature_names)
        self.num_batches = number_of_batches

    def __len__(self):
        return self.num_batches

    def op(self):
        pipeline = tf.data.Dataset.zip((self._label, self._numerical, self._categorical))
        pipeline = pipeline.map(self.decode_batch, num_parallel_calls=tf.data.AUTOTUNE)
        pipeline = pipeline.batch(batch_size=1)
        # Only one gpu is set to visible
        pipeline = pipeline.apply(tf.data.experimental.prefetch_to_device(f'/gpu:0'))
        pipeline = pipeline.unbatch()
        pipeline = pipeline.repeat()
        return pipeline

    @tf.function
    def decode_batch(self, labels, numerical_features, categorical_features, concat_features=False):
        labels = tf.io.decode_raw(labels, out_type=tf.int8)
        labels = labels[self.dp_begin_idx:self.dp_end_idx]

        if self._number_of_numerical_features > 0:
            numerical_features = tf.io.decode_raw(numerical_features, out_type=tf.float16)
            numerical_features = tf.reshape(numerical_features,
                                            shape=[-1, self._number_of_numerical_features])
            numerical_features = numerical_features[self.dp_begin_idx:self.dp_end_idx, :]

        if self._categorical:
            cat_data = []
            for dtype, feature in zip(self._categorical_types_tf, categorical_features):
                feature = tf.io.decode_raw(feature, out_type=dtype)
                feature = tf.cast(feature, dtype=tf.int32)
                feature = tf.expand_dims(feature, axis=1)
                feature = tf.reshape(feature, [self._batch_size, 1])
                if self._data_parallel_categoricals:
                    feature = feature[self.dp_begin_idx:self.dp_end_idx]
                cat_data.append(feature)
            if self._concat_features:
                cat_data = tf.concat(cat_data, axis=1)
            else:
                cat_data = tuple(cat_data)

        return (numerical_features, cat_data), labels

    @staticmethod
    def generate(src_train, src_test, feature_spec, dst_dir, dst_feature_spec,
                 max_batches_train, max_batches_test):

        categorical_sizes = feature_spec.get_categorical_sizes()
        num_numerical = feature_spec.get_number_of_numerical_features()
        feature_spec = FeatureSpec.get_default_feature_spec(number_of_numerical_features=num_numerical,
                                                            categorical_feature_cardinalities=categorical_sizes)

        feature_spec.to_yaml(output_path=os.path.join(dst_dir, dst_feature_spec))
        sources = [(src_train, 'train', max_batches_train), (src_test, 'test', max_batches_test)]

        cat_feature_types = [get_categorical_feature_type(cat_size) for cat_size in categorical_sizes]

        for src_dataset, split, max_batches in sources:
            os.makedirs(os.path.join(dst_dir, split), exist_ok=True)

            categorical_fs = []
            for i in range(len(categorical_sizes)):
                fs = open(os.path.join(dst_dir, split, f'cat_{i}.bin'), 'wb+')
                categorical_fs.append(fs)

            label_f = open(os.path.join(dst_dir, split, 'label.bin'), 'wb+')
            numerical_f = open(os.path.join(dst_dir, split, "numerical.bin"), "wb+")

            for batch_idx, src_batch in tqdm.tqdm(enumerate(src_dataset),
                                                  total=max_batches,
                                                  desc=f'Generating the {split} data'):
                if batch_idx == max_batches:
                    break

                (numerical_features, categorical_features), label = src_batch

                for ftype, stream, feature in zip(cat_feature_types, categorical_fs, categorical_features):
                    if isinstance(feature, tf.RaggedTensor):
                        feature = feature.values
                    raw_data = feature.numpy().astype(ftype).tobytes()
                    stream.write(raw_data)

                label_f.write(label.numpy().astype(bool).tobytes())
                numerical_f.write(numerical_features.numpy().astype(np.float16).tobytes())

            for stream in chain(*categorical_fs, [label_f, numerical_f]):
                stream.close()

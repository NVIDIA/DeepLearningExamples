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

from functools import partial
from multiprocessing import cpu_count

import tensorflow as tf

from data.outbrain.features import get_features_keys


def _consolidate_batch(elem):
    label = elem.pop('label')
    reshaped_label = tf.reshape(label, [-1, label.shape[-1]])
    features = get_features_keys()

    reshaped_elem = {
        key: tf.reshape(elem[key], [-1, elem[key].shape[-1]])
        for key in elem
        if key in features
    }

    return reshaped_elem, reshaped_label


def get_parse_function(feature_spec):
    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_spec)

    return _parse_function


def train_input_fn(
        filepath_pattern,
        feature_spec,
        records_batch_size,
        num_gpus=1,
        id=0):
    _parse_function = get_parse_function(feature_spec)

    dataset = tf.data.Dataset.list_files(
        file_pattern=filepath_pattern
    )

    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=cpu_count() // num_gpus,
        block_length=1
    )

    dataset = dataset.map(
        map_func=_parse_function,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    dataset = dataset.shard(num_gpus, id)

    dataset = dataset.shuffle(records_batch_size * 8)

    dataset = dataset.repeat(
        count=None
    )

    dataset = dataset.batch(
        batch_size=records_batch_size,
        drop_remainder=False
    )

    dataset = dataset.map(
        map_func=partial(
            _consolidate_batch
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    dataset = dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE
    )

    return dataset


def eval_input_fn(
        filepath_pattern,
        feature_spec,
        records_batch_size,
        num_gpus=1,
        repeat=1,
        id=0):
    dataset = tf.data.Dataset.list_files(
        file_pattern=filepath_pattern,
        shuffle=False
    )

    dataset = tf.data.TFRecordDataset(
        filenames=dataset,
        num_parallel_reads=1
    )

    dataset = dataset.shard(num_gpus, id)

    dataset = dataset.repeat(
        count=repeat
    )

    dataset = dataset.batch(
        batch_size=records_batch_size,
        drop_remainder=False
    )

    dataset = dataset.apply(
        transformation_func=tf.data.experimental.parse_example_dataset(
            features=feature_spec,
            num_parallel_calls=1
        )
    )

    dataset = dataset.map(
        map_func=partial(
            _consolidate_batch
        ),
        num_parallel_calls=None
    )
    dataset = dataset.prefetch(
        buffer_size=1
    )

    return dataset

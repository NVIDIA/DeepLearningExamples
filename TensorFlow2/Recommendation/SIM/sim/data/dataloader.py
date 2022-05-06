# Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
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

import multiprocessing
from functools import partial

import tensorflow as tf

from sim.data.defaults import (DIMENSIONS_SELECTOR, LABEL_CHANNEL, NEGATIVE_HISTORY_CHANNEL, POSITIVE_HISTORY_CHANNEL,
                               TARGET_ITEM_FEATURES_CHANNEL, USER_FEATURES_CHANNEL)


def _pad_ragged_infront(x, pad_length):
    x = tf.reverse(x, axis=[1])
    return tf.reverse(x.to_tensor(shape=(None, pad_length)), axis=[1])


def _remap_column_values_tfrecord(sample, feature_spec, long_seq_length):

    channel_spec = feature_spec.channel_spec
    features = feature_spec.feature_spec

    user_features = {
        f_name: sample[f_name] for f_name in channel_spec[USER_FEATURES_CHANNEL]
    }

    target_item_features = {
        f_name: sample[f_name] for f_name in channel_spec[TARGET_ITEM_FEATURES_CHANNEL]
    }

    padded_positive = {
        f_name: _pad_ragged_infront(sample[f_name], features[f_name][DIMENSIONS_SELECTOR][0])
        for f_name in channel_spec[POSITIVE_HISTORY_CHANNEL]
    }

    padded_negative = {
        f_name: _pad_ragged_infront(sample[f_name], features[f_name][DIMENSIONS_SELECTOR][0])
        for f_name in channel_spec[NEGATIVE_HISTORY_CHANNEL]
    }

    long_sequence_features = {
        f_name: val[:, :long_seq_length] for f_name, val in padded_positive.items()
    }

    short_sequence_features = {
        f_name: val[:, long_seq_length:] for f_name, val in padded_positive.items()
    }

    short_neg_sequence_features = {
        f_name: val[:, long_seq_length:] for f_name, val in padded_negative.items()
    }

    first_positive_feature_name = channel_spec[POSITIVE_HISTORY_CHANNEL][0]
    first_positive_feature = padded_positive[first_positive_feature_name]

    history_mask = tf.cast(tf.greater(first_positive_feature, 0), tf.float32)

    long_sequence_mask = history_mask[:, :long_seq_length]
    short_sequence_mask = history_mask[:, long_seq_length:]

    label_name = channel_spec[LABEL_CHANNEL][0]
    target = sample[label_name]

    return {
        "user_features": user_features,
        "target_item_features": target_item_features,
        "long_sequence_features": long_sequence_features,
        "short_sequence_features": short_sequence_features,
        "short_neg_sequence_features": short_neg_sequence_features,
        "long_sequence_mask": long_sequence_mask,
        "short_sequence_mask": short_sequence_mask,
        "other_features": None
    }, target


def get_dataloader_tfrecord(
    file_paths,
    feature_spec,
    batch_size,
    long_seq_length,
    num_gpus=1,
    id=0,
    drop_remainder=False,
    repeat_count=0,
    prefetch_buffer_size=90,
    disable_cache=False):

    features = feature_spec.feature_spec

    tf_feature_spec = {}
    for name, feature in features.items():
        dimensions = feature.get(DIMENSIONS_SELECTOR)
        if dimensions is None:
            tf_feature_spec[name] = tf.io.FixedLenFeature([], tf.int64)
        else:
            tf_feature_spec[name] = tf.io.RaggedFeature(tf.int64)

    num_cpus = multiprocessing.cpu_count()

    dataset = tf.data.TFRecordDataset(file_paths)

    dataset = dataset.shard(num_gpus, id)

    dataset = dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size, drop_remainder=drop_remainder)
    )

    dataset = dataset.map(
        map_func=partial(tf.io.parse_example, features=tf_feature_spec),
        num_parallel_calls=num_cpus//num_gpus
    )

    dataset = dataset.map(
        map_func=partial(_remap_column_values_tfrecord, feature_spec=feature_spec, long_seq_length=long_seq_length),
        num_parallel_calls=num_cpus//num_gpus
    )

    if repeat_count > 0:
        dataset = dataset.repeat(
            count=repeat_count
        )

    if prefetch_buffer_size > 0:
        dataset = dataset.prefetch(
            buffer_size=prefetch_buffer_size
        )

    if not disable_cache:
        dataset = dataset.cache()

    return dataset

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import tensorflow as tf
from trainer import features

# rough approximation for MAP metric for measuring ad quality
# roughness comes from batch sizes falling between groups of
# display ids
# hack because of name clashes. Probably makes sense to rename features
DISPLAY_ID_COLUMN = features.DISPLAY_ID_COLUMN


def map_custom_metric(features, labels, predictions):
    display_ids = tf.reshape(features[DISPLAY_ID_COLUMN], [-1])
    predictions = predictions['probabilities'][:, 1]
    labels = labels[:, 0]

    # Processing unique display_ids, indexes and counts
    # Sorting needed in case the same display_id occurs in two different places
    sorted_ids = tf.argsort(display_ids)
    display_ids = tf.gather(display_ids, indices=sorted_ids)
    predictions = tf.gather(predictions, indices=sorted_ids)
    labels = tf.gather(labels, indices=sorted_ids)

    _, display_ids_idx, display_ids_ads_count = tf.unique_with_counts(
        display_ids, out_idx=tf.int64)
    pad_length = 30 - tf.reduce_max(display_ids_ads_count)
    pad_fn = lambda x: tf.pad(x, [(0, 0), (0, pad_length)])

    preds = tf.RaggedTensor.from_value_rowids(
        predictions, display_ids_idx).to_tensor()
    labels = tf.RaggedTensor.from_value_rowids(
        labels, display_ids_idx).to_tensor()

    labels = tf.argmax(labels, axis=1)

    return {
        'map': tf.compat.v1.metrics.average_precision_at_k(
            predictions=pad_fn(preds),
            labels=labels,
            k=12,
            name="streaming_map")}


IS_LEAK_COLUMN = features.IS_LEAK_COLUMN


def map_custom_metric_with_leak(features, labels, predictions):
    display_ids = features[DISPLAY_ID_COLUMN]
    display_ids = tf.reshape(display_ids, [-1])
    is_leak_tf = features[IS_LEAK_COLUMN]
    is_leak_tf = tf.reshape(is_leak_tf, [-1])

    predictions = predictions['probabilities'][:, 1]
    predictions = predictions + tf.cast(is_leak_tf, tf.float32)
    labels = labels[:, 0]

    # Processing unique display_ids, indexes and counts
    # Sorting needed in case the same display_id occurs in two different places
    sorted_ids = tf.argsort(display_ids)
    display_ids = tf.gather(display_ids, indices=sorted_ids)
    predictions = tf.gather(predictions, indices=sorted_ids)
    labels = tf.gather(labels, indices=sorted_ids)

    _, display_ids_idx, display_ids_ads_count = tf.unique_with_counts(
        display_ids, out_idx=tf.int64)
    pad_length = 30 - tf.reduce_max(display_ids_ads_count)
    pad_fn = lambda x: tf.pad(x, [(0, 0), (0, pad_length)])

    preds = tf.RaggedTensor.from_value_rowids(predictions, display_ids_idx).to_tensor()
    labels = tf.RaggedTensor.from_value_rowids(labels, display_ids_idx).to_tensor()
    labels = tf.argmax(labels, axis=1)

    return {
        'map_with_leak': tf.compat.v1.metrics.average_precision_at_k(
            predictions=pad_fn(preds),
            labels=labels,
            k=12,
            name="streaming_map_with_leak")}

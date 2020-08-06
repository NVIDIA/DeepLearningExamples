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
from tensorflow.compat.v1 import logging


def separate_input_fn(
        tf_transform_output,
        transformed_examples,
        create_batches,
        mode,
        reader_num_threads=1,
        parser_num_threads=2,
        shuffle_buffer_size=10,
        prefetch_buffer_size=1,
        print_display_ids=False):
    """
    A version of the training + eval input function that uses dataset operations.
    (For more straightforward tweaking.)
    """

    logging.warn('Shuffle buffer size: {}'.format(shuffle_buffer_size))

    filenames_dataset = tf.data.Dataset.list_files(
        transformed_examples,
        shuffle=False
    )

    raw_dataset = tf.data.TFRecordDataset(
        filenames_dataset,
        num_parallel_reads=reader_num_threads
    )

    if mode == tf.estimator.ModeKeys.TRAIN and shuffle_buffer_size > 1:
        raw_dataset = raw_dataset.shuffle(shuffle_buffer_size)

    raw_dataset = raw_dataset.repeat()
    raw_dataset = raw_dataset.batch(create_batches)

    # this function appears to require each element to be a vector
    # batching should mean that this is always true
    # one possible alternative for any problematic case is tf.io.parse_single_example
    parsed_dataset = raw_dataset.apply(
        tf.data.experimental.parse_example_dataset(
            tf_transform_output.transformed_feature_spec(),
            num_parallel_calls=parser_num_threads
        )
    )

    # a function mapped over each dataset element
    # will separate label, ensure that elements are two-dimensional (batch size, elements per record)
    # adds print_display_ids injection
    def consolidate_batch(elem):
        label = elem.pop('label')
        reshaped_label = tf.reshape(label, [-1, label.shape[-1]])
        reshaped_elem = {
            key: tf.reshape(elem[key], [-1, elem[key].shape[-1]])
            for key in elem
        }

        if print_display_ids:
            elem['ad_id'] = tf.Print(input_=elem['ad_id'],
                                     data=[tf.reshape(elem['display_id'], [-1])],
                                     message='display_id', name='print_display_ids',
                                     summarize=elem['ad_id'].shape[1])
            elem['ad_id'] = tf.Print(input_=elem['ad_id'],
                                     data=[tf.reshape(elem['ad_id'], [-1])],
                                     message='ad_id', name='print_ad_ids',
                                     summarize=elem['ad_id'].shape[1])
            elem['ad_id'] = tf.Print(input_=elem['ad_id'],
                                     data=[tf.reshape(elem['is_leak'], [-1])],
                                     message='is_leak', name='print_is_leak',
                                     summarize=elem['ad_id'].shape[1])

        return reshaped_elem, reshaped_label

    if mode == tf.estimator.ModeKeys.EVAL:
        parsed_dataset = parsed_dataset.map(
            consolidate_batch,
            num_parallel_calls=None
        )
    else:
        parsed_dataset = parsed_dataset.map(
            consolidate_batch,
            num_parallel_calls=parser_num_threads
        )
        parsed_dataset = parsed_dataset.prefetch(prefetch_buffer_size)

    return parsed_dataset

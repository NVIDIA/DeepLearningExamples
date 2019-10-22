#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

import sys

import tensorflow as tf
import horovod.tensorflow as hvd

from utils import image_processing
from utils import hvd_utils
from utils import dali_utils

__all__ = ["get_synth_input_fn", "normalized_inputs"]

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

_NUM_CHANNELS = 3


def get_synth_input_fn(batch_size, height, width, num_channels, data_format, num_classes, dtype=tf.float32):
    """Returns an input function that returns a dataset with random data.

    This input_fn returns a data set that iterates over a set of random data and
    bypasses all preprocessing, e.g. jpeg decode and copy. The host to device
    copy is still included. This used to find the upper throughput bound when
    tunning the full input pipeline.

    Args:
        height: Integer height that will be used to create a fake image tensor.
        width: Integer width that will be used to create a fake image tensor.
        num_channels: Integer depth that will be used to create a fake image tensor.
        num_classes: Number of classes that should be represented in the fake labels
            tensor
        dtype: Data type for features/images.

    Returns:
        An input_fn that can be used in place of a real one to return a dataset
        that can be used for iteration.
    """

    if data_format not in ["NHWC", "NCHW"]:
        raise ValueError("Unknown data_format: %s" % str(data_format))

    if data_format == "NHWC":
        input_shape = [batch_size, height, width, num_channels]
    else:
        input_shape = [batch_size, num_channels, height, width]

    # Convert the inputs to a Dataset.
    inputs = tf.truncated_normal(input_shape, dtype=dtype, mean=127, stddev=60, name='synthetic_inputs')
    labels = tf.random_uniform([batch_size], minval=0, maxval=num_classes - 1, dtype=tf.int32, name='synthetic_labels')

    data = tf.data.Dataset.from_tensors((inputs, labels))

    data = data.repeat()

    data = data.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    return data


def get_tfrecords_input_fn(filenames, batch_size, height, width, training, distort_color, num_threads, deterministic):

    shuffle_buffer_size = 4096

    if deterministic:
        if hvd_utils.is_using_hvd():
            seed = 13 * (1 + hvd.rank())
        else:
            seed = 13
    else:
        seed = None

    ds = tf.data.Dataset.from_tensor_slices(filenames)

    if hvd_utils.is_using_hvd() and training:
        ds = ds.shard(hvd.size(), hvd.rank())

    ds = ds.apply(
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset,
            cycle_length=10,
            block_length=8,
            sloppy=not deterministic,
            prefetch_input_elements=16
        )
    )

    counter = tf.data.Dataset.range(sys.maxsize)
    ds = tf.data.Dataset.zip((ds, counter))

    def preproc_func(record, counter_):
        return image_processing.preprocess_image_record(record, height, width, _NUM_CHANNELS, training)

    ds = ds.cache()
    
    if training:

        ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=shuffle_buffer_size, seed=seed))

    else:
        ds = ds.repeat()

    ds = ds.apply(
        tf.data.experimental.map_and_batch(
            map_func=preproc_func,
            num_parallel_calls=num_threads,
            batch_size=batch_size,
            drop_remainder=True,
        )
    )

    ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    return ds

def get_inference_input_fn(filenames, height, width, num_threads):
    
    ds = tf.data.Dataset.from_tensor_slices(filenames)

    counter = tf.data.Dataset.range(sys.maxsize)
    ds = tf.data.Dataset.zip((ds, counter))

    def preproc_func(record, counter_):
        return image_processing.preprocess_image_file(record, height, width, _NUM_CHANNELS, is_training=False)
    
    ds = ds.apply(
        tf.data.experimental.map_and_batch(
            map_func=preproc_func,
            num_parallel_calls=num_threads,
            batch_size=1
        )
    )

    ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    
    return ds

    
    
def get_dali_input_fn(filenames, idx_filenames, batch_size, height, width, training, distort_color, num_threads, deterministic):

    if idx_filenames is None:
        raise ValueError("Must provide idx_filenames for DALI's reader")
        
    preprocessor = dali_utils.DALIPreprocessor(
        filenames,
        idx_filenames,
        height, width,
        batch_size,
        num_threads,
        dali_cpu=False,
        deterministic=deterministic,
        training=training)
    
    images, labels = preprocessor.get_device_minibatches()
    
    return (images, labels)


def normalized_inputs(inputs):

    num_channels = inputs.get_shape()[-1]

    if inputs.get_shape().ndims != 4:
        raise ValueError('Input must be of size [batch_size, height, width, C>0]')

    if len(_CHANNEL_MEANS) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    # We have a 1-D tensor of means; convert to 3-D.
    means_per_channel = tf.reshape(_CHANNEL_MEANS, [1, 1, num_channels])
    means_per_channel = tf.cast(means_per_channel, dtype=inputs.dtype)

    inputs = tf.subtract(inputs, means_per_channel)

    return tf.divide(inputs, 255.0)

def get_serving_input_receiver_fn(batch_size, height, width, num_channels, data_format, dtype=tf.float32):
    
    if data_format not in ["NHWC", "NCHW"]:
        raise ValueError("Unknown data_format: %s" % str(data_format))

    if data_format == "NHWC":
        input_shape = [batch_size] + [height, width, num_channels]
    else:
        input_shape = [batch_size] + [num_channels, height, width]
        
    def serving_input_receiver_fn():
        features = tf.placeholder(dtype=dtype, shape=input_shape, name='input_tensor')
        return tf.estimator.export.TensorServingInputReceiver(features=features, receiver_tensors=features)
    
    return serving_input_receiver_fn

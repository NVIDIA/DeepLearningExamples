# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

# -*- coding: utf-8 -*-
""" Contains a set of utilities that allow building the UNet model

"""

import tensorflow as tf


def _crop_and_concat(inputs, residual_input):
    """ Perform a central crop of ``residual_input`` and concatenate to ``inputs``

    Args:
        inputs (tf.Tensor): Tensor with input
        residual_input (tf.Tensor): Residual input

    Return:
        Concatenated tf.Tensor with the size of ``inputs``

    """

    factor = inputs.shape[1].value / residual_input.shape[1].value
    return tf.concat([inputs, tf.image.central_crop(residual_input, factor)], axis=-1)


def downsample_block(inputs, filters, idx):
    """ UNet downsample block

    Perform 2 unpadded convolutions with a specified number of filters and downsample
    through max-pooling

    Args:
        inputs (tf.Tensor): Tensor with inputs
        filters (int): Number of filters in convolution

    Return:
        Tuple of convolved ``inputs`` after and before downsampling

    """

    out = inputs

    with tf.name_scope('downsample_block_{}'.format(idx)):
        out = tf.layers.conv2d(inputs=out,
                               filters=filters,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu)
        out = tf.layers.conv2d(inputs=out,
                               filters=filters,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu)
        return tf.layers.max_pooling2d(inputs=out, pool_size=(2, 2), strides=2), out


def upsample_block(inputs, residual_input, filters, idx):
    """ UNet upsample block

    Perform 2 unpadded convolutions with a specified number of filters and upsample

    Args:
        inputs (tf.Tensor): Tensor with inputs
        residual_input (tf.Tensor): Residual input
        filters (int): Number of filters in convolution

    Return:
       Convolved ``inputs`` after upsampling

    """
    out = _crop_and_concat(inputs, residual_input)

    with tf.name_scope('upsample_block_{}'.format(idx)):
        out = tf.layers.conv2d(inputs=out,
                               filters=filters,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu)
        out = tf.layers.conv2d(inputs=out,
                               filters=int(filters),
                               kernel_size=(3, 3),
                               activation=tf.nn.relu)
        return tf.layers.conv2d_transpose(inputs=out,
                                          filters=int(filters // 2),
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='same',
                                          activation=tf.nn.relu)


def bottleneck(inputs, filters, mode):
    """ UNet central block

    Perform 2 unpadded convolutions with a specified number of filters and upsample
    including dropout before upsampling for training

    Args:
        inputs (tf.Tensor): Tensor with inputs
        filters (int): Number of filters in convolution

    Return:
        Convolved ``inputs`` after upsampling

    """
    out = inputs

    with tf.name_scope('bottleneck'):
        out = tf.layers.conv2d(inputs=out,
                               filters=filters,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu)
        out = tf.layers.conv2d(inputs=out,
                               filters=filters,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu)

        training = (mode == tf.estimator.ModeKeys.TRAIN)

        out = tf.layers.dropout(out, rate=0.5, training=training)

        return tf.layers.conv2d_transpose(inputs=out,
                                          filters=filters // 2,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='same',
                                          activation=tf.nn.relu)


def output_block(inputs, residual_input, filters, n_classes):
    """ UNet output

    Perform 3 unpadded convolutions, the last one with the same number
    of channels as classes we want to classify

    Args:
        inputs (tf.Tensor): Tensor with inputs
        residual_input (tf.Tensor): Residual input
        filters (int): Number of filters in convolution
        n_classes (int): Number of output classes

    Return:
        Convolved ``inputs`` with as many channels as classes

    """

    out = _crop_and_concat(inputs, residual_input)

    with tf.name_scope('output'):
        out = tf.layers.conv2d(inputs=out,
                               filters=filters,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu)
        out = tf.layers.conv2d(inputs=out,
                               filters=filters,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu)

        return tf.layers.conv2d(inputs=out,
                                filters=n_classes,
                                kernel_size=(1, 1),
                                activation=tf.nn.relu)


def input_block(inputs, filters):
    """ UNet input block

    Perform 2 unpadded convolutions with a specified number of filters and downsample
    through max-pooling. First convolution

    Args:
        inputs (tf.Tensor): Tensor with inputs
        filters (int): Number of filters in convolution

    Return:
        Tuple of convolved ``inputs`` after and before downsampling

    """

    out = inputs

    with tf.name_scope('input'):
        out = tf.layers.conv2d(inputs=out,
                               filters=filters,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu)
        out = tf.layers.conv2d(inputs=out,
                               filters=filters,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu)
        return tf.layers.max_pooling2d(inputs=out, pool_size=(2, 2), strides=2), out

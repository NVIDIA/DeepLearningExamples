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
    factor = inputs.shape[1] / residual_input.shape[1]
    return tf.concat([inputs, tf.image.central_crop(residual_input, factor)], axis=-1)


class InputBlock(tf.keras.Model):
    def __init__(self, filters):
        """ UNet input block

        Perform two unpadded convolutions with a specified number of filters and downsample
        through max-pooling. First convolution

        Args:
            filters (int): Number of filters in convolution
        """
        super().__init__(self)
        with tf.name_scope('input_block'):
            self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                                                kernel_size=(3, 3),
                                                activation=tf.nn.relu)
            self.conv2 = tf.keras.layers.Conv2D(filters=filters,
                                                kernel_size=(3, 3),
                                                activation=tf.nn.relu)
            self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

    def call(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        mp = self.maxpool(out)
        return mp, out


class DownsampleBlock(tf.keras.Model):
    def __init__(self, filters, idx):
        """ UNet downsample block

        Perform two unpadded convolutions with a specified number of filters and downsample
        through max-pooling

        Args:
            filters (int): Number of filters in convolution
            idx (int): Index of block

        Return:
            Tuple of convolved ``inputs`` after and before downsampling

        """
        super().__init__(self)
        with tf.name_scope('downsample_block_{}'.format(idx)):
            self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                                                kernel_size=(3, 3),
                                                activation=tf.nn.relu)
            self.conv2 = tf.keras.layers.Conv2D(filters=filters,
                                                kernel_size=(3, 3),
                                                activation=tf.nn.relu)
            self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

    def call(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        mp = self.maxpool(out)
        return mp, out


class BottleneckBlock(tf.keras.Model):
    def __init__(self, filters):
        """ UNet central block

        Perform two unpadded convolutions with a specified number of filters and upsample
        including dropout before upsampling for training

        Args:
            filters (int): Number of filters in convolution
        """
        super().__init__(self)
        with tf.name_scope('bottleneck_block'):
            self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                                                kernel_size=(3, 3),
                                                activation=tf.nn.relu)
            self.conv2 = tf.keras.layers.Conv2D(filters=filters,
                                                kernel_size=(3, 3),
                                                activation=tf.nn.relu)
            self.dropout = tf.keras.layers.Dropout(rate=0.5)
            self.conv_transpose = tf.keras.layers.Conv2DTranspose(filters=filters // 2,
                                                                  kernel_size=(3, 3),
                                                                  strides=(2, 2),
                                                                  padding='same',
                                                                  activation=tf.nn.relu)

    def call(self, inputs, training):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.dropout(out, training=training)
        out = self.conv_transpose(out)
        return out


class UpsampleBlock(tf.keras.Model):
    def __init__(self, filters, idx):
        """ UNet upsample block

        Perform two unpadded convolutions with a specified number of filters and upsample

        Args:
            filters (int): Number of filters in convolution
            idx (int): Index of block
        """
        super().__init__(self)
        with tf.name_scope('upsample_block_{}'.format(idx)):
            self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                                                kernel_size=(3, 3),
                                                activation=tf.nn.relu)
            self.conv2 = tf.keras.layers.Conv2D(filters=filters,
                                                kernel_size=(3, 3),
                                                activation=tf.nn.relu)
            self.conv_transpose = tf.keras.layers.Conv2DTranspose(filters=filters // 2,
                                                                  kernel_size=(3, 3),
                                                                  strides=(2, 2),
                                                                  padding='same',
                                                                  activation=tf.nn.relu)

    def call(self, inputs, residual_input):
        out = _crop_and_concat(inputs, residual_input)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv_transpose(out)
        return out


class OutputBlock(tf.keras.Model):
    def __init__(self, filters, n_classes):
        """ UNet output block

        Perform three unpadded convolutions, the last one with the same number
        of channels as classes we want to classify

        Args:
            filters (int): Number of filters in convolution
            n_classes (int): Number of output classes
        """
        super().__init__(self)
        with tf.name_scope('output_block'):
            self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                                                kernel_size=(3, 3),
                                                activation=tf.nn.relu)
            self.conv2 = tf.keras.layers.Conv2D(filters=filters,
                                                kernel_size=(3, 3),
                                                activation=tf.nn.relu)
            self.conv3 = tf.keras.layers.Conv2D(filters=n_classes,
                                                kernel_size=(1, 1),
                                                activation=tf.nn.relu)

    def call(self, inputs, residual_input):
        out = _crop_and_concat(inputs, residual_input)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

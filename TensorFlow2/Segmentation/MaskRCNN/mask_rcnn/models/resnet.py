#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Resnet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.keras import backend

from mask_rcnn.models.keras_utils import KerasMockLayer

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-4


class BNReLULayer(KerasMockLayer):
    def __init__(self, trainable, relu=True, init_zero=False, data_format='channels_last'):
        """Performs a batch normalization followed by a ReLU.

        Args:
        inputs: `Tensor` of shape `[batch, channels, ...]`.
        trainable: `bool` for whether to finetune the batchnorm layer.
        relu: `bool` if False, omits the ReLU operation.
        init_zero: `bool` if True, initializes scale parameter of batch
            normalization with 0 instead of 1 (default).
        data_format: `str` either "channels_first" for `[batch, channels, height,
            width]` or "channels_last for `[batch, height, width, channels]`.
        name: the name of the batch normalization layer

        Returns:
        A normalized `Tensor` with the same `data_format`.
        """
        super(BNReLULayer, self).__init__(trainable=trainable)

        if init_zero:
            gamma_initializer = tf.keras.initializers.Zeros()
        else:
            gamma_initializer = tf.keras.initializers.Ones()

        if data_format == 'channels_first':
            axis = 1
        else:
            axis = 3

        self._local_layers = dict()
        self._local_layers["batchnorm"] = tf.keras.layers.BatchNormalization(
            axis=axis,
            momentum=_BATCH_NORM_DECAY,
            epsilon=_BATCH_NORM_EPSILON,
            center=True,
            scale=True,
            trainable=self._trainable,
            gamma_initializer=gamma_initializer,
            fused=True,
            name="batch_normalization"
        )

        if relu:
            self._local_layers["relu"] = tf.keras.layers.ReLU()

    def __call__(self, inputs, training=False, *args, **kwargs):

        net = self._local_layers["batchnorm"](inputs, training=training and self._trainable)

        try:
            return self._local_layers["relu"](net)
        except KeyError:
            return net


class FixedPaddingLayer(KerasMockLayer):
    def __init__(self, kernel_size, data_format='channels_last', trainable=True):
        """Pads the input along the spatial dimensions independently of input size.

        Args:
        kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
            operations. Should be a positive integer.
        data_format: `str` either "channels_first" for `[batch, channels, height,
            width]` or "channels_last for `[batch, height, width, channels]`.
        """
        super(FixedPaddingLayer, self).__init__(trainable=trainable)

        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        if data_format == 'channels_first':
            self._paddings = [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]]
        else:
            self._paddings = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]

    def __call__(self, inputs, *args, **kwargs):
        """
      Args:
        inputs: `Tensor` of size `[batch, channels, height, width]` or
            `[batch, height, width, channels]` depending on `data_format`.
      Returns:
        A padded `Tensor` of the same `data_format` with size either intact
        (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
        :param **kwargs:
      """

        return tf.pad(tensor=inputs, paddings=self._paddings)


class Conv2dFixedPadding(KerasMockLayer):
    def __init__(self, filters, kernel_size, strides, data_format='channels_last', trainable=False):
        """Strided 2-D convolution with explicit padding.

        The padding is consistent and is based only on `kernel_size`, not on the
        dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

        Args:
        inputs: `Tensor` of size `[batch, channels, height_in, width_in]`.
        filters: `int` number of filters in the convolution.
        kernel_size: `int` size of the kernel to be used in the convolution.
        strides: `int` strides of the convolution.
        data_format: `str` either "channels_first" for `[batch, channels, height,
            width]` or "channels_last for `[batch, height, width, channels]`.

        Returns:
            A `Tensor` of shape `[batch, filters, height_out, width_out]`.
        """
        super(Conv2dFixedPadding, self).__init__(trainable=trainable)

        if strides > 1:
            self._local_layers["fixed_padding"] = FixedPaddingLayer(kernel_size=kernel_size, data_format=data_format)

        self._local_layers["conv2d"] = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'),
            use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            data_format=data_format,
            trainable=self._trainable,
            name="conv2d"
        )

    def __call__(self, inputs, *args, **kwargs):

        try:
            net = self._local_layers["fixed_padding"](inputs)
        except KeyError:
            net = inputs

        return self._local_layers["conv2d"](net)


class ResidualBlock(KerasMockLayer):
    def __init__(self, filters, trainable, finetune_bn, strides, use_projection=False, data_format='channels_last'):
        """Standard building block for residual networks with BN after convolutions.

        Args:
        filters: `int` number of filters for the first two convolutions. Note that
            the third and final convolution will use 4 times as many filters.
        finetune_bn: `bool` for whether the model is in training.
        strides: `int` block stride. If greater than 1, this block will ultimately downsample the input.
        use_projection: `bool` for whether this block should use a projection
            shortcut (versus the default identity shortcut). This is usually `True`
            for the first block of a block group, which may change the number of
            filters and the resolution.
        data_format: `str` either "channels_first" for `[batch, channels, height, width]`
            or "channels_last for `[batch, height, width, channels]`.
        """
        super(ResidualBlock, self).__init__(trainable=trainable)

        self._finetune_bn = finetune_bn

        if use_projection:
            self._local_layers["projection"] = dict()

            self._local_layers["projection"]["conv2d"] = Conv2dFixedPadding(
                filters=filters,
                kernel_size=1,
                strides=strides,
                data_format=data_format,
                trainable=trainable
            )

            self._local_layers["projection"]["batchnorm"] = BNReLULayer(
                trainable=finetune_bn and trainable,
                relu=False,
                init_zero=False,
                data_format=data_format,
            )

        self._local_layers["conv2d_1"] = Conv2dFixedPadding(
            trainable=trainable,
            filters=filters,
            kernel_size=3,
            strides=strides,
            data_format=data_format,
        )

        self._local_layers["conv2d_2"] = Conv2dFixedPadding(
            trainable=trainable,
            filters=filters,
            kernel_size=3,
            strides=1,
            data_format=data_format,
        )

        self._local_layers["batchnorm_1"] = BNReLULayer(
            trainable=finetune_bn and trainable,
            relu=True,
            init_zero=False,
            data_format=data_format,
        )

        self._local_layers["batchnorm_2"] = BNReLULayer(
            trainable=finetune_bn and trainable,
            relu=False,
            init_zero=True,
            data_format=data_format,
        )

        self._local_layers["activation"] = tf.keras.layers.ReLU()

    def __call__(self, inputs, training=False):
        """
        Args:
        inputs: `Tensor` of size `[batch, channels, height, width]`.

        Returns:
        The output `Tensor` of the block.
        """

        try:
            # Projection shortcut in first layer to match filters and strides
            shortcut = self._local_layers["projection"]["conv2d"](inputs=inputs)

            shortcut = self._local_layers["projection"]["batchnorm"](
                inputs=shortcut,
                training=training and self._trainable and self._finetune_bn
            )

        except KeyError:
            shortcut = inputs

        net = inputs

        for i in range(1, 3):
            net = self._local_layers["conv2d_%d" % i](inputs=net)

            net = self._local_layers["batchnorm_%d" % i](
                inputs=net,
                training=training and self._trainable and self._finetune_bn
            )

        return self._local_layers["activation"](net + shortcut)


class BottleneckBlock(KerasMockLayer):
    def __init__(self, filters, trainable, finetune_bn, strides, use_projection=False, data_format='channels_last'):
        """Bottleneck block variant for residual networks with BN after convolutions.

        Args:
        filters: `int` number of filters for the first two convolutions. Note that
            the third and final convolution will use 4 times as many filters.
        finetune_bn: `bool` for whether the model is in training.
        strides: `int` block stride. If greater than 1, this block will ultimately downsample the input.
        use_projection: `bool` for whether this block should use a projection
            shortcut (versus the default identity shortcut). This is usually `True`
            for the first block of a block group, which may change the number of
            filters and the resolution.
        data_format: `str` either "channels_first" for `[batch, channels, height, width]`
            or "channels_last for `[batch, height, width, channels]`.
        """
        super(BottleneckBlock, self).__init__(trainable=trainable)

        self._finetune_bn = finetune_bn

        if use_projection:
            # Projection shortcut only in first block within a group. Bottleneck blocks
            # end with 4 times the number of filters.
            filters_out = 4 * filters

            self._local_layers["projection"] = dict()

            self._local_layers["projection"]["conv2d"] = Conv2dFixedPadding(
                filters=filters_out,
                kernel_size=1,
                strides=strides,
                data_format=data_format,
                trainable=trainable
            )

            self._local_layers["projection"]["batchnorm"] = BNReLULayer(
                trainable=finetune_bn and trainable,
                relu=False,
                init_zero=False,
                data_format=data_format,
            )

        self._local_layers["conv2d_1"] = Conv2dFixedPadding(
            filters=filters,
            kernel_size=1,
            strides=1,
            data_format=data_format,
            trainable=trainable
        )

        self._local_layers["conv2d_2"] = Conv2dFixedPadding(
            filters=filters,
            kernel_size=3,
            strides=strides,
            data_format=data_format,
            trainable=trainable
        )

        self._local_layers["conv2d_3"] = Conv2dFixedPadding(
            filters=4 * filters,
            kernel_size=1,
            strides=1,
            data_format=data_format,
            trainable=trainable
        )

        self._local_layers["batchnorm_1"] = BNReLULayer(
            trainable=finetune_bn and trainable,
            relu=True,
            init_zero=False,
            data_format=data_format,
        )

        self._local_layers["batchnorm_2"] = BNReLULayer(
            trainable=finetune_bn and trainable,
            relu=True,
            init_zero=False,
            data_format=data_format,
        )

        self._local_layers["batchnorm_3"] = BNReLULayer(
            trainable=finetune_bn and trainable,
            relu=False,
            init_zero=True,
            data_format=data_format,
        )

        self._local_layers["activation"] = tf.keras.layers.ReLU()

    def __call__(self, inputs, training=False):
        """
        Args:
        inputs: `Tensor` of size `[batch, channels, height, width]`.

        Returns:
        The output `Tensor` of the block.
        """

        try:
            # Projection shortcut in first layer to match filters and strides
            shortcut = self._local_layers["projection"]["conv2d"](inputs=inputs)

            shortcut = self._local_layers["projection"]["batchnorm"](
                inputs=shortcut,
                training=training and self._trainable and self._finetune_bn
            )

        except KeyError:
            shortcut = inputs

        net = inputs

        for i in range(1, 4):
            net = self._local_layers["conv2d_%d" % i](inputs=net)

            net = self._local_layers["batchnorm_%d" % i](
                inputs=net,
                training=training and self._trainable and self._finetune_bn
            )

        return self._local_layers["activation"](net + shortcut)


class BlockGroup(KerasMockLayer):
    def __init__(self, filters, block_layer, n_blocks, strides, trainable, finetune_bn, data_format='channels_last'):
        """Creates one group of blocks for the ResNet model.

        Args:
        inputs: `Tensor` of size `[batch, channels, height, width]`.
        filters: `int` number of filters for the first convolution of the layer.
        block_layer: `layer` for the block to use within the model
        n_blocks: `int` number of blocks contained in the layer.
        strides: `int` stride to use for the first convolution of the layer. If
            greater than 1, this layer will downsample the input.
        finetune_bn: `bool` for whether the model is training.
        name: `str`name for the Tensor output of the block layer.
        data_format: `str` either "channels_first" for `[batch, channels, height,
            width]` or "channels_last for `[batch, height, width, channels]`.

        Returns:
        The output `Tensor` of the block layer.
        """
        super(BlockGroup, self).__init__(trainable=trainable)

        self._finetune_bn = finetune_bn

        self._n_blocks = n_blocks

        for block_id in range(self._n_blocks):
            # Only the first block per block_group uses projection shortcut and strides.
            self._local_layers["block_%d" % (block_id + 1)] = block_layer(
                filters=filters,
                finetune_bn=finetune_bn,
                trainable=trainable,
                strides=strides if block_id == 0 else 1,
                use_projection=block_id == 0,
                data_format=data_format
            )

    def __call__(self, inputs, training=False):

        net = inputs

        for block_id in range(self._n_blocks):
            net = self._local_layers["block_%d" % (block_id + 1)](net, training=training and self._trainable)

        return net


class Resnet_Model(KerasMockLayer, tf.keras.models.Model):
    def __init__(self, resnet_model, data_format='channels_last', trainable=True, finetune_bn=False, *args, **kwargs):
        """
        Our actual ResNet network.  We return the output of c2, c3,c4,c5
        N.B. batch norm is always run with trained parameters, as we use very small
        batches when training the object layers.

        Args:
        resnet_model: model type. Authorized Values: (resnet18, resnet34, resnet50, resnet101, resnet152, resnet200)
        data_format: `str` either "channels_first" for
          `[batch, channels, height, width]` or "channels_last for `[batch, height, width, channels]`.
        finetune_bn: `bool` for whether the model is training.

        Returns the ResNet model for a given size and number of output classes.
        """
        model_params = {
            'resnet18': {'block': ResidualBlock, 'layers': [2, 2, 2, 2]},
            'resnet34': {'block': ResidualBlock, 'layers': [3, 4, 6, 3]},
            'resnet50': {'block': BottleneckBlock, 'layers': [3, 4, 6, 3]},
            'resnet101': {'block': BottleneckBlock, 'layers': [3, 4, 23, 3]},
            'resnet152': {'block': BottleneckBlock, 'layers': [3, 8, 36, 3]},
            'resnet200': {'block': BottleneckBlock, 'layers': [3, 24, 36, 3]}
        }

        if resnet_model not in model_params:
            raise ValueError('Not a valid resnet_model: %s' % resnet_model)

        super(Resnet_Model, self).__init__(trainable=trainable, name=resnet_model, *args, **kwargs)

        self._finetune_bn = finetune_bn

        self._data_format = data_format
        self._block_layer = model_params[resnet_model]['block']
        self._n_layers = model_params[resnet_model]['layers']

        self._local_layers["conv2d"] = Conv2dFixedPadding(
            filters=64,
            kernel_size=7,
            strides=2,
            data_format=self._data_format,
            # Freeze at conv2d and batchnorm first 11 layers based on reference model.
            # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/core/config.py#L194
            trainable=False
        )

        self._local_layers["batchnorm"] = BNReLULayer(
            relu=True,
            init_zero=False,
            data_format=self._data_format,
            # Freeze at conv2d and batchnorm first 11 layers based on reference model.
            # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/core/config.py#L194
            trainable=False
        )

        self._local_layers["maxpool2d"] = tf.keras.layers.MaxPool2D(
            pool_size=3,
            strides=2,
            padding='SAME',
            data_format=self._data_format
        )

        self._local_layers["block_1"] = BlockGroup(
            filters=64,
            strides=1,
            n_blocks=self._n_layers[0],
            block_layer=self._block_layer,
            data_format=self._data_format,
            # Freeze at conv2d and batchnorm first 11 layers based on reference model.
            # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/core/config.py#L194
            trainable=False,
            finetune_bn=False
        )

        self._local_layers["block_2"] = BlockGroup(
            filters=128,
            strides=2,
            n_blocks=self._n_layers[1],
            block_layer=self._block_layer,
            data_format=self._data_format,
            # Freeze at conv2d and batchnorm first 11 layers based on reference model.
            # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/core/config.py#L194
            trainable=self._trainable,
            finetune_bn=self._finetune_bn
        )

        self._local_layers["block_3"] = BlockGroup(
            filters=256,
            strides=2,
            n_blocks=self._n_layers[2],
            block_layer=self._block_layer,
            data_format=self._data_format,
            # Freeze at conv2d and batchnorm first 11 layers based on reference model.
            # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/core/config.py#L194
            trainable=self._trainable,
            finetune_bn=self._finetune_bn
        )

        self._local_layers["block_4"] = BlockGroup(
            filters=512,
            strides=2,
            n_blocks=self._n_layers[3],
            block_layer=self._block_layer,
            data_format=self._data_format,
            # Freeze at conv2d and batchnorm first 11 layers based on reference model.
            # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/core/config.py#L194
            trainable=self._trainable,
            finetune_bn=self._finetune_bn
        )

    def call(self, inputs, training=True, *args, **kwargs):
        """Creation of the model graph."""
        net = self._local_layers["conv2d"](inputs=inputs)

        net = self._local_layers["batchnorm"](
            inputs=net,
            training=False
        )

        net = self._local_layers["maxpool2d"](net)

        c2 = self._local_layers["block_1"](
            inputs=net,
            training=False,
        )

        c3 = self._local_layers["block_2"](
            inputs=c2,
            training=training,
        )

        c4 = self._local_layers["block_3"](
            inputs=c3,
            training=training,
        )

        c5 = self._local_layers["block_4"](
            inputs=c4,
            training=training,
        )

        return {2: c2, 3: c3, 4: c4, 5: c5}

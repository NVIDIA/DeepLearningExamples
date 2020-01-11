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

import tensorflow as tf


def normalization_layer(inputs, name, mode):
    if name == 'batchnorm':
        return tf.layers.batch_normalization(inputs=inputs,
                                             axis=-1,
                                             training=(mode == tf.estimator.ModeKeys.TRAIN),
                                             trainable=True,
                                             virtual_batch_size=None)
    elif name == 'none':
        return inputs
    else:
        raise ValueError('Invalid normalization layer')


def activation_layer(x, activation):
    if activation == 'relu':
        return tf.nn.relu(x)
    elif activation == 'none':
        return x
    else:
        raise ValueError("Unkown activation {}".format(activation))


def convolution_layer(inputs, filters, kernel_size, stride, normalization, activation, mode):
    x = tf.layers.conv3d(inputs=inputs,
                         filters=filters,
                         kernel_size=kernel_size,
                         strides=stride,
                         activation=None,
                         padding='same',
                         data_format='channels_last',
                         use_bias=True,
                         kernel_initializer=tf.glorot_uniform_initializer(),
                         bias_initializer=tf.zeros_initializer(),
                         bias_regularizer=None)

    x = normalization_layer(x, normalization, mode)

    return activation_layer(x, activation)


def downsample_layer(inputs, pooling, normalization, activation, mode):
    if pooling == 'conv_pool':
        return convolution_layer(inputs=inputs,
                                 filters=inputs.get_shape()[-1] * 2,
                                 kernel_size=2,
                                 stride=2,
                                 normalization=normalization,
                                 activation=activation,
                                 mode=mode)
    else:
        raise ValueError('Invalid downsampling method: {}'.format(pooling))


def upsample_layer(inputs, filters, upsampling, normalization, activation, mode):
    if upsampling == 'transposed_conv':
        x = tf.layers.conv3d_transpose(inputs=inputs,
                                       filters=filters,
                                       kernel_size=2,
                                       strides=2,
                                       activation=None,
                                       padding='same',
                                       data_format='channels_last',
                                       use_bias=True,
                                       kernel_initializer=tf.glorot_uniform_initializer(),
                                       bias_initializer=tf.zeros_initializer(),
                                       bias_regularizer=None)

        x = normalization_layer(x, normalization, mode)

        return activation_layer(x, activation)

    else:
        raise ValueError('Unsupported upsampling: {}'.format(upsampling))


def residual_block(input_0, input_1, kernel_size, depth, normalization, activation, mode):
    with tf.name_scope('residual_block'):
        x = input_0
        if input_1 is not None:
            x = tf.concat([input_0, input_1], axis=-1)

        inputs = x
        n_input_channels = inputs.get_shape()[-1]

        for i in range(depth):
            x = convolution_layer(inputs=x,
                                  filters=n_input_channels,
                                  kernel_size=kernel_size,
                                  stride=1,
                                  normalization=normalization,
                                  activation=activation,
                                  mode=mode)

        return x + inputs


def input_block(inputs, filters, kernel_size, normalization, activation, mode):
    with tf.name_scope('conversion_block'):
        x = inputs
        return convolution_layer(inputs=inputs,
                                 filters=filters,
                                 kernel_size=kernel_size,
                                 stride=1,
                                 normalization=normalization,
                                 activation=activation,
                                 mode=mode) + x


def downsample_block(inputs, depth, kernel_size, pooling, normalization, activation, mode):
    with tf.name_scope('downsample_block'):
        x = downsample_layer(inputs,
                             pooling=pooling,
                             normalization=normalization,
                             activation=activation,
                             mode=mode)

        return residual_block(input_0=x,
                              input_1=None,
                              depth=depth,
                              kernel_size=kernel_size,
                              normalization=normalization,
                              activation=activation,
                              mode=mode)


def upsample_block(inputs, residual_inputs, depth, kernel_size, upsampling, normalization, activation, mode):
    with tf.name_scope('upsample_block'):
        x = upsample_layer(inputs,
                           filters=residual_inputs.get_shape()[-1],
                           upsampling=upsampling,
                           normalization=normalization,
                           activation=activation,
                           mode=mode)

        return residual_block(input_0=x,
                              input_1=residual_inputs,
                              depth=depth,
                              kernel_size=kernel_size,
                              normalization=normalization,
                              activation=activation,
                              mode=mode)


def output_block(inputs, residual_inputs, n_classes, kernel_size, upsampling, normalization, activation, mode):
    with tf.name_scope('output_block'):
        x = upsample_layer(inputs,
                           filters=residual_inputs.get_shape()[-1],
                           upsampling=upsampling,
                           normalization=normalization,
                           activation=activation,
                           mode=mode)

        return convolution_layer(inputs=x,
                                 filters=n_classes,
                                 kernel_size=kernel_size,
                                 stride=1,
                                 mode=mode,
                                 activation='none',
                                 normalization='none')

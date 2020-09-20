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


def _normalization(inputs, name, mode):
    training = mode == tf.estimator.ModeKeys.TRAIN

    if name == 'instancenorm':
        gamma_initializer = tf.constant_initializer(1.0)
        return tf.contrib.layers.instance_norm(
            inputs,
            center=True,
            scale=True,
            epsilon=1e-6,
            param_initializers={'gamma': gamma_initializer},
            reuse=None,
            variables_collections=None,
            outputs_collections=None,
            trainable=True,
            data_format='NHWC',
            scope=None)

    if name == 'groupnorm':
        return tf.contrib.layers.group_norm(inputs=inputs,
                                            groups=16,
                                            channels_axis=-1,
                                            reduction_axes=(-4, -3, -2),
                                            activation_fn=None,
                                            trainable=True)

    if name == 'batchnorm':
        return tf.keras.layers.BatchNormalization(axis=-1,
                                                  trainable=True,
                                                  virtual_batch_size=None)(inputs, training=training)
    elif name == 'none':
        return inputs
    else:
        raise ValueError('Invalid normalization layer')


def _activation(x, activation):
    if activation == 'relu':
        return tf.nn.relu(x)
    elif activation == 'leaky_relu':
        return tf.nn.leaky_relu(x, alpha=0.01)
    elif activation == 'sigmoid':
        return tf.nn.sigmoid(x)
    elif activation == 'softmax':
        return tf.nn.softmax(x, axis=-1)
    elif activation == 'none':
        return x
    else:
        raise ValueError("Unknown activation {}".format(activation))


def convolution(x,
                out_channels,
                kernel_size=3,
                stride=1,
                mode=tf.estimator.ModeKeys.TRAIN,
                normalization='batchnorm',
                activation='leaky_relu',
                transpose=False):

    if transpose:
        conv = tf.keras.layers.Conv3DTranspose
    else:
        conv = tf.keras.layers.Conv3D
    regularizer = None#tf.keras.regularizers.l2(1e-5)

    x = conv(filters=out_channels,
             kernel_size=kernel_size,
             strides=stride,
             activation=None,
             padding='same',
             data_format='channels_last',
             kernel_initializer=tf.glorot_uniform_initializer(),
             kernel_regularizer=regularizer,
             bias_initializer=tf.zeros_initializer(),
             bias_regularizer=regularizer)(x)

    x = _normalization(x, normalization, mode)

    return _activation(x, activation)


def upsample_block(x, skip_connection, out_channels, normalization, mode):
    x = convolution(x, kernel_size=2, out_channels=out_channels, stride=2,
                    normalization='none', activation='none', transpose=True)
    x = tf.keras.layers.Concatenate(axis=-1)([x, skip_connection])

    x = convolution(x, out_channels=out_channels, normalization=normalization, mode=mode)
    x = convolution(x, out_channels=out_channels, normalization=normalization, mode=mode)
    return x


def input_block(x, out_channels, normalization, mode):
    x = convolution(x, out_channels=out_channels, normalization=normalization, mode=mode)
    x = convolution(x, out_channels=out_channels, normalization=normalization, mode=mode)
    return x


def downsample_block(x, out_channels, normalization, mode):
    x = convolution(x, out_channels=out_channels, normalization=normalization, mode=mode, stride=2)
    return convolution(x, out_channels=out_channels, normalization=normalization, mode=mode)


def linear_block(x, out_channels, mode, activation='leaky_relu', normalization='none'):
    x = convolution(x, out_channels=out_channels, normalization=normalization, mode=mode)
    return convolution(x, out_channels=out_channels, activation=activation, mode=mode, normalization=normalization)


def output_layer(x, out_channels, activation):
    x = tf.keras.layers.Conv3D(out_channels,
                               kernel_size=3,
                               activation=None,
                               padding='same',
                               kernel_regularizer=None,
                               kernel_initializer=tf.glorot_uniform_initializer(),
                               bias_initializer=tf.zeros_initializer(),
                               bias_regularizer=None)(x)
    return _activation(x, activation)

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

import nv_norms
import tensorflow as tf
import tensorflow_addons as tfa

convolutions = {
    "Conv2d": tf.keras.layers.Conv2D,
    "Conv3d": tf.keras.layers.Conv3D,
    "ConvTranspose2d": tf.keras.layers.Conv2DTranspose,
    "ConvTranspose3d": tf.keras.layers.Conv3DTranspose,
}


class KaimingNormal(tf.keras.initializers.VarianceScaling):
    def __init__(self, negative_slope, seed=None):
        super().__init__(
            scale=2.0 / (1 + negative_slope**2), mode="fan_in", distribution="untruncated_normal", seed=seed
        )

    def get_config(self):
        return {"seed": self.seed}


def get_norm(name):
    if "group" in name:
        return tfa.layers.GroupNormalization(32, axis=-1, center=True, scale=True)
    elif "batch" in name:
        return tf.keras.layers.BatchNormalization(axis=-1, center=True, scale=True)
    elif "atex_instance" in name:
        return nv_norms.InstanceNormalization(axis=-1)
    elif "instance" in name:
        return tfa.layers.InstanceNormalization(axis=-1, center=True, scale=True)
    elif "none" in name:
        return tf.identity
    else:
        raise ValueError("Invalid normalization layer")


def extract_args(kwargs):
    args = {}
    if "input_shape" in kwargs:
        args["input_shape"] = kwargs["input_shape"]
    return args


def get_conv(filters, kernel_size, stride, dim, use_bias=False, **kwargs):
    conv = convolutions[f"Conv{dim}d"]
    return conv(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=KaimingNormal(kwargs["negative_slope"]),
        data_format="channels_last",
        **extract_args(kwargs),
    )


def get_transp_conv(filters, kernel_size, stride, dim, **kwargs):
    conv = convolutions[f"ConvTranspose{dim}d"]
    return conv(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=True,
        data_format="channels_last",
        **extract_args(kwargs),
    )


class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, stride, **kwargs):
        super().__init__()
        self.conv = get_conv(filters, kernel_size, stride, **kwargs)
        self.norm = get_norm(kwargs["norm"])
        self.lrelu = tf.keras.layers.LeakyReLU(alpha=kwargs["negative_slope"])

    def call(self, data):
        out = self.conv(data)
        out = self.norm(out)
        out = self.lrelu(out)
        return out


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, stride, **kwargs):
        super().__init__()
        self.conv1 = ConvLayer(filters, kernel_size, stride, **kwargs)
        kwargs.pop("input_shape", None)
        self.conv2 = ConvLayer(filters, kernel_size, 1, **kwargs)

    def call(self, input_data):
        out = self.conv1(input_data)
        out = self.conv2(out)
        return out


class UpsampleBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, stride, **kwargs):
        super().__init__()
        self.transp_conv = get_transp_conv(filters, stride, stride, **kwargs)
        self.conv_block = ConvBlock(filters, kernel_size, 1, **kwargs)

    def call(self, input_data, skip_data):
        out = self.transp_conv(input_data)
        out = tf.concat((out, skip_data), axis=-1)
        out = self.conv_block(out)
        return out


class OutputBlock(tf.keras.layers.Layer):
    def __init__(self, filters, dim, negative_slope):
        super().__init__()
        self.conv = get_conv(
            filters,
            kernel_size=1,
            stride=1,
            dim=dim,
            use_bias=True,
            negative_slope=negative_slope,
        )

    def call(self, data):
        return self.conv(data)

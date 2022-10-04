# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

import numpy as np
import torch
import torch.nn as nn

normalizations = {
    "instancenorm3d": nn.InstanceNorm3d,
    "instancenorm2d": nn.InstanceNorm2d,
    "batchnorm3d": nn.BatchNorm3d,
    "batchnorm2d": nn.BatchNorm2d,
}

convolutions = {
    "Conv2d": nn.Conv2d,
    "Conv3d": nn.Conv3d,
    "ConvTranspose2d": nn.ConvTranspose2d,
    "ConvTranspose3d": nn.ConvTranspose3d,
}


def get_norm(name, out_channels, groups=32):
    if "groupnorm" in name:
        return nn.GroupNorm(groups, out_channels, affine=True)
    return normalizations[name](out_channels, affine=True)


def get_conv(in_channels, out_channels, kernel_size, stride, dim=3, bias=False):
    conv = convolutions[f"Conv{dim}d"]
    padding = get_padding(kernel_size, stride)
    return conv(in_channels, out_channels, kernel_size, stride, padding, bias=bias)


def get_transp_conv(in_channels, out_channels, kernel_size, stride, dim):
    conv = convolutions[f"ConvTranspose{dim}d"]
    padding = get_padding(kernel_size, stride)
    output_padding = get_output_padding(kernel_size, stride, padding)
    return conv(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True)


def get_padding(kernel_size, stride):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    padding = tuple(int(p) for p in padding_np)
    return padding if len(padding) > 1 else padding[0]


def get_output_padding(kernel_size, stride, padding):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)
    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    out_padding = tuple(int(p) for p in out_padding_np)
    return out_padding if len(out_padding) > 1 else out_padding[0]


class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(InputBlock, self).__init__()
        self.conv1 = get_conv(in_channels, out_channels, 3, 1)
        self.conv2 = get_conv(out_channels, out_channels, 3, 1)
        self.norm = get_norm(kwargs["norm"], out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(ConvLayer, self).__init__()
        self.conv = get_conv(in_channels, out_channels, kernel_size, stride)
        self.norm = get_norm(kwargs["norm"], in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, 1, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(UpsampleBlock, self).__init__()
        self.conv_block = ConvBlock(out_channels + in_channels, out_channels, kernel_size, 1, **kwargs)

    def forward(self, x, x_skip):
        x = nn.functional.interpolate(x, scale_factor=2, mode="trilinear", align_corners=True)
        x = torch.cat((x, x_skip), dim=1)
        x = self.conv_block(x)
        return x


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim):
        super(OutputBlock, self).__init__()
        self.conv = get_conv(in_channels, out_channels, kernel_size=1, stride=1, dim=dim, bias=True)

    def forward(self, input_data):
        return self.conv(input_data)


class UNet3D(nn.Module):
    def __init__(
        self,
        kernels,
        strides,
    ):
        super(UNet3D, self).__init__()
        self.dim = 3
        self.n_class = 3
        self.deep_supervision = True
        self.norm = "instancenorm3d"
        self.filters = [64, 128, 256, 512, 768, 1024, 2048][: len(strides)]

        down_block = ConvBlock
        self.input_block = InputBlock(5, self.filters[0], norm=self.norm)
        self.downsamples = self.get_module_list(
            conv_block=down_block,
            in_channels=self.filters[:-1],
            out_channels=self.filters[1:],
            kernels=kernels[1:-1],
            strides=strides[1:-1],
        )
        self.bottleneck = self.get_conv_block(
            conv_block=down_block,
            in_channels=self.filters[-2],
            out_channels=self.filters[-1],
            kernel_size=kernels[-1],
            stride=strides[-1],
        )
        self.upsamples = self.get_module_list(
            conv_block=UpsampleBlock,
            in_channels=self.filters[1:][::-1],
            out_channels=self.filters[:-1][::-1],
            kernels=kernels[1:][::-1],
            strides=strides[1:][::-1],
        )
        self.output_block = self.get_output_block(decoder_level=0)
        self.deep_supervision_heads = self.get_deep_supervision_heads()
        self.apply(self.initialize_weights)

    def forward(self, input_data):
        out = self.input_block(input_data)
        encoder_outputs = [out]
        for downsample in self.downsamples:
            out = downsample(out)
            encoder_outputs.append(out)
        out = self.bottleneck(out)
        decoder_outputs = []
        for upsample, skip in zip(self.upsamples, reversed(encoder_outputs)):
            out = upsample(out, skip)
            decoder_outputs.append(out)
        out = self.output_block(out)
        if self.training and self.deep_supervision:
            out = [out]
            for i, decoder_out in enumerate(decoder_outputs[-3:-1][::-1]):
                out.append(self.deep_supervision_heads[i](decoder_out))
        return out

    def get_conv_block(self, conv_block, in_channels, out_channels, kernel_size, stride, drop_block=False):
        return conv_block(
            dim=self.dim,
            stride=stride,
            norm=self.norm,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def get_output_block(self, decoder_level):
        return OutputBlock(in_channels=self.filters[decoder_level], out_channels=self.n_class, dim=self.dim)

    def get_deep_supervision_heads(self):
        return nn.ModuleList([self.get_output_block(1), self.get_output_block(2)])

    def get_module_list(self, in_channels, out_channels, kernels, strides, conv_block):
        layers = []
        for in_channel, out_channel, kernel, stride in zip(in_channels, out_channels, kernels, strides):
            conv_layer = self.get_conv_block(conv_block, in_channel, out_channel, kernel, stride)
            layers.append(conv_layer)
        return nn.ModuleList(layers)

    def initialize_weights(self, module):
        name = module.__class__.__name__.lower()
        if name in ["conv2d", "conv3d"]:
            nn.init.kaiming_normal_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, 0)

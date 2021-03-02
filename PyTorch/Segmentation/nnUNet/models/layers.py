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

import numpy as np
import torch
import torch.nn as nn
from dropblock import DropBlock3D, LinearScheduler

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


def get_norm(name, out_channels):
    if "groupnorm" in name:
        return nn.GroupNorm(32, out_channels, affine=True)
    return normalizations[name](out_channels, affine=True)


def get_conv(in_channels, out_channels, kernel_size, stride, dim, bias=False):
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


def get_drop_block():
    return LinearScheduler(
        DropBlock3D(block_size=5, drop_prob=0.0),
        start_value=0.0,
        stop_value=0.1,
        nr_steps=10000,
    )


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(ConvLayer, self).__init__()
        self.conv = get_conv(in_channels, out_channels, kernel_size, stride, kwargs["dim"])
        self.norm = get_norm(kwargs["norm"], out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=kwargs["negative_slope"], inplace=True)
        self.use_drop_block = kwargs["drop_block"]
        if self.use_drop_block:
            self.drop_block = get_drop_block()

    def forward(self, data):
        out = self.conv(data)
        if self.use_drop_block:
            self.drop_block.step()
            out = self.drop_block(out)
        out = self.norm(out)
        out = self.lrelu(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, 1, **kwargs)

    def forward(self, input_data):
        out = self.conv1(input_data)
        out = self.conv2(out)
        return out


class ResidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(ResidBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.conv2 = get_conv(out_channels, out_channels, kernel_size, 1, kwargs["dim"])
        self.norm = get_norm(kwargs["norm"], out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=kwargs["negative_slope"], inplace=True)
        self.use_drop_block = kwargs["drop_block"]
        if self.use_drop_block:
            self.drop_block = get_drop_block()
            self.skip_drop_block = get_drop_block()
        self.downsample = None
        if max(stride) > 1 or in_channels != out_channels:
            self.downsample = get_conv(in_channels, out_channels, kernel_size, stride, kwargs["dim"])
            self.norm_res = get_norm(kwargs["norm"], out_channels)

    def forward(self, input_data):
        residual = input_data
        out = self.conv1(input_data)
        out = self.conv2(out)
        if self.use_drop_block:
            out = self.drop_block(out)
        out = self.norm(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
            if self.use_drop_block:
                residual = self.skip_drop_block(residual)
            residual = self.norm_res(residual)
        out = self.lrelu(out + residual)
        return out


class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm, dim):
        super(AttentionLayer, self).__init__()
        self.conv = get_conv(in_channels, out_channels, kernel_size=3, stride=1, dim=dim)
        self.norm = get_norm(norm, out_channels)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(UpsampleBlock, self).__init__()
        self.transp_conv = get_transp_conv(in_channels, out_channels, stride, stride, kwargs["dim"])
        self.conv_block = ConvBlock(2 * out_channels, out_channels, kernel_size, 1, **kwargs)
        self.attention = kwargs["attention"]
        if self.attention:
            att_out, norm, dim = out_channels // 2, kwargs["norm"], kwargs["dim"]
            self.conv_o = AttentionLayer(out_channels, att_out, norm, dim)
            self.conv_s = AttentionLayer(out_channels, att_out, norm, dim)
            self.psi = AttentionLayer(att_out, 1, norm, dim)
            self.sigmoid = nn.Sigmoid()
            self.relu = nn.ReLU(inplace=True)

    def forward(self, input_data, skip_data):
        out = self.transp_conv(input_data)
        if self.attention:
            out_a = self.conv_o(out)
            skip_a = self.conv_s(skip_data)
            psi_a = self.psi(self.relu(out_a + skip_a))
            attention = self.sigmoid(psi_a)
            skip_data = skip_data * attention
        out = torch.cat((out, skip_data), dim=1)
        out = self.conv_block(out)
        return out


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim):
        super(OutputBlock, self).__init__()
        self.conv = get_conv(in_channels, out_channels, kernel_size=1, stride=1, dim=dim, bias=True)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, input_data):
        return self.conv(input_data)

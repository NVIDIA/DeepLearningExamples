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

import torch
import torch.nn as nn

from models.layers import ConvBlock, OutputBlock, UpsampleBlock


class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        n_class,
        kernels,
        strides,
        normalization_layer,
        negative_slope,
        dimension,
        deep_supervision,
        more_chn,
    ):
        super(UNet, self).__init__()
        self.more_chn = more_chn
        self.dim = dimension
        self.n_class = n_class
        self.negative_slope = negative_slope
        self.norm = normalization_layer + f"norm{dimension}d"
        self.deep_supervision = deep_supervision
        self.depth = len(strides)
        if self.more_chn:
            self.filters = [64, 96, 128, 192, 256, 384, 512, 768, 1024][: self.depth]
        else:
            self.filters = [min(2 ** (5 + i), 320 if dimension == 3 else 512) for i in range(self.depth)]

        self.input_block = self.get_conv_block(
            conv_block=ConvBlock,
            in_channels=in_channels,
            out_channels=self.filters[0],
            kernel_size=kernels[0],
            stride=strides[0],
        )
        self.downsamples = self.get_module_list(
            conv_block=ConvBlock,
            in_channels=self.filters[:-1],
            out_channels=self.filters[1:],
            kernels=kernels[1:-1],
            strides=strides[1:-1],
        )
        self.bottleneck = self.get_conv_block(
            conv_block=ConvBlock,
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
        self.deep_supervision_head1 = self.get_output_block(1)
        self.deep_supervision_head2 = self.get_output_block(2)
        self.output_block = self.get_output_block(decoder_level=0)
        self.apply(self.initialize_weights)
        self.n_layers = len(self.upsamples) - 1

    def forward(self, input_data):
        out = self.input_block(input_data)
        encoder_outputs = [out]
        for downsample in self.downsamples:
            out = downsample(out)
            encoder_outputs.append(out)
        out = self.bottleneck(out)
        decoder_outputs = []
        for i, upsample in enumerate(self.upsamples):
            out = upsample(out, encoder_outputs[self.depth - i - 2])
            decoder_outputs.append(out)
        out = self.output_block(out)
        if self.training and self.deep_supervision:
            out1 = self.deep_supervision_head1(decoder_outputs[-2])
            out2 = self.deep_supervision_head2(decoder_outputs[-3])
            out1 = nn.functional.interpolate(out1, out.shape[2:], mode="trilinear", align_corners=True)
            out2 = nn.functional.interpolate(out2, out.shape[2:], mode="trilinear", align_corners=True)
            return torch.stack([out, out1, out2])
        return out

    def get_conv_block(self, conv_block, in_channels, out_channels, kernel_size, stride):
        return conv_block(
            dim=self.dim,
            stride=stride,
            norm=self.norm,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            negative_slope=self.negative_slope,
        )

    def get_output_block(self, decoder_level):
        return OutputBlock(in_channels=self.filters[decoder_level], out_channels=self.n_class, dim=self.dim)

    def get_module_list(self, in_channels, out_channels, kernels, strides, conv_block):
        layers = []
        for in_channel, out_channel, kernel, stride in zip(in_channels, out_channels, kernels, strides):
            conv_layer = self.get_conv_block(conv_block, in_channel, out_channel, kernel, stride)
            layers.append(conv_layer)
        return nn.ModuleList(layers)

    def initialize_weights(self, module):
        name = module.__class__.__name__.lower()
        if name in ["conv2d", "conv3d"]:
            nn.init.kaiming_normal_(module.weight, a=self.negative_slope)

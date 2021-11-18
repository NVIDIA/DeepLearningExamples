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

import math
from collections import namedtuple

import torch
from torch import nn

BlockParameters = namedtuple('BlockParameters',
                             ['kernel_size', 'stride', 'num_repeat', 'in_channels', 'out_channels', 'expand_ratio'])
GlobalParameters = namedtuple('GlobalParameters',
                              ['squeeze_excitation_ratio', 'batchnorm_momentum', 'batchnorm_epsilon',
                               'stochastic_depth_survival_prob', 'feature_channels', "weights_init_mode"])

efficientnet_configs = {
    "fanin": GlobalParameters(
        squeeze_excitation_ratio=0.25,
        batchnorm_momentum=1-0.99,  # batchnorm momentum definition is different in pytorch and original paper
        batchnorm_epsilon=1e-3,
        stochastic_depth_survival_prob=0.8,
        feature_channels=1280,
        weights_init_mode="fan_in"
    ),
    "fanout": GlobalParameters(
        squeeze_excitation_ratio=0.25,
        batchnorm_momentum=1-0.99,
        batchnorm_epsilon=1e-3,
        stochastic_depth_survival_prob=0.8,
        feature_channels=1280,
        weights_init_mode="fan_out"
    ),
}

BASE_EFFICIENTNET_BLOCKS_CONFIG = [
    BlockParameters(kernel_size=3, stride=1, num_repeat=1, in_channels=32, out_channels=16, expand_ratio=1),
    BlockParameters(kernel_size=3, stride=2, num_repeat=2, in_channels=16, out_channels=24, expand_ratio=6),
    BlockParameters(kernel_size=5, stride=2, num_repeat=2, in_channels=24, out_channels=40, expand_ratio=6),
    BlockParameters(kernel_size=3, stride=2, num_repeat=3, in_channels=40, out_channels=80, expand_ratio=6),
    BlockParameters(kernel_size=5, stride=1, num_repeat=3, in_channels=80, out_channels=112, expand_ratio=6),
    BlockParameters(kernel_size=5, stride=2, num_repeat=4, in_channels=112, out_channels=192, expand_ratio=6),
    BlockParameters(kernel_size=3, stride=1, num_repeat=1, in_channels=192, out_channels=320, expand_ratio=6)
]


def _scale_width(num_channels, width_coeff, divisor=8):
    num_channels *= width_coeff
    # Rounding should not go down by more than 10%
    rounded_num_channels = max(divisor, int(num_channels + divisor / 2) // divisor * divisor)
    if rounded_num_channels < 0.9 * num_channels:
        rounded_num_channels += divisor
    return rounded_num_channels


def scaled_efficientnet_config(width_coeff, depth_coeff):
    config = [
        block._replace(
            num_repeat=int(math.ceil(block.num_repeat * depth_coeff)),
            in_channels=_scale_width(block.in_channels, width_coeff),
            out_channels=_scale_width(block.out_channels, width_coeff),
        )
        for block in BASE_EFFICIENTNET_BLOCKS_CONFIG
    ]
    return config


class SqueezeAndExcitation(nn.Module):
    def __init__(self, in_channels, squeeze, activation):
        super(SqueezeAndExcitation, self).__init__()
        self.squeeze = nn.Linear(in_channels, squeeze)
        self.expand = nn.Linear(squeeze, in_channels)
        self.activation = activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
        out = self.squeeze(out)
        out = self.activation(out)
        out = self.expand(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(2).unsqueeze(3)
        return out


# Since torch.nn.SiLU is not supported in ONNX,
# it is required to use this implementation in exported model (15-20% more GPU memory is needed)
class MemoryInefficientSiLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MemoryInefficientSiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvBN(nn.Sequential):
    def __init__(self, kernel_size, stride, in_channels, out_channels, activation,
                 bn_momentum, bn_epsilon, groups=1):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      groups=groups, bias=False, padding=int((kernel_size - 1) / 2)),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum, eps=bn_epsilon),
        ]
        if activation is not None:
            layers.append(activation)
        super(ConvBN, self).__init__(*layers)


class MBConvBlock(nn.Module):
    def __init__(self, block_config, global_config, survival_prob, activation):
        super(MBConvBlock, self).__init__()

        self.in_channels = block_config.in_channels
        self.out_channels = block_config.out_channels
        self.hidden_dim = self.in_channels * block_config.expand_ratio
        self.squeeze_dim = max(1, int(self.in_channels * global_config.squeeze_excitation_ratio))
        self.kernel_size = block_config.kernel_size
        self.stride = block_config.stride
        self.stochastic_depth_survival_prob = survival_prob

        bn_momentum = global_config.batchnorm_momentum
        bn_epsilon = global_config.batchnorm_epsilon

        if self.in_channels != self.hidden_dim:
            self.expand_conv = ConvBN(1, 1, self.in_channels, self.hidden_dim, activation(),
                                      bn_momentum=bn_momentum, bn_epsilon=bn_epsilon)

        self.squeeze_and_excitation = SqueezeAndExcitation(self.hidden_dim, self.squeeze_dim, activation())

        self.depthwise_conv = ConvBN(self.kernel_size, self.stride, self.hidden_dim, self.hidden_dim, activation(),
                                     groups=self.hidden_dim, bn_momentum=bn_momentum, bn_epsilon=bn_epsilon)

        self.project_conv = ConvBN(1, 1, self.hidden_dim, self.out_channels,
                                   activation=None,  bn_momentum=bn_momentum, bn_epsilon=bn_epsilon)

    def _drop_connections(self, x, synchronized=False):
        if not self.training:
            return x
        random_mask = torch.rand([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
        if synchronized:
            torch.distributed.broadcast(random_mask, 0)
        random_mask = (self.stochastic_depth_survival_prob + random_mask).floor()
        scaled_x = x / self.stochastic_depth_survival_prob
        return scaled_x * random_mask

    def forward(self, inputs):
        x = inputs
        if self.in_channels != self.hidden_dim:
            x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = x * self.squeeze_and_excitation(x)
        x = self.project_conv(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            if self.stochastic_depth_survival_prob != 1:
                x = self._drop_connections(x)
            x = x + inputs
        return x


class EfficientNet(nn.Module):
    def __init__(self, width_coeff, depth_coeff, dropout, num_classes, global_config, features_only=True, out_indices=None, onnx_exportable=False):
        super(EfficientNet, self).__init__()
        self.features_only = features_only
        self.efficientnet_blocks_config = scaled_efficientnet_config(width_coeff, depth_coeff)
        self.global_config = global_config
        self.in_channels = 3
        self.feature_channels = _scale_width(self.global_config.feature_channels, width_coeff)
        self.activation = torch.nn.SiLU if not onnx_exportable else MemoryInefficientSiLU

        self.input_conv = ConvBN(3, 2, self.in_channels, self.efficientnet_blocks_config[0].in_channels,
                                 activation=self.activation(),
                                 bn_momentum=self.global_config.batchnorm_momentum,
                                 bn_epsilon=self.global_config.batchnorm_epsilon)

        self.feature_info = []
        self.mbconv_blocks = nn.Sequential(*self.mbconv_blocks_generator())
        if not self.features_only:
            self.features_conv = ConvBN(1, 1, self.efficientnet_blocks_config[-1].out_channels, self.feature_channels,
                                        activation=self.activation(),
                                        bn_momentum=self.global_config.batchnorm_momentum,
                                        bn_epsilon=self.global_config.batchnorm_epsilon)
            self.avg_pooling = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(self.feature_channels, num_classes)
        if out_indices is not None:
            self.feature_info = [v for i, v in enumerate(self.feature_info) if i in out_indices]

    def mbconv_blocks_generator(self):
        num_blocks = sum([block_config.num_repeat for block_config in self.efficientnet_blocks_config])
        drop_rate = 1.0 - self.global_config.stochastic_depth_survival_prob
        idx = 0
        current_stride = 2
        prev_block_config = None
        for config_idx, block_config in enumerate(self.efficientnet_blocks_config):
            for i in range(block_config.num_repeat):
                # Conditions for feature extraction
                if config_idx == len(self.efficientnet_blocks_config)-1 and i == block_config.num_repeat-1:
                    self.feature_info.append(dict(block_idx=idx, reduction=current_stride, num_chs=block_config.out_channels))
                elif prev_block_config is not None and block_config.stride > 1:
                    self.feature_info.append(dict(block_idx=idx-1, reduction=current_stride, num_chs=prev_block_config.out_channels))
                # Calculating the current stride
                if block_config.stride > 1:
                    current_stride = current_stride * block_config.stride

                survival_prob = 1.0 - drop_rate * float(idx) / num_blocks
                yield MBConvBlock(block_config, self.global_config,
                                  survival_prob=survival_prob, activation=self.activation)
                idx += 1
                prev_block_config = block_config
                block_config = block_config._replace(in_channels=block_config.out_channels, stride=1)

    def forward(self, inputs):
        x = inputs
        x = self.input_conv(x)
        features = []
        extraction_idx = 0
        for i, b in enumerate(self.mbconv_blocks):
            x = b(x)
            if i == self.feature_info[extraction_idx]['block_idx']:
                features.append(x)
                extraction_idx += 1
        return x, features

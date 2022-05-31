# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
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
import paddle
from paddle import ParamAttr
import paddle.nn as nn
from paddle.nn import Conv2D, BatchNorm, Linear
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform, Constant, KaimingNormal

MODELS = ["ResNet50"]

__all__ = MODELS


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None,
                 lr_mult=1.0,
                 data_format="NCHW",
                 bn_weight_decay=True):
        super().__init__()
        self.act = act
        self.avg_pool = AvgPool2D(
            kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(
                learning_rate=lr_mult, initializer=KaimingNormal()),
            bias_attr=False,
            data_format=data_format)
        self.bn = BatchNorm(
            num_filters,
            param_attr=ParamAttr(
                learning_rate=lr_mult,
                regularizer=None
                if bn_weight_decay else paddle.regularizer.L2Decay(0.0),
                initializer=Constant(1.0)),
            bias_attr=ParamAttr(
                learning_rate=lr_mult,
                regularizer=None
                if bn_weight_decay else paddle.regularizer.L2Decay(0.0),
                initializer=Constant(0.0)),
            data_layout=data_format)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act:
            x = self.relu(x)
        return x


class BottleneckBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True,
                 lr_mult=1.0,
                 data_format="NCHW",
                 bn_weight_decay=True):
        super().__init__()

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act="relu",
            lr_mult=lr_mult,
            data_format=data_format,
            bn_weight_decay=bn_weight_decay)
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act="relu",
            lr_mult=lr_mult,
            data_format=data_format,
            bn_weight_decay=bn_weight_decay)
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
            lr_mult=lr_mult,
            data_format=data_format,
            bn_weight_decay=bn_weight_decay)

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride,
                lr_mult=lr_mult,
                data_format=data_format,
                bn_weight_decay=bn_weight_decay)
        self.relu = nn.ReLU()
        self.shortcut = shortcut

    def forward(self, x):
        identity = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        if self.shortcut:
            short = identity
        else:
            short = self.short(identity)
        x = paddle.add(x=x, y=short)
        x = self.relu(x)
        return x


class ResNet(nn.Layer):
    def __init__(self,
                 class_num=1000,
                 data_format="NCHW",
                 input_image_channel=3,
                 use_pure_fp16=False,
                 bn_weight_decay=True):
        super().__init__()

        self.class_num = class_num
        self.num_filters = [64, 128, 256, 512]
        self.block_depth = [3, 4, 6, 3]
        self.num_channels = [64, 256, 512, 1024]
        self.channels_mult = 1 if self.num_channels[-1] == 256 else 4
        self.use_pure_fp16 = use_pure_fp16

        self.stem_cfg = {
            #num_channels, num_filters, filter_size, stride
            "vb": [[input_image_channel, 64, 7, 2]],
        }
        self.stem = nn.Sequential(* [
            ConvBNLayer(
                num_channels=in_c,
                num_filters=out_c,
                filter_size=k,
                stride=s,
                act="relu",
                data_format=data_format,
                bn_weight_decay=bn_weight_decay)
            for in_c, out_c, k, s in self.stem_cfg['vb']
        ])

        self.max_pool = MaxPool2D(
            kernel_size=3, stride=2, padding=1, data_format=data_format)
        block_list = []
        for block_idx in range(len(self.block_depth)):
            shortcut = False
            for i in range(self.block_depth[block_idx]):
                block_list.append(
                    BottleneckBlock(
                        num_channels=self.num_channels[block_idx] if i == 0
                        else self.num_filters[block_idx] * self.channels_mult,
                        num_filters=self.num_filters[block_idx],
                        stride=2 if i == 0 and block_idx != 0 else 1,
                        shortcut=shortcut,
                        data_format=data_format,
                        bn_weight_decay=bn_weight_decay))
                shortcut = True
        self.blocks = nn.Sequential(*block_list)

        self.avg_pool = AdaptiveAvgPool2D(1, data_format=data_format)
        self.flatten = nn.Flatten()
        self.avg_pool_channels = self.num_channels[-1] * 2
        stdv = 1.0 / math.sqrt(self.avg_pool_channels * 1.0)
        self.fc = Linear(
            self.avg_pool_channels,
            self.class_num,
            weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)))

    def forward(self, x):
        if self.use_pure_fp16:
            with paddle.static.amp.fp16_guard():
                x = self.stem(x)
                x = self.max_pool(x)
                x = self.blocks(x)
                x = self.avg_pool(x)
                x = self.flatten(x)
                x = self.fc(x)
        else:
            x = self.stem(x)
            x = self.max_pool(x)
            x = self.blocks(x)
            x = self.avg_pool(x)
            x = self.flatten(x)
            x = self.fc(x)

        return x


def ResNet50(**kwargs):
    model = ResNet(**kwargs)
    return model

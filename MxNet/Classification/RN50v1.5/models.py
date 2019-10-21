# Copyright 2017-2018 The Apache Software Foundation
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# -----------------------------------------------------------------------
#
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


import copy

import mxnet as mx
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn

def add_model_args(parser):
    model = parser.add_argument_group('Model')
    model.add_argument('--arch', default='resnetv15',
                       choices=['resnetv1', 'resnetv15',
                                'resnextv1', 'resnextv15',
                                'xception'],
                       help='model architecture')
    model.add_argument('--num-layers', type=int, default=50,
                       help='number of layers in the neural network, \
                             required by some networks such as resnet')
    model.add_argument('--num-groups', type=int, default=32,
                       help='number of groups for grouped convolutions, \
                             required by some networks such as resnext')
    model.add_argument('--num-classes', type=int, default=1000,
                       help='the number of classes')
    model.add_argument('--batchnorm-eps', type=float, default=1e-5,
                       help='the amount added to the batchnorm variance to prevent output explosion.')
    model.add_argument('--batchnorm-mom', type=float, default=0.9,
                       help='the leaky-integrator factor controling the batchnorm mean and variance.')
    model.add_argument('--fuse-bn-relu', type=int, default=0,
                       help='have batchnorm kernel perform activation relu')
    model.add_argument('--fuse-bn-add-relu', type=int, default=0,
                       help='have batchnorm kernel perform add followed by activation relu')
    return model

class Builder:
    def __init__(self, dtype, input_layout, conv_layout, bn_layout,
                 pooling_layout, bn_eps, bn_mom, fuse_bn_relu, fuse_bn_add_relu):
        self.dtype = dtype
        self.input_layout = input_layout
        self.conv_layout = conv_layout
        self.bn_layout = bn_layout
        self.pooling_layout = pooling_layout
        self.bn_eps = bn_eps
        self.bn_mom = bn_mom
        self.fuse_bn_relu = fuse_bn_relu
        self.fuse_bn_add_relu = fuse_bn_add_relu

        self.act_type = 'relu'
        self.bn_gamma_initializer = lambda last: 'zeros' if last else 'ones'
        self.linear_initializer = lambda groups=1: mx.init.Xavier(rnd_type='gaussian', factor_type="in",
                                                                  magnitude=2 * (groups ** 0.5))

        self.last_layout = self.input_layout

    def copy(self):
        return copy.copy(self)

    def batchnorm(self, last=False):
        gamma_initializer = self.bn_gamma_initializer(last)
        bn_axis = 3 if self.bn_layout == 'NHWC' else 1
        return self.sequence(
            self.transpose(self.bn_layout),
            nn.BatchNorm(axis=bn_axis, momentum=self.bn_mom, epsilon=self.bn_eps,
                         gamma_initializer=gamma_initializer,
                         running_variance_initializer=gamma_initializer)
        )

    def batchnorm_add_relu(self, last=False):
        gamma_initializer = self.bn_gamma_initializer(last)
        if self.fuse_bn_add_relu:
            bn_axis = 3 if self.bn_layout == 'NHWC' else 1
            return self.sequence(
                self.transpose(self.bn_layout),
                BatchNormAddRelu(axis=bn_axis, momentum=self.bn_mom,
                                 epsilon=self.bn_eps, act_type=self.act_type,
                                 gamma_initializer=gamma_initializer,
                                 running_variance_initializer=gamma_initializer)
            )
        return NonFusedBatchNormAddRelu(self, last=last)

    def batchnorm_relu(self, last=False):
        gamma_initializer = self.bn_gamma_initializer(last)
        if self.fuse_bn_relu:
            bn_axis = 3 if self.bn_layout == 'NHWC' else 1
            return self.sequence(
                self.transpose(self.bn_layout),
                nn.BatchNorm(axis=bn_axis, momentum=self.bn_mom,
                             epsilon=self.bn_eps, act_type=self.act_type,
                             gamma_initializer=gamma_initializer,
                             running_variance_initializer=gamma_initializer)
            )

        return self.sequence(self.batchnorm(last=last), self.activation())

    def activation(self):
        return nn.Activation(self.act_type)

    def global_avg_pool(self):
        return self.sequence(
            self.transpose(self.pooling_layout),
            nn.GlobalAvgPool2D(layout=self.pooling_layout)
        )

    def max_pool(self, pool_size, strides=1, padding=True):
        padding = pool_size // 2 if padding is True else int(padding)
        return self.sequence(
            self.transpose(self.pooling_layout),
            nn.MaxPool2D(pool_size, strides=strides, padding=padding,
                         layout=self.pooling_layout)
        )

    def conv(self, channels, kernel_size, padding=True, strides=1, groups=1, in_channels=0):
        padding = kernel_size // 2 if padding is True else int(padding)
        initializer = self.linear_initializer(groups=groups)
        return self.sequence(
            self.transpose(self.conv_layout),
            nn.Conv2D(channels, kernel_size=kernel_size, strides=strides,
                      padding=padding, use_bias=False, groups=groups,
                      in_channels=in_channels, layout=self.conv_layout,
                      weight_initializer=initializer)
        )

    def separable_conv(self, channels, kernel_size, in_channels, padding=True, strides=1):
        return self.sequence(
            self.conv(in_channels, kernel_size, padding=padding,
                      strides=strides, groups=in_channels, in_channels=in_channels),
            self.conv(channels, 1, in_channels=in_channels)
        )

    def dense(self, units, in_units=0):
        return nn.Dense(units, in_units=in_units,
                        weight_initializer=self.linear_initializer())

    def transpose(self, to_layout):
        if self.last_layout == to_layout:
            return None
        ret = Transpose(self.last_layout, to_layout)
        self.last_layout = to_layout
        return ret

    def sequence(self, *seq):
        seq = list(filter(lambda x: x is not None, seq))
        if len(seq) == 1:
            return seq[0]
        ret = nn.HybridSequential()
        ret.add(*seq)
        return ret


class Transpose(HybridBlock):
    def __init__(self, from_layout, to_layout):
        super().__init__()
        supported_layouts = ['NCHW', 'NHWC']
        if from_layout not in supported_layouts:
            raise ValueError('Not prepared to handle layout: {}'.format(from_layout))
        if to_layout not in supported_layouts:
            raise ValueError('Not prepared to handle layout: {}'.format(to_layout))
        self.from_layout = from_layout
        self.to_layout = to_layout

    def hybrid_forward(self, F, x):
        # Insert transpose if from_layout and to_layout don't match
        if self.from_layout == 'NCHW' and self.to_layout == 'NHWC':
            return F.transpose(x, axes=(0, 2, 3, 1))
        elif self.from_layout == 'NHWC' and self.to_layout == 'NCHW':
            return F.transpose(x, axes=(0, 3, 1, 2))
        else:
            return x

    def __repr__(self):
        s = '{name}({content})'
        if self.from_layout == self.to_layout:
            content = 'passthrough ' + self.from_layout
        else:
            content = self.from_layout + ' -> ' + self.to_layout
        return s.format(name=self.__class__.__name__,
                        content=content)

class LayoutWrapper(HybridBlock):
    def __init__(self, op, io_layout, op_layout, **kwargs):
        super(LayoutWrapper, self).__init__(**kwargs)
        with self.name_scope():
            self.layout1 = Transpose(io_layout, op_layout)
            self.op = op
            self.layout2 = Transpose(op_layout, io_layout)

    def hybrid_forward(self, F, *x):
        return self.layout2(self.op(*(self.layout1(y) for y in x)))

class BatchNormAddRelu(nn.BatchNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._kwargs.pop('act_type') != 'relu':
            raise ValueError('BatchNormAddRelu can be used only with ReLU as activation')

    def hybrid_forward(self, F, x, y, gamma, beta, running_mean, running_var):
        return F.BatchNormAddRelu(data=x, addend=y, gamma=gamma, beta=beta,
                moving_mean=running_mean, moving_var=running_var, name='fwd', **self._kwargs)

class NonFusedBatchNormAddRelu(HybridBlock):
    def __init__(self, builder, **kwargs):
        super().__init__()
        self.bn = builder.batchnorm(**kwargs)
        self.act = builder.activation()

    def hybrid_forward(self, F, x, y):
        return self.act(self.bn(x) + y)


# Blocks
class ResNetBasicBlock(HybridBlock):
    def __init__(self, builder, channels, stride, downsample=False, in_channels=0,
                 version='1', resnext_groups=None, **kwargs):
        super().__init__()
        assert not resnext_groups

        self.transpose = builder.transpose(builder.conv_layout)
        builder_copy = builder.copy()

        body = [
            builder.conv(channels, 3, strides=stride, in_channels=in_channels),
            builder.batchnorm_relu(),
            builder.conv(channels, 3),
        ]

        self.body = builder.sequence(*body)
        self.bn_add_relu = builder.batchnorm_add_relu(last=True)

        builder = builder_copy
        if downsample:
            self.downsample = builder.sequence(
                builder.conv(channels, 1, strides=stride, in_channels=in_channels),
                builder.batchnorm()
            )
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        if self.transpose is not None:
            x = self.transpose(x)
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = self.bn_add_relu(x, residual)
        return x


class ResNetBottleNeck(HybridBlock):
    def __init__(self, builder, channels, stride, downsample=False, in_channels=0,
                 version='1', resnext_groups=None):
        super().__init__()
        stride1 = stride if version == '1' else 1
        stride2 = 1 if version == '1' else stride

        mult = 2 if resnext_groups else 1
        groups = resnext_groups or 1

        self.transpose = builder.transpose(builder.conv_layout)
        builder_copy = builder.copy()

        body = [
            builder.conv(channels * mult // 4, 1, strides=stride1, in_channels=in_channels),
            builder.batchnorm_relu(),
            builder.conv(channels * mult // 4, 3, strides=stride2),
            builder.batchnorm_relu(),
            builder.conv(channels, 1)
        ]

        self.body = builder.sequence(*body)
        self.bn_add_relu = builder.batchnorm_add_relu(last=True)

        builder = builder_copy
        if downsample:
            self.downsample = builder.sequence(
                builder.conv(channels, 1, strides=stride, in_channels=in_channels),
                builder.batchnorm()
            )
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        if self.transpose is not None:
            x = self.transpose(x)
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = self.bn_add_relu(x, residual)
        return x


class XceptionBlock(HybridBlock):
    def __init__(self, builder, definition, in_channels, relu_at_beginning=True):
        super().__init__()

        self.transpose = builder.transpose(builder.conv_layout)
        builder_copy = builder.copy()

        body = []
        if relu_at_beginning:
            body.append(builder.activation())

        last_channels = in_channels
        for channels1, channels2 in zip(definition, definition[1:] + [0]):
            if channels1 > 0:
                body.append(builder.separable_conv(channels1, 3, in_channels=last_channels))
                if channels2 > 0:
                    body.append(builder.batchnorm_relu())
                else:
                    body.append(builder.batchnorm(last=True))

                last_channels = channels1
            else:
                body.append(builder.max_pool(3, 2))

        self.body = builder.sequence(*body)

        builder = builder_copy
        if any(map(lambda x: x <= 0, definition)):
            self.shortcut = builder.sequence(
                builder.conv(last_channels, 1, strides=2, in_channels=in_channels),
                builder.batchnorm(),
            )
        else:
            self.shortcut = builder.sequence()

    def hybrid_forward(self, F, x):
        return self.shortcut(x) + self.body(x)

# Nets
class ResNet(HybridBlock):
    def __init__(self, builder, block, layers, channels, classes=1000,
                 version='1', resnext_groups=None):
        super().__init__()
        assert len(layers) == len(channels) - 1

        self.version = version
        with self.name_scope():
            features = [
                builder.conv(channels[0], 7, strides=2),
                builder.batchnorm_relu(),
                builder.max_pool(3, 2),
            ]

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                features.append(self.make_layer(builder, block, num_layer, channels[i+1],
                                                stride, in_channels=channels[i],
                                                resnext_groups=resnext_groups))
            features.append(builder.global_avg_pool())

            self.features = builder.sequence(*features)
            self.output = builder.dense(classes, in_units=channels[-1])

    def make_layer(self, builder, block, layers, channels, stride,
                    in_channels=0, resnext_groups=None):
        layer = []
        layer.append(block(builder, channels, stride, channels != in_channels,
                            in_channels=in_channels, version=self.version,
                            resnext_groups=resnext_groups))
        for _ in range(layers-1):
            layer.append(block(builder, channels, 1, False, in_channels=channels,
                               version=self.version, resnext_groups=resnext_groups))
        return builder.sequence(*layer)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


class Xception(HybridBlock):
    def __init__(self, builder,
                 definition=([32, 64],
                             [[128, 128, 0], [256, 256, 0], [728, 728, 0],
                              *([[728, 728, 728]] * 8), [728, 1024, 0]],
                             [1536, 2048]),
                 classes=1000):
        super().__init__()

        definition1, definition2, definition3 = definition

        with self.name_scope():
            features = []
            last_channels = 0
            for i, channels in enumerate(definition1):
                features += [
                    builder.conv(channels, 3, strides=(2 if i == 0 else 1), in_channels=last_channels),
                    builder.batchnorm_relu(),
                ]
                last_channels = channels

            for i, block_definition in enumerate(definition2):
                features.append(XceptionBlock(builder, block_definition, in_channels=last_channels,
                                              relu_at_beginning=False if i == 0 else True))
                last_channels = list(filter(lambda x: x > 0, block_definition))[-1]

            for i, channels in enumerate(definition3):
                features += [
                    builder.separable_conv(channels, 3, in_channels=last_channels),
                    builder.batchnorm_relu(),
                ]
                last_channels = channels

            features.append(builder.global_avg_pool())

            self.features = builder.sequence(*features)
            self.output = builder.dense(classes, in_units=last_channels)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)

        return x


resnet_spec = {18: (ResNetBasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: (ResNetBasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: (ResNetBottleNeck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: (ResNetBottleNeck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: (ResNetBottleNeck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}

def create_resnet(builder, version, num_layers=50, resnext=False, classes=1000):
    assert num_layers in resnet_spec, \
        "Invalid number of layers: {}. Options are {}".format(
            num_layers, str(resnet_spec.keys()))
    block_class, layers, channels = resnet_spec[num_layers]
    assert not resnext or num_layers >= 50, \
        "Cannot create resnext with less then 50 layers"
    net = ResNet(builder, block_class, layers, channels, version=version,
                 resnext_groups=args.num_groups if resnext else None)
    return net

class fp16_model(mx.gluon.block.HybridBlock):
    def __init__(self, net, **kwargs):
        super(fp16_model, self).__init__(**kwargs)
        with self.name_scope():
            self._net = net

    def hybrid_forward(self, F, x):
        y = self._net(x)
        y = F.cast(y, dtype='float32')
        return y

def get_model(arch, num_classes, num_layers, image_shape, dtype, amp,
              input_layout, conv_layout, batchnorm_layout, pooling_layout,
              batchnorm_eps, batchnorm_mom, fuse_bn_relu, fuse_bn_add_relu, **kwargs):

    builder = Builder(
            dtype               = dtype,
            input_layout        = input_layout,
            conv_layout         = conv_layout,
            bn_layout           = batchnorm_layout,
            pooling_layout      = pooling_layout,
            bn_eps              = batchnorm_eps,
            bn_mom              = batchnorm_mom,
            fuse_bn_relu        = fuse_bn_relu,
            fuse_bn_add_relu    = fuse_bn_add_relu,
    )

    if arch.startswith('resnet') or arch.startswith('resnext'):
        version = '1' if arch in {'resnetv1', 'resnextv1'} else '1.5'
        net = create_resnet(
                builder         = builder,
                version         = version,
                resnext         = arch.startswith('resnext'),
                num_layers      = num_layers,
                classes         = num_classes,
        )
    elif arch == 'xception':
        net = Xception(builder, classes=num_classes)
    else:
        raise ValueError('Wrong model architecture')

    net.hybridize(static_shape=True, static_alloc=True)

    if not amp:
        net.cast(dtype)
        if dtype == 'float16':
            net = fp16_model(net)

    return net

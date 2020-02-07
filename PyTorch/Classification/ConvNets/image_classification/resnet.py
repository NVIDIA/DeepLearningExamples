# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import math
import torch
import torch.nn as nn
import numpy as np

__all__ = ['ResNet', 'build_resnet', 'resnet_versions', 'resnet_configs']

# ResNetBuilder {{{

class ResNetBuilder(object):
    def __init__(self, version, config):
        self.conv3x3_cardinality = 1 if 'cardinality' not in version.keys() else version['cardinality']
        self.config = config

    def conv(self, kernel_size, in_planes, out_planes, groups=1, stride=1):
        conv = nn.Conv2d(
                in_planes, out_planes,
                kernel_size=kernel_size, groups=groups,
                stride=stride, padding=int((kernel_size - 1)/2),
                bias=False)

        if self.config['nonlinearity'] == 'relu': 
            nn.init.kaiming_normal_(conv.weight,
                    mode=self.config['conv_init'],
                    nonlinearity=self.config['nonlinearity'])

        return conv

    def conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        c = self.conv(3, in_planes, out_planes, groups=self.conv3x3_cardinality, stride=stride)
        return c

    def conv1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution with padding"""
        c = self.conv(1, in_planes, out_planes, stride=stride)
        return c

    def conv7x7(self, in_planes, out_planes, stride=1):
        """7x7 convolution with padding"""
        c = self.conv(7, in_planes, out_planes, stride=stride)
        return c

    def conv5x5(self, in_planes, out_planes, stride=1):
        """5x5 convolution with padding"""
        c = self.conv(5, in_planes, out_planes, stride=stride)
        return c

    def batchnorm(self, planes, last_bn=False):
        bn = nn.BatchNorm2d(planes)
        gamma_init_val = 0 if last_bn and self.config['last_bn_0_init'] else 1
        nn.init.constant_(bn.weight, gamma_init_val)
        nn.init.constant_(bn.bias, 0)

        return bn

    def activation(self):
        return self.config['activation']()

# ResNetBuilder }}}

# BasicBlock {{{
class BasicBlock(nn.Module):
    def __init__(self, builder, inplanes, planes, expansion, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = builder.conv3x3(inplanes, planes, stride)
        self.bn1 = builder.batchnorm(planes)
        self.relu = builder.activation()
        self.conv2 = builder.conv3x3(planes, planes*expansion)
        self.bn2 = builder.batchnorm(planes*expansion, last_bn=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        if self.bn2 is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
# BasicBlock }}}

# SqueezeAndExcitation {{{
class SqueezeAndExcitation(nn.Module):
    def __init__(self, planes, squeeze):
        super(SqueezeAndExcitation, self).__init__()
        self.squeeze = nn.Linear(planes, squeeze)
        self.expand = nn.Linear(squeeze, planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
        out = self.squeeze(out)
        out = self.relu(out)
        out = self.expand(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(2).unsqueeze(3)

        return out

# }}}

# Bottleneck {{{
class Bottleneck(nn.Module):
    def __init__(self, builder, inplanes, planes, expansion, stride=1, se=False, se_squeeze=16, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = builder.conv1x1(inplanes, planes)
        self.bn1 = builder.batchnorm(planes)
        self.conv2 = builder.conv3x3(planes, planes, stride=stride)
        self.bn2 = builder.batchnorm(planes)
        self.conv3 = builder.conv1x1(planes, planes * expansion)
        self.bn3 = builder.batchnorm(planes * expansion, last_bn=True)
        self.relu = builder.activation()
        self.downsample = downsample
        self.stride = stride
        self.squeeze = SqueezeAndExcitation(planes*expansion, se_squeeze) if se else None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.squeeze is None:
            out += residual
        else:
            out = torch.addcmul(residual, 1.0, out, self.squeeze(out))

        out = self.relu(out)

        return out

def SEBottleneck(builder, inplanes, planes, expansion, stride=1, downsample=None):
    return Bottleneck(builder, inplanes, planes, expansion, stride=stride, se=True, se_squeeze=16, downsample=downsample)
# Bottleneck }}}

# ResNet {{{
class ResNet(nn.Module):
    def __init__(self, builder, block, expansion, layers, widths, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = builder.conv7x7(3, 64, stride=2)
        self.bn1 = builder.batchnorm(64)
        self.relu = builder.activation()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(builder, block, expansion, widths[0], layers[0])
        self.layer2 = self._make_layer(builder, block, expansion, widths[1], layers[1], stride=2)
        self.layer3 = self._make_layer(builder, block, expansion, widths[2], layers[2], stride=2)
        self.layer4 = self._make_layer(builder, block, expansion, widths[3], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[3] * expansion, num_classes)

    def _make_layer(self, builder, block, expansion, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * expansion:
            dconv = builder.conv1x1(self.inplanes, planes * expansion,
                                    stride=stride)
            dbn = builder.batchnorm(planes * expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        layers = []
        layers.append(block(builder, self.inplanes, planes, expansion, stride=stride, downsample=downsample))
        self.inplanes = planes * expansion
        for i in range(1, blocks):
            layers.append(block(builder, self.inplanes, planes, expansion))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
# ResNet }}}

resnet_configs = {
        'classic' : {
            'conv' : nn.Conv2d,
            'conv_init' : 'fan_out',
            'nonlinearity' : 'relu',
            'last_bn_0_init' : False,
            'activation' : lambda: nn.ReLU(inplace=True),
            },
        'fanin' : {
            'conv' : nn.Conv2d,
            'conv_init' : 'fan_in',
            'nonlinearity' : 'relu',
            'last_bn_0_init' : False,
            'activation' : lambda: nn.ReLU(inplace=True),
            },
        'grp-fanin' : {
            'conv' : nn.Conv2d,
            'conv_init' : 'fan_in',
            'nonlinearity' : 'relu',
            'last_bn_0_init' : False,
            'activation' : lambda: nn.ReLU(inplace=True),
            },
        'grp-fanout' : {
            'conv' : nn.Conv2d,
            'conv_init' : 'fan_out',
            'nonlinearity' : 'relu',
            'last_bn_0_init' : False,
            'activation' : lambda: nn.ReLU(inplace=True),
            },
        }

resnet_versions = {
        'resnet18' : {
            'net' : ResNet,
            'block' : BasicBlock,
            'layers' : [2, 2, 2, 2],
            'widths' : [64, 128, 256, 512],
            'expansion' : 1,
            'num_classes' : 1000,
            },
         'resnet34' : {
            'net' : ResNet,
            'block' : BasicBlock,
            'layers' : [3, 4, 6, 3],
            'widths' : [64, 128, 256, 512],
            'expansion' : 1,
            'num_classes' : 1000,
            },
         'resnet50' : {
            'net' : ResNet,
            'block' : Bottleneck,
            'layers' : [3, 4, 6, 3],
            'widths' : [64, 128, 256, 512],
            'expansion' : 4,
            'num_classes' : 1000,
            },
        'resnet101' : {
            'net' : ResNet,
            'block' : Bottleneck,
            'layers' : [3, 4, 23, 3],
            'widths' : [64, 128, 256, 512],
            'expansion' : 4,
            'num_classes' : 1000,
            },
        'resnet152' : {
            'net' : ResNet,
            'block' : Bottleneck,
            'layers' : [3, 8, 36, 3],
            'widths' : [64, 128, 256, 512],
            'expansion' : 4,
            'num_classes' : 1000,
            },
        'resnext101-32x4d' : {
            'net' : ResNet,
            'block' : Bottleneck,
            'cardinality' : 32,
            'layers' : [3, 4, 23, 3],
            'widths' : [128, 256, 512, 1024],
            'expansion' : 2,
            'num_classes' : 1000,
            },
        'se-resnext101-32x4d' : {
            'net' : ResNet,
            'block' : SEBottleneck,
            'cardinality' : 32,
            'layers' : [3, 4, 23, 3],
            'widths' : [128, 256, 512, 1024],
            'expansion' : 2,
            'num_classes' : 1000,
            },
        }


def build_resnet(version, config, verbose=True):
    version = resnet_versions[version]
    config = resnet_configs[config]

    builder = ResNetBuilder(version, config)
    if verbose:
        print("Version: {}".format(version))
        print("Config: {}".format(config))
    model = version['net'](builder,
                           version['block'],
                           version['expansion'],
                           version['layers'],
                           version['widths'],
                           version['num_classes'])

    return model

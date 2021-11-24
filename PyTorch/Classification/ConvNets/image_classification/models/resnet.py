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

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Dict, Callable, Any, Type

import torch
import torch.nn as nn

from .common import (
    SqueezeAndExcitation,
    LayerBuilder,
    SqueezeAndExcitationTRT,
)

from .model import (
    Model,
    ModelParams,
    ModelArch,
    EntryPoint,
)


__all__ = ["ResNet", "resnet_configs"]

# BasicBlock {{{
class BasicBlock(nn.Module):
    def __init__(
        self,
        builder,
        inplanes,
        planes,
        expansion,
        stride=1,
        cardinality=1,
        downsample=None,
        last_bn_0_init=False,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = builder.conv3x3(inplanes, planes, stride, cardinality=cardinality)
        self.bn1 = builder.batchnorm(planes)
        self.relu = builder.activation()
        self.conv2 = builder.conv3x3(
            planes, planes * expansion, cardinality=cardinality
        )
        self.bn2 = builder.batchnorm(planes * expansion, zero_init=last_bn_0_init)
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

# Bottleneck {{{
class Bottleneck(nn.Module):
    def __init__(
        self,
        builder,
        inplanes,
        planes,
        expansion,
        stride=1,
        cardinality=1,
        se=False,
        se_squeeze=16,
        downsample=None,
        fused_se=True,
        last_bn_0_init=False,
        trt=False,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = builder.conv1x1(inplanes, planes)
        self.bn1 = builder.batchnorm(planes)
        self.conv2 = builder.conv3x3(planes, planes, groups=cardinality, stride=stride)
        self.bn2 = builder.batchnorm(planes)
        self.conv3 = builder.conv1x1(planes, planes * expansion)
        self.bn3 = builder.batchnorm(planes * expansion, zero_init=last_bn_0_init)
        self.relu = builder.activation()
        self.downsample = downsample
        self.stride = stride

        self.fused_se = fused_se
        if se:
            self.squeeze = (
                SqueezeAndExcitation(
                    planes * expansion, se_squeeze, builder.activation()
                )
                if not trt
                else SqueezeAndExcitationTRT(
                    planes * expansion, se_squeeze, builder.activation()
                )
            )
        else:
            self.squeeze = None

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
            if self.fused_se:
                out = torch.addcmul(residual, out, self.squeeze(out), value=1)
            else:
                out = residual + out * self.squeeze(out)

        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    def __init__(
        self,
        builder,
        inplanes,
        planes,
        expansion,
        stride=1,
        cardinality=1,
        downsample=None,
        fused_se=True,
        last_bn_0_init=False,
        trt=False,
    ):
        super(SEBottleneck, self).__init__(
            builder,
            inplanes,
            planes,
            expansion,
            stride=stride,
            cardinality=cardinality,
            se=True,
            se_squeeze=16,
            downsample=downsample,
            fused_se=fused_se,
            last_bn_0_init=last_bn_0_init,
            trt=trt,
        )


# Bottleneck }}}


class ResNet(nn.Module):
    @dataclass
    class Arch(ModelArch):
        block: Type[Bottleneck]
        layers: List[int]  # arch
        widths: List[int]  # arch
        expansion: int
        cardinality: int = 1
        stem_width: int = 64
        activation: str = "relu"
        default_image_size: int = 224

    @dataclass
    class Params(ModelParams):
        num_classes: int = 1000
        last_bn_0_init: bool = False
        conv_init: str = "fan_in"
        trt: bool = False
        fused_se: bool = True

        def parser(self, name):
            p = super().parser(name)

            p.add_argument(
                "--num_classes",
                metavar="N",
                default=self.num_classes,
                type=int,
                help="number of classes",
            )
            p.add_argument(
                "--last_bn_0_init",
                metavar="True|False",
                default=self.last_bn_0_init,
                type=bool,
            )
            p.add_argument(
                "--conv_init",
                default=self.conv_init,
                choices=["fan_in", "fan_out"],
                type=str,
                help="initialization mode for convolutional layers, see https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_",
            )
            p.add_argument("--trt", metavar="True|False", default=self.trt, type=bool)
            p.add_argument(
                "--fused_se", metavar="True|False", default=self.fused_se, type=bool
            )

            return p

    def __init__(
        self,
        arch: Arch,
        num_classes: int = 1000,
        last_bn_0_init: bool = False,
        conv_init: str = "fan_in",
        trt: bool = False,
        fused_se: bool = True,
    ):

        super(ResNet, self).__init__()
        self.arch = arch
        self.builder = LayerBuilder(
            LayerBuilder.Config(activation=arch.activation, conv_init=conv_init)
        )
        self.last_bn_0_init = last_bn_0_init
        self.conv1 = self.builder.conv7x7(3, arch.stem_width, stride=2)
        self.bn1 = self.builder.batchnorm(arch.stem_width)
        self.relu = self.builder.activation()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        inplanes = arch.stem_width
        assert len(arch.widths) == len(arch.layers)
        self.num_layers = len(arch.widths)
        layers = []
        for i, (w, l) in enumerate(zip(arch.widths, arch.layers)):
            layer, inplanes = self._make_layer(
                arch.block,
                arch.expansion,
                inplanes,
                w,
                l,
                cardinality=arch.cardinality,
                stride=1 if i == 0 else 2,
                trt=trt,
                fused_se=fused_se,
            )
            layers.append(layer)

        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(arch.widths[-1] * arch.expansion, num_classes)

    def stem(self, x):
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def classifier(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.classifier(x)
        return x

    def extract_features(self, x, layers=None):
        if layers is None:
            layers = [f"layer{i+1}" for i in range(self.num_layers)] + ["classifier"]

        run = [
            i
            for i in range(self.num_layers)
            if "classifier" in layers
            or any([f"layer{j+1}" in layers for j in range(i, self.num_layers)])
        ]

        output = {}
        x = self.stem(x)
        for l in run:
            fn = self.layers[l]
            x = fn(x)
            if f"layer{l+1}" in layers:
                output[f"layer{l+1}"] = x

        if "classifier" in layers:
            output["classifier"] = self.classifier(x)

        return output

    # helper functions {{{
    def _make_layer(
        self,
        block,
        expansion,
        inplanes,
        planes,
        blocks,
        stride=1,
        cardinality=1,
        trt=False,
        fused_se=True,
    ):
        downsample = None
        if stride != 1 or inplanes != planes * expansion:
            dconv = self.builder.conv1x1(inplanes, planes * expansion, stride=stride)
            dbn = self.builder.batchnorm(planes * expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        layers = []
        for i in range(blocks):
            layers.append(
                block(
                    self.builder,
                    inplanes,
                    planes,
                    expansion,
                    stride=stride if i == 0 else 1,
                    cardinality=cardinality,
                    downsample=downsample if i == 0 else None,
                    fused_se=fused_se,
                    last_bn_0_init=self.last_bn_0_init,
                    trt=trt,
                )
            )
            inplanes = planes * expansion

        return nn.Sequential(*layers), inplanes

    def ngc_checkpoint_remap(self, url=None, version=None):
        if version is None:
            version = url.split("/")[8]

        def to_sequential_remap(s):
            splited = s.split(".")
            if splited[0].startswith("layer"):
                return ".".join(
                    ["layers." + str(int(splited[0][len("layer") :]) - 1)] + splited[1:]
                )
            else:
                return s

        def no_remap(s):
            return s

        return {"20.06.0": to_sequential_remap}.get(version, no_remap)

    # }}}


__models: Dict[str, Model] = {
    "resnet50": Model(
        constructor=ResNet,
        arch=ResNet.Arch(
            stem_width=64,
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            widths=[64, 128, 256, 512],
            expansion=4,
            default_image_size=224,
        ),
        params=ResNet.Params(),
        checkpoint_url="https://api.ngc.nvidia.com/v2/models/nvidia/resnet50_pyt_amp/versions/20.06.0/files/nvidia_resnet50_200821.pth.tar",
    ),
    "resnext101-32x4d": Model(
        constructor=ResNet,
        arch=ResNet.Arch(
            stem_width=64,
            block=Bottleneck,
            layers=[3, 4, 23, 3],
            widths=[128, 256, 512, 1024],
            expansion=2,
            cardinality=32,
            default_image_size=224,
        ),
        params=ResNet.Params(),
        checkpoint_url="https://api.ngc.nvidia.com/v2/models/nvidia/resnext101_32x4d_pyt_amp/versions/20.06.0/files/nvidia_resnext101-32x4d_200821.pth.tar",
    ),
    "se-resnext101-32x4d": Model(
        constructor=ResNet,
        arch=ResNet.Arch(
            stem_width=64,
            block=SEBottleneck,
            layers=[3, 4, 23, 3],
            widths=[128, 256, 512, 1024],
            expansion=2,
            cardinality=32,
            default_image_size=224,
        ),
        params=ResNet.Params(),
        checkpoint_url="https://api.ngc.nvidia.com/v2/models/nvidia/seresnext101_32x4d_pyt_amp/versions/20.06.0/files/nvidia_se-resnext101-32x4d_200821.pth.tar",
    ),
}

_ce = lambda n: EntryPoint.create(n, __models[n])
resnet50 = _ce("resnet50")
resnext101_32x4d = _ce("resnext101-32x4d")
se_resnext101_32x4d = _ce("se-resnext101-32x4d")

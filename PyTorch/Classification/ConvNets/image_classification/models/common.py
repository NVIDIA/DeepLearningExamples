import copy
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn

# LayerBuilder {{{
class LayerBuilder(object):
    @dataclass
    class Config:
        activation: str = "relu"
        conv_init: str = "fan_in"
        bn_momentum: Optional[float] = None
        bn_epsilon: Optional[float] = None

    def __init__(self, config: "LayerBuilder.Config"):
        self.config = config

    def conv(
        self,
        kernel_size,
        in_planes,
        out_planes,
        groups=1,
        stride=1,
        bn=False,
        zero_init_bn=False,
        act=False,
    ):
        conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            groups=groups,
            stride=stride,
            padding=int((kernel_size - 1) / 2),
            bias=False,
        )

        nn.init.kaiming_normal_(
            conv.weight, mode=self.config.conv_init, nonlinearity="relu"
        )
        layers = [("conv", conv)]
        if bn:
            layers.append(("bn", self.batchnorm(out_planes, zero_init_bn)))
        if act:
            layers.append(("act", self.activation()))

        if bn or act:
            return nn.Sequential(OrderedDict(layers))
        else:
            return conv

    def convDepSep(
        self, kernel_size, in_planes, out_planes, stride=1, bn=False, act=False
    ):
        """3x3 depthwise separable convolution with padding"""
        c = self.conv(
            kernel_size,
            in_planes,
            out_planes,
            groups=in_planes,
            stride=stride,
            bn=bn,
            act=act,
        )
        return c

    def conv3x3(self, in_planes, out_planes, stride=1, groups=1, bn=False, act=False):
        """3x3 convolution with padding"""
        c = self.conv(
            3, in_planes, out_planes, groups=groups, stride=stride, bn=bn, act=act
        )
        return c

    def conv1x1(self, in_planes, out_planes, stride=1, groups=1, bn=False, act=False):
        """1x1 convolution with padding"""
        c = self.conv(
            1, in_planes, out_planes, groups=groups, stride=stride, bn=bn, act=act
        )
        return c

    def conv7x7(self, in_planes, out_planes, stride=1, groups=1, bn=False, act=False):
        """7x7 convolution with padding"""
        c = self.conv(
            7, in_planes, out_planes, groups=groups, stride=stride, bn=bn, act=act
        )
        return c

    def conv5x5(self, in_planes, out_planes, stride=1, groups=1, bn=False, act=False):
        """5x5 convolution with padding"""
        c = self.conv(
            5, in_planes, out_planes, groups=groups, stride=stride, bn=bn, act=act
        )
        return c

    def batchnorm(self, planes, zero_init=False):
        bn_cfg = {}
        if self.config.bn_momentum is not None:
            bn_cfg["momentum"] = self.config.bn_momentum
        if self.config.bn_epsilon is not None:
            bn_cfg["eps"] = self.config.bn_epsilon

        bn = nn.BatchNorm2d(planes, **bn_cfg)
        gamma_init_val = 0 if zero_init else 1
        nn.init.constant_(bn.weight, gamma_init_val)
        nn.init.constant_(bn.bias, 0)

        return bn

    def activation(self):
        return {
            "silu": lambda: nn.SiLU(inplace=True),
            "relu": lambda: nn.ReLU(inplace=True),
            "onnx-silu": ONNXSiLU,
        }[self.config.activation]()


# LayerBuilder }}}

# LambdaLayer {{{
class LambdaLayer(nn.Module):
    def __init__(self, lmbd):
        super().__init__()
        self.lmbd = lmbd

    def forward(self, x):
        return self.lmbd(x)


# }}}

# SqueezeAndExcitation {{{
class SqueezeAndExcitation(nn.Module):
    def __init__(self, in_channels, squeeze, activation):
        super(SqueezeAndExcitation, self).__init__()
        self.squeeze = nn.Linear(in_channels, squeeze)
        self.expand = nn.Linear(squeeze, in_channels)
        self.activation = activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = torch.mean(x, [2, 3])
        out = self.squeeze(out)
        out = self.activation(out)
        out = self.expand(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(2).unsqueeze(3)
        return out


# }}}

# EMA {{{
class EMA:
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def state_dict(self):
        return copy.deepcopy(self.shadow)

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

    def __len__(self):
        return len(self.shadow)

    def __call__(self, module, step=None):
        if step is None:
            mu = self.mu
        else:
            mu = min(self.mu, (1.0 + step) / (10 + step))

        for name, x in module.state_dict().items():
            if name in self.shadow:
                new_average = (1.0 - mu) * x + mu * self.shadow[name]
                self.shadow[name] = new_average.clone()
            else:
                self.shadow[name] = x.clone()


# }}}

# ONNXSiLU {{{
# Since torch.nn.SiLU is not supported in ONNX,
# it is required to use this implementation in exported model (15-20% more GPU memory is needed)
class ONNXSiLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ONNXSiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


# }}}


class SequentialSqueezeAndExcitation(SqueezeAndExcitation):
    def forward(self, x):
        return super().forward(x) * x

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

# Copyright 2019-2022 Ross Wightman
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
from typing import List, Tuple

import torch
import torch.nn as nn
from timm.models.layers import create_act_layer
from torch.nn import functional as F


# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


# Can SAME padding for given args be done statically?
def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


# Dynamically pad input x with 'SAME' padding for conv with specified args
def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(
        iw, k[1], s[1], d[1]
    )
    if pad_h > 0 or pad_w > 0:
        x = F.pad(
            x,
            [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
            value=value,
        )
    return x


def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == "same":
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == "valid":
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop("padding", "")
    kwargs.setdefault("bias", False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)


def create_conv2d(in_channels, out_channels, kernel_size, **kwargs):
    """Select a 2d convolution implementation based on arguments
    Creates and returns one of torch.nn.Conv2d, Conv2dSame, MixedConv2d, or CondConv2d.

    Used extensively by EfficientNet, MobileNetv3 and related networks.
    """
    if isinstance(kernel_size, list):
        raise NotImplementedError
    else:
        depthwise = kwargs.pop("depthwise", False)
        # for DW out_channels must be multiple of in_channels as must have out_channels % groups == 0
        groups = in_channels if depthwise else kwargs.pop("groups", 1)
        if "num_experts" in kwargs and kwargs["num_experts"] > 0:
            raise NotImplementedError
        else:
            m = create_conv2d_pad(
                in_channels, out_channels, kernel_size, groups=groups, **kwargs
            )
    return m


def get_act(actType: str = ""):
    if actType == "swish":
        return nn.SiLU
    elif actType == "relu":
        return nn.ReLU
    else:
        raise NotImplementedError


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation w/ specific features for EfficientNet/MobileNet family

    Args:
        in_chs (int): input channels to layer
        rd_ratio (float): ratio of squeeze reduction
        act_layer (nn.Module): activation layer of containing block
        gate_layer (Callable): attention gate function
        force_act_layer (nn.Module): override block's activation fn if this is set/bound
        rd_round_fn (Callable): specify a fn to calculate rounding of reduced chs
    """

    def __init__(
        self,
        in_chs,
        rd_ratio=0.25,
        rd_channels=None,
        act_layer=nn.ReLU,
        gate_layer=nn.Sigmoid,
        force_act_layer=None,
        rd_round_fn=None,
    ):
        super(SqueezeExcite, self).__init__()
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = rd_round_fn(in_chs * rd_ratio)
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = create_act_layer(act_layer, inplace=True)
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class ConvBnAct(nn.Module):
    """Conv + Norm Layer + Activation w/ optional skip connection"""

    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size,
        stride=1,
        dilation=1,
        pad_type="",
        skip=False,
        act_layer="relu",
        norm_layer=nn.BatchNorm2d,
        drop_path_rate=0.0,
    ):
        super(ConvBnAct, self).__init__()
        self.has_residual = skip and stride == 1 and in_chs == out_chs
        self.drop_path_rate = drop_path_rate
        self.conv = create_conv2d(
            in_chs,
            out_chs,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=pad_type,
        )
        self.bn1 = norm_layer(out_chs, eps=0.001)
        self.act1 = get_act(act_layer)(inplace=True)

        # for representation.
        self.in_channels = in_chs
        self.out_channels = out_chs
        self.kernel_size = kernel_size
        self.stride = stride
        self.act_layer = act_layer

    def feature_info(self, location):
        if location == "expansion":
            # output of conv after act, same as block coutput
            info = dict(
                module="act1", hook_type="forward", num_chs=self.conv.out_channels
            )
        else:
            info = dict(module="", hook_type="", num_chs=self.conv.out_channels)
        return info

    def __repr__(self):
        name = "conv_k{}_i{}_o{}_s{}_{}".format(
            self.kernel_size,
            self.in_channels,
            self.out_channels,
            self.stride,
            self.act_layer,
        )
        return name

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.has_residual:
            if self.drop_path_rate > 0.0:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
        return x


class DepthwiseSeparableConv(nn.Module):
    """DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """

    def __init__(
        self,
        in_chs,
        out_chs,
        dw_kernel_size=3,
        stride=1,
        dilation=1,
        pad_type="",
        noskip=False,
        pw_kernel_size=1,
        pw_act=False,
        act_layer="relu",
        norm_layer=nn.BatchNorm2d,
        se_layer=None,
        drop_path_rate=0.0,
    ):
        super(DepthwiseSeparableConv, self).__init__()
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv
        self.drop_path_rate = drop_path_rate

        self.conv_dw = create_conv2d(
            in_chs,
            in_chs,
            dw_kernel_size,
            stride=stride,
            dilation=dilation,
            padding=pad_type,
            depthwise=True,
        )
        self.bn1 = norm_layer(in_chs, eps=0.001)
        self.act1 = get_act(act_layer)(inplace=True)

        # Squeeze-and-excitation
        self.se = (
            se_layer(in_chs, act_layer=get_act(act_layer))
            if se_layer
            else nn.Identity()
        )

        self.conv_pw = create_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_layer(out_chs, eps=0.001)
        self.act2 = act_layer(inplace=True) if self.has_pw_act else nn.Identity()

    def feature_info(self, location):
        if location == "expansion":  # after SE, input to PW
            info = dict(
                module="conv_pw",
                hook_type="forward_pre",
                num_chs=self.conv_pw.in_channels,
            )
        else:  # location == 'bottleneck', block output
            info = dict(module="", hook_type="", num_chs=self.conv_pw.out_channels)
        return info

    def forward(self, x):
        shortcut = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.act2(x)

        if self.has_residual:
            if self.drop_path_rate > 0.0:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
        return x


class InvertedResidual(nn.Module):
    """Inverted residual block w/ optional SE

    Originally used in MobileNet-V2 - https://arxiv.org/abs/1801.04381v4, this layer is often
    referred to as 'MBConv' for (Mobile inverted bottleneck conv) and is also used in
      * MNasNet - https://arxiv.org/abs/1807.11626
      * EfficientNet - https://arxiv.org/abs/1905.11946
      * MobileNet-V3 - https://arxiv.org/abs/1905.02244
    """

    def __init__(
        self,
        in_chs,
        out_chs,
        dw_kernel_size=3,
        stride=1,
        dilation=1,
        pad_type="",
        noskip=False,
        exp_ratio=1.0,
        exp_kernel_size=1,
        pw_kernel_size=1,
        act_layer="relu",
        norm_layer=nn.BatchNorm2d,
        use_se=None,
        se_ratio=0.25,
        conv_kwargs=None,
        drop_path_rate=0.0,
    ):
        super(InvertedResidual, self).__init__()
        conv_kwargs = conv_kwargs or {}
        mid_chs = make_divisible(in_chs * exp_ratio)
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate

        # Point-wise expansion
        self.conv_pw = create_conv2d(
            in_chs, mid_chs, exp_kernel_size, padding=pad_type, **conv_kwargs
        )
        self.bn1 = norm_layer(mid_chs, eps=0.001)
        self.act1 = get_act(act_layer)(inplace=True)

        # Depth-wise convolution
        self.conv_dw = create_conv2d(
            mid_chs,
            mid_chs,
            dw_kernel_size,
            stride=stride,
            dilation=dilation,
            padding=pad_type,
            depthwise=True,
            **conv_kwargs
        )
        self.bn2 = norm_layer(mid_chs, eps=0.001)
        self.act2 = get_act(act_layer)(inplace=True)

        # Squeeze-and-excitation
        self.use_se = use_se
        if use_se:
            rd_ratio = se_ratio / exp_ratio
            self.se = SqueezeExcite(
                mid_chs, act_layer=get_act(act_layer), rd_ratio=rd_ratio
            )
        else:
            self.se = nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(
            mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs
        )
        self.bn3 = norm_layer(out_chs, eps=0.001)

        # For representation
        self.in_channels = in_chs
        self.out_channels = out_chs
        self.kernel_size = dw_kernel_size
        self.expansion = exp_ratio
        self.stride = stride
        self.act_layer = act_layer

    def feature_info(self, location):
        if location == "expansion":  # after SE, input to PWL
            info = dict(
                module="conv_pwl",
                hook_type="forward_pre",
                num_chs=self.conv_pwl.in_channels,
            )
        else:  # location == 'bottleneck', block output
            info = dict(module="", hook_type="", num_chs=self.conv_pwl.out_channels)
        return info

    def __repr__(self):
        name = "irb_k{}_e{}_i{}_o{}_s{}_{}_se_{}".format(
            self.kernel_size,
            self.expansion,
            self.in_channels,
            self.out_channels,
            self.stride,
            self.act_layer,
            self.use_se,
        )
        return name

    def forward(self, x):
        shortcut = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            if self.drop_path_rate > 0.0:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut

        return x


class EdgeResidual(nn.Module):
    """Residual block with expansion convolution followed by pointwise-linear w/ stride

    Originally introduced in `EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML`
        - https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html

    This layer is also called FusedMBConv in the MobileDet, EfficientNet-X, and EfficientNet-V2 papers
      * MobileDet - https://arxiv.org/abs/2004.14525
      * EfficientNet-X - https://arxiv.org/abs/2102.05610
      * EfficientNet-V2 - https://arxiv.org/abs/2104.00298
    """

    def __init__(
        self,
        in_chs,
        out_chs,
        exp_kernel_size=3,
        stride=1,
        dilation=1,
        pad_type="",
        force_in_chs=0,
        noskip=False,
        exp_ratio=1.0,
        pw_kernel_size=1,
        act_layer="relu",
        norm_layer=nn.BatchNorm2d,
        use_se=False,
        se_ratio=0.25,
        drop_path_rate=0.0,
    ):
        super(EdgeResidual, self).__init__()
        if force_in_chs > 0:
            mid_chs = make_divisible(force_in_chs * exp_ratio)
        else:
            mid_chs = make_divisible(in_chs * exp_ratio)
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate

        # Expansion convolution
        self.conv_exp = create_conv2d(
            in_chs,
            mid_chs,
            exp_kernel_size,
            stride=stride,
            dilation=dilation,
            padding=pad_type,
        )
        self.bn1 = norm_layer(mid_chs, eps=0.001)
        self.act1 = get_act(act_layer)(inplace=True)

        # Squeeze-and-excitation
        self.use_se = use_se
        if use_se:
            rd_ratio = se_ratio / exp_ratio
            self.se = SqueezeExcite(
                mid_chs, act_layer=get_act(act_layer), rd_ratio=rd_ratio
            )
        else:
            self.se = nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(
            mid_chs, out_chs, pw_kernel_size, padding=pad_type
        )
        self.bn2 = norm_layer(out_chs, eps=0.001)

        self.kernel_size = exp_kernel_size
        self.expansion = exp_ratio
        self.in_channels = in_chs
        self.out_channels = out_chs
        self.stride = stride
        self.act_layer = act_layer

    def feature_info(self, location):
        if location == "expansion":  # after SE, before PWL
            info = dict(
                module="conv_pwl",
                hook_type="forward_pre",
                num_chs=self.conv_pwl.in_channels,
            )
        else:  # location == 'bottleneck', block output
            info = dict(module="", hook_type="", num_chs=self.conv_pwl.out_channels)
        return info

    def __repr__(self):
        name = "er_k{}_e{}_i{}_o{}_s{}_{}_se_{}".format(
            self.kernel_size,
            self.expansion,
            self.in_channels,
            self.out_channels,
            self.stride,
            self.act_layer,
            self.use_se,
        )
        return name

    def forward(self, x):
        shortcut = x

        # Expansion convolution
        x = self.conv_exp(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn2(x)

        if self.has_residual:
            if self.drop_path_rate > 0.0:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut

        return x


class ProloguePool(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, act_layer="relu"):
        super().__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.net = nn.Sequential(
            nn.Conv2d(
                self.num_in_channels,
                self.num_out_channels,
                3,
                padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(self.num_out_channels, eps=1e-03),
            get_act(act_layer)(inplace=True),
            nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
            ),
        )
        # for representation
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.act_layer = act_layer

    def __repr__(self):
        name = "prologue_i{}_o{}_s{}_{}".format(
            self.num_in_channels, self.num_out_channels, 2, self.act_layer
        )
        return name

    def forward(self, x):
        return self.net(x)


class Prologue(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, act_layer="relu"):
        super().__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.net = nn.Sequential(
            nn.Conv2d(
                self.num_in_channels,
                self.num_out_channels,
                3,
                padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(self.num_out_channels, eps=1e-03),
            get_act(act_layer)(inplace=True),
        )
        # for representation
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.act_layer = act_layer

    def __repr__(self):
        name = "prologue_i{}_o{}_s{}_{}".format(
            self.num_in_channels, self.num_out_channels, 2, self.act_layer
        )
        return name

    def forward(self, x):
        return self.net(x)


class Epilogue(nn.Module):
    def __init__(
        self, num_in_channels, num_out_channels, num_classes, act_layer="relu"
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_in_channels, num_out_channels, 1, bias=False),
            nn.BatchNorm2d(num_out_channels, eps=1e-03),
            get_act(act_layer)(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(num_out_channels, num_classes),
        )

        # for representation
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.act_layer = act_layer

    def __repr__(self):
        name = "epilogue_i{}_o{}_s{}_{}".format(
            self.num_in_channels, self.num_out_channels, 1, self.act_layer
        )
        return name

    def forward(self, x):
        x = self.net(x)
        return x


# modules for distilled GPUNet
class PrologueD(nn.Module):
    def __init__(self, num_in_channels, num_out_channels):
        super().__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.net = nn.Sequential(
            nn.Conv2d(
                self.num_in_channels,
                self.num_out_channels,
                3,
                padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(self.num_out_channels),
            nn.ReLU(),
        )

    def __repr__(self):
        return "Prologue"

    def forward(self, x):
        return self.net(x)


class PrologueLargeD(nn.Module):
    def __init__(self, num_in_channels, num_out_channels):
        super().__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.net = nn.Sequential(
            nn.Conv2d(
                self.num_in_channels,
                self.num_out_channels,
                3,
                padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(self.num_out_channels),
            nn.ReLU(),
            nn.Conv2d(
                self.num_out_channels,
                self.num_out_channels,
                3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.num_out_channels),
            nn.ReLU(),
            nn.Conv2d(
                self.num_out_channels,
                self.num_out_channels,
                3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.num_out_channels),
            nn.ReLU(),
        )

    def __repr__(self):
        return "PrologueLarge"

    def forward(self, x):
        return self.net(x)


class Fused_IRB(nn.Module):
    def __init__(
        self,
        num_in_channels: int = 1,
        num_out_channels: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        expansion: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.drop_connect_rate = 0.0
        self.in_channels = num_in_channels
        self.out_channels = num_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expansion = expansion
        self.groups = groups

        self.body = nn.Sequential(
            # merge pw and dw
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels * self.expansion,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.in_channels * self.expansion, eps=0.001),
            nn.ReLU(),
            # pw
            nn.Conv2d(
                in_channels=self.in_channels * self.expansion,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels, eps=0.001),
        )
        if self.stride == 1 and self.in_channels == self.out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = None

    def drop_connect(self, inputs, training=False, drop_connect_rate=0.0):
        """Apply drop connect."""
        if not training:
            return inputs

        keep_prob = 1 - drop_connect_rate
        random_tensor = keep_prob + torch.rand(
            (inputs.size()[0], 1, 1, 1), dtype=inputs.dtype, device=inputs.device
        )
        random_tensor.floor_()  # binarize
        output = inputs.div(keep_prob) * random_tensor
        return output

    def forward(self, x):
        res = self.body(x)
        if self.shortcut is not None:
            if self.drop_connect_rate > 0 and self.training:
                res = self.drop_connect(res, self.training, self.drop_connect_rate)
            res = res + self.shortcut(x)
            return res
        else:
            return res

    def __repr__(self):
        name = "k{}_e{}_g{}_i{}_o{}_s{}".format(
            self.kernel_size,
            self.expansion,
            self.groups,
            self.in_channels,
            self.out_channels,
            self.stride,
        )
        return name


class Inverted_Residual_Block(nn.Module):
    def __init__(
        self, num_in_channels, num_out_channels, kernel_size, stride, expansion, groups
    ):
        super().__init__()
        self.drop_connect_rate = 0.0
        self.in_channels = num_in_channels
        self.out_channels = num_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expansion = expansion
        self.groups = groups

        self.body = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.in_channels * self.expansion,
                1,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(self.in_channels * self.expansion),
            nn.ReLU(),
            nn.Conv2d(
                self.in_channels * self.expansion,
                self.in_channels * self.expansion,
                kernel_size,
                padding=kernel_size // 2,
                stride=stride,
                groups=self.in_channels * self.expansion,
                bias=False,
            ),
            nn.BatchNorm2d(self.in_channels * self.expansion),
            nn.ReLU(),
            nn.Conv2d(
                self.in_channels * self.expansion,
                self.out_channels,
                1,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels),
        )
        if self.stride == 1 and self.in_channels == self.out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = None

    def drop_connect(self, inputs, training=False, drop_connect_rate=0.0):
        """Apply drop connect."""
        if not training:
            return inputs

        keep_prob = 1 - drop_connect_rate
        random_tensor = keep_prob + torch.rand(
            (inputs.size()[0], 1, 1, 1), dtype=inputs.dtype, device=inputs.device
        )
        random_tensor.floor_()  # binarize
        output = inputs.div(keep_prob) * random_tensor
        return output

    def forward(self, x):
        res = self.body(x)
        if self.shortcut is not None:
            if self.drop_connect_rate > 0 and self.training:
                res = self.drop_connect(res, self.training, self.drop_connect_rate)
            res = res + self.shortcut(x)
            return res
        else:
            return res

    def __repr__(self):
        name = "k{}_e{}_g{}_i{}_o{}_s{}".format(
            self.kernel_size,
            self.expansion,
            self.groups,
            self.in_channels,
            self.out_channels,
            self.stride,
        )
        return name


class EpilogueD(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_in_channels, 1152, 1, bias=False),
            nn.BatchNorm2d(1152),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1152, num_out_channels, 1, bias=False),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(num_out_channels, num_classes),
        )

    def __repr__(self):
        return "Epilogue"

    def forward(self, x):
        x = self.net(x)
        return x

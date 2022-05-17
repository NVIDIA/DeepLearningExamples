# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from maskrcnn_benchmark import _C

class _ROIAlign(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio, is_nhwc):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        ctx.is_nhwc = is_nhwc
        output = _C.roi_align_forward(
            input, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio, is_nhwc
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.roi_align_backward(
            grad_output,
            rois,
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            h,
            w,
            sampling_ratio,
            ctx.is_nhwc
        )
        return grad_input, None, None, None, None, None


roi_align = _ROIAlign.apply

class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio, is_nhwc):
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.nhwc = is_nhwc

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, rois):
        return roi_align(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio, self.nhwc
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ", is_nhwc=" + str(self.nhwc)
        tmpstr += ")"
        return tmpstr

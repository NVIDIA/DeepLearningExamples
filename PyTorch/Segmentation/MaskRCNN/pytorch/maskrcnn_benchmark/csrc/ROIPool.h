// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif


std::tuple<at::Tensor, at::Tensor> ROIPool_forward(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width) {
  if (input.is_cuda()) {
#ifdef WITH_CUDA
    return ROIPool_forward_cuda(input, rois, spatial_scale, pooled_height, pooled_width);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

at::Tensor ROIPool_backward(const at::Tensor& grad,
                                 const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const at::Tensor& argmax,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int batch_size,
                                 const int channels,
                                 const int height,
                                 const int width) {
  if (grad.is_cuda()) {
#ifdef WITH_CUDA
    return ROIPool_backward_cuda(grad, input, rois, argmax, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}




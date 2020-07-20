// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif


at::Tensor nms(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold) {

  if (dets.is_cuda()) {
#ifdef WITH_CUDA
    // TODO raise error if not compiled with CUDA
    if (dets.numel() == 0)
      return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
    auto b = at::cat({dets, scores.unsqueeze(1)}, 1);
    return nms_cuda(b, threshold);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  at::Tensor result = nms_cpu(dets, scores, threshold);
  return result;
}

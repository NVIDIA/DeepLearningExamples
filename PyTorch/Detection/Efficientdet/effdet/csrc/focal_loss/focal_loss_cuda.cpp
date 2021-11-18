// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> focal_loss_forward_cuda(
    const at::Tensor &cls_output, const at::Tensor &cls_targets_at_level,
    const at::Tensor &num_positives_sum, const int64_t num_real_classes,
    const float alpha, const float gamma, const float smoothing_factor);

at::Tensor focal_loss_backward_cuda(const at::Tensor &grad_output,
                                    const at::Tensor &partial_grad,
                                    const at::Tensor &num_positives_sum);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> focal_loss_forward(
    const at::Tensor &cls_output, const at::Tensor &cls_targets_at_level,
    const at::Tensor &num_positives_sum, const int64_t num_real_classes,
    const float alpha, const float gamma, const float smoothing_factor) {
  CHECK_INPUT(cls_output);
  CHECK_INPUT(cls_targets_at_level);
  CHECK_INPUT(num_positives_sum);

  return focal_loss_forward_cuda(cls_output, cls_targets_at_level,
                                 num_positives_sum, num_real_classes, alpha,
                                 gamma, smoothing_factor);
}

at::Tensor focal_loss_backward(const at::Tensor &grad_output,
                               const at::Tensor &partial_grad,
                               const at::Tensor &num_positives_sum) {
  CHECK_INPUT(grad_output);
  CHECK_INPUT(partial_grad);

  return focal_loss_backward_cuda(grad_output, partial_grad, num_positives_sum);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &focal_loss_forward,
        "Focal loss calculation forward (CUDA)");
  m.def("backward", &focal_loss_backward,
        "Focal loss calculation backward (CUDA)");
}

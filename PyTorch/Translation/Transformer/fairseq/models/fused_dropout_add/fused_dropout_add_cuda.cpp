// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

std::vector<at::Tensor> fused_dropout_add_cuda(
    const at::Tensor& input, 
    const at::Tensor& input_add, 
    double prob);

at::Tensor fused_dropout_add_backward_cuda(
    const at::Tensor& grad, 
    const at::Tensor& mask, 
    double scale);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> fused_dropout_add(
    const at::Tensor& input, 
    const at::Tensor& input_add, 
    double prob) {
  CHECK_CUDA(input);
  CHECK_CUDA(input_add);
  AT_ASSERTM(input.dim() == input_add.dim(), "Expected inputs to have matching number of dimensions");
  for (int i = 0; i < input.dim(); ++i) {
    AT_ASSERTM(input.size(i) == input_add.size(i), "Expected inputs have a matching dimension size");
    AT_ASSERTM(input.stride(i) == input_add.stride(i), "Expected inputs have a matching dimension stride");
  }
  return fused_dropout_add_cuda(input, input_add, prob);
}

at::Tensor fused_dropout_add_backward(
    const at::Tensor& grad, 
    const at::Tensor& mask, 
    double scale) {
  CHECK_CUDA(grad);
  CHECK_CUDA(mask);
  return fused_dropout_add_backward_cuda(grad, mask, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fused_dropout_add, "Fused Droput and Add forward (CUDA)");
  m.def("backward", &fused_dropout_add_backward, "Fused Dropout and Add backward (CUDA)");
}

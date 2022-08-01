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

at::Tensor strided_batched_gemm_cuda(
    float beta,
    at::Tensor in_result,
    float alpha,
    at::Tensor batch1,
    at::Tensor batch2);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor strided_batched_gemm(
    float beta,
    at::Tensor in_result,
    float alpha,
    at::Tensor batch1,
    at::Tensor batch2) {
  //CHECK_INPUT(in_result); 
  //CHECK_INPUT(batch1);
  //CHECK_INPUT(batch2);

  AT_ASSERTM(in_result.dim()   == 3, "expected 3D tensor");
  AT_ASSERTM(batch1.dim()      == 3, "expected 3D tensor");
  AT_ASSERTM(batch2.dim()      == 3, "expected 3D tensor");

  AT_ASSERTM(in_result.size(0) == batch1.size(0), "equal number of batches expected");
  AT_ASSERTM(in_result.size(0) == batch2.size(0), "equal number of batches expected");

  AT_ASSERTM(in_result.size(1) == batch1.size(1), "wrong matrix size");
  AT_ASSERTM(in_result.size(2) == batch2.size(2), "wrong matrix size");
  AT_ASSERTM(batch1.size(2)    == batch2.size(1), "wrong matrix size");

  AT_ASSERTM(batch1.dtype()    == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(batch2.dtype()    == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(in_result.dtype() == at::ScalarType::Half, "Only HALF is supported");
  
  return strided_batched_gemm_cuda(beta, in_result, alpha, batch1, batch2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("strided_batched_gemm", &strided_batched_gemm, "Special strided batched gemm.");
}


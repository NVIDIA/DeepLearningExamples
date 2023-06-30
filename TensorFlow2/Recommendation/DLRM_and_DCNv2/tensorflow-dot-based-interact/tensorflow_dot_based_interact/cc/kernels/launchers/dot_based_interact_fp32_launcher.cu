// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
//


#ifndef FP32_LAUNCHER_CU
#define FP32_LAUNCHER_CU

#include "../cuda_kernels/dot_based_interact_fp32.cu"

inline void dotBasedInteractFP32Fwd(const void *input,
                                         const void *bottom_mlp_output,
                                         void *output,
                                         uint batch_size,
                                         uint num_rows,
                                         uint num_cols,
                                         cudaStream_t stream) {
  const uint kNumThreads = 128;
  uint num_blocks = batch_size;

  // Output
  uint interaction_output_size = (num_rows * (num_rows - 1)) >> 1;
  uint output_size = ((interaction_output_size+num_cols-1)/8 + 1)*8; //round up to multiple of 8

  // Input
  uint input_size = num_rows * num_cols;

  uint shared_mem_size_elems = input_size;
  uint shared_mem_size_bytes = shared_mem_size_elems << 2;  // F32 Kernel

  bool float4_predicate = !((num_cols & 3) || (output_size & 3));

  if (float4_predicate) {
    dotBasedInteractF32FwdKernel<kNumThreads>
        <<<num_blocks, kNumThreads, shared_mem_size_bytes, stream>>>((const float *)input,
                                                                     (float *)output,
                                                                     batch_size,
                                                                     num_rows,
                                                                     num_cols,
                                                                     input_size,
                                                                     output_size,
                                                                     interaction_output_size);
  } else {
    dotBasedInteractF32FwdKernelNonAligned<kNumThreads>
        <<<num_blocks, kNumThreads, shared_mem_size_bytes, stream>>>((const float *)input,
                                                                     (float *)output,
                                                                     batch_size,
                                                                     num_rows,
                                                                     num_cols,
                                                                     input_size,
                                                                     output_size,
                                                                     interaction_output_size);
  }
}

inline void dotBasedInteractFP32Bwd(const void *input,
                                         const void *upstream_grad,
                                         void *grad,
                                         void *bottom_mlp_grad,
                                         uint batch_size,
                                         uint num_rows,
                                         uint num_cols,
                                         cudaStream_t stream) {
  const uint kNumThreads = 128;

  uint num_blocks = batch_size;

  uint input_size = num_rows * num_cols;

  // 1D ugrad size
  uint interaction_ugrad_size = num_rows * (num_rows - 1) >> 1; //this IS supposed to be without padding

  // this has to be the same padding that we applied in forward

  uint unpadded_ugrad_size = num_cols + interaction_ugrad_size;
  // this has to be the same padding that we applied in forward
  uint padded_ugrad_size = ((unpadded_ugrad_size-1)/8 + 1)*8;  //round up to multiple of 8

  // input space + upstream grad space
  // We copy the whole input plus just the unpadded interaction part of the upstream grad
  uint smem_size_elems = input_size + interaction_ugrad_size;
  uint smem_size_bytes = smem_size_elems << 2;  // F32 Kernel

  // we use the fact that padded_ugrad_size is always divisible by 4 - we just made it.
  bool float4_predicate = !(num_cols & 3);
  if (float4_predicate) {
    dotBasedInteractF32BwdKernel<kNumThreads>
        <<<num_blocks, kNumThreads, smem_size_bytes, stream>>>((const float *)input,
                                                               (const float *)upstream_grad,
                                                               (float *)grad,
                                                               (float *)bottom_mlp_grad,
                                                               batch_size,
                                                               num_rows,
                                                               num_cols,
                                                               input_size,
                                                               padded_ugrad_size,
                                                               interaction_ugrad_size);
  } else {
    dotBasedInteractF32BwdKernelNonAligned<kNumThreads>
        <<<num_blocks, kNumThreads, smem_size_bytes, stream>>>((const float *)input,
                                                               (const float *)upstream_grad,
                                                               (float *)grad,
                                                               (float *)bottom_mlp_grad,
                                                               batch_size,
                                                               num_rows,
                                                               num_cols,
                                                               input_size,
                                                               padded_ugrad_size,
                                                               interaction_ugrad_size);
  }
}

#endif /* FP32_LAUNCHER_CU */

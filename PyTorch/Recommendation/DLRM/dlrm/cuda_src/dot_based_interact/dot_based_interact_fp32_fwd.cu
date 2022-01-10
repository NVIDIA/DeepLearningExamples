#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <mma.h>
#include <cuda_fp16.hpp>

#include <math.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "shared_utils.cuh"

using namespace nvcuda;

template <uint THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void dotBasedInteractF32FwdKernelNonAligned(const float *__restrict input,
                                                float *__restrict output,
                                                uint batch_size,
                                                uint num_rows,
                                                uint num_cols,
                                                uint input_size,
                                                uint output_size,
                                                uint interaction_output_size) {
  extern __shared__ float smem_f32_fwd[];
  float *smem_in = &smem_f32_fwd[0];

  uint input_batch_offset = blockIdx.x * input_size;
  const float *gmem_in = &input[input_batch_offset];

  // The layout of each output row is bottom_mlp | interactions | padding
  uint output_batch_offset = blockIdx.x * output_size;
  float *gmem_out_bottom_mlp = &output[output_batch_offset];
  float *gmem_out_interaction = &output[output_batch_offset + num_cols];

  // Load the input - one sample per block
  for (uint idx = threadIdx.x; idx < input_size; idx += blockDim.x) {
    smem_in[idx] = gmem_in[idx];
  }
  __syncthreads();

  // Copy bottom MLP output to output
  for (uint idx = threadIdx.x; idx < num_cols; idx += blockDim.x) {
    gmem_out_bottom_mlp[idx] = smem_in[idx];
  }

  for (uint idx = threadIdx.x; idx < (interaction_output_size); idx += blockDim.x) {
    uint elems_per_row = 1;
    uint index = idx;
    while (index >= elems_per_row) {
      index -= elems_per_row;
      elems_per_row++;
    }
    uint target_row = elems_per_row;
    uint target_col = index;

    float sum = 0;
    for (uint i = 0; i < num_cols; i++) {
      float tmp1 = smem_in[target_row * num_cols + i];
      float tmp2 = smem_in[target_col * num_cols + i];
      sum = fmaf(tmp1, tmp2, sum);
    }

    gmem_out_interaction[idx] = sum;
  }

   // Zero out the padding
  uint zeroout_index = num_cols + interaction_output_size + threadIdx.x;
  if(zeroout_index < output_size){
  gmem_out_bottom_mlp[zeroout_index] = 0;
  }
}

template <uint THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void dotBasedInteractF32FwdKernel(const float *__restrict input,
                                                                                 float *__restrict output,
                                                                                 uint batch_size,
                                                                                 uint num_rows,
                                                                                 uint num_cols,
                                                                                 uint input_size,
                                                                                 uint output_size,
                                                                                 uint interaction_output_size) {
  extern __shared__ float smem_f32_fwd[];
  float *smem_in = &smem_f32_fwd[0];
  //launch one block per sample in batch

  uint input_batch_offset = blockIdx.x * input_size;
  const float *gmem_in = &input[input_batch_offset];

  // The layout of each output row is bottom_mlp | interactions | padding
  uint output_batch_offset = blockIdx.x * output_size;
  float *gmem_out_bottom_mlp = &output[output_batch_offset];
  float *gmem_out_interaction = &output[output_batch_offset + num_cols];

  // Load the input - one sample per block
  uint input_size_float4 = input_size >> 2;
  for (uint idx = threadIdx.x; idx < input_size_float4; idx += blockDim.x) {
    ((float4 *)smem_in)[idx] = ((float4 *)gmem_in)[idx];
  }
  __syncthreads();

  // Copy bottom MLP output to output
  uint btm_mlp_out_size_float4 = num_cols >> 2;
  for (uint idx = threadIdx.x; idx < btm_mlp_out_size_float4; idx += blockDim.x) {
    ((float4 *)gmem_out_bottom_mlp)[idx] = ((float4 *)smem_in)[idx];
  }

  for (uint idx = threadIdx.x; idx < (interaction_output_size); idx += blockDim.x) {
    uint elems_per_row = 1;
    uint index = idx;
    while (index >= elems_per_row) {
      index -= elems_per_row;
      elems_per_row++;
    }
    uint target_row = elems_per_row;
    uint target_col = index;

    float4 sum;
    sum.x = 0;
    sum.y = 0;
    sum.z = 0;
    sum.w = 0;
    uint num_cols_float4 = num_cols >> 2;
    for (uint i = 0; i < num_cols_float4; i++) {
      float4 tmp1 = ((float4 *)smem_in)[target_row * num_cols_float4 + i];
      float4 tmp2 = ((float4 *)smem_in)[target_col * num_cols_float4 + i];
      sum.x = fmaf(tmp1.x, tmp2.x, sum.x);
      sum.y = fmaf(tmp1.y, tmp2.y, sum.y);
      sum.z = fmaf(tmp1.z, tmp2.z, sum.z);
      sum.w = fmaf(tmp1.w, tmp2.w, sum.w);
    }

    gmem_out_interaction[idx] = sum.x + sum.y + sum.z + sum.w;
  }

  // Zero out the padding
  uint zeroout_index = num_cols + interaction_output_size + threadIdx.x;
  if(zeroout_index < output_size){
  gmem_out_bottom_mlp[zeroout_index] = 0;
  }
}

inline void dotBasedInteractF32Fwd(const void *input,
                                   const void *bottom_mlp_output,
                                   const void *output,
                                   uint batch_size,
                                   uint num_rows,
                                   uint num_cols) {
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
        <<<num_blocks, kNumThreads, shared_mem_size_bytes, at::cuda::getCurrentCUDAStream()>>>((const float *)input,
                                                             (float *)output,
                                                             batch_size,
                                                             num_rows,
                                                             num_cols,
                                                             input_size,
                                                             output_size,
                                                             interaction_output_size);
  } else {
    dotBasedInteractF32FwdKernelNonAligned<kNumThreads>
        <<<num_blocks, kNumThreads, shared_mem_size_bytes,
           at::cuda::getCurrentCUDAStream()>>>((const float *)input,
                                              (float *)output,
                                              batch_size,
                                              num_rows,
                                              num_cols,
                                              input_size,
                                              output_size,
                                              interaction_output_size);
  }
}
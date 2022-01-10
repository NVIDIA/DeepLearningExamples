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
    void dotBasedInteractF32BwdKernelNonAligned(const float *__restrict input,
                                                const float *__restrict upstream_grad,
                                                float *__restrict grad,
                                                float *__restrict bottom_mlp_grad,
                                                uint batch_size,
                                                uint num_rows,
                                                uint num_cols,
                                                uint input_size,
                                                uint padded_ugrad_size,
                                                uint interaction_ugrad_size) {
  extern __shared__ float smem_f32_bwd[];
  float *smem_in = &smem_f32_bwd[0];
  float *smem_interaction_ugrad = &smem_f32_bwd[input_size]; //skip over the part where we copy in the input

  // Input
  uint input_batch_offset = blockIdx.x * input_size;
  const float *gmem_in = &input[input_batch_offset];

  // Gradient
  const uint &grad_batch_offset = input_batch_offset;
  float *gmem_mlp_grad = &bottom_mlp_grad[blockIdx.x * num_cols]; //where the bottom mlp grad of our sample will land
  float *gmem_interaction_grad = &grad[grad_batch_offset]; //where the interaction grads of our sample will land

  // Upstream Gradient
  uint upstream_grad_batch_offset = blockIdx.x * padded_ugrad_size;
  const float *gmem_mlp_ugrad = &upstream_grad[upstream_grad_batch_offset];
  // fwd output contained mlp at the start, so the gradient has mlp grad at the start
  const float *gmem_interaction_ugrad = &upstream_grad[upstream_grad_batch_offset + num_cols];

  // input -> shared memory
  for (uint idx = threadIdx.x; idx < input_size; idx += blockDim.x) {
    smem_in[idx] = gmem_in[idx];
  }

  // Interaction Upstream Grad -> Shared Memory
  for (uint idx = threadIdx.x; idx < interaction_ugrad_size; idx += blockDim.x) {
    smem_interaction_ugrad[idx] = gmem_interaction_ugrad[idx];
  }
  __syncthreads();

  // Copy the upstream gradient w.r.t to mlp to it's corresponding memory location.
  for (uint idx = threadIdx.x; idx < num_cols; idx += blockDim.x) {
    gmem_mlp_grad[idx] = gmem_mlp_ugrad[idx];
  }

  for (uint idx = threadIdx.x; idx < num_cols; idx += blockDim.x) {
    size_t grad_idx = idx;
    // Calculate a single column (1...128) of the output
    for (uint row_idx = 0; row_idx < num_rows; row_idx++) {
      // Pick a row: now we calculating a single value of the gradient
      float sum = 0;
      // Jump to our row in (flattened) triangular matrix of  upstream gradients
      size_t upstream_grad_offset = (row_idx * (row_idx - 1)) >> 1;
      // Iterate over all the interactions we took part in
      // Sum upstream gradient for that interaction multiplied with the right element of the other vector in the interaction
      // We need to do this in two passes because we only keep the triangular part of the matrix, so the row "bends"
      for (int k = 0; k < row_idx; k++) {
        sum = fmaf(smem_in[k * num_cols + idx], smem_interaction_ugrad[upstream_grad_offset + k], sum);
      }
      for (int k = row_idx + 1; k < num_rows; k++) {
        upstream_grad_offset = (k * (k - 1)) >> 1;  // TODO: this can become a sum
        sum = fmaf(smem_in[k * num_cols + idx], smem_interaction_ugrad[upstream_grad_offset + row_idx], sum);
      }
      gmem_interaction_grad[grad_idx] = sum;
      grad_idx += num_cols;
    }
  }
}

template <uint THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void dotBasedInteractF32BwdKernel(const float *__restrict input,
                                                                                 const float *__restrict upstream_grad,
                                                                                 float *__restrict grad,
                                                                                 float *__restrict bottom_mlp_grad,
                                                                                 uint batch_size,
                                                                                 uint num_rows,
                                                                                 uint num_cols,
                                                                                 uint input_size,
                                                                                 uint padded_ugrad_size,
                                                                                 uint interaction_ugrad_size) {

  // This kernel assumes that:
  // input_size is divisible by 4
  // num_cols is divisible by 4

  extern __shared__ float smem_f32_bwd[];
  float *smem_in = &smem_f32_bwd[0];
  float *smem_interaction_ugrad = &smem_f32_bwd[input_size];

  // Input
  uint input_batch_offset = blockIdx.x * input_size;
  const float *gmem_in = &input[input_batch_offset];

  // Gradient
  const uint &grad_batch_offset = input_batch_offset;
  float *gmem_mlp_grad = &bottom_mlp_grad[blockIdx.x * num_cols];
  float *gmem_interaction_grad = &grad[grad_batch_offset];

  // Upstream Gradient
  uint upstream_grad_batch_offset = blockIdx.x * padded_ugrad_size;
  const float *gmem_mlp_ugrad = &upstream_grad[upstream_grad_batch_offset];
  const float *gmem_interaction_ugrad = &upstream_grad[upstream_grad_batch_offset + num_cols];

  // input -> shared memory
  uint input_size_float4 = input_size >> 2;
  for (uint idx = threadIdx.x; idx < input_size_float4; idx += blockDim.x) {
    ((float4 *)smem_in)[idx] = ((float4 *)gmem_in)[idx];
  }


  // Interaction Upstream Grad -> Shared Memory
  uint upstream_grad_size_float4 = interaction_ugrad_size >> 2;
  for (uint idx = threadIdx.x; idx < upstream_grad_size_float4; idx += blockDim.x) {
    ((float4 *)smem_interaction_ugrad)[idx] = ((float4 *)gmem_interaction_ugrad)[idx];
  }

   // This may seem like it may never be activated, but it will
   // interaction_ugrad_size is the unpadded size, so it will probably not align to 4
   // This loop copies the part that is left over from the vectorized copy above
  uint vectorized_load_offset = (upstream_grad_size_float4 << 2);
  for (uint idx = vectorized_load_offset + threadIdx.x; idx < interaction_ugrad_size; idx += blockDim.x) {
    smem_interaction_ugrad[idx] = gmem_interaction_ugrad[idx];
  }
  __syncthreads();

  // Copy the upstream gradient w.r.t to mlp to it's corresponding memory location.
  for (uint idx = threadIdx.x; idx < (num_cols >> 2); idx += blockDim.x) {
    ((float4 *)gmem_mlp_grad)[idx] = ((float4 *)gmem_mlp_ugrad)[idx];
  }

  for (uint idx = threadIdx.x; idx < num_cols; idx += blockDim.x) {
    size_t grad_idx = idx;
    for (uint row_idx = 0; row_idx < num_rows; row_idx++) {
      float sum = 0;
      size_t upstream_grad_offset = (row_idx * (row_idx - 1)) >> 1;
      for (int k = 0; k < row_idx; k++) {
        sum = fmaf(smem_in[k * num_cols + idx], smem_interaction_ugrad[upstream_grad_offset + k], sum);
      }
      for (int k = row_idx + 1; k < num_rows; k++) {
        upstream_grad_offset = (k * (k - 1)) >> 1;  // TODO: this can become a sum
        sum = fmaf(smem_in[k * num_cols + idx], smem_interaction_ugrad[upstream_grad_offset + row_idx], sum);
      }
      gmem_interaction_grad[grad_idx] = sum;
      grad_idx += num_cols;
    }
  }
}

inline void dotBasedInteractF32Bwd(const void *input,
                                   const void *upstream_grad,
                                   void *grad,
                                   void *bottom_mlp_grad,
                                   uint batch_size,
                                   uint num_rows,
                                   uint num_cols) {
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
        <<<num_blocks, kNumThreads, smem_size_bytes,
           at::cuda::getCurrentCUDAStream()>>>((const float *)input,
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
        <<<num_blocks, kNumThreads, smem_size_bytes,
           at::cuda::getCurrentCUDAStream()>>>((const float *)input,
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


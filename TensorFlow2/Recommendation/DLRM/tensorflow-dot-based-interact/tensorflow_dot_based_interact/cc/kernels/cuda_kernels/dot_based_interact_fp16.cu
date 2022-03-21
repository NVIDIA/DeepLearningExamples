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


#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <mma.h>
#include <cuda_fp16.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "dot_based_interact_shared_utils.cuh"

struct __align__(8) half4 {
  half2 vals[2];
};

using namespace nvcuda;

template <uint WARPS_PER_BLOCK,
          uint THREADBLOCK_SIZE,
          uint M_BLOCKS,
          uint K_BLOCKS,
          uint SMEM_STRIDE,
          uint SMEM_STRIDE_ACC,
          uint WARP_SIZE,
          uint WARP_SIZE_LOG_2,
          uint TILE_DIM,
          uint TILE_DIM_LOG_2,
          bool IS_ALIGNED>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void dotBasedInteractFwdKernelFP16(const __half *__restrict input,
                                                                                        __half *__restrict output,
                                                                                        uint batch_size,
                                                                                        uint num_rows,
                                                                                        uint num_cols,
                                                                                        uint num_rows_after_padding, //nearest greater multiple of 16
                                                                                        uint num_cols_after_padding, //nearest greater multiple of 16
                                                                                        uint smem_elems_per_warp, //size of shmem allocation for each warp
                                                                                        uint smem_rows_per_warp, //how many rows are multiplied in a single pass
                                                                                        uint output_size, //this is with padding
                                                                                        uint num_row_steps,
                                                                                        uint num_col_steps,
                                                                                        uint padding_size) {
                                                                                      
  //right now, there are two solutions implemented for padding out the matrices to have dims that are multiples of 16
  //first, they are both padded out to be multiples: we can just take it and continue
  //second, in the multiplication we inset the second tile so that it overlaps with the first one, then skip the doubled up part when unwinding the resulting matrix
  
  uint warp_id = (threadIdx.x >> WARP_SIZE_LOG_2);
  int sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  if (sample_id >= batch_size) {
    return;
  }
  int lane_id = threadIdx.x & (WARP_SIZE - 1);

  extern __shared__ half shmem_dynamic[];
  half *shmem = shmem_dynamic + (warp_id * smem_elems_per_warp);

  const half *sample_input = input + num_rows * num_cols * sample_id;

  // Input -> shmem

  if (IS_ALIGNED) { //TODO use contexpr if when makefile is upgraded to C++17
    for(uint idx = lane_id; idx < (num_cols >> 2); idx+=WARP_SIZE){
      for (int i = 0; i < num_rows; ++i, sample_input += num_cols) {
        ((float2 *)(shmem + i * SMEM_STRIDE))[idx] = ((float2 *)sample_input)[idx];//todo will this not read OOB for last sample?
      }
    }
  }
  else {
    for (uint i = 0; i < num_rows; ++i, sample_input += num_cols) {
      for (uint idx = lane_id; idx < num_cols; idx += WARP_SIZE) {
        (shmem + i * SMEM_STRIDE)[idx] = sample_input[idx];
      }
    }
  }

  // pad out each row to have a multiple of 16 columns

  uint idx = lane_id + num_cols;
  if (idx < num_cols_after_padding) {
    for (int i = 0; i < num_rows; ++i) {
      (shmem + i * SMEM_STRIDE)[idx] = __float2half(0);
    }
  }

  // pad out with zero rows until we have a multiple of 16 rows

  half4 zeros;
  zeros.vals[0].x = __float2half(0);
  zeros.vals[0].y = __float2half(0);
  zeros.vals[1].x = __float2half(0);
  zeros.vals[1].y = __float2half(0);
  for (uint idx = lane_id; idx < (num_cols_after_padding >> 2); idx += WARP_SIZE) {
    for (int i = num_rows; i < num_rows_after_padding; i++) {
      ((half4 *)(shmem + i * SMEM_STRIDE))[idx] = zeros;
    }
  }
  __syncwarp();

  // copy bottom mlp output into the output tensor
  half *gmem_output = output + output_size * sample_id;

  if (IS_ALIGNED) {
    for(uint idx = lane_id; idx < (num_cols >> 2); idx+=WARP_SIZE){
        ((float2 *)gmem_output)[idx] = ((float2 *)shmem)[idx];
    }
  }
  else {
    for (uint idx = lane_id; idx < num_cols; idx += WARP_SIZE) {
      gmem_output[idx] = shmem[idx];
    }
  }

  // perform tiled multiplication using tensor cores

  wmma::fragment<wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float> acc[M_BLOCKS][M_BLOCKS];

  for (int i = 0; i < M_BLOCKS; i++) {
    for (int j = 0; j < M_BLOCKS; j++) {
      wmma::fill_fragment(acc[i][j], 0);
    }
  }

  for (int k_step = 0; k_step < num_col_steps; k_step++) {
    wmma::fragment<wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, half, wmma::row_major> a[M_BLOCKS];
    wmma::fragment<wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, half, wmma::col_major> b[M_BLOCKS];
    for (int j = 0; j < M_BLOCKS; j++) {
      int base_row = (j < M_BLOCKS - 1) ? j * 16 : smem_rows_per_warp - 16;
      const half *tile_ptr = shmem + (base_row * SMEM_STRIDE + k_step * 16);
      wmma::load_matrix_sync(a[j], tile_ptr, SMEM_STRIDE);
      wmma::load_matrix_sync(b[j], tile_ptr, SMEM_STRIDE);
    }
    for (int i = 0; i < M_BLOCKS; i++) {
      for (int j = 0; j < M_BLOCKS; j++) {
        wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]); //todo we can get a speedup by skipping one of the multiplications - they are redundant
      }
    }
  }
  // dump accumulators back into shared memory
  float *shmem_store = reinterpret_cast<float *>(shmem);
  for (int i = 0; i < M_BLOCKS; i++) {
    for (int j = 0; j < M_BLOCKS; j++) {
      float *tile_ptr = shmem_store + (i * 16 * SMEM_STRIDE_ACC + j * 16);
      wmma::store_matrix_sync(tile_ptr, acc[i][j], SMEM_STRIDE_ACC, wmma::mem_row_major);
    }
  }

  // unwind accumulators into the output memory
  half *gmem_interact_output = gmem_output + num_cols;
  int lastRowBlockOffset = M_BLOCKS * 16 - smem_rows_per_warp;
  int srcLine = 0;
  for (int i = 0; i < num_rows; ++i, ++srcLine) {
    if (i == ((M_BLOCKS - 1) * 16)) {
      srcLine += lastRowBlockOffset;
    }
    for(uint idx = lane_id; idx < i; idx+=WARP_SIZE){
      uint offset = (i * (i - 1)) >> 1; //sum of previously unwound rows
      gmem_interact_output[offset + idx] = __float2half(shmem_store[srcLine * SMEM_STRIDE_ACC + idx]);
    }
  }
  // Add padding to the output vectors
  if (lane_id < padding_size) {
    gmem_output[output_size - lane_id - 1] = __float2half(0);
  }
}

template <uint WARPS_PER_BLOCK,
          uint THREADBLOCK_SIZE,
          uint ROW_TILES_PER_STEP,
          uint COL_TILES_PER_STEP,
          uint WARP_SIZE,
          uint WARP_SIZE_LOG_2,
          uint TILE_DIM,
          uint TILE_DIM_LOG_2,
          bool IS_ALIGNED>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void dotBasedInteractBwdKernelFP16(const __half *__restrict input,
                                             const __half *__restrict upstream_grad,
                                             half __restrict *grad,
                                             half __restrict *bottom_mlp_grad,
                                             uint batch_size,
                                             uint num_rows,
                                             uint num_cols,
                                             uint num_rows_after_padding,
                                             uint num_cols_after_padding,
                                             uint sample_size,
                                             uint interaction_ugrad_size,
                                             uint interaction_ugrad_size_with_padding,
                                             uint interaction_ugrad_2D_size_elems,
                                             uint interaction_ugrad_2D_stride,
                                             uint input_size_elems,
                                             uint input_stride,
                                             uint num_row_steps,
                                             uint num_col_steps,
                                             uint row_tiles_per_step,
                                             uint shared_mem_per_warp_size_byte) {
  extern __shared__ half shared_mem[];
  uint warp_id = (threadIdx.x >> WARP_SIZE_LOG_2);
  uint sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  if (sample_id >= batch_size) {
    return;
  }
  uint lane_id = threadIdx.x & (WARP_SIZE - 1);
  // ">> 1" to convert to half pointer
  uint smem_warp_offset = warp_id * (shared_mem_per_warp_size_byte >> 1);

  half *smem_in = &shared_mem[smem_warp_offset];
  half *smem_temp = &shared_mem[smem_warp_offset + input_size_elems];
  float *smem_out = reinterpret_cast<float *>(smem_temp);

  // Global memory pointers for the current sample
  // Input
  uint gmem_input_sample_offset = sample_id * sample_size;
  const half *gmem_input = &input[gmem_input_sample_offset];

  // Interaction Gradient
  const uint &gmem_grad_sample_offset = gmem_input_sample_offset;
  half *gmem_grad = &grad[gmem_grad_sample_offset];

  // Bottom MLP gradient
  half *gmem_mlp_grad = &bottom_mlp_grad[sample_id * num_cols];

  // Upstream gradient vector
  uint gmem_ugrad_sample_offset = sample_id * (num_cols + interaction_ugrad_size_with_padding);
  const half *gmem_ugrad = &upstream_grad[gmem_ugrad_sample_offset];

  // Upstream gradient vector for interactions
  const half *gmem_ugrad_interactions = &gmem_ugrad[num_cols];

  // upstream grad -> shared memory (place in input section temporarily)
  if(IS_ALIGNED){
    #pragma unroll
    for (uint idx = lane_id; idx < (interaction_ugrad_size >> 3); idx += WARP_SIZE) {
      ((float4 *)smem_in)[idx] = ((float4 *)gmem_ugrad_interactions)[idx];
    }
    uint offset = (interaction_ugrad_size >> 3) << 3;
    for (uint idx = lane_id + offset; idx < interaction_ugrad_size; idx += WARP_SIZE) {
      smem_in[idx] = gmem_ugrad_interactions[idx];
    }
  }
  else {
    #pragma unroll
    for (uint idx = lane_id; idx < interaction_ugrad_size; idx += WARP_SIZE) {
      smem_in[idx] = gmem_ugrad_interactions[idx];
    }
  }
  __syncwarp();
  // Form the 2D ugrad matrix.
  for(uint idx = lane_id; idx < num_rows_after_padding; idx+=WARP_SIZE){
    uint ugrad_flat_index = ((idx * (idx - 1)) >> 1);
    uint ugrad_offset_1 = idx * interaction_ugrad_2D_stride;
    for (uint row = 0; row < num_rows; row++) {
      half ugrad_val = __float2half(0.0f);
      if (row < idx && idx < num_rows) {
        ugrad_val = smem_in[ugrad_flat_index + row];
        smem_temp[ugrad_offset_1 + row] = ugrad_val;
      }
      if (row <= idx && idx < num_rows_after_padding) {
        smem_temp[row * interaction_ugrad_2D_stride + idx] = ugrad_val;
      }
    }
    for (uint row = num_rows; row < num_rows_after_padding; row++) {
      smem_temp[row * interaction_ugrad_2D_stride + idx] = __float2half(0.0f);
    }
  }
  __syncwarp();

  // Input -> Shared Memory
  if (IS_ALIGNED) {
    for (uint idx = lane_id; idx < (num_cols >> 2); idx+=WARP_SIZE) {
      for (uint row = 0; row < num_rows; row++) {
        half *smem_row_ptr = &smem_in[row * input_stride];
        const half *gmem_row_ptr = &gmem_input[row * num_cols];
        ((float2 *)smem_row_ptr)[idx] = ((float2 *)gmem_row_ptr)[idx];
      }
    }

    uint idx = lane_id + num_cols;
    if (idx < num_cols_after_padding) {
      for (uint row = 0; row < num_rows; row++) {
        half *smem_row_ptr = &smem_in[row * input_stride];
        smem_row_ptr[idx] = __float2half(0);
      }
    }

    half4 zeros;
    zeros.vals[0].x = __float2half(0);
    zeros.vals[0].y = __float2half(0);
    zeros.vals[1].x = __float2half(0);
    zeros.vals[1].y = __float2half(0);
    for (uint idx = lane_id; idx < (num_cols_after_padding >> 2); idx +=WARP_SIZE){
  #pragma unroll 2
      for (uint row = num_rows; row < num_rows_after_padding; row++) {
        half *smem_row_ptr = &smem_in[row * input_stride];
        ((half4 *)smem_row_ptr)[idx] = zeros;
      }
    }
  }
  else {
  //UNALIGNED
    for (uint row = 0; row < num_rows; row++) {
      half *smem_row_ptr = &smem_in[row * input_stride];
      const half *gmem_row_ptr = &gmem_input[row * num_cols];
      for (uint idx = lane_id; idx < num_cols; idx += WARP_SIZE) {
        smem_row_ptr[idx] = gmem_row_ptr[idx];
      }
      uint idx = lane_id + num_cols;
      if (idx < num_cols_after_padding) {
        smem_row_ptr[idx] = __float2half(0);
      }
    }

  #pragma unroll 2
    for (uint row = num_rows; row < num_rows_after_padding; row++) {
      half *smem_row_ptr = &smem_in[row * input_stride];
      for (uint idx = lane_id; idx < num_cols_after_padding; idx += WARP_SIZE) {
        smem_row_ptr[idx] = __float2half(0);
      }
    }
  }
  __syncwarp();

  wmma::fragment<wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, half, wmma::row_major> a[ROW_TILES_PER_STEP]
                                                                                       [ROW_TILES_PER_STEP];
  for (uint i = 0; i < ROW_TILES_PER_STEP; i++) {
    for (uint j = 0; j < ROW_TILES_PER_STEP; j++) {
      const half *tile_ptr = smem_temp + ((i * interaction_ugrad_2D_stride + j) << TILE_DIM_LOG_2);
      wmma::load_matrix_sync(a[i][j], tile_ptr, interaction_ugrad_2D_stride);
    }
  }

  wmma::fragment<wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float> acc[ROW_TILES_PER_STEP];
  wmma::fragment<wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, half, wmma::row_major> b[ROW_TILES_PER_STEP];
  for (int col_step = 0; col_step < num_col_steps; col_step++) {
    for (uint i = 0; i < ROW_TILES_PER_STEP; i++) {
      const half *tile_ptr = smem_in + ((i * input_stride + col_step) << TILE_DIM_LOG_2);
      wmma::fill_fragment(acc[i], 0);
      wmma::load_matrix_sync(b[i], tile_ptr, input_stride);
    }
    for (uint i = 0; i < ROW_TILES_PER_STEP; i++) {
      for (uint j = 0; j < ROW_TILES_PER_STEP; j++) {
        wmma::mma_sync(acc[i], a[i][j], b[j], acc[i]);
      }
    }
    for (uint i = 0; i < ROW_TILES_PER_STEP; i++) {
      float *tile_ptr = smem_out + i * TILE_DIM * TILE_DIM;
      wmma::store_matrix_sync(tile_ptr, acc[i], TILE_DIM, wmma::mem_row_major);
    }
    __syncwarp();
    // gmem_grad_col = col_step * tile_dim  + lane_id
    // this is correct because lane_id spans 0..tile_dim-1
    uint gmem_grad_col = (col_step << TILE_DIM_LOG_2) + lane_id;
    // only copy over result if we are not in the padding
    if (gmem_grad_col < num_cols) {
      for (uint i = 0; i < num_rows; i++) {
        gmem_grad[i * num_cols + gmem_grad_col] = __float2half(smem_out[(i << TILE_DIM_LOG_2) + lane_id]);
      }
    }
  }
  if (IS_ALIGNED) {
    for(uint idx = lane_id; idx < (num_cols >> 2); idx+=WARP_SIZE){
        ((float2 *)gmem_mlp_grad)[idx] = ((float2 *)gmem_ugrad)[idx];
    }
  }
  else {
    for (uint idx = lane_id; idx < num_cols; idx += WARP_SIZE) {
      gmem_mlp_grad[idx] = gmem_ugrad[idx];
    }
  }
}
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

#include <math.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "dot_based_interact_shared_utils.cuh"

using namespace nvcuda;

template <uint WARPS_PER_BLOCK,
          uint THREADBLOCK_SIZE,
          uint WARP_SIZE,
          uint WARP_SIZE_LOG_2,
          uint TILE_LENGTH,
          uint TILE_LENGTH_LOG_2,
          uint TILE_WIDTH,
          uint TILE_WIDTH_LOG_2,
          uint ROW_TILES_PER_STEP>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void dotBasedInteractTF32FwdKernel(const float *__restrict input,
                                                                                  float *__restrict output,
                                                                                  uint batch_size,
                                                                                  uint num_rows,
                                                                                  uint num_cols,
                                                                                  uint num_rows_after_padding,
                                                                                  uint num_cols_after_padding,
                                                                                  uint smem_elems_per_warp,
                                                                                  uint output_size,
                                                                                  uint num_row_steps,
                                                                                  uint num_col_steps,
                                                                                  uint smem_stride,
                                                                                  uint smem_stride_acc,
                                                                                  uint padding_size) {
  // The only support sizes for TF32.
  const uint kWmmaM = 16;
  const uint kWmmaN = 16;
  const uint kWmmaK = 8;

  uint warp_id = threadIdx.x >> WARP_SIZE_LOG_2;
  uint sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  if (sample_id >= batch_size) {
    return;
  }
  int lane_id = threadIdx.x & (WARP_SIZE - 1);

  extern __shared__ float shmem_dynamic_float[];
  float *shmem = shmem_dynamic_float + (warp_id * smem_elems_per_warp);

  const float *gmem_input = input + num_rows * num_cols * sample_id;
  if (lane_id < (num_cols >> 2)) {
    for (int i = 0; i < num_rows; ++i, gmem_input += num_cols) {
      float4 tmp = ((float4 *)gmem_input)[lane_id];
      tmp.x = wmma::__float_to_tf32(tmp.x);
      tmp.y = wmma::__float_to_tf32(tmp.y);
      tmp.z = wmma::__float_to_tf32(tmp.z);
      tmp.w = wmma::__float_to_tf32(tmp.w);
      ((float4 *)(shmem + i * smem_stride))[lane_id] = tmp;
    }
  }

  float zero = wmma::__float_to_tf32(0.0f);
  float4 zero4;
  zero4.x = zero;
  zero4.y = zero;
  zero4.z = zero;
  zero4.w = zero;

  uint idx = lane_id + num_cols;
  if (idx < num_cols_after_padding) {
    for (uint i = 0; i < num_rows; ++i) {
      (shmem + i * smem_stride)[idx] = zero;
    }
  }

  if (lane_id < (num_cols_after_padding >> 2)) {
    for (int i = num_rows; i < num_rows_after_padding; i++) {
      ((float4 *)(shmem + i * smem_stride))[lane_id] = zero4;
    }
  }
  __syncwarp();
  // TODO: MTMD - Copy directly without using shared memory
  float *gmem_output = output + output_size * sample_id;
  if (lane_id < (num_cols >> 2)) {
    ((float4 *)gmem_output)[lane_id] = ((float4 *)shmem)[lane_id];
  }

  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc[ROW_TILES_PER_STEP][ROW_TILES_PER_STEP];

  for (int i = 0; i < ROW_TILES_PER_STEP; i++) {
    for (int j = 0; j < ROW_TILES_PER_STEP; j++) {
      wmma::fill_fragment(acc[i][j], zero);
    }
  }

  // TODO: MTMD - Loop promotion
  for (int k_step = 0; k_step < num_col_steps; k_step++) {
    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, wmma::precision::tf32, wmma::row_major>
        a[ROW_TILES_PER_STEP];
    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, wmma::precision::tf32, wmma::col_major>
        b[ROW_TILES_PER_STEP];
    for (int j = 0; j < ROW_TILES_PER_STEP; j++) {
      int base_row = (j < ROW_TILES_PER_STEP - 1) ? j * 16 : num_rows_after_padding - 16;
      const float *tile_ptr = shmem + (base_row * smem_stride + k_step * kWmmaK);
      wmma::load_matrix_sync(a[j], tile_ptr, smem_stride);
      wmma::load_matrix_sync(b[j], tile_ptr, smem_stride);
    }
    for (int i = 0; i < ROW_TILES_PER_STEP; i++) {
      for (int j = 0; j < ROW_TILES_PER_STEP; j++) {
        wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
      }
    }
  }

  for (int i = 0; i < ROW_TILES_PER_STEP; i++) {
    for (int j = 0; j < ROW_TILES_PER_STEP; j++) {
      float *tile_ptr = shmem + (i * kWmmaM * smem_stride_acc + j * kWmmaN);
      wmma::store_matrix_sync(tile_ptr, acc[i][j], smem_stride_acc, wmma::mem_row_major);
    }
  }

  float *gmem_interact_output = gmem_output + num_cols;
  int lastRowBlockOffset = ROW_TILES_PER_STEP * 16 - num_rows_after_padding;
  int src_line = 0;
  for (int i = 0; i < num_rows; ++i, ++src_line) {
    if (i == ((ROW_TILES_PER_STEP - 1) * 16)) {
      src_line += lastRowBlockOffset;
    }
    if (lane_id < i) {
      uint offset = (i * (i - 1)) >> 1;
      gmem_interact_output[offset + lane_id] = shmem[src_line * smem_stride_acc + lane_id];
    }
  }
  // Add padding to the output vectors
  if (lane_id < padding_size) {
    gmem_output[output_size - lane_id - 1] = __float2half(0);
  }
}

template <uint WARPS_PER_BLOCK,
          uint THREADBLOCK_SIZE,
          uint WARP_SIZE,
          uint WARP_SIZE_LOG_2,
          uint FRAG_A_ROWS,
          uint FRAG_B_COLS,
          uint TILE_LENGTH,
          uint TILE_LENGTH_LOG_2,
          uint TILE_WIDTH,
          uint TILE_WIDTH_LOG_2>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void dotBasedInteractTF32BwdKernel(const float *__restrict input,
                                       const float *__restrict upstream_grad,
                                       float *__restrict grad,
                                       float *__restrict bottom_mlp_grad,
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
                                       uint shared_mem_per_warp_size_elems,
                                       uint num_k_steps,
                                       uint num_n_steps) {
  // The only support sizes for TF32.
  const uint kWmmaM = 16;
  const uint kWmmaN = 16;
  const uint kWmmaK = 8;

  extern __shared__ float shared_mem_float[];
  uint warp_id = threadIdx.x >> WARP_SIZE_LOG_2;
  uint sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  if (sample_id >= batch_size) {
    return;
  }
  uint lane_id = threadIdx.x & (WARP_SIZE - 1);
  uint smem_warp_offset = warp_id * shared_mem_per_warp_size_elems;

  float *smem_in = &shared_mem_float[smem_warp_offset];
  float *smem_ugrad = &shared_mem_float[smem_warp_offset + input_size_elems];
  float *smem_out = &shared_mem_float[smem_warp_offset + input_size_elems + interaction_ugrad_2D_size_elems];

  // Global memory pointers for the current sample
  // Input
  uint gmem_input_sample_offset = sample_id * sample_size;
  const float *gmem_input = &input[gmem_input_sample_offset];

  // Interaction Gradient
  const uint &gmem_grad_sample_offset = gmem_input_sample_offset;
  float *gmem_grad = &grad[gmem_grad_sample_offset];

  // Bottom MLP gradient
  float *gmem_mlp_grad = &bottom_mlp_grad[sample_id * num_cols];

  // Upstream gradient vector
  uint gmem_ugrad_sample_offset = sample_id * (num_cols + interaction_ugrad_size_with_padding);
  const float *gmem_ugrad = &upstream_grad[gmem_ugrad_sample_offset];

  // Upstream gradient vector for interactions
  const float *gmem_ugrad_interactions = &gmem_ugrad[num_cols];

  // upstream grad -> shared memory (place in input section temporarily)
#pragma unroll
  for (uint idx = lane_id; idx < (interaction_ugrad_size >> 2); idx += WARP_SIZE) {
    float4 tmp = ((float4 *)gmem_ugrad_interactions)[idx];
    tmp.x = wmma::__float_to_tf32(tmp.x);
    tmp.y = wmma::__float_to_tf32(tmp.y);
    tmp.z = wmma::__float_to_tf32(tmp.z);
    tmp.w = wmma::__float_to_tf32(tmp.w);
    ((float4 *)smem_in)[idx] = tmp;
  }
  uint offset = (interaction_ugrad_size >> 2) << 2;
  for (uint idx = lane_id + offset; idx < interaction_ugrad_size; idx += WARP_SIZE) {
    smem_in[idx] = wmma::__float_to_tf32(gmem_ugrad_interactions[idx]);
  }
  __syncwarp();

  float zero = wmma::__float_to_tf32(0.0f);
  float4 zero4;
  zero4.x = zero;
  zero4.y = zero;
  zero4.z = zero;
  zero4.w = zero;
  // Form the 2D ugrad matrix.
  if (lane_id < num_rows_after_padding) {
    uint ugrad_flat_index = ((lane_id * (lane_id - 1)) >> 1);
    uint ugrad_offset_1 = lane_id * interaction_ugrad_2D_stride;
    for (uint row = 0; row < num_rows; row++) {
      float ugrad_val = zero;
      if (row < lane_id && lane_id < num_rows) {
        ugrad_val = smem_in[ugrad_flat_index + row];
        smem_ugrad[ugrad_offset_1 + row] = ugrad_val;
      }
      if (row <= lane_id && lane_id < num_rows_after_padding) {
        smem_ugrad[row * interaction_ugrad_2D_stride + lane_id] = ugrad_val;
      }
    }
    for (uint row = num_rows; row < num_rows_after_padding; row++) {
      smem_ugrad[row * interaction_ugrad_2D_stride + lane_id] = zero;
    }
  }
  __syncwarp();

  // Input -> Shared Memory

  if (lane_id < (num_cols >> 2)) {
    for (uint row = 0; row < num_rows; row++) {
      float *smem_row_ptr = &smem_in[row * input_stride];
      const float *gmem_row_ptr = &gmem_input[row * num_cols];
      float4 tmp = ((float4 *)gmem_row_ptr)[lane_id];
      tmp.x = wmma::__float_to_tf32(tmp.x);
      tmp.y = wmma::__float_to_tf32(tmp.y);
      tmp.z = wmma::__float_to_tf32(tmp.z);
      tmp.w = wmma::__float_to_tf32(tmp.w);
      ((float4 *)smem_row_ptr)[lane_id] = tmp;
    }
  }

  uint idx = lane_id + num_cols;
  if (idx < num_cols_after_padding) {
    for (uint row = 0; row < num_rows; row++) {
      float *smem_row_ptr = &smem_in[row * input_stride];
      smem_row_ptr[idx] = zero;
    }
  }

  if (lane_id < (num_cols_after_padding >> 2)) {
#pragma unroll 2
    for (uint row = num_rows; row < num_rows_after_padding; row++) {
      float *smem_row_ptr = &smem_in[row * input_stride];
      ((float4 *)smem_row_ptr)[lane_id] = zero4;
    }
  }
  __syncwarp();

  wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, wmma::precision::tf32, wmma::row_major> a[FRAG_A_ROWS];
  wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, wmma::precision::tf32, wmma::row_major> b[FRAG_B_COLS];
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc[FRAG_A_ROWS][FRAG_B_COLS];
  for (uint n = 0; n < num_n_steps; n++) {
    for (uint i = 0; i < FRAG_A_ROWS; i++) {
      for (uint j = 0; j < FRAG_B_COLS; j++) {
        wmma::fill_fragment(acc[i][j], zero);
      }
    }
    for (uint k = 0; k < num_k_steps; k++) {
      for (uint i = 0; i < FRAG_A_ROWS; i++) {
        const float *mat_a_tile_ptr =
            smem_ugrad + (i << TILE_LENGTH_LOG_2) * interaction_ugrad_2D_stride + (k << TILE_WIDTH_LOG_2);
        wmma::load_matrix_sync(a[i], mat_a_tile_ptr, interaction_ugrad_2D_stride);
      }
      for (uint j = 0; j < FRAG_B_COLS; j++) {
        const float *mat_b_tile_ptr =
            smem_in + (k << TILE_WIDTH_LOG_2) * input_stride + ((2 * n + j) << TILE_LENGTH_LOG_2);
        wmma::load_matrix_sync(b[j], mat_b_tile_ptr, input_stride);
      }
      for (uint i = 0; i < FRAG_A_ROWS; i++) {
        for (uint j = 0; j < FRAG_B_COLS; j++) {
          wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
      }
    }
    // __syncwarp(); ?
    uint out_stride = FRAG_B_COLS << TILE_LENGTH_LOG_2;
    for (uint i = 0; i < FRAG_A_ROWS; i++) {
      for (uint j = 0; j < FRAG_B_COLS; j++) {
        float *out_tile_ptr = smem_out + (i << TILE_LENGTH_LOG_2) * out_stride + (j << TILE_LENGTH_LOG_2);
        wmma::store_matrix_sync(out_tile_ptr, acc[i][j], out_stride, wmma::mem_row_major);
      }
    }
    uint gmem_grad_col = n * (FRAG_B_COLS << TILE_LENGTH_LOG_2) + lane_id;
    for (uint i = 0; i < num_rows; i++) {
      gmem_grad[i * num_cols + gmem_grad_col] = smem_out[i * out_stride + lane_id];
    }
  }

  if (lane_id < (num_cols >> 2)) {
    ((float4 *)gmem_mlp_grad)[lane_id] = ((float4 *)gmem_ugrad)[lane_id];
  }
}

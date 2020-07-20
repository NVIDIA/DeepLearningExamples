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

#include "shared_utils.cuh"

using namespace nvcuda;

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
          uint TILE_DIM_LOG_2>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void dotBasedInteractFwdKernelNonAligned_(const __half *__restrict input,
                                                                                         __half *__restrict output,
                                                                                         uint batch_size,
                                                                                         uint num_rows,
                                                                                         uint num_cols,
                                                                                         uint num_rows_after_padding,
                                                                                         uint num_cols_after_padding,
                                                                                         uint smem_elems_per_warp,
                                                                                         uint smem_rows_per_warp,
                                                                                         uint output_size,
                                                                                         uint num_row_steps,
                                                                                         uint num_col_steps) {
  uint warp_id = (threadIdx.x >> WARP_SIZE_LOG_2);
  int sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  if (sample_id >= batch_size) {
    return;
  }
  int lane_id = threadIdx.x & (WARP_SIZE - 1);

  extern __shared__ half shmem_dynamic_[];
  half *shmem = shmem_dynamic_ + (warp_id * smem_elems_per_warp);

  const half *sample_input = input + num_rows * num_cols * sample_id;
  for (uint i = 0; i < num_rows; ++i, sample_input += num_cols) {
    for (uint idx = lane_id; idx < num_cols; idx += WARP_SIZE) {
      (shmem + i * SMEM_STRIDE)[idx] = sample_input[idx];
    }
  }

  uint idx = lane_id + num_cols;
  if (idx < num_cols_after_padding) {
    for (int i = 0; i < num_rows; ++i) {
      (shmem + i * SMEM_STRIDE)[idx] = __float2half(0);
    }
  }

  half4 zeros;
  zeros.vals[0].x = __float2half(0);
  zeros.vals[0].y = __float2half(0);
  zeros.vals[1].x = __float2half(0);
  zeros.vals[1].y = __float2half(0);
  if (lane_id < (num_cols_after_padding >> 2)) {
    for (int i = num_rows; i < num_rows_after_padding; i++) {
      ((half4 *)(shmem + i * SMEM_STRIDE))[lane_id] = zeros;
    }
  }
  __syncwarp();
  half *gmem_output = output + output_size * sample_id;

  for (uint idx = lane_id; idx < num_cols; idx += WARP_SIZE) {
    gmem_output[idx] = shmem[idx];
  }

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
        wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
      }
    }
  }
  float *shmem_store = reinterpret_cast<float *>(shmem);
  for (int i = 0; i < M_BLOCKS; i++) {
    for (int j = 0; j < M_BLOCKS; j++) {
      float *tile_ptr = shmem_store + (i * 16 * SMEM_STRIDE_ACC + j * 16);
      wmma::store_matrix_sync(tile_ptr, acc[i][j], SMEM_STRIDE_ACC, wmma::mem_row_major);
    }
  }

  half *gmem_interact_output = gmem_output + num_cols;
  int lastRowBlockOffset = M_BLOCKS * 16 - smem_rows_per_warp;
  int srcLine = 0;
  for (int i = 0; i < num_rows; ++i, ++srcLine) {
    if (i == ((M_BLOCKS - 1) * 16)) {
      srcLine += lastRowBlockOffset;
    }
    if (lane_id < i) {
      uint offset = (i * (i - 1)) >> 1;
      gmem_interact_output[offset + lane_id] = __float2half(shmem_store[srcLine * SMEM_STRIDE_ACC + lane_id]);
    }
  }
  // Padding
  if (lane_id == 0) {
    gmem_output[output_size - 1] = __float2half(0);
  }
}

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
                                                                                  uint smem_stride_acc) {
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
  // Padding
  if (lane_id == 0) {
    gmem_output[output_size - 1] = 0;
  }
}

template <uint WARPS_PER_BLOCK,
          uint THREADBLOCK_SIZE,
          uint ROW_TILES_PER_STEP,
          uint COL_TILES_PER_STEP,
          uint WARP_SIZE,
          uint WARP_SIZE_LOG_2,
          uint TILE_DIM,
          uint TILE_DIM_LOG_2>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void dotBasedInteractBwdKernelNonAligned_(const __half *__restrict input,
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
#pragma unroll
  for (uint idx = lane_id; idx < interaction_ugrad_size; idx += WARP_SIZE) {
    smem_in[idx] = gmem_ugrad_interactions[idx];
  }
  __syncwarp();
  // Form the 2D ugrad matrix.
  if (lane_id < num_rows_after_padding) {
    uint ugrad_flat_index = ((lane_id * (lane_id - 1)) >> 1);
    uint ugrad_offset_1 = lane_id * interaction_ugrad_2D_stride;
    for (uint row = 0; row < num_rows; row++) {
      half ugrad_val = __float2half(0.0f);
      if (row < lane_id && lane_id < num_rows) {
        ugrad_val = smem_in[ugrad_flat_index + row];
        smem_temp[ugrad_offset_1 + row] = ugrad_val;
      }
      if (row <= lane_id && lane_id < num_rows_after_padding) {
        smem_temp[row * interaction_ugrad_2D_stride + lane_id] = ugrad_val;
      }
    }
    for (uint row = num_rows; row < num_rows_after_padding; row++) {
      smem_temp[row * interaction_ugrad_2D_stride + lane_id] = __float2half(0.0f);
    }
  }
  __syncwarp();

  // Input -> Shared Memory

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
    uint gmem_grad_col = (col_step << TILE_DIM_LOG_2) + lane_id;
    if (gmem_grad_col < num_cols) {
      for (uint i = 0; i < num_rows; i++) {
        gmem_grad[i * num_cols + gmem_grad_col] = __float2half(smem_out[(i << TILE_DIM_LOG_2) + lane_id]);
      }
    }
  }

  for (uint idx = lane_id; idx < num_cols; idx += WARP_SIZE) {
    gmem_mlp_grad[idx] = gmem_ugrad[idx];
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

inline void dotBasedInteractTF32Fwd(
    const void *input, const void *bottom_mlp_output, void *output, uint batch_size, uint num_rows, uint num_cols) {
  const uint kWarpSize = 32;
  const uint kWarpSizeLog2 = Log2<kWarpSize>::value;
  const uint kTileLength = 16;
  const uint kTileLengthLog2 = Log2<kTileLength>::value;
  const uint kTileWidth = 8;
  const uint kTileWidthLog2 = Log2<kTileWidth>::value;
  const uint kWarpsPerBlock = 2;
  const uint kThreadBlockSize = kWarpsPerBlock * kWarpSize;
  const uint kPaddingSize = 1;
  const uint kRowTilesPerStep = 2;
  const uint kColTilesPerStep = 1;
  const uint kSkewFloat = 4;  // Ensures we are 16 byte align as required by nvcuda::wmma::load_matrix_sync

  // num tiles
  uint mat_a_num_row_tiles = (num_rows + kTileLength - 1) >> kTileLengthLog2;
  uint mat_a_num_col_tiles = (num_cols + kTileWidth - 1) >> kTileWidthLog2;

  const uint &mat_b_num_row_tiles = mat_a_num_col_tiles;
  const uint &mat_b_num_col_tiles = mat_a_num_row_tiles;

  // number of rows and columns after padding
  uint num_rows_after_padding = mat_a_num_row_tiles << kTileLengthLog2;
  uint num_cols_after_padding = mat_a_num_col_tiles << kTileWidthLog2;

  uint num_row_steps = mat_a_num_row_tiles / kRowTilesPerStep;
  uint num_col_steps = mat_a_num_col_tiles / kColTilesPerStep;

  const uint smem_stride = num_cols_after_padding + kSkewFloat;
  const uint smem_elems_per_warp_mat = num_rows_after_padding * smem_stride;

  const uint smem_stride_acc = num_rows_after_padding + kSkewFloat;
  const uint smem_elems_per_warp_acc = num_rows_after_padding * smem_stride_acc;

  const uint smem_elems_per_warp =
      smem_elems_per_warp_mat > smem_elems_per_warp_acc ? smem_elems_per_warp_mat : smem_elems_per_warp_acc;

  uint output_size = num_cols + ((num_rows * (num_rows - 1)) >> 1) + kPaddingSize;
  bool float4_predicate = !((num_cols & 7) || (output_size & 7));

  // TODO: MTMD - Clean Up
  // std::cout << "mat_a_num_row_tiles    " << mat_a_num_row_tiles << std::endl;
  // std::cout << "mat_a_num_col_tiles    " << mat_a_num_col_tiles << std::endl;
  // std::cout << "mat_b_num_row_tiles    " << mat_b_num_row_tiles << std::endl;
  // std::cout << "mat_b_num_col_tiles    " << mat_b_num_col_tiles << std::endl;
  // std::cout << "num_rows_after_padding " << num_rows_after_padding << std::endl;
  // std::cout << "num_cols_after_padding " << num_cols_after_padding << std::endl;
  // std::cout << "num_row_steps          " << num_row_steps << std::endl;
  // std::cout << "num_col_steps          " << num_col_steps << std::endl;
  // std::cout << "smem_stride            " << smem_stride << std::endl;
  // std::cout << "smem_elems_per_warp_mat" << smem_elems_per_warp_mat << std::endl;
  // std::cout << "smem_stride_acc        " << smem_stride_acc << std::endl;
  // std::cout << "smem_elems_per_warp_acc" << smem_elems_per_warp_acc << std::endl;
  // std::cout << "===================================================================" << std::endl;

  if (float4_predicate) {
    dotBasedInteractTF32FwdKernel<kWarpsPerBlock,
                                  kThreadBlockSize,
                                  kWarpSize,
                                  kWarpSizeLog2,
                                  kTileLength,
                                  kTileLengthLog2,
                                  kTileWidth,
                                  kTileWidthLog2,
                                  kRowTilesPerStep>
        <<<(batch_size + kWarpsPerBlock - 1) / kWarpsPerBlock,
           kThreadBlockSize,
           kWarpsPerBlock * smem_elems_per_warp * sizeof(float)>>>((const float *)input,
                                                                   (float *)output,
                                                                   batch_size,
                                                                   num_rows,
                                                                   num_cols,
                                                                   num_rows_after_padding,
                                                                   num_cols_after_padding,
                                                                   smem_elems_per_warp,
                                                                   output_size,
                                                                   num_row_steps,
                                                                   num_col_steps,
                                                                   smem_stride,
                                                                   smem_stride_acc);
  } else {
    std::cout << "GENERIC VERSION IS UNFINISHED." << std::endl;
#ifdef GENERIC_IS_DONE
    dotBasedInteractFwdKernelNonAligned<warps_per_threadblock,
                                        threadblock_size,
                                        M_BLOCKS,
                                        K_BLOCKS,
                                        SMEM_STRIDE,
                                        SMEM_STRIDE_ACC,
                                        kWarpSize,
                                        kWarpSizeLog2,
                                        kTileDim,
                                        kTileDimLog2>
        <<<(batch_size + warps_per_threadblock - 1) / warps_per_threadblock,
           threadblock_size,
           warps_per_threadblock * smem_elems_per_warp * sizeof(__half)>>>((const __half *)input,
                                                                           (half *)output,
                                                                           batch_size,
                                                                           num_rows,
                                                                           num_cols,
                                                                           num_rows_after_padding,
                                                                           num_cols_after_padding,
                                                                           smem_elems_per_warp,
                                                                           smem_rows_per_warp,
                                                                           output_size,
                                                                           num_row_steps,
                                                                           num_col_steps);
#endif
  }
}

inline void dotBasedInteractTF32Bwd(void *input,
                                    void *upstream_grad,
                                    void *grad,
                                    void *bottom_mlp_grad,
                                    uint batch_size,
                                    uint num_rows,
                                    uint num_cols) {
  // Fragment Settings
  const uint kFragARows = 2;
  const uint kFragBCols = 2;
  const uint kTileLength = 16;
  const uint kTileLengthLog2 = Log2<kTileLength>::value;
  const uint kTileWidth = 8;
  const uint kTileWidthLog2 = Log2<kTileWidth>::value;

  const uint kWarpSize = 32;
  const uint kWarpSizeLog2 = Log2<kWarpSize>::value;
  const uint kSkewFloat = 4;
  const uint kPaddingSize = 1;
  const uint kWarpsPerBlock = 1;
  const uint kWarpsPerBlockLog2 = Log2<kWarpsPerBlock>::value;
  const uint kNumThreads = kWarpsPerBlock * kWarpSize;

  // num tiles
  uint mat_a_num_row_tiles = (num_rows + kTileLength - 1) >> kTileLengthLog2;
  uint mat_a_num_col_tiles = (num_rows + kTileWidth - 1) >> kTileWidthLog2;

  const uint &mat_b_num_row_tiles = mat_a_num_col_tiles;
  uint mat_b_num_col_tiles = (num_cols + kTileLength - 1) >> kTileLengthLog2;

  // number of rows and columns after padding
  uint num_rows_after_padding = mat_a_num_row_tiles << kTileLengthLog2;
  uint num_cols_after_padding = mat_b_num_col_tiles << kTileLengthLog2;

  // 2D ugrad size and stride
  uint interaction_ugrad_2D_stride = num_rows_after_padding + kSkewFloat;
  uint interaction_ugrad_2D_size_elems = num_rows_after_padding * interaction_ugrad_2D_stride;

  // 1D ugrad size
  uint interaction_ugrad_size = num_rows * (num_rows - 1) >> 1;
  uint interaction_ugrad_size_with_padding = interaction_ugrad_size + kPaddingSize;

  // in_out place size and stride
  uint input_stride = num_cols_after_padding + kSkewFloat;
  uint input_size_elems = num_rows_after_padding * input_stride;

  // sample size
  uint sample_size = num_rows * num_cols;

  // output size
  uint output_size_elems = kTileLength * kTileLength * kFragARows * kFragBCols;

  // Shared memory size
  uint shared_mem_per_warp_size_elems = interaction_ugrad_2D_size_elems + input_size_elems + output_size_elems;
  uint shared_mem_size_elems = kWarpsPerBlock * shared_mem_per_warp_size_elems;
  uint shared_mem_size_bytes = shared_mem_size_elems * sizeof(float);

  uint num_blocks = (batch_size + kWarpsPerBlock - 1) >> kWarpsPerBlockLog2;
  uint num_k_steps = mat_a_num_col_tiles;
  uint num_n_steps = mat_b_num_col_tiles / kFragBCols;

  bool float4_predicate = !((interaction_ugrad_size_with_padding & 7) || (num_cols & 7));
  if (float4_predicate) {
    dotBasedInteractTF32BwdKernel<kWarpsPerBlock,
                                  kNumThreads,
                                  kWarpSize,
                                  kWarpSizeLog2,
                                  kFragARows,
                                  kFragBCols,
                                  kTileLength,
                                  kTileLengthLog2,
                                  kTileWidth,
                                  kTileWidthLog2>
        <<<num_blocks, kNumThreads, shared_mem_size_bytes>>>((const float *)input,
                                                             (const float *)upstream_grad,
                                                             (float *)grad,
                                                             (float *)bottom_mlp_grad,
                                                             batch_size,
                                                             num_rows,
                                                             num_cols,
                                                             num_rows_after_padding,
                                                             num_cols_after_padding,
                                                             sample_size,
                                                             interaction_ugrad_size,
                                                             interaction_ugrad_size_with_padding,
                                                             interaction_ugrad_2D_size_elems,
                                                             interaction_ugrad_2D_stride,
                                                             input_size_elems,
                                                             input_stride,
                                                             shared_mem_per_warp_size_elems,
                                                             num_k_steps,
                                                             num_n_steps);
  } else {
    std::cout << "GENERIC VERSION IS UNFINISHED." << std::endl;
#ifdef GENERIC_IS_DONE
    dotBasedInteractBwdKernelNonAligned<kWarpsPerBlock,
                                        kNumThreads,
                                        kRowTilesPerStep,
                                        kColTilesPerStep,
                                        kWarpSize,
                                        kWarpSizeLog2,
                                        kTileDim,
                                        kTileDimLog2>
        <<<num_blocks, kNumThreads, shared_mem_size_bytes>>>((const half *)input,
                                                             (const half *)upstream_grad,
                                                             (half *)grad,
                                                             (half *)bottom_mlp_grad,
                                                             batch_size,
                                                             num_rows,
                                                             num_cols,
                                                             num_rows_after_padding,
                                                             num_cols_after_padding,
                                                             sample_size,
                                                             interaction_ugrad_size,
                                                             interaction_ugrad_size_with_padding,
                                                             interaction_ugrad_2D_size_elems,
                                                             interaction_ugrad_2D_stride,
                                                             input_size_elems,
                                                             input_stride,
                                                             num_row_steps,
                                                             num_col_steps,
                                                             row_tiles_per_step,
                                                             shared_mem_per_warp_size_byte);
#endif
  }
}

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

using namespace nvcuda;

#define CHK_CUDA(expression)                                                                                        \
  {                                                                                                                 \
    cudaError_t status = (expression);                                                                              \
    if (status != cudaSuccess) {                                                                                    \
      std::cerr << "Error in file: " << __FILE__ << ", on line: " << __LINE__ << ": " << cudaGetErrorString(status) \
                << std::endl;                                                                                       \
      std::exit(EXIT_FAILURE);                                                                                      \
    }                                                                                                               \
  }

template <uint x>
struct Log2 {
  static constexpr uint value = 1 + Log2<x / 2>::value;
};
template <>
struct Log2<1> {
  static constexpr uint value = 0;
};

struct __align__(8) half4 {
  half2 vals[2];
};

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
__launch_bounds__(THREADBLOCK_SIZE) __global__ void dotBasedInteractFwdKernelNonAligned(const __half *__restrict input,
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

  extern __shared__ half shmem_dynamic[];
  half *shmem = shmem_dynamic + (warp_id * smem_elems_per_warp);

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
          uint M_BLOCKS,
          uint K_BLOCKS,
          uint SMEM_STRIDE,
          uint SMEM_STRIDE_ACC,
          uint WARP_SIZE,
          uint WARP_SIZE_LOG_2,
          uint TILE_DIM,
          uint TILE_DIM_LOG_2>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void dotBasedInteractFwdKernel(const __half *__restrict input,
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

  extern __shared__ half shmem_dynamic[];
  half *shmem = shmem_dynamic + (warp_id * smem_elems_per_warp);

  const half *sample_input = input + num_rows * num_cols * sample_id;
  if (lane_id < (num_cols >> 2)) {
    for (int i = 0; i < num_rows; ++i, sample_input += num_cols) {
      ((float2 *)(shmem + i * SMEM_STRIDE))[lane_id] = ((float2 *)sample_input)[lane_id];
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
  if (lane_id < (num_cols >> 2)) {
    ((float2 *)gmem_output)[lane_id] = ((float2 *)shmem)[lane_id];
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
          uint ROW_TILES_PER_STEP,
          uint COL_TILES_PER_STEP,
          uint WARP_SIZE,
          uint WARP_SIZE_LOG_2,
          uint TILE_DIM,
          uint TILE_DIM_LOG_2>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void dotBasedInteractBwdKernelNonAligned(const __half *__restrict input,
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
          uint ROW_TILES_PER_STEP,
          uint COL_TILES_PER_STEP,
          uint WARP_SIZE,
          uint WARP_SIZE_LOG_2,
          uint TILE_DIM,
          uint TILE_DIM_LOG_2>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void dotBasedInteractBwdKernel(const __half *__restrict input,
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
  for (uint idx = lane_id; idx < (interaction_ugrad_size >> 3); idx += WARP_SIZE) {
    ((float4 *)smem_in)[idx] = ((float4 *)gmem_ugrad_interactions)[idx];
  }
  uint offset = (interaction_ugrad_size >> 3) << 3;
  for (uint idx = lane_id + offset; idx < interaction_ugrad_size; idx += WARP_SIZE) {
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

  if (lane_id < (num_cols >> 2)) {
    for (uint row = 0; row < num_rows; row++) {
      half *smem_row_ptr = &smem_in[row * input_stride];
      const half *gmem_row_ptr = &gmem_input[row * num_cols];
      ((float2 *)smem_row_ptr)[lane_id] = ((float2 *)gmem_row_ptr)[lane_id];
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
  if (lane_id < (num_cols_after_padding >> 2)) {
#pragma unroll 2
    for (uint row = num_rows; row < num_rows_after_padding; row++) {
      half *smem_row_ptr = &smem_in[row * input_stride];
      ((half4 *)smem_row_ptr)[lane_id] = zeros;
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
  if (lane_id < (num_cols >> 2)) {
    ((float2 *)gmem_mlp_grad)[lane_id] = ((float2 *)gmem_ugrad)[lane_id];
  }
}

inline void dotBasedInteractFwd(
    const void *input, const void *bottom_mlp_output, void *output, uint batch_size, uint num_rows, uint num_cols) {
  const uint kWarpSize = 32;
  const uint kWarpSizeLog2 = Log2<kWarpSize>::value;
  const uint kTileDim = 16;
  const uint kTileDimLog2 = Log2<kTileDim>::value;
  const uint warps_per_threadblock = 4;
  const uint threadblock_size = warps_per_threadblock * 32;
  const uint kPaddingSize = 1;
  const uint kRowTilesPerStep = 2;
  const uint kColTilesPerStep = 1;

  // num tiles
  uint num_row_tiles = (num_rows + kTileDim - 1) >> kTileDimLog2;
  uint num_col_tiles = (num_cols + kTileDim - 1) >> kTileDimLog2;

  // number of rows and columns after padding
  uint num_rows_after_padding = kTileDim << 1;
  uint num_cols_after_padding = num_col_tiles << kTileDimLog2;

  uint num_row_steps = num_row_tiles / kRowTilesPerStep;
  uint num_col_steps = num_col_tiles / kColTilesPerStep;

  const uint K_BLOCKS = 8;
  const uint M_BLOCKS = 2;
  const uint SKEW_HALF = ((K_BLOCKS % 2) == 0) ? 8 : 0;
  const uint SMEM_STRIDE = (K_BLOCKS * 16 + SKEW_HALF);
  // multiple of 2 to guarantee 256-bit alignment for start of the row, at least 16 to safeload a tile
  const uint smem_rows_per_warp = M_BLOCKS << 4;
  const uint smem_elems_per_warp_mat = smem_rows_per_warp * SMEM_STRIDE;
  const uint SKEW_HALF_ACC = ((M_BLOCKS % 2) == 0) ? 8 : 0;
  const uint SMEM_STRIDE_ACC = (M_BLOCKS * 16 + SKEW_HALF_ACC);
  const uint smem_elems_per_warp_acc = M_BLOCKS * 16 * SMEM_STRIDE_ACC * 2;  // output in FP32
  const uint smem_elems_per_warp =
      (smem_elems_per_warp_mat > smem_elems_per_warp_acc) ? smem_elems_per_warp_mat : smem_elems_per_warp_acc;
  uint output_size = num_cols + ((num_rows * (num_rows - 1)) >> 1) + kPaddingSize;

  bool float4_predicate = !((num_cols & 7) || (output_size & 7));

  if (float4_predicate) {
    dotBasedInteractFwdKernel<warps_per_threadblock,
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
  } else {
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
  }
}

inline void dotBasedInteractBwd(void *input,
                                void *upstream_grad,
                                void *grad,
                                void *bottom_mlp_grad,
                                uint batch_size,
                                uint num_rows,
                                uint num_cols) {
  const uint kWarpSize = 32;
  const uint kWarpSizeLog2 = Log2<kWarpSize>::value;
  const uint kTileDim = 16;
  const uint kTileDimLog2 = Log2<kTileDim>::value;
  const uint mem_skew_size = 8;
  const uint kPaddingSize = 1;
  const uint kWarpsPerBlock = 4;
  const uint kWarpsPerBlockLog2 = Log2<kWarpsPerBlock>::value;
  const uint kNumThreads = kWarpsPerBlock * kWarpSize;
  const uint kRowTilesPerStep = 2;
  const uint kColTilesPerStep = 1;

  uint row_tiles_per_step = num_rows > kTileDim ? kRowTilesPerStep : 1;

  // num tiles
  uint num_row_tiles = (num_rows + kTileDim - 1) >> kTileDimLog2;
  uint num_col_tiles = (num_cols + kTileDim - 1) >> kTileDimLog2;

  // number of rows and columns after padding
  uint num_rows_after_padding = kTileDim << 1;
  uint num_cols_after_padding = num_col_tiles << kTileDimLog2;

  // 2D ugrad size and stride
  uint interaction_ugrad_2D_stride = num_rows_after_padding + mem_skew_size;
  uint interaction_ugrad_2D_size_elems = num_rows_after_padding * interaction_ugrad_2D_stride;
  uint interaction_ugrad_2D_size_bytes = interaction_ugrad_2D_size_elems * sizeof(half);

  // 1D ugrad size
  uint interaction_ugrad_size = num_rows * (num_rows - 1) >> 1;
  uint interaction_ugrad_size_with_padding = interaction_ugrad_size + kPaddingSize;

  // in_out place size and stride
  uint input_stride = num_cols_after_padding + mem_skew_size;
  uint input_size_elems = num_rows_after_padding * input_stride;
  uint input_size_bytes = input_size_elems * sizeof(half);

  // sample size
  uint sample_size = num_rows * num_cols;

  // output size
  uint output_size_elems = kTileDim * kTileDim * kRowTilesPerStep * kColTilesPerStep;
  uint output_size_bytes = output_size_elems * sizeof(float);

  // staging area size
  uint staging_area_size_bytes =
      output_size_bytes > interaction_ugrad_2D_size_bytes ? output_size_bytes : interaction_ugrad_2D_size_bytes;

  // Shared memory size
  uint shared_mem_per_warp_size_byte = input_size_bytes + staging_area_size_bytes;
  uint shared_mem_size_bytes = kWarpsPerBlock * shared_mem_per_warp_size_byte;

  uint num_blocks = (batch_size + kWarpsPerBlock - 1) >> kWarpsPerBlockLog2;
  uint num_row_steps = num_row_tiles / row_tiles_per_step;
  uint num_col_steps = num_col_tiles / kColTilesPerStep;

  bool float4_predicate = !((interaction_ugrad_size_with_padding & 7) || (num_cols & 7));
  if (float4_predicate) {
    dotBasedInteractBwdKernel<kWarpsPerBlock,
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
  } else {
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
  }
}

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

  gmem_out_interaction[interaction_output_size] = 0;
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

  uint input_batch_offset = blockIdx.x * input_size;
  const float *gmem_in = &input[input_batch_offset];

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

  gmem_out_interaction[interaction_output_size] = 0;
}

inline void dotBasedInteractF32Fwd(const void *input,
                                   const void *bottom_mlp_output,
                                   const void *output,
                                   uint batch_size,
                                   uint num_rows,
                                   uint num_cols) {
  const uint kPaddingSize = 1;
  const uint kNumThreads = 128;
  uint num_blocks = batch_size;

  // Output
  uint interaction_output_size = (num_rows * (num_rows - 1)) >> 1;
  uint output_size = num_cols + interaction_output_size + kPaddingSize;

  // Input
  uint input_size = num_rows * num_cols;

  uint shared_mem_size_elems = input_size;
  uint shared_mem_size_bytes = shared_mem_size_elems << 2;  // F32 Kernel

  bool float4_predicate = !((num_cols & 3) || (output_size & 3));

  if (float4_predicate) {
    dotBasedInteractF32FwdKernel<kNumThreads>
        <<<num_blocks, kNumThreads, shared_mem_size_bytes>>>((const float *)input,
                                                             (float *)output,
                                                             batch_size,
                                                             num_rows,
                                                             num_cols,
                                                             input_size,
                                                             output_size,
                                                             interaction_output_size);
  } else {
    dotBasedInteractF32FwdKernelNonAligned<kNumThreads>
        <<<num_blocks, kNumThreads, shared_mem_size_bytes>>>((const float *)input,
                                                             (float *)output,
                                                             batch_size,
                                                             num_rows,
                                                             num_cols,
                                                             input_size,
                                                             output_size,
                                                             interaction_output_size);
  }
}

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
                                                uint ugrad_size,
                                                uint interaction_ugrad_size) {
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
  uint upstream_grad_batch_offset = blockIdx.x * ugrad_size;
  const float *gmem_mlp_ugrad = &upstream_grad[upstream_grad_batch_offset];
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

template <uint THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void dotBasedInteractF32BwdKernel(const float *__restrict input,
                                                                                 const float *__restrict upstream_grad,
                                                                                 float *__restrict grad,
                                                                                 float *__restrict bottom_mlp_grad,
                                                                                 uint batch_size,
                                                                                 uint num_rows,
                                                                                 uint num_cols,
                                                                                 uint input_size,
                                                                                 uint ugrad_size,
                                                                                 uint interaction_ugrad_size) {
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
  uint upstream_grad_batch_offset = blockIdx.x * ugrad_size;
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
  const uint kPaddingSize = 1;
  const uint kNumThreads = 128;

  uint num_blocks = batch_size;

  uint input_size = num_rows * num_cols;

  // 1D ugrad size
  uint interaction_ugrad_size = num_rows * (num_rows - 1) >> 1;
  uint interaction_ugrad_size_with_padding = interaction_ugrad_size + kPaddingSize;
  uint ugrad_size = num_cols + interaction_ugrad_size_with_padding;

  // input space + upstream grad space
  uint smem_size_elems = input_size + interaction_ugrad_size;
  uint smem_size_bytes = smem_size_elems << 2;  // F32 Kernel

  bool float4_predicate = !((interaction_ugrad_size_with_padding & 3) || (num_cols & 3));
  if (float4_predicate) {
    dotBasedInteractF32BwdKernel<kNumThreads>
        <<<num_blocks, kNumThreads, smem_size_bytes>>>((const float *)input,
                                                       (const float *)upstream_grad,
                                                       (float *)grad,
                                                       (float *)bottom_mlp_grad,
                                                       batch_size,
                                                       num_rows,
                                                       num_cols,
                                                       input_size,
                                                       ugrad_size,
                                                       interaction_ugrad_size);
  } else {
    dotBasedInteractF32BwdKernelNonAligned<kNumThreads>
        <<<num_blocks, kNumThreads, smem_size_bytes>>>((const float *)input,
                                                       (const float *)upstream_grad,
                                                       (float *)grad,
                                                       (float *)bottom_mlp_grad,
                                                       batch_size,
                                                       num_rows,
                                                       num_cols,
                                                       input_size,
                                                       ugrad_size,
                                                       interaction_ugrad_size);
  }
}


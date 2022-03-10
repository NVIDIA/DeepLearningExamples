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
__launch_bounds__(THREADBLOCK_SIZE) __global__ void dotBasedInteractTF32FwdKernelNonAligned_(const __half *__restrict input,
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
                                                                                         uint num_col_steps,
                                                                                         uint padding_size) {
  uint warp_id = (threadIdx.x >> WARP_SIZE_LOG_2); //each threadblock covers multiple samples
  int sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id; //each warp covers one sample
  if (sample_id >= batch_size) {
    return;
  }
  int lane_id = threadIdx.x & (WARP_SIZE - 1); //0...32 within a sample

  extern __shared__ half shmem_dynamic_[];
  half *shmem = shmem_dynamic_ + (warp_id * smem_elems_per_warp);

  //skip to the input for our warp
  const half *sample_input = input + num_rows * num_cols * sample_id;

  //copy all rows of our sample into shmem
  for (uint i = 0; i < num_rows; ++i, sample_input += num_cols) {
    for (uint idx = lane_id; idx < num_cols; idx += WARP_SIZE) {
      (shmem + i * SMEM_STRIDE)[idx] = sample_input[idx];
    }
  }

  uint idx = lane_id + num_cols;
  //pad each row (embedding) to num_cols_after_padding
  //this assumes that num_cols_after_padding-num_cols<=WARP_SIZE
  if (idx < num_cols_after_padding) {
    for (int i = 0; i < num_rows; ++i) {
      (shmem + i * SMEM_STRIDE)[idx] = __float2half(0);
    }
  }

  //add more zero-filled rows (embeddings) to pad to tile width
  //zero out 4 cells at once, hence the >>2
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

  //copy over the bottom_mlp_output into the final result
  //assumes bottom_mlp_output is at the start of the input
  for (uint idx = lane_id; idx < num_cols; idx += WARP_SIZE) {
    gmem_output[idx] = shmem[idx];
  }

  //compute the dot product
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

  // skip over the part where we copied the bottom_mlp_output
  half *gmem_interact_output = gmem_output + num_cols;

  // copy over the dot product result into the output
  int lastRowBlockOffset = M_BLOCKS * 16 - smem_rows_per_warp;
  int srcLine = 0;
  for (int i = 0; i < num_rows; ++i, ++srcLine) {
    if (i == ((M_BLOCKS - 1) * 16)) {
      srcLine += lastRowBlockOffset;
    }
    if (lane_id < i) { //this assumes we have num_categorical_features<WARP_SIZE
      uint offset = (i * (i - 1)) >> 1;
      gmem_interact_output[offset + lane_id] = __float2half(shmem_store[srcLine * SMEM_STRIDE_ACC + lane_id]);
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

  uint warp_id = threadIdx.x >> WARP_SIZE_LOG_2; //each threadblock covers multiple samples
  uint sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id; //each warp covers a single sample
  if (sample_id >= batch_size) {
    return;
  }
  int lane_id = threadIdx.x & (WARP_SIZE - 1); //0...32 within a sample

  extern __shared__ float shmem_dynamic_float[];
  float *shmem = shmem_dynamic_float + (warp_id * smem_elems_per_warp);

  const float *gmem_input = input + num_rows * num_cols * sample_id; //jump to our input sample
   //loop over embeddings, and copy each into shmem (but assume size is <=128)
   //convert to tf32 while copying
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

  //pad each embedding to num_cols_after_padding
  //this assumes that num_cols_after_padding-num_cols<= WARP_SIZE
  uint idx = lane_id + num_cols;
  if (idx < num_cols_after_padding) {
    for (uint i = 0; i < num_rows; ++i) {
      (shmem + i * smem_stride)[idx] = zero;
    }
  }

  //add more fake embeddings filled with zeros so we can better use cores
  //zero out 4 cells at once, hence the >>2
  if (lane_id < (num_cols_after_padding >> 2)) {
    for (int i = num_rows; i < num_rows_after_padding; i++) {
      ((float4 *)(shmem + i * smem_stride))[lane_id] = zero4;
    }
  }
  __syncwarp();
  // TODO: MTMD - Copy directly without using shared memory
  //copy over bottom_mlp_output into the output array
  float *gmem_output = output + output_size * sample_id;
  if (lane_id < (num_cols >> 2)) {
    ((float4 *)gmem_output)[lane_id] = ((float4 *)shmem)[lane_id];
  }

  //compute the dot product
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

  // skip over the part where we copied the bottom_mlp_output
  float *gmem_interact_output = gmem_output + num_cols;

  // copy over the dot product result into the output
  int lastRowBlockOffset = ROW_TILES_PER_STEP * 16 - num_rows_after_padding;
  int src_line = 0;
  for (int i = 0; i < num_rows; ++i, ++src_line) {
    if (i == ((ROW_TILES_PER_STEP - 1) * 16)) {
      src_line += lastRowBlockOffset;
    }
    if (lane_id < i) { //this assumes we have num_categorical_features (num_rows-1)<WARP_SIZE
      uint offset = (i * (i - 1)) >> 1;
      gmem_interact_output[offset + lane_id] = shmem[src_line * smem_stride_acc + lane_id];
    }
  }

  // Add padding to the output vectors
  if (lane_id < padding_size) {
    gmem_output[output_size - lane_id - 1] = zero;
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

  uint raw_output_size = num_cols + ((num_rows * (num_rows - 1)) >> 1);
  uint output_size = ((raw_output_size-1)/8 + 1)*8; //round up to multiple of 8
  uint padding_size = output_size-raw_output_size;

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
           kWarpsPerBlock * smem_elems_per_warp * sizeof(float),
           at::cuda::getCurrentCUDAStream()>>>((const float *)input,
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
                                              smem_stride_acc,
                                              padding_size);
  } else {
    std::cout << "GENERIC VERSION IS UNFINISHED." << std::endl;
#ifdef GENERIC_IS_DONE
    dotBasedInteractTF32FwdKernelNonAligned<warps_per_threadblock,
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
           warps_per_threadblock * smem_elems_per_warp * sizeof(__half),
           at::cuda::getCurrentCUDAStream()>>>((const __half *)input,
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
                                              num_col_steps,
                                              padding_size);
#endif
  }
}
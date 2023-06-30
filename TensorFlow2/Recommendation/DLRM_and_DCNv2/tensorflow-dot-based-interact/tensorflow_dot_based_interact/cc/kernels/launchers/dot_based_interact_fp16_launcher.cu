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


#ifndef FP16_LAUNCHER_CU
#define FP16_LAUNCHER_CU

#include "../cuda_kernels/dot_based_interact_fp16.cu"

inline void dotBasedInteractFP16Fwd(const void *input,
                                 const void *bottom_mlp_output,
                                 void *output,
                                 uint batch_size,
                                 uint num_rows,
                                 uint num_cols,
                                 cudaStream_t stream) {
  // num_rows is num cat variables + 1
  // num_cols is embedding dim
  const uint kWarpSize = 32;
  const uint kWarpSizeLog2 = Log2<kWarpSize>::value;
  const uint kTileDim = 16;
  const uint kTileDimLog2 = Log2<kTileDim>::value;
  const uint warps_per_threadblock = 4;
  const uint threadblock_size = warps_per_threadblock * 32;
  const uint kRowTilesPerStep = 2;
  const uint kColTilesPerStep = 1;

  // num tiles - divide rounding up
  uint num_row_tiles = (num_rows + kTileDim - 1) >> kTileDimLog2;
  uint num_col_tiles = (num_cols + kTileDim - 1) >> kTileDimLog2;

  // number of rows and columns after padding
  uint num_rows_after_padding = kTileDim << 1; //TODO here we assume that kTileDim*2 >= num_rows.
  uint num_cols_after_padding = num_col_tiles << kTileDimLog2; //

  uint num_row_steps = num_row_tiles / kRowTilesPerStep;
  uint num_col_steps = num_col_tiles / kColTilesPerStep;

  const uint K_BLOCKS = 8; //todo what is this? is this 128//16, ie num_col_tiles?
  const uint M_BLOCKS = 2; //todo what is this? is this 32//16, ie num_row_tiles?
  const uint SKEW_HALF = ((K_BLOCKS % 2) == 0) ? 8 : 0;
  const uint SMEM_STRIDE = (K_BLOCKS * 16 + SKEW_HALF); //num_cols padded for tensor cores, then padded again for alignment
  // multiple of 2 to guarantee 256-bit alignment for start of the row, at least 16 to safeload a tile
  const uint smem_rows_per_warp = M_BLOCKS << 4; //is this m_blocks * tile dim? how many rows we multiply in a single pass
  const uint smem_elems_per_warp_mat = smem_rows_per_warp * SMEM_STRIDE;
  const uint SKEW_HALF_ACC = ((M_BLOCKS % 2) == 0) ? 8 : 0;
  const uint SMEM_STRIDE_ACC = (M_BLOCKS * 16 + SKEW_HALF_ACC);
  const uint smem_elems_per_warp_acc = M_BLOCKS * 16 * SMEM_STRIDE_ACC * 2;
  const uint smem_elems_per_warp =
      (smem_elems_per_warp_mat > smem_elems_per_warp_acc) ? smem_elems_per_warp_mat : smem_elems_per_warp_acc;
  uint raw_output_size = ((num_rows * (num_rows - 1)) >> 1) + num_cols;
  uint output_size = ((raw_output_size-1)/8 + 1)*8; //round up to multiple of 8
  uint padding_size = output_size-raw_output_size;

  bool float4_predicate = !((num_cols & 7) || (output_size & 7));

  if (float4_predicate) {
    //aligned
    dotBasedInteractFwdKernelFP16<warps_per_threadblock,
                              threadblock_size,
                              M_BLOCKS,
                              K_BLOCKS,
                              SMEM_STRIDE,
                              SMEM_STRIDE_ACC,
                              kWarpSize,
                              kWarpSizeLog2,
                              kTileDim,
                              kTileDimLog2,
                              true>
        <<<(batch_size + warps_per_threadblock - 1) / warps_per_threadblock,
           threadblock_size,
           warps_per_threadblock * smem_elems_per_warp * sizeof(__half), stream>>>((const __half *)input,
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
  } else {
    //not aligned
    dotBasedInteractFwdKernelFP16<warps_per_threadblock,
                                        threadblock_size,
                                        M_BLOCKS,
                                        K_BLOCKS,
                                        SMEM_STRIDE,
                                        SMEM_STRIDE_ACC,
                                        kWarpSize,
                                        kWarpSizeLog2,
                                        kTileDim,
                                        kTileDimLog2,
                                        false>
        <<<(batch_size + warps_per_threadblock - 1) / warps_per_threadblock,
           threadblock_size,
           warps_per_threadblock * smem_elems_per_warp * sizeof(__half), stream>>>((const __half *)input,
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
  }
}

inline void dotBasedInteractFP16Bwd(const void *input,
                                  const void *upstream_grad,
                                  void *grad,
                                  void *bottom_mlp_grad,
                                  uint batch_size,
                                  uint num_rows,
                                  uint num_cols,
                                  cudaStream_t stream) {
  const uint kWarpSize = 32;
  const uint kWarpSizeLog2 = Log2<kWarpSize>::value;
  const uint kTileDim = 16;
  const uint kTileDimLog2 = Log2<kTileDim>::value;
  const uint mem_skew_size = 8;
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
  uint interaction_ugrad_size_with_padding = ((interaction_ugrad_size-1)/8 + 1)*8; //round up to multiple of 8

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
    //aligned
    dotBasedInteractBwdKernelFP16<kWarpsPerBlock,
                              kNumThreads,
                              kRowTilesPerStep,
                              kColTilesPerStep,
                              kWarpSize,
                              kWarpSizeLog2,
                              kTileDim,
                              kTileDimLog2,
                              true>
        <<<num_blocks, kNumThreads, shared_mem_size_bytes, stream>>>((const half *)input,
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
    //unaligned
    dotBasedInteractBwdKernelFP16<kWarpsPerBlock,
                                        kNumThreads,
                                        kRowTilesPerStep,
                                        kColTilesPerStep,
                                        kWarpSize,
                                        kWarpSizeLog2,
                                        kTileDim,
                                        kTileDimLog2,
                                        false>
        <<<num_blocks, kNumThreads, shared_mem_size_bytes, stream>>>((const half *)input,
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

#endif /* FP16_LAUNCHER_CU */

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


#ifndef TF32_LAUNCHER_CU
#define TF32_LAUNCHER_CU

#include "../cuda_kernels/dot_based_interact_tf32.cu"
#include "../launchers/dot_based_interact_fp32_launcher.cu"

inline void dotBasedInteractTF32Fwd(const void *input,
                                   const void *bottom_mlp_output,
                                   void *output,
                                   uint batch_size,
                                   uint num_rows,
                                   uint num_cols,
                                   cudaStream_t stream) {
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

  // const uint &mat_b_num_row_tiles = mat_a_num_col_tiles;
  // const uint &mat_b_num_col_tiles = mat_a_num_row_tiles;

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
           kWarpsPerBlock * smem_elems_per_warp * sizeof(float), stream>>>((const float *)input,
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
    // GENERIC VERSION IS UNFINISHED: Use FP32 instead for now
    dotBasedInteractFP32Fwd(input,
                                 bottom_mlp_output,
                                 output,
                                 batch_size,
                                 num_rows,
                                 num_cols,
                                 stream);
  }
}

inline void dotBasedInteractTF32Bwd(const void *input,
                                   const void *upstream_grad,
                                   void *grad,
                                   void *bottom_mlp_grad,
                                   uint batch_size,
                                   uint num_rows,
                                   uint num_cols,
                                   cudaStream_t stream) {
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
  const uint kWarpsPerBlock = 1;
  const uint kWarpsPerBlockLog2 = Log2<kWarpsPerBlock>::value;
  const uint kNumThreads = kWarpsPerBlock * kWarpSize;

  // num tiles
  uint mat_a_num_row_tiles = (num_rows + kTileLength - 1) >> kTileLengthLog2;
  uint mat_a_num_col_tiles = (num_rows + kTileWidth - 1) >> kTileWidthLog2;

  // const uint &mat_b_num_row_tiles = mat_a_num_col_tiles;
  uint mat_b_num_col_tiles = (num_cols + kTileLength - 1) >> kTileLengthLog2;

  // number of rows and columns after padding
  uint num_rows_after_padding = mat_a_num_row_tiles << kTileLengthLog2;
  uint num_cols_after_padding = mat_b_num_col_tiles << kTileLengthLog2;

  // 2D ugrad size and stride
  uint interaction_ugrad_2D_stride = num_rows_after_padding + kSkewFloat;
  uint interaction_ugrad_2D_size_elems = num_rows_after_padding * interaction_ugrad_2D_stride;

  // 1D ugrad size
  uint interaction_ugrad_size = num_rows * (num_rows - 1) >> 1;
  uint interaction_ugrad_size_with_padding = ((interaction_ugrad_size-1)/8 + 1)*8;

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
        <<<num_blocks, kNumThreads, shared_mem_size_bytes, stream>>>((const float *)input,
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
    // GENERIC VERSION IS UNFINISHED: Use FP32 instead for now
    dotBasedInteractFP32Bwd(input,
                                 upstream_grad,
                                 grad,
                                 bottom_mlp_grad,
                                 batch_size,
                                 num_rows,
                                 num_cols,
                                 stream);
  }
}
#endif /* TF32_LAUNCHER_CU */

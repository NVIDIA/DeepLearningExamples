/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fastertransformer/th_op/weight_quantize_op.h"


namespace {
int index_CUBLASLT_ORDER_COL4_4R2_8C(int col_id, int row_id, int m_32){
  int new_col = col_id >> 5;
  int new_row =   //CUBLASLT_ORDER_COL4_4R2_8C
                  ////row_id/8 is the number of tile of (8 rows 32 columns) -- column-major
                  ////row_id%2 is even row, otherwise odd row
                  ////col_id%COL32_/8 is the number tile of (8 rows 8 columns)
                  (
                  ((((row_id >> 3) << 3) + ((row_id&1) << 2) + ((col_id&31) >> 3)) << 5) +
                  ////col_id%8 >= 4 is the right half of (8 rows 8 columns) tile
                  ////(row_id%8/2) is (the row id of alternating 4 rows) - 1
                  (((((col_id&7) >= 4)?4:0) + ((row_id&7) >> 1)) << 2) +
                  ////col_id%4 is the id of 4 cols
                  (col_id&3)
                  )
                  ;
  return new_col*m_32 + new_row;
}

int index_CUBLASLT_ORDER_COL32_2R_4R4(int col_id, int row_id, int m_32){
  int new_col = col_id >> 5;
  int row_in_tile = row_id & 31;
  int col_in_tile = col_id & 31;
  int new_row =   //CUBLASLT_ORDER_COL32_2R_4R4
                  (
                  ((row_id >> 5) << 10) +
                  //(((row%8)/2*4+row/8)*2+row%2)*32+col
                  (((((((row_in_tile&7)>>1)<<2)+(row_in_tile>>3))<<1)+(row_in_tile&1))<<5)+col_in_tile
                  )
                  ;
  return new_col*m_32 + new_row;
}

//be consistent with FasterTransformer
int8_t float_to_int8_rn_host(float x){
  int8_t res;
  int32_t tmp;
  if (x >= 0){
    tmp = int(x + 0.5);
    tmp = tmp > 127 ? 127 : tmp;
    res = int8_t(tmp);
  }
  else{
    tmp = int(x - 0.5);
    tmp = tmp < -127 ? -127 : tmp;
    res = int8_t(tmp);
  }
  return res;
}

template <typename T>
void quantization_CUBLASLT_ORDER_COL4_4R2_8C(T *dst, float* amaxs, const T* weight, const float* quant_max, const float *quant_min, int n, int k, bool per_channel_quantization){
  //quantization
  int8_t* int8_dst = (int8_t*)dst;
  float element;
  float amax;
  float amax_in_all = fabs(quant_max[0]);
  if (per_channel_quantization){
    for (int i = 0 ; i < n ; i++){
      amaxs[i] = fabs(quant_min[i]);
      if (fabs(quant_max[i]) > amaxs[i])
        amaxs[i] = fabs(quant_max[i]);
      if (amaxs[i] > amax_in_all)
        amax_in_all = amaxs[i];
    }
  }
  if (!per_channel_quantization){
    for (int i = 0 ; i < n ; i++){
      amaxs[i] = amax_in_all;
    }
  }
  int idx_in_COL4;
  int tmp, tmpI;
  for (int col = 0 ; col < k ; col++){
    tmp = col*n;
    for (int row = 0 ; row < n ; row++){
      amax = amaxs[row];
      element = float(weight[tmp+row]);
      idx_in_COL4 = index_CUBLASLT_ORDER_COL4_4R2_8C(col, row, 32*n);
      int8_dst[idx_in_COL4] = float_to_int8_rn_host(element*127.0/amax);
    }
  }
}

template <typename T>
void quantization_CUBLASLT_ORDER_COL32_2R_4R4(T *dst, float* amaxs, const T* weight, const float* quant_max, const float *quant_min, int n, int k, bool per_channel_quantization){
  //quantization
  int8_t* int8_dst = (int8_t*)dst;
  float element;
  float amax;
  float amax_in_all = fabs(quant_max[0]);
  if (per_channel_quantization){
    for (int i = 0 ; i < n ; i++){
      amaxs[i] = fabs(quant_min[i]);
      if (fabs(quant_max[i]) > amaxs[i])
        amaxs[i] = fabs(quant_max[i]);
      if (amaxs[i] > amax_in_all)
        amax_in_all = amaxs[i];
    }
  }
  if (!per_channel_quantization){
    for (int i = 0 ; i < n ; i++){
      amaxs[i] = amax_in_all;
    }
  }
  int idx_in_COL32_2R_4R4;
  int tmp, tmpI;
  for (int col = 0 ; col < k ; col++){
    tmp = col*n;
    for (int row = 0 ; row < n ; row++){
      amax = amaxs[row];
      element = float(weight[tmp+row]);
      idx_in_COL32_2R_4R4 = index_CUBLASLT_ORDER_COL32_2R_4R4(col, row, 32*n);
      int8_dst[idx_in_COL32_2R_4R4] = float_to_int8_rn_host(element*127.0/amax);
    }
  }
}
} // namespace


namespace torch_ext
{
using torch::Tensor;

std::vector<Tensor> weight_quantize(Tensor weight, Tensor quant_max, Tensor quant_min, bool if_per_channel) {
  bool use_ORDER_COL32_2R_4R4 = false;
#ifdef CUDA11_MODE
  int device{-1};
  cudaGetDevice(&device);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);
  if (props.major * 10 + props.minor >= 80){
    use_ORDER_COL32_2R_4R4 = true;
  }
#endif

  CHECK_CPU(weight); CHECK_CONTIGUOUS(weight);
  TORCH_CHECK(weight.dtype()==torch::kFloat32, "weight dtype should be float32");
  TORCH_CHECK(weight.numel()!=0, "weight should not be empty tensor");
  TORCH_CHECK(weight.dim()==2, "Invalid rank. The rank of weight should be 2");

  int k = weight.size(0);
  int n = weight.size(1);

  CHECK_CPU(quant_max); CHECK_CONTIGUOUS(quant_max);
  TORCH_CHECK(quant_max.dtype()==torch::kFloat32, "quant_max dtype should be float32");
  TORCH_CHECK(((if_per_channel&&quant_max.numel()==n)||(!if_per_channel&&quant_max.numel()==1)), "quant_max wrong shape");
  CHECK_CPU(quant_min); CHECK_CONTIGUOUS(quant_min);
  TORCH_CHECK(quant_min.dtype()==torch::kFloat32, "quant_min dtype should be float32");
  TORCH_CHECK(((if_per_channel&&quant_min.numel()==n)||(!if_per_channel&&quant_min.numel()==1)), "quant_max wrong shape");

  const float* weight_ = get_ptr<float>(weight);
  const float* quant_max_ = get_ptr<float>(quant_max);
  const float* quant_min_ = get_ptr<float>(quant_min);

  auto output = torch::empty({k * n}, torch::dtype(torch::kFloat16).device(torch::kCPU).requires_grad(false));
  auto output2 = torch::empty({n}, torch::dtype(torch::kFloat32).device(torch::kCPU).requires_grad(false));

  float* transform_out = get_ptr<float>(output);
  float* transform_out2 = get_ptr<float>(output2);

  if (use_ORDER_COL32_2R_4R4)
    quantization_CUBLASLT_ORDER_COL32_2R_4R4(transform_out, transform_out2, weight_, quant_max_, quant_min_, n, k, if_per_channel);
  else
    quantization_CUBLASLT_ORDER_COL4_4R2_8C(transform_out, transform_out2, weight_, quant_max_, quant_min_, n, k, if_per_channel);
  
  return std::vector<Tensor>{output, output2};
}

} //namespace torch_ext

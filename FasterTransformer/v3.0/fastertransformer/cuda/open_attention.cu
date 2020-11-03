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
/**
* Open sourced multi-head attention
**/

#include "fastertransformer/allocator.h"
#include "fastertransformer/cuda/multi_head_attention.h"
#include "fastertransformer/cuda/open_attention.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
namespace fastertransformer{
namespace cuda{

/**
* Multi-head attetion open sourced
*/
#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
  #pragma unroll
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

/* Calculate the sum of all elements in a block */
template <typename T>
  __inline__ __device__
T blockReduceSum(T val)
{
  static __shared__ T shared[32]; 
  int lane = threadIdx.x & 0x1f; 
  int wid = threadIdx.x >> 5;  

  val = warpReduceSum<T>(val);

  if(lane == 0)
    shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);
                              
  return val;
}

template <typename T>
  __inline__ __device__
T warpReduceMax(T val)
{
  #pragma unroll
  for(int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
  __inline__ __device__
T blockReduceMax(T val)
{
  static __shared__ T shared[32]; 
  int lane = threadIdx.x & 0x1f; // in-warp idx
  int wid = threadIdx.x >> 5;  // warp idx

  val = warpReduceMax(val); // get maxx in each warp

  if(lane == 0) // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();


  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : -1e20f;
  val = warpReduceMax(val);

  return val;
}

  __inline__ __device__
int target_index(int id1, int id2, int id3, int id4, int dim_1, int dim_2, int dim_3, int dim_4)
{
  return id1 * (dim_2 * dim_3 * dim_4) + id3 * (dim_2 * dim_4) + id2 * dim_4 + id4;
}

//build a mapping for fullData to removePaddingData
//grid((valid_word_num+63)/64)
//block(64)
__global__ void mappingRemovePaddingData(int *mapping, const int* sequence_id_offset, const int valid_word_num){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < valid_word_num)
    mapping[idx + __ldg(sequence_id_offset + idx)] = idx;
}

//add_QK_bias_transform for batch int8 cublasLtMatmul & per axis quantization for weight
//1.add QK bias
//2.transform each Q K CUBLASLT_ORDER_COL32 matrixes into a series of sub-matrix (with CUBLASLT_ORDER_COL32/CUBLASLT_ORDER_COL4_4R2_8C layout)
//  Q, K are CUBLASLT_ORDER_COL32 matrixes of m = batch_size * seq_len, n = head_num * size_per_head
//  q_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL32
//  k_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL4_4R2_8C
//only for int32 input & int8 output
//seq_len, size_per_head must be a multiple of 32
//grid.x = batch_size * seq_len * 2;
//block.x = head_num * size_per_head / 4;
//using char4
template <typename T>
__global__
void add_QK_bias_transform(int8_t *q_buf_, int8_t *k_buf_, const int32_t* Q, const T* bias_Q, 
                           const int32_t* K, const T* bias_K, const int m, const int batch_size, 
                           const int seq_len, const int head_num, const int size_per_head, int stride, 
                           const float * q_weight_amax, const float *q_input_deQFactor_div127_ptr, const float * k_weight_amax, 
                           const float *k_input_deQFactor_div127_ptr, const float *q_output_scale_ptr, const float *k_output_scale_ptr)
{
  const int32_t* data_ptr;
  char4* buf_ptr4;
  const T* bias_ptr;
  const float* weight_amax;
  int qk_id = blockIdx.x / m;

  data_ptr = qk_id == 0 ? Q : K;
  buf_ptr4 = qk_id == 0 ? (char4*)q_buf_ : (char4*)k_buf_;
  bias_ptr = qk_id == 0 ? bias_Q : bias_K;
  const float input_deQFactor_div127 = qk_id == 0 ? __ldg(q_input_deQFactor_div127_ptr) : __ldg(k_input_deQFactor_div127_ptr);
  weight_amax = qk_id == 0 ? q_weight_amax : k_weight_amax;
  const float output_scale = qk_id == 0 ? __ldg(q_output_scale_ptr) : __ldg(k_output_scale_ptr);

  int threadIdx4 = threadIdx.x << 2;
  int batch_id = (blockIdx.x % m) / seq_len;
  int head_id = threadIdx4 / size_per_head;
  int id_in_head = threadIdx4 % size_per_head;
  int word_id = blockIdx.x % seq_len;

  int data_id = (((threadIdx4 >> 5) << 5)*m + ((blockIdx.x%m) << 5) + (threadIdx4&31));

  float scale;
  float tmp;
  char4 tmp4;
  scale = static_cast<float>(__ldg(data_ptr+data_id)) * __ldg(weight_amax+threadIdx4) * input_deQFactor_div127;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.x = float_to_int8_rn(tmp*output_scale);

  data_id = data_id+1;
  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(__ldg(data_ptr+data_id)) * __ldg(weight_amax+threadIdx4)* input_deQFactor_div127;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.y = float_to_int8_rn(tmp*output_scale);

  data_id = data_id+1;
  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(__ldg(data_ptr+data_id)) * __ldg(weight_amax+threadIdx4) * input_deQFactor_div127;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.z = float_to_int8_rn(tmp*output_scale);

  data_id = data_id+1;
  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(__ldg(data_ptr+data_id)) * __ldg(weight_amax+threadIdx4) * input_deQFactor_div127;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.w = float_to_int8_rn(tmp*output_scale);


  //row_id, col_id of sub-matrix (m = seq_len, n = size_per_head), column-major

  int row_id = word_id;
  int col_id = id_in_head;
  //new (row, rol) of LtTrans COL32/COL4 sub-matrix, leading dim = (COL32_ * seq_len)
  int new_col = col_id >> 5;
  int new_row = (qk_id != 1) ?
                  //COL32
                  ((row_id << 5) + (col_id&31))
               :
                  //COL4
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
  buf_ptr4[(((batch_id*head_num + head_id) * stride + (new_col << 5)*seq_len + new_row) >> 2)] = tmp4;
}

//add_QK_bias_transform & rebuild padding for batch int8 cublasLtMatmul & per axis quantization for weight
//1.add QK bias
//2.transform each Q K CUBLASLT_ORDER_COL32 matrixes into a series of sub-matrix (with CUBLASLT_ORDER_COL32/CUBLASLT_ORDER_COL4_4R2_8C layout)
//  Q, K are CUBLASLT_ORDER_COL32 matrixes of m = valid_word_num, n = head_num * size_per_head
//  q_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL32
//  k_buf_ is of batchCount = batch_size * head_num, m = seq_len, n = size_per_head, CUBLASLT_ORDER_COL4_4R2_8C
//only for int32 input & int8 output
//seq_len, size_per_head must be a multiple of 32
//grid.x = valid_word_num * 2;
//block.x = head_num * size_per_head / 4;
//using char4
template <typename T>
__global__
void add_QK_bias_transform_rebuild_padding(int8_t *q_buf_, int8_t *k_buf_, const int32_t* Q, const T* bias_Q, 
                                           const int32_t* K, const T* bias_K, const int* sequence_id_offset, 
                                           const int valid_word_num, const int m, const int batch_size, const int seq_len, 
                                           const int head_num, const int size_per_head, int stride, const float * q_weight_amax, 
                                           const float *q_input_deQFactor_div127_ptr, const float * k_weight_amax, 
                                           const float *k_input_deQFactor_div127_ptr, const float *q_output_scale_ptr, const float *k_output_scale_ptr)
{
  const int32_t* data_ptr;
  char4* buf_ptr4;
  const T* bias_ptr;
  const float* weight_amax;
  int qk_id = blockIdx.x / valid_word_num;

  data_ptr = qk_id == 0 ? Q : K;
  buf_ptr4 = qk_id == 0 ? (char4*)q_buf_ : (char4*)k_buf_;
  bias_ptr = qk_id == 0 ? bias_Q : bias_K;
  
  int threadIdx4 = threadIdx.x << 2;
  int m_full_idx = blockIdx.x % valid_word_num;
  m_full_idx = (valid_word_num != m) ? (m_full_idx + __ldg(sequence_id_offset+m_full_idx)) : m_full_idx;
  int batch_id = m_full_idx / seq_len;
  int head_id = threadIdx4 / size_per_head;
  int id_in_head = threadIdx4 % size_per_head;
  int word_id = m_full_idx % seq_len;
  
  const float input_deQFactor_div127 = qk_id == 0 ? __ldg(q_input_deQFactor_div127_ptr) : __ldg(k_input_deQFactor_div127_ptr);
  weight_amax = qk_id == 0 ? q_weight_amax : k_weight_amax;
  const float output_scale = qk_id == 0 ? __ldg(q_output_scale_ptr) : __ldg(k_output_scale_ptr);

  int data_id = (((threadIdx4 >> 5) << 5)*valid_word_num + ((blockIdx.x%valid_word_num) << 5) + (threadIdx4&31));
    
  float scale;
  float tmp;
  char4 tmp4;
  scale = static_cast<float>(__ldg(data_ptr+data_id)) * __ldg(weight_amax+threadIdx4) * input_deQFactor_div127;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.x = float_to_int8_rn(tmp*output_scale);

  data_id = data_id+1;
  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(__ldg(data_ptr+data_id)) * __ldg(weight_amax+threadIdx4)* input_deQFactor_div127;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.y = float_to_int8_rn(tmp*output_scale);

  data_id = data_id+1;
  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(__ldg(data_ptr+data_id)) * __ldg(weight_amax+threadIdx4) * input_deQFactor_div127;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.z = float_to_int8_rn(tmp*output_scale);

  data_id = data_id+1;
  threadIdx4 = threadIdx4+1;
  scale = static_cast<float>(__ldg(data_ptr+data_id)) * __ldg(weight_amax+threadIdx4) * input_deQFactor_div127;
  tmp = static_cast<float>(__ldg(bias_ptr+threadIdx4)) + scale;
  tmp4.w = float_to_int8_rn(tmp*output_scale);

  //row_id, col_id of sub-matrix (m = seq_len, n = size_per_head), column-major
  int row_id = word_id;
  int col_id = id_in_head;
  //new (row, rol) of LtTrans COL32/COL4 sub-matrix, leading dim = (COL32_ * seq_len)
  int new_col = col_id >> 5;
  int new_row = (qk_id != 1) ?
                  //COL32
                  ((row_id << 5) + (col_id&31))
               :
                  //COL4
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
  buf_ptr4[(((batch_id*head_num + head_id) * stride + (new_col << 5)*seq_len + new_row) >> 2)] = tmp4;
}

//input matrix a matrix of m = batch_size*seq_len , n = head_num*size_per_head, CUBLASLT_ORDER_COL32
//output matrixes are a series of sub-matrixes with size of m = size_per_head, n = seq_len , CUBLASLT_ORDER_COL4_4R2_8C
//only for int32_t Input int8_t Output
//seq_len, size_per_head must be a multiple of 32
//grid = (size_per_head/32, seq_len/32, batch_size*head_num)
//block = (8, 32);
//using char4
//per axis quantization for weight
template <typename T>
__global__
void add_V_bias_transform(int8_t *v_buf_, const int32_t *V, const T *V_bias, const int batch_size, const int seq_len, 
                          const int head_num, const int size_per_head, int stride, const float* weight_amax, 
                          const float *input_deQFactor_div127_ptr, const float *out_scale_ptr)
{
  const float input_deQFactor_div127 = __ldg(input_deQFactor_div127_ptr);
  const float out_scale = __ldg(out_scale_ptr);
  __shared__ int8_t shm[32][33];
  const int32_t* data_ptr = V;
  char4* buf_ptr4 = (char4*) v_buf_;
  const T* bias_ptr = V_bias;

  int threadIdx4 = threadIdx.x << 2;

  //for src of (seq_len, size_per_head)
  int batch_id = blockIdx.z/head_num;
  int head_id = blockIdx.z%head_num;
  int word_id = (blockIdx.y << 5) + threadIdx.y;
  int id_in_size = (blockIdx.x << 5) + threadIdx4;

  //for V layout (batch_size*seq_len, head_num*size_per_head)
  int col = head_id*size_per_head + id_in_size;
  int row = batch_id*seq_len + word_id;
  int inIdx = (((col >> 5) << 5)*batch_size*seq_len + ((row << 5) + (col&31)));
  //for shm row-major
  int sh_col = threadIdx4;
  int sh_row = threadIdx.y;
  
  float tmp;
  float scale;

  //const half2* bias_ptr2 = (const half2*)bias_ptr;
  //half2 tmp2;

  //tmp2 = __ldg(&bias_ptr2[col >> 1]);
  
  scale = __ldg(data_ptr + inIdx) * __ldg(weight_amax + col) * input_deQFactor_div127;
  tmp = scale + static_cast<float>(__ldg(bias_ptr + col));//(tmp2.x);
  shm[sh_row][sh_col] = float_to_int8_rn(tmp*out_scale);
  
  scale = __ldg(data_ptr + inIdx + 1) * __ldg(weight_amax + col + 1) * input_deQFactor_div127;
  tmp = scale + static_cast<float>(__ldg(bias_ptr+col+1));//(tmp2.y);
  shm[sh_row][sh_col+1] = float_to_int8_rn(tmp*out_scale);
  
  //tmp2 = __ldg(&bias_ptr2[(col >> 1) + 1]);

  scale = __ldg(data_ptr+inIdx+2) * __ldg(weight_amax+col+2) * input_deQFactor_div127;
  tmp = scale + static_cast<float>(__ldg(bias_ptr+col+2));//(tmp2.x);
  shm[sh_row][sh_col+2] = float_to_int8_rn(tmp*out_scale);
  
  scale = __ldg(data_ptr+inIdx + 3) * __ldg(weight_amax+col+3) * input_deQFactor_div127;
  tmp = scale + static_cast<float>(__ldg(bias_ptr+col+3));//(tmp2.y);
  shm[sh_row][sh_col+3] = float_to_int8_rn(tmp*out_scale);

  __syncthreads();

  //for dst of (size_per_head, seq_len)
  word_id = (blockIdx.y << 5) + threadIdx4;
  id_in_size = (blockIdx.x << 5) + threadIdx.y;
  col = (word_id >> 5);
  row = (
        //COL4
        ////id_in_size/8 is the number of tile of (8 rows 32 columns) -- column-major
        ////id_in_size%2 is even row, otherwise odd row
        ////word_id%COL32_/8 is the number tile of (8 rows 8 columns)
        ((((id_in_size >> 3) << 3) + ((id_in_size&1) << 2) + ((word_id&31) >> 3)) << 5) +
        ////word_id%8 >= 4 is the right half of (8 rows 8 columns) tile
        ////(id_in_size%8/2) is (the row id of alternating 4 rows) - 1
        (((((word_id&7) >= 4)?4:0) + ((id_in_size&7) >> 1)) << 2) +
        ////word_id%4 is the id of 4 cols
        (word_id&3)
        );
        
  char4 dataTmp;
  dataTmp.x = shm[sh_col][sh_row];
  dataTmp.y = shm[sh_col+1][sh_row];
  dataTmp.z = shm[sh_col+2][sh_row];
  dataTmp.w = shm[sh_col+3][sh_row];
  buf_ptr4[(blockIdx.z*stride + (col << 5)*size_per_head + row) >> 2] = dataTmp;
}

template <>
__global__
void add_V_bias_transform(int8_t *v_buf_, const int32_t *V, const half *V_bias, const int batch_size, const int seq_len, 
                          const int head_num, const int size_per_head, int stride, const float* weight_amax, 
                          const float *input_deQFactor_div127_ptr, const float *out_scale_ptr)
{
  const float input_deQFactor_div127 = __ldg(input_deQFactor_div127_ptr);
  const float out_scale = __ldg(out_scale_ptr);
  __shared__ int8_t shm[32][33];
  const int32_t* data_ptr = V;
  char4* buf_ptr4 = (char4*) v_buf_;

  int threadIdx4 = threadIdx.x << 2;

  //for src of (seq_len, size_per_head)
  int batch_id = blockIdx.z/head_num;
  int head_id = blockIdx.z%head_num;
  
  int blockIdy32 = (blockIdx.y << 5);
  int blockIdx32 = (blockIdx.x << 5);
  int word_id = blockIdy32 + threadIdx.y;
  int id_in_size = blockIdx32 + threadIdx4;

  //for V layout (batch_size*seq_len, head_num*size_per_head)
  int col = head_id*size_per_head + id_in_size;
  int row = batch_id*seq_len + word_id;
  int inIdx = ((col & 0xffffffe0)*batch_size*seq_len + ((row << 5) + (col&31)));
  //for shm row-major
  int sh_col = threadIdx4;
  int sh_row = threadIdx.y;
  
  int col_2 = col >> 1;
  float scale;

  const half2* bias_ptr2 = (const half2*)V_bias;
  half2 tmp2;

  tmp2 = __ldg(bias_ptr2+col_2);
  
  scale = __ldg(data_ptr+inIdx) * __ldg(weight_amax+col) * input_deQFactor_div127;
  scale = scale + static_cast<float>(tmp2.x);
  shm[sh_row][sh_col] = float_to_int8_rn(scale*out_scale);
  
  scale = __ldg(data_ptr+inIdx+1) * __ldg(weight_amax+col+1) * input_deQFactor_div127;
  scale = scale + static_cast<float>(tmp2.y);
  shm[sh_row][sh_col+1] = float_to_int8_rn(scale*out_scale);
  
  tmp2 = __ldg(bias_ptr2 + col_2 + 1);

  scale = __ldg(data_ptr + inIdx + 2) * __ldg(weight_amax + col + 2) * input_deQFactor_div127;
  scale = scale + static_cast<float>(tmp2.x);
  shm[sh_row][sh_col+2] = float_to_int8_rn(scale*out_scale);
  
  scale = __ldg(data_ptr + inIdx + 3) * __ldg(weight_amax + col + 3) * input_deQFactor_div127;
  scale = scale + static_cast<float>(tmp2.y);
  shm[sh_row][sh_col+3] = float_to_int8_rn(scale*out_scale);

  __syncthreads();

  //for dst of (size_per_head, seq_len)
  word_id = blockIdy32 + threadIdx4;
  id_in_size = blockIdx32 + threadIdx.y;
  col = (word_id >> 5);
  row = (
        //COL4
        ////id_in_size/8 is the number of tile of (8 rows 32 columns) -- column-major
        ////id_in_size%2 is even row, otherwise odd row
        ////word_id%COL32_/8 is the number tile of (8 rows 8 columns)
        (((id_in_size & 0xfffffff8) + ((id_in_size&1) << 2) + ((word_id&31) >> 3)) << 5) +
        ////word_id%8 >= 4 is the right half of (8 rows 8 columns) tile
        ////(id_in_size%8/2) is (the row id of alternating 4 rows) - 1
        (((((word_id&7) >= 4)?4:0) + ((id_in_size&7) >> 1)) << 2) +
        ////word_id%4 is the id of 4 cols
        (word_id&3)
        );
        
  char4 dataTmp;
  dataTmp.x = shm[sh_col][sh_row];
  dataTmp.y = shm[sh_col+1][sh_row];
  dataTmp.z = shm[sh_col+2][sh_row];
  dataTmp.w = shm[sh_col+3][sh_row];
  buf_ptr4[(blockIdx.z*stride + (col << 5)*size_per_head + row) >> 2] = dataTmp;
}

//add bias into V & rebuild padding 
//input matrix a matrix of m = valid_word_num, n = head_num*size_per_head, CUBLASLT_ORDER_COL32
//output matrixes are a series of sub-matrixes with size of m = size_per_head, n = seq_len , CUBLASLT_ORDER_COL4_4R2_8C
//only for int32_t Input int8_t Output
//seq_len, size_per_head must be a multiple of 32
//grid = (size_per_head/32, seq_len/32, batch_size*head_num)
//block = (8, 32);
//using char4
//per axis quantization for weight
template <typename T>
__global__
void add_V_bias_transform_rebuild_padding(int8_t *v_buf_, const int32_t *V, const T *V_bias, const int* sequence_id_map, const int valid_word_num, 
                                          const int batch_size, const int seq_len, const int head_num, const int size_per_head, int stride, 
                                          const float* weight_amax, const float *input_deQFactor_div127_ptr, const float *out_scale_ptr)
{
  __shared__ int8_t shm[32][33];
  const int32_t* data_ptr = V;
  char4* buf_ptr4 = (char4*) v_buf_;
  const T* bias_ptr = V_bias;

  int threadIdx4 = threadIdx.x << 2;

  //for src of (seq_len, size_per_head)
  int batch_id = blockIdx.z/head_num;
  int head_id = blockIdx.z%head_num;
  int word_id = (blockIdx.y << 5) + threadIdx.y;
  int id_in_size = (blockIdx.x << 5) + threadIdx4;

  //for shm row-major
  int sh_col = threadIdx4;
  int sh_row = threadIdx.y;
  
  //for V layout (batch_size*seq_len, head_num*size_per_head)
  int col;
  int row = __ldg(sequence_id_map + batch_id*seq_len + word_id);
  
  if (row != -1){
    col = head_id*size_per_head + id_in_size;  
    int inIdx = ((col & 0xffffffe0)*valid_word_num + ((row << 5) + (col&31)));
  
    float tmp;
    float scale;
  
    const float input_deQFactor_div127 = __ldg(input_deQFactor_div127_ptr);
    const float out_scale = __ldg(out_scale_ptr);
  
    scale = __ldg(data_ptr + inIdx) * __ldg(weight_amax + col) * input_deQFactor_div127;
    tmp = scale + static_cast<float>(__ldg(bias_ptr + col));
    shm[sh_row][sh_col] = float_to_int8_rn(tmp*out_scale);
  
    scale = __ldg(data_ptr + inIdx + 1) * __ldg(weight_amax + col + 1) * input_deQFactor_div127;
    tmp = scale + static_cast<float>(__ldg(bias_ptr+col+1));
    shm[sh_row][sh_col+1] = float_to_int8_rn(tmp*out_scale);

    scale = __ldg(data_ptr+inIdx+2) * __ldg(weight_amax+col+2) * input_deQFactor_div127;
    tmp = scale + static_cast<float>(__ldg(bias_ptr+col+2));
    shm[sh_row][sh_col+2] = float_to_int8_rn(tmp*out_scale);
  
    scale = __ldg(data_ptr+inIdx + 3) * __ldg(weight_amax+col+3) * input_deQFactor_div127;
    tmp = scale + static_cast<float>(__ldg(bias_ptr+col+3));
    shm[sh_row][sh_col+3] = float_to_int8_rn(tmp*out_scale);
  }
  else{
    shm[sh_row][sh_col] = shm[sh_row][sh_col + 1] = shm[sh_row][sh_col + 2] = shm[sh_row][sh_col + 3] = 0;
  }
  __syncthreads();

  char4 dataTmp;
  dataTmp.x = shm[sh_col][sh_row];
  dataTmp.y = shm[sh_col+1][sh_row];
  dataTmp.z = shm[sh_col+2][sh_row];
  dataTmp.w = shm[sh_col+3][sh_row];

  //for dst of (size_per_head, seq_len)
  word_id = (blockIdx.y << 5) + threadIdx4;
  id_in_size = (blockIdx.x << 5) + threadIdx.y;
  col = (word_id >> 5);
  row = (
        //COL4
        ////id_in_size/8 is the number of tile of (8 rows 32 columns) -- column-major
        ////id_in_size%2 is even row, otherwise odd row
        ////word_id%COL32_/8 is the number tile of (8 rows 8 columns)
        (((id_in_size & 0xfffffff8) + ((id_in_size&1) << 2) + ((word_id&31) >> 3)) << 5) +
        ////word_id%8 >= 4 is the right half of (8 rows 8 columns) tile
        ////(id_in_size%8/2) is (the row id of alternating 4 rows) - 1
        (((((word_id&7) >= 4)?4:0) + ((id_in_size&7) >> 1)) << 2) +
        ////word_id%4 is the id of 4 cols
        (word_id&3)
        );
        
  buf_ptr4[(blockIdx.z*stride + (col << 5)*size_per_head + row) >> 2] = dataTmp;
}

template <>
__global__
void add_V_bias_transform_rebuild_padding(int8_t *v_buf_, const int32_t *V, const half *V_bias, const int* sequence_id_map, const int valid_word_num, 
                                          const int batch_size, const int seq_len, const int head_num, const int size_per_head, int stride, 
                                          const float* weight_amax, const float *input_deQFactor_div127_ptr, const float *out_scale_ptr)
{
  __shared__ int8_t shm[32][33];
  const int32_t* data_ptr = V;
  char4* buf_ptr4 = (char4*) v_buf_;

  int threadIdx4 = threadIdx.x << 2;

  //for src of (seq_len, size_per_head)
  int batch_id = blockIdx.z/head_num;
  int head_id = blockIdx.z%head_num;
  
  int blockIdy32 = (blockIdx.y << 5);
  int blockIdx32 = (blockIdx.x << 5);
  int word_id = blockIdy32 + threadIdx.y;
  int id_in_size = blockIdx32 + threadIdx4;

  
  //for shm row-major
  int sh_col = threadIdx4;
  int sh_row = threadIdx.y;
  
  //for V layout (batch_size*seq_len, head_num*size_per_head)
  int col;
  int row = __ldg(sequence_id_map + batch_id*seq_len + word_id);
  
  if (row >= 0){
    const float input_deQFactor_div127 = __ldg(input_deQFactor_div127_ptr);
    const float out_scale = __ldg(out_scale_ptr);
    col = head_id*size_per_head + id_in_size;
    int inIdx = ((col & 0xffffffe0)*valid_word_num + ((row << 5) + (col&31)));
    int col_2 = col >> 1;
    float scale;

    const half2* bias_ptr2 = (const half2*)V_bias;
    half2 tmp2;

    tmp2 = __ldg(bias_ptr2+col_2);
  
    scale = __ldg(data_ptr+inIdx) * __ldg(weight_amax+col) * input_deQFactor_div127;
    scale = scale + static_cast<float>(tmp2.x);
    shm[sh_row][sh_col] = float_to_int8_rn(scale*out_scale);
  
    scale = __ldg(data_ptr+inIdx+1) * __ldg(weight_amax+col+1) * input_deQFactor_div127;
    scale = scale + static_cast<float>(tmp2.y);
    shm[sh_row][sh_col+1] = float_to_int8_rn(scale*out_scale);
  
    tmp2 = __ldg(bias_ptr2 + col_2 + 1);

    scale = __ldg(data_ptr + inIdx + 2) * __ldg(weight_amax + col + 2) * input_deQFactor_div127;
    scale = scale + static_cast<float>(tmp2.x);
    shm[sh_row][sh_col+2] = float_to_int8_rn(scale*out_scale);
  
    scale = __ldg(data_ptr + inIdx + 3) * __ldg(weight_amax + col + 3) * input_deQFactor_div127;
    scale = scale + static_cast<float>(tmp2.y);
    shm[sh_row][sh_col+3] = float_to_int8_rn(scale*out_scale);
  }
  else{
    shm[sh_row][sh_col] = shm[sh_row][sh_col + 1] = shm[sh_row][sh_col + 2] = shm[sh_row][sh_col + 3] = 0;
  }
  __syncthreads();

  char4 dataTmp;
  dataTmp.x = shm[sh_col][sh_row];
  dataTmp.y = shm[sh_col+1][sh_row];
  dataTmp.z = shm[sh_col+2][sh_row];
  dataTmp.w = shm[sh_col+3][sh_row];

  //for dst of (size_per_head, seq_len)
  word_id = blockIdy32 + threadIdx4;
  id_in_size = blockIdx32 + threadIdx.y;
  col = (word_id >> 5);
  row = (
        //COL4
        ////id_in_size/8 is the number of tile of (8 rows 32 columns) -- column-major
        ////id_in_size%2 is even row, otherwise odd row
        ////word_id%COL32_/8 is the number tile of (8 rows 8 columns)
        (((id_in_size & 0xfffffff8) + ((id_in_size&1) << 2) + ((word_id&31) >> 3)) << 5) +
        ////word_id%8 >= 4 is the right half of (8 rows 8 columns) tile
        ////(id_in_size%8/2) is (the row id of alternating 4 rows) - 1
        (((((word_id&7) >= 4)?4:0) + ((id_in_size&7) >> 1)) << 2) +
        ////word_id%4 is the id of 4 cols
        (word_id&3)
        );
        
  buf_ptr4[(blockIdx.z*stride + (col << 5)*size_per_head + row) >> 2] = dataTmp;
}


template<typename T>
__global__
void add_QKV_bias(T* Q, const T* bias_Q, T* K, const T* bias_K, T* V, const T* bias_V, T* q_buf_, T* k_buf_, T* v_buf_, 
  const int batch_size, const int seq_len, const int head_num, const int size_per_head, const int word_per_block)
{

  T* data_ptr;
  T* buf_ptr;
  const T* bias_ptr;
  
  int m = batch_size * seq_len;
  int n = head_num * size_per_head;

  int qkv_id = blockIdx.x * word_per_block / m;
  int row_offset = (blockIdx.x * word_per_block % m) * n;

  if(qkv_id == 0)
  {
    data_ptr = Q + row_offset;
    buf_ptr = q_buf_;
    bias_ptr = bias_Q;
  }
  else if(qkv_id == 1)
  {
    data_ptr = K + row_offset;
    buf_ptr = k_buf_;
    bias_ptr = bias_K;
  }
  else
  {
    data_ptr = V + row_offset;
    buf_ptr = v_buf_;
    bias_ptr = bias_V;
  }

  int batch_id = (blockIdx.x * word_per_block % m) / seq_len;
  int head_id = threadIdx.x / size_per_head;
  int id_in_head = threadIdx.x % size_per_head;
  int word_start_id = (blockIdx.x * word_per_block) % seq_len;

  T bias = __ldg(&bias_ptr[threadIdx.x]);

  for(int i = word_start_id; i < word_start_id + word_per_block; ++i)
  {
    T tmp = data_ptr[threadIdx.x] + bias;

    int target_id = batch_id * (seq_len * head_num * size_per_head) + head_id * seq_len * size_per_head + 
      i * size_per_head + id_in_head;

    buf_ptr[target_id] = tmp;
    data_ptr += n;
  }
}

template <>
__global__
void add_QKV_bias(half* Q, const half* bias_Q, half* K, const half* bias_K, half* V, const half* bias_V, 
  half* q_buf_, half* k_buf_, half* v_buf_, 
  const int batch_size, const int seq_len, const int head_num, const int size_per_head, const int word_per_block)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_id = tid / (head_num * seq_len * size_per_head);
  int seq_id = (tid % (head_num * seq_len * size_per_head)) / (head_num * size_per_head);
  int head_id = (tid % (head_num * size_per_head)) / size_per_head;
  int id = tid % size_per_head;
  int target_id = target_index(batch_id, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);

  int bias_id = threadIdx.x;

  half2* src_ptr = (half2*)Q;
  half2* dst_ptr = (half2*)q_buf_;
  const half2* bias_ptr = (const half2*)bias_Q;

  dst_ptr[target_id] = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));

  src_ptr = (half2*)K;
  dst_ptr = (half2*)k_buf_;
  bias_ptr = (const half2*)bias_K;
  dst_ptr[target_id] = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));

  src_ptr = (half2*)V;
  dst_ptr = (half2*)v_buf_;
  bias_ptr = (const half2*)bias_V;
  dst_ptr[target_id] = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));
}

template<typename T>
__global__
void add_QKV_bias_rebuild_padding(T* Q, const T* bias_Q, T* K, const T* bias_K, T* V, const T* bias_V, T* q_buf_, T* k_buf_, T* v_buf_, 
  const int batch_size, const int seq_len, const int head_num, const int size_per_head, const int* mask_offset)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int bdim = blockDim.x;

  const int tgt_batch_id = (bid + mask_offset[bid]) / seq_len;
  const int tgt_seq_id = (bid + mask_offset[bid]) % seq_len;
  const int tgt_head_id = tid / size_per_head;
  const int tgt_hidden_id = tid % size_per_head;

  const int src_id = bid * bdim + tid;
  const int tgt_id = tgt_batch_id * head_num * seq_len * size_per_head + \
                    tgt_head_id * seq_len * size_per_head + \
                    tgt_seq_id * size_per_head + \
                    tgt_hidden_id;
  
  q_buf_[tgt_id] = Q[src_id] + bias_Q[tid];
  k_buf_[tgt_id] = K[src_id] + bias_K[tid];
  v_buf_[tgt_id] = V[src_id] + bias_V[tid];
}

template <typename T>
__global__
void softmax_kernel(T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num, const int seq_len, 
  const T scalar)
{
    int batch_id = blockIdx.x / head_num;
    int qk_offset = blockIdx.x * seq_len * seq_len;
    int mask_offset = batch_id * seq_len * seq_len;

    __shared__ float s_sum, s_max;

    for(int i = 0; i < seq_len; ++i)
    {
      float qk = threadIdx.x < seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
      float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x + mask_offset] : 0.0f;
      
      mask_val = (1.0f - mask_val) * -10000.0f;

      float tmp = threadIdx.x < seq_len ? (float)(qk * (float)scalar + mask_val): -1e20f;

      float max_val = blockReduceMax<float>(tmp);

      if(threadIdx.x == 0)
        s_max = max_val;
      __syncthreads();

      qk = threadIdx.x < seq_len ? __expf(tmp - s_max) : 0.0f;

      float sum_val = blockReduceSum<float>(qk);

      if(threadIdx.x == 0)
      {
        s_sum = sum_val + 1e-6f;
      }
      __syncthreads();

      if(threadIdx.x < seq_len)
        qk_buf_[threadIdx.x + qk_offset] = (T)(qk / s_sum);

      qk_offset += seq_len;
      mask_offset += seq_len;
    }
}


template <typename T>
__global__
void softmax_kernel_v2(T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num, 
  const int seq_len, const float scalar)
{
    int batch_id = blockIdx.x / head_num / seq_len;
    int seq_id = blockIdx.x % seq_len;
    int qk_offset = blockIdx.x * seq_len;
    int mask_offset = batch_id * seq_len * seq_len + seq_id * seq_len;

    __shared__ float s_sum, s_max;

    float qk = threadIdx.x < seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
    float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x + mask_offset] : 0.0f;
      
    mask_val = (1.0f - mask_val) * -10000.0f;

    float tmp = threadIdx.x < seq_len ? (float)(qk * (float)scalar + mask_val) : -1e20f;
    float max_val = blockReduceMax<float>(tmp);
    if(threadIdx.x == 0)
      s_max = max_val;
    __syncthreads();

    float qk_tmp = threadIdx.x < seq_len ? __expf((float)(tmp - s_max)) : 0.0f;
    float sum_val = blockReduceSum<float>(qk_tmp);

    if(threadIdx.x == 0)
    {
      s_sum = sum_val + 1e-6f;
    }
    __syncthreads();

    if(threadIdx.x < seq_len)
      qk_buf_[threadIdx.x + qk_offset] = (T)(qk_tmp / s_sum);
}

//grid = (seq_len/word_per_thread, batch_size, head_num)
//block.x = max(32, (seq_len + 31)/32*32)
template <typename T>
__global__
void softmax_kernel_v3(T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num, const int seq_len, const T scalar)
{
    
  bool qual = threadIdx.x < seq_len;
  for (int seq_id = blockIdx.x ; seq_id < seq_len ; seq_id += gridDim.x){
    float tmp = -1e20f;
    int qk_offset;
    __shared__ float s_mean, s_max;
    if (qual){
      qk_offset = ((blockIdx.y*head_num + blockIdx.z)*seq_len + seq_id) *seq_len + threadIdx.x;
      int mask_offset = (blockIdx.y * seq_len + seq_id) * seq_len + threadIdx.x;

      float qk = static_cast<float>(qk_buf_[qk_offset]);
      float mask_val = static_cast<float>(__ldg(&attr_mask[mask_offset]));

      mask_val = (1.0f - mask_val) * -10000.0f;

      tmp = qk * static_cast<float>(scalar) + mask_val;
    }

    float max_val = blockReduceMax<float>(tmp);
    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();
    
    float qk_tmp = qual ? __expf(tmp - s_max) : 0.0f;
    float sum_val = blockReduceSum<float>(qk_tmp);
    if (threadIdx.x == 0){
      s_mean = sum_val + 1e-6f;
      s_mean = __fdividef(1.0f, s_mean);
    }
    __syncthreads();
    
    if(qual)
      qk_buf_[qk_offset] = (T)(qk_tmp * s_mean);
  }
}  


//grid = (seq_len/word_per_thread, batch_size, head_num)
//block.x = max(32, (seq_len/2 + 31)/32*32)
//seq_len % 2 == 0
template <>
__global__
void softmax_kernel_v3(half* qk_buf_, const half* attr_mask, 
                      const int batch_size, const int head_num, 
                      const int seq_len, const half scalar)
{
  int threadIdx2 = threadIdx.x << 1;
  bool qual = threadIdx2 < seq_len;
  half2* qk_buf_half2Ptr = (half2*) qk_buf_;
  const half2* attr_mask_half2Ptr = (const half2*) attr_mask;
  __shared__ float s_mean, s_max;
  for (int seq_id = blockIdx.x ; seq_id < seq_len ; seq_id += gridDim.x){
    int qk_offset;
    half2 tmp = __float2half2_rn(0.0f);

    float max_val = -1e20f;
    half2 qk;
    if (qual){ 
      qk_offset = ((((blockIdx.y*head_num + blockIdx.z)*seq_len + seq_id) *seq_len) >> 1) + threadIdx.x;
      int mask_offset = (((blockIdx.y * seq_len + seq_id) * seq_len) >> 1) + threadIdx.x;

      qk = qk_buf_half2Ptr[qk_offset];
      half2 mask_val = __ldg(&attr_mask_half2Ptr[mask_offset]);
      half2 mask_val_tmp = __hmul2(__hsub2(__float2half2_rn(1.0f), mask_val), __float2half2_rn(-10000.0f));
      tmp = __hadd2(__hmul2(__half2half2(scalar), qk), mask_val_tmp);
      max_val = fmax((float)tmp.x, (float)tmp.y);
    }
    
    max_val = blockDim.x <= 32 ? warpReduceMax(max_val) : blockReduceMax<float>(max_val);

    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();
    
    if (qual){
      tmp = h2exp(__hsub2(tmp, __float2half2_rn(s_max)));
    }
    float sum_val = blockDim.x <= 32 ? warpReduceSum((float)(tmp.x + tmp.y)) : blockReduceSum<float>((float)(tmp.x + tmp.y));

    if (threadIdx.x == 0){
      s_mean = sum_val + 1e-6f;
      s_mean = __fdividef(1.0f, s_mean);
    }
    __syncthreads();

    if(qual){
      qk = __hmul2(tmp, __float2half2_rn(s_mean));
      qk_buf_half2Ptr[qk_offset] = qk;
    }
  }
}

//grid = (seq_len/word_per_thread, batch_size, head_num)
//block.x = max(32, (seq_len + 31)/32*32)
//for seq_len not larger than 32
template <typename T>
__global__
void softmax_kernel_v3_LE32(T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num, const int seq_len, const T scalar)
{
  bool qual = threadIdx.x < seq_len;
  for (int seq_id = blockIdx.x ; seq_id < seq_len ; seq_id += gridDim.x){
    int qk_offset;
    __shared__ float s_mean, s_max;
    float tmp = -1e20f;
    if (qual){
      qk_offset = ((blockIdx.y*head_num + blockIdx.z)*seq_len + seq_id) *seq_len + threadIdx.x;
      int mask_offset = (blockIdx.y * seq_len + seq_id) * seq_len + threadIdx.x;

      float qk = static_cast<float>(qk_buf_[qk_offset]);
      float mask_val = static_cast<float>(__ldg(&attr_mask[mask_offset]));

      mask_val = (1.0f - mask_val) * -10000.0f;

      tmp = static_cast<float>(qk) * static_cast<float>(scalar) + mask_val;
    }
    float max_val = warpReduceMax<float>(tmp);

    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();

    tmp = qual ? __expf(tmp - s_max) : 0.0f;
    float sum_val = warpReduceSum<float>(tmp);

    if (threadIdx.x == 0){
      s_mean = sum_val + 1e-6f;
      s_mean = __fdividef(1.0f, s_mean);
    }
    __syncthreads();

    if(qual)
      qk_buf_[qk_offset] = (T)(tmp * s_mean);
  }
}

//int_buf are a series of sub-matrixes of m = seq_len, n = seq_len, CUBLASLT_ORDER_COL32
//grid = (seq_len, batch_size, head_num)
//block.x = max(32, (seq_len/4 + 31)/32*32)
//for int32_t I; int8 O;
template <typename T>
__global__
void softmax_COL32(int8_t* qk_buf_, const int32_t* int_buf, const T* attr_mask, const int batch_size, 
                   const int head_num, const int seq_len, const float scalar1a, const float *scalar1b, 
                   const float *scalar1c, const float *amax_ptr, const int head_num_x_seq_len, const int seq_len_x_seq_len)
{
  const float amax = __ldg(amax_ptr);
  const float scalar1 = scalar1a * __ldg(scalar1b) * __ldg(scalar1c);
  int mask_id;
  int threadIdx4 = threadIdx.x << 2;

  char4* buf4Ptr = (char4 *)qk_buf_;

  bool qual = threadIdx4 < seq_len;
  for (int seq_id = blockIdx.x ; seq_id < seq_len ; seq_id += gridDim.x){
    char4 tmp4;
    float4 floatTmp4 = {0.0f, 0.0f, 0.0f, 0.0f};
    int inIdx = (blockIdx.y * head_num + blockIdx.z) * (seq_len_x_seq_len) +
                (threadIdx4 & 0xffffffe0) * seq_len +
                (seq_id << 5) + (threadIdx4 & 31);

    if (qual){
      floatTmp4.x = static_cast<float>(__ldg(int_buf + inIdx)) * scalar1;
      floatTmp4.y = static_cast<float>(__ldg(int_buf+inIdx+1)) * scalar1;
      floatTmp4.z = static_cast<float>(__ldg(int_buf+inIdx+2)) * scalar1;
      floatTmp4.w = static_cast<float>(__ldg(int_buf+inIdx+3)) * scalar1;
    }

    float mask_val, max_val;
    max_val = -1e20f;

    __shared__ float s_max, s_sum;

    if (qual){
      mask_id = threadIdx4 + blockIdx.y * seq_len_x_seq_len + seq_id * seq_len;
      //for x
      mask_val = (1.0f - static_cast<float>(__ldg(attr_mask+mask_id))) * -10000.0f;
      floatTmp4.x = floatTmp4.x + mask_val;
      max_val = fmaxf(max_val, floatTmp4.x);

      //for y
      mask_val = (1.0f - static_cast<float>(__ldg(attr_mask+mask_id+1))) * -10000.0f;
      floatTmp4.y = floatTmp4.y + mask_val;
      max_val = fmaxf(max_val, floatTmp4.y);

      //for z
      mask_val = (1.0f - static_cast<float>(__ldg(attr_mask+mask_id+2))) * -10000.0f;
      floatTmp4.z = floatTmp4.z + mask_val;
      max_val = fmaxf(max_val, floatTmp4.z);

      //for w
      mask_val = (1.0f - static_cast<float>(__ldg(attr_mask+mask_id+3))) * -10000.0f;
      floatTmp4.w = floatTmp4.w + mask_val;
      max_val = fmaxf(max_val, floatTmp4.w);
    }

    max_val = blockDim.x <= 32 ? warpReduceMax(max_val) : blockReduceMax<float>(max_val);

    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();

    float sum_val = 0.0f;

    if (qual){
      floatTmp4.x = __expf(floatTmp4.x - s_max);
      sum_val += floatTmp4.x;
      floatTmp4.y = __expf(floatTmp4.y - s_max);
      sum_val += floatTmp4.y;
      floatTmp4.z = __expf(floatTmp4.z - s_max);
      sum_val += floatTmp4.z;
      floatTmp4.w = __expf(floatTmp4.w - s_max);
      sum_val += floatTmp4.w;
    }
    
    sum_val = blockDim.x <= 32 ? warpReduceSum(sum_val) : blockReduceSum<float>(sum_val);

    if (threadIdx.x == 0){
      s_sum = __fdividef(127.0f, (sum_val + 1e-6f));
      s_sum = __fdividef(s_sum, amax);
    }
    __syncthreads();

    if (qual){

      tmp4.x = float_to_int8_rn(floatTmp4.x*s_sum);
      tmp4.y = float_to_int8_rn(floatTmp4.y*s_sum);
      tmp4.z = float_to_int8_rn(floatTmp4.z*s_sum);
      tmp4.w = float_to_int8_rn(floatTmp4.w*s_sum);

      buf4Ptr[inIdx >> 2] = tmp4;
    }
  }
}

//int_buf are a series of sub-matrixes of m = seq_len, n = seq_len, CUBLASLT_ORDER_COL32
//grid = (seq_len, batch_size, head_num)
//block.x = (seq_len + 31)/32
//for int32_t I; int8 O;
//for seq_len <= 32
template <typename T>
__global__
void softmax_COL32_LE32(int8_t* qk_buf_, const int32_t* int_buf, const T* attr_mask, const int batch_size, 
                        const int head_num, const int seq_len, const float scalar1a, const float *scalar1b, 
                        const float *scalar1c, const float *amax_ptr, const int head_num_x_seq_len, const int seq_len_x_seq_len)
{
  const float amax = __ldg(amax_ptr);
  const float scalar1 = scalar1a * __ldg(scalar1b) * __ldg(scalar1c);
  int mask_id;
  int threadIdxx = threadIdx.x;
  bool qual = threadIdxx < seq_len;
  for (int seq_id = blockIdx.x ; seq_id < seq_len ; seq_id += gridDim.x){
    int inIdx = (blockIdx.y * head_num + blockIdx.z) * (seq_len_x_seq_len) +
                (threadIdxx & 0xffffffe0) * seq_len +
                (seq_id << 5) + (threadIdxx & 31);

    float floatTmp = qual ? static_cast<float>(__ldg(int_buf + inIdx)) * scalar1 : 0.0f;

    float mask_val, max_val;

    __shared__ float s_max, s_sum;

    mask_id = qual ? threadIdxx + blockIdx.y * seq_len_x_seq_len + seq_id * seq_len : 0;
    mask_val = qual ? (1.0f - static_cast<float>(__ldg(attr_mask+mask_id))) * -10000.0f : 0.0f;
    floatTmp = qual ? floatTmp + mask_val : 0.0f;
    max_val = qual ? floatTmp : -1e20f;

    max_val = blockDim.x <= 32 ? warpReduceMax(max_val) : blockReduceMax<float>(max_val);

    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();

    floatTmp = qual ? __expf(floatTmp - s_max) : 0.0f;
    
    float sum_val = blockDim.x <= 32 ? warpReduceSum(floatTmp) : blockReduceSum<float>(floatTmp);

    if (threadIdx.x == 0){
      s_sum = __fdividef(127.0f, (sum_val + 1e-6f));
      s_sum = __fdividef(s_sum, amax);
    }
    __syncthreads();

    
    if (qual){
      qk_buf_[inIdx] = float_to_int8_rn(floatTmp*s_sum);
    }
  }
}

//int_buf are a series of sub-matrixes of m = seq_len, n = seq_len, CUBLASLT_ORDER_COL32
//grid = (seq_len, batch_size, head_num)
//block.x = max(32, (seq_len/2 + 31)/32*32)
//for int32_t I; int8 O;
//for seq_len in (32, 64]
template <typename T>
__global__
void softmax_COL32_LE64(int8_t* qk_buf_, const int32_t* int_buf, const T* attr_mask, const int batch_size, 
                        const int head_num, const int seq_len, const float scalar1a, const float *scalar1b, 
                        const float *scalar1c, const float *amax_ptr, const int head_num_x_seq_len, const int seq_len_x_seq_len)
{
  const float amax = __ldg(amax_ptr);
  const float scalar1 = scalar1a * __ldg(scalar1b) * __ldg(scalar1c);
  int mask_id;
  int threadIdx2 = threadIdx.x << 1;

  char2* buf2Ptr = (char2 *)qk_buf_;

  bool qual = threadIdx2 < seq_len;
  for (int seq_id = blockIdx.x ; seq_id < seq_len ; seq_id += gridDim.x){
    char2 tmp2;
    float2 floatTmp2 = {0.0f, 0.0f};
    int inIdx = (blockIdx.y * head_num + blockIdx.z) * (seq_len_x_seq_len) +
                (threadIdx2 & 0xffffffe0) * seq_len +
                (seq_id << 5) + (threadIdx2 & 31);

    if (qual){
      floatTmp2.x = static_cast<float>(__ldg(int_buf + inIdx)) * scalar1;
      floatTmp2.y = static_cast<float>(__ldg(int_buf + inIdx + 1)) * scalar1;
    }

    float mask_val, max_val;
    max_val = -1e20f;

    __shared__ float s_max, s_sum;

    if (qual){
      mask_id = threadIdx2 + blockIdx.y * seq_len_x_seq_len + seq_id * seq_len;
      //for x
      mask_val = (1.0f - static_cast<float>(__ldg(attr_mask+mask_id))) * -10000.0f;
      floatTmp2.x = floatTmp2.x + mask_val;

      //for y
      mask_val = (1.0f - static_cast<float>(__ldg(attr_mask+mask_id+1))) * -10000.0f;
      floatTmp2.y = floatTmp2.y + mask_val;
            
      max_val = fmaxf(floatTmp2.x, floatTmp2.y);
    }

    max_val = blockDim.x <= 32 ? warpReduceMax(max_val) : blockReduceMax<float>(max_val);

    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();

    float sum_val = 0.0f;

    if (qual){
      floatTmp2.x = __expf(floatTmp2.x - s_max);
      sum_val += floatTmp2.x;
      floatTmp2.y = __expf(floatTmp2.y - s_max);
      sum_val += floatTmp2.y;
    }
    
    sum_val = blockDim.x <= 32 ? warpReduceSum(sum_val) : blockReduceSum<float>(sum_val);

    if (threadIdx.x == 0){
      s_sum = __fdividef(127.0f, (sum_val + 1e-6f));
      s_sum = __fdividef(s_sum, amax);
    }
    __syncthreads();

    if (qual){
      tmp2.x = float_to_int8_rn(floatTmp2.x*s_sum);
      tmp2.y = float_to_int8_rn(floatTmp2.y*s_sum);
      buf2Ptr[inIdx >> 1] = tmp2;
    }
  }
}


template<typename T>
__global__
void transpose(T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
  int batch_id = blockIdx.x / (head_num * seq_len);
  int seq_id = blockIdx.x % seq_len;
  int head_id = (blockIdx.x % (head_num * seq_len))/ seq_len;
  dst[batch_id * (head_num * seq_len * size_per_head) + seq_id * head_num * size_per_head
    + head_id * size_per_head + threadIdx.x] = src[blockIdx.x * size_per_head + threadIdx.x];
}

template<>
  __global__
void transpose(half* src, half* dst,
    const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int batch_id = tid / (head_num * seq_len * size_per_head);
  int head_id = (tid % (head_num * seq_len * size_per_head)) / (seq_len * size_per_head);
  int seq_id = (tid % (seq_len * size_per_head)) / size_per_head;
  int id = tid % size_per_head;

  int target_id = target_index(batch_id, head_id, seq_id, id, batch_size, head_num, seq_len, size_per_head);
  half2* src_ptr = (half2*)src;
  half2* dst_ptr = (half2*)dst;

  dst_ptr[target_id] = src_ptr[tid];
}


template<typename T>
__global__
void transpose_rebuild_padding(T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head,
  const int* mask_offset)
{
  // TODO: optimize this kernel? 
  // do remove_sequence_length_padding
  const int tid = threadIdx.x; // batch * seq_len or valid_word_num
  const int bid = blockIdx.x; // head_num * size_per_head

  const int src_batch_id = (bid + mask_offset[bid]) / seq_len;
  const int src_seq_id = (bid + mask_offset[bid]) % seq_len;

  const int dst_seq_id = bid;

  const int head_id = tid / size_per_head;
  const int hidden_id = tid % size_per_head;
  dst[dst_seq_id * head_num * size_per_head + tid] = src[ src_batch_id * head_num * seq_len * size_per_head +
    head_id * seq_len * size_per_head + src_seq_id * size_per_head + hidden_id];
}

template<typename T>
__global__ void rebuild_sequence_length_padding(const T* src, T* tgt,
                                            const int* mask_offset,
                                            const int n)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int tgt_seq_id = bid + mask_offset[bid];
  const int src_seq_id = bid;

  for(int i = tid; i < n; i += blockDim.x)
  {
    tgt[tgt_seq_id * n + i] = src[src_seq_id * n + i];
  }
}

template<OperationType OpType_>
void OpenMultiHeadAttention<OpType_>::multiHeadAttr_nofuse_kernelLauncher(
      cudaStream_t stream,
      cublasHandle_t cublas_handle,
      cublasLtHandle_t cublaslt_handle,
      DataType_* Q,
      const DataType_* bias_Q,
      DataType_* K,
      const DataType_* bias_K,
      DataType_* V,
      const DataType_* bias_V,
      const DataType_* attr_mask,
      DataType_* dst,
      const int batch_size,
      const int seq_len,
      const int head_num,
      const int size_per_head,
      const int int8_mode_,
      const DataType_ scalar)
{
    const int k = head_num * size_per_head;

    dim3 grid;
    dim3 block;

    
    if (int8_mode_ != 0){
      //var for int8
      const float*q_buf_addBias_amax_ptr, *k_buf_addBias_amax_ptr, *v_buf_addBias_amax_ptr, *qk_afterSM_amax_ptr, *qkv_amax_ptr, *in_amax_ptr;
      q_buf_addBias_amax_ptr = param_.amaxList+4;
      k_buf_addBias_amax_ptr = param_.amaxList + 8;
      v_buf_addBias_amax_ptr = param_.amaxList + 12;
      qk_afterSM_amax_ptr = param_.amaxList + 16;
      qkv_amax_ptr = param_.amaxList + 20;
      in_amax_ptr = param_.amaxList;

      assert(seq_len % COL32_ == 0 && size_per_head%COL32_ == 0);

      if(param_.sequence_id_offset == nullptr || param_.valid_word_num == batch_size * seq_len){
        add_QK_bias_transform<<<dim3(batch_size*seq_len*2), dim3((head_num * size_per_head)/4), 0, stream>>>((int8_t*)q_buf_, (int8_t*)k_buf_, (const int32_t*) Q, bias_Q, (const int32_t*) K, 
                       bias_K, batch_size * seq_len, batch_size, seq_len, head_num, size_per_head, 
                       seq_len*size_per_head, query_weight_amax_list, in_amax_ptr+2, key_weight_amax_list, 
                       in_amax_ptr+2, q_buf_addBias_amax_ptr+3, k_buf_addBias_amax_ptr+3);
        add_V_bias_transform<<<dim3(size_per_head/32, seq_len/32, batch_size*head_num), dim3(8, 32), 0, stream>>>((int8_t*)v_buf_, (const int32_t *)V, bias_V, batch_size, seq_len, 
                            head_num, size_per_head, seq_len*size_per_head, value_weight_amax_list, 
                            in_amax_ptr+2, v_buf_addBias_amax_ptr+3);
      }
      else{
        cudaMemset(sequence_id_map_, -1, batch_size * seq_len * sizeof(int));
        mappingRemovePaddingData<<<dim3((param_.valid_word_num + 63)/64), dim3(64)>>>(sequence_id_map_, param_.sequence_id_offset, param_.valid_word_num);
        add_QK_bias_transform_rebuild_padding<<<dim3(param_.valid_word_num*2), dim3((head_num * size_per_head)/4), 0, stream>>>((int8_t*)q_buf_, (int8_t*)k_buf_, (const int32_t*) Q, bias_Q, 
                                          (const int32_t*) K, bias_K, param_.sequence_id_offset, param_.valid_word_num, 
                                          batch_size * seq_len, batch_size, seq_len, head_num, size_per_head, seq_len*size_per_head, 
                                          query_weight_amax_list, in_amax_ptr+2, key_weight_amax_list, in_amax_ptr+2, 
                                          q_buf_addBias_amax_ptr+3, k_buf_addBias_amax_ptr+3);
        
        add_V_bias_transform_rebuild_padding<<<dim3(size_per_head/32, seq_len/32, batch_size*head_num), dim3(8, 32), 0, stream>>>((int8_t*)v_buf_, (const int32_t *)V, bias_V, sequence_id_map_, 
                                            param_.valid_word_num, batch_size, seq_len, head_num, 
                                            size_per_head, seq_len*size_per_head, value_weight_amax_list, 
                                            in_amax_ptr+2, v_buf_addBias_amax_ptr+3);
      }
      
      int batchCount = batch_size * head_num;
      cublasLtMM_withAlgo(qk_int_buf_, batchCount, seq_len, seq_len, size_per_head, 
                          size_per_head*seq_len, size_per_head*seq_len, seq_len*seq_len, 
                          (int8_t*)q_buf_, (int8_t*)k_buf_, cublaslt_handle, stream, cublasLtAlgoMap);

      grid.x = seq_len;
      grid.y = batch_size;
      grid.z = head_num;

      if (seq_len <= 32){
        if (batch_size * head_num > 960)
          grid.x = ceil(float(seq_len)/32.0f);
        block.x = (seq_len + 31)/32*32;
        softmax_COL32_LE32<<<grid, block, 0, stream>>>((int8_t*)qk_buf_, qk_int_buf_, attr_mask, batch_size, head_num, 
                                                       seq_len, float(scalar), q_buf_addBias_amax_ptr + 1, k_buf_addBias_amax_ptr + 1, 
                                                       qk_afterSM_amax_ptr, seq_len*head_num, seq_len*seq_len);
      }
      else if (seq_len <= 64){
        assert(seq_len % 2 == 0);
        block.x = (seq_len/2 + 31)/32*32;
        if (batch_size * head_num > 960)
          grid.x = ceil(float(seq_len)/32.0f);
        softmax_COL32_LE64<<<grid, block, 0, stream>>>((int8_t*)qk_buf_, qk_int_buf_, attr_mask, batch_size, head_num, 
                                                       seq_len, float(scalar), q_buf_addBias_amax_ptr + 1, k_buf_addBias_amax_ptr + 1, 
                                                       qk_afterSM_amax_ptr, seq_len*head_num, seq_len*seq_len);
      }
      else
      {
        assert(seq_len % 4 == 0);
        block.x = (seq_len/4 + 31)/32*32;
        softmax_COL32<<<grid, block, 0, stream>>>((int8_t*)qk_buf_, qk_int_buf_, attr_mask, batch_size, head_num, 
                                                  seq_len, float(scalar), q_buf_addBias_amax_ptr + 1, k_buf_addBias_amax_ptr + 1, 
                                                  qk_afterSM_amax_ptr, seq_len*head_num, seq_len*seq_len);
      }

      cublasLtMM_withAlgo(transpose_dst_int_buf_, batchCount, seq_len, size_per_head, seq_len, 
                          seq_len*seq_len, size_per_head*seq_len, size_per_head*seq_len, (int8_t*)qk_buf_, 
                          (int8_t*)v_buf_, cublaslt_handle, stream, cublasLtAlgoMap);
    
      if(param_.sequence_id_offset == nullptr || param_.valid_word_num == batch_size * seq_len){
        transpose_COL32_kernelLauncher((int8_t*)dst, (const int*)transpose_dst_int_buf_, batch_size, seq_len, head_num, 
                                       size_per_head, v_buf_addBias_amax_ptr+1, qk_afterSM_amax_ptr+1, qkv_amax_ptr+3, stream);
      }
      else{
        transpose_COL32_rebuild_padding_kernelLauncher((int8_t*)dst, (const int*)transpose_dst_int_buf_, sequence_id_map_, 
                                                       param_.valid_word_num, batch_size, seq_len, head_num, size_per_head, 
                                                       v_buf_addBias_amax_ptr+1, qk_afterSM_amax_ptr+1, qkv_amax_ptr+3, stream);     
      }
    }
    //FP32/FP16
    else{
      if(OpType_ == OperationType::FP32)
      {
        if(param_.sequence_id_offset == nullptr || param_.valid_word_num == batch_size * seq_len)
        {
          const int m = batch_size * seq_len;
          const int word_per_block = 1;
          assert(k <= 1024);
          assert(m / word_per_block * 3 <= 65536);
  
          dim3 grid(m / word_per_block * 3);
          dim3 block(k);
          add_QKV_bias<DataType_><<<grid, block, 0, stream>>>(Q, bias_Q, K, bias_K, V, bias_V, q_buf_, k_buf_, v_buf_,
            batch_size, seq_len, head_num, size_per_head, word_per_block);
        }
        else
        {
          add_QKV_bias_rebuild_padding<DataType_><<<param_.valid_word_num, k, 0, stream>>>(Q, bias_Q, K, bias_K, 
            V, bias_V, q_buf_, k_buf_, v_buf_, 
            batch_size, seq_len, head_num, size_per_head, param_.sequence_id_offset);
        }
      }
      else
      {
        if(param_.sequence_id_offset == nullptr || param_.valid_word_num == batch_size * seq_len)
        {
          const int word_per_block = 1;
          grid.x = batch_size * seq_len / word_per_block;
          block.x = head_num * size_per_head * word_per_block / 2;
  
          assert(block.x <= 1024);
  
          add_QKV_bias<DataType_><<<grid, block, 0, stream>>>(Q, bias_Q, K, bias_K, V, bias_V, q_buf_, k_buf_, 
          v_buf_, batch_size, seq_len, head_num, size_per_head / 2, word_per_block);
        }
        else
        {
          add_QKV_bias_rebuild_padding<half2><<<param_.valid_word_num, k / 2, 0, stream>>>((half2*)Q, (const half2*)bias_Q, 
            (half2*)K, (const half2*)bias_K, (half2*)V, (const half2*)bias_V, 
            (half2*)q_buf_, (half2*)k_buf_, (half2*)v_buf_,
            batch_size, seq_len, head_num, size_per_head / 2, param_.sequence_id_offset);
        }
      }

      DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;
      
      check_cuda_error(cublasGemmStridedBatchedEx(cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        seq_len, seq_len, size_per_head,
        &alpha,
        k_buf_, AType_, size_per_head, seq_len * size_per_head,
        q_buf_, BType_, size_per_head, seq_len * size_per_head,
        &beta,
        qk_buf_, CType_, seq_len, seq_len * seq_len,
        batch_size * head_num,
        computeType_,
        static_cast<cublasGemmAlgo_t>(cublasAlgo_[1])));
        
      //deal with odd seq_len
      if (seq_len % 2 != 0){
        if(seq_len <= 32)
          block.x = 32;
        else if(seq_len > 32 && seq_len <= 64)
          block.x = 64;
        else if(seq_len > 64 && seq_len <= 128)
          block.x = 128;
        else if(seq_len > 128 && seq_len <= 256)
          block.x = 256;
        else if(seq_len > 256 && seq_len <= 512)
          block.x = 512;
        else
          block.x = 1024;

        if(batch_size * head_num <= 120)
        {
          grid.x = batch_size * head_num * seq_len;
          softmax_kernel_v2<DataType_><<<grid, block, 0, stream>>>(qk_buf_, attr_mask, batch_size, head_num, seq_len, scalar);
        }
        else
        {
          grid.x = batch_size * head_num;
          softmax_kernel<DataType_><<<grid, block, 0, stream>>>(qk_buf_, attr_mask, batch_size, head_num, seq_len, scalar);
        }
      }
      //deal with even seq_len 
      else{
        grid.x = seq_len;
        if (batch_size * head_num > 360)
          grid.x = ceil(float(seq_len)/32.0f);
        grid.y = batch_size;
        grid.z = head_num;
        if (seq_len <= 32){
          block.x = 32;
          softmax_kernel_v3_LE32<DataType_><<<grid, block, 0, stream>>>(qk_buf_, attr_mask, batch_size, head_num, seq_len, scalar);
        }
        else{
          if (OpType_ == OperationType::FP16){
            block.x = (seq_len/2 + 31)/32*32;
            softmax_kernel_v3<<<grid, block, 0, stream>>>(qk_buf_, attr_mask, batch_size, head_num, seq_len, scalar);
          }
          else{
            block.x = (seq_len + 31)/32*32;
            softmax_kernel_v3<DataType_><<<grid, block, 0, stream>>>(qk_buf_, attr_mask, batch_size, head_num, seq_len, scalar);
          }
        }
        grid.x = grid.y = grid.z = 1;
      }

      check_cuda_error(cublasGemmStridedBatchedEx(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_per_head, seq_len, seq_len,
        &alpha,
        v_buf_, AType_, size_per_head, seq_len * size_per_head,
        qk_buf_, BType_, seq_len, seq_len * seq_len,
        &beta,
        transpose_dst_, CType_, size_per_head, seq_len * size_per_head,
        batch_size * head_num,
        computeType_,
        static_cast<cublasGemmAlgo_t>(cublasAlgo_[2])));
        
      /* for half2 only */
      if(OpType_ == OperationType::FP16)
      {
        if(param_.sequence_id_offset == nullptr || param_.valid_word_num == batch_size * seq_len)
        {
          const int seq_per_block = 4;
          grid.x = batch_size * head_num * seq_len / seq_per_block;
          block.x = seq_per_block * size_per_head / 2;
    
          assert(grid.x * seq_per_block == batch_size * head_num * seq_len);
    
          transpose<DataType_><<<grid, block, 0, stream>>>(transpose_dst_, dst, 
              batch_size, seq_len, head_num, size_per_head / 2);
        }
        else
        {
          transpose_rebuild_padding<half2><<<param_.valid_word_num, k / 2, 0, stream>>>(
            (half2*)transpose_dst_, (half2*)dst, 
            batch_size, seq_len, head_num, size_per_head / 2, param_.sequence_id_offset);
        }
      }
      else
      {
        if(param_.sequence_id_offset == nullptr || param_.valid_word_num == batch_size * seq_len)
        {
          const int seq_per_block = 1;
          grid.x = batch_size * head_num * seq_len / seq_per_block;
          block.x = seq_per_block * size_per_head;
          transpose<DataType_><<<grid, block, 0, stream>>>(transpose_dst_, dst, 
            batch_size, seq_len, head_num, size_per_head);
        }
        else
        {
          transpose_rebuild_padding<DataType_><<<param_.valid_word_num, k, 0, stream>>>(transpose_dst_, dst, 
            batch_size, seq_len, head_num, size_per_head, param_.sequence_id_offset);
        }
      }
    }
}

template void OpenMultiHeadAttention<OperationType::FP32>::multiHeadAttr_nofuse_kernelLauncher(
      cudaStream_t stream,
      cublasHandle_t handle,
      cublasLtHandle_t cublaslt_handle,
      float* Q,
      const float* bias_Q,
      float* K,
      const float* bias_K,
      float* V,
      const float* bias_V,
      const float* attr_mask,
      float* dst,
      const int batch_size,
      const int seq_len,
      const int head_num,
      const int size_per_head,
      const int int8_mode_,
      const float scalar);

template void OpenMultiHeadAttention<OperationType::FP16>::multiHeadAttr_nofuse_kernelLauncher(
      cudaStream_t stream,
      cublasHandle_t handle,
      cublasLtHandle_t cublaslt_handle,
      half* Q,
      const half* bias_Q,
      half* K,
      const half* bias_K,
      half* V,
      const half* bias_V,
      const half* attr_mask,
      half* dst,
      const int batch_size,
      const int seq_len,
      const int head_num,
      const int size_per_head,
      const int int8_mode_,
      const half scalar);
}//namespace cuda
}//namespace fastertransformer

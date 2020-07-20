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

#include "fastertransformer/open_decoder.h"

#include "cub/cub.cuh"

namespace fastertransformer{

const int WARP_SIZE = 32;
const bool ATTENION_OPT = true;
const int ATTENTION_BLOCK_SIZE = 256;

///////////////////////////////////////////////////////////////////////////////////////////////////

template <int HALF_ELEMENTS_PER_WARP_LOAD>
using Copy_half_t =
    typename std::conditional<HALF_ELEMENTS_PER_WARP_LOAD == 32, half,
        typename std::conditional<HALF_ELEMENTS_PER_WARP_LOAD == 64, int,
            typename std::conditional<HALF_ELEMENTS_PER_WARP_LOAD == 128, int2, int4
            >::type
        >::type
    >::type;

template <typename T, int ELEMENTS_PER_WARP_LOAD>
using Copy_t = Copy_half_t<sizeof(T) / sizeof(half) * ELEMENTS_PER_WARP_LOAD>;

///////////////////////////////////////////////////////////////////////////////////////////////////

/**
  masked multi-head attention
 */
#define FINAL_MASK 0xffffffff
template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
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
  // __shared__ T shared[32]; 
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
__global__ 
void add_bias_relu(T* out, const T* bias, int m, int n)
{
  T val, reg_bias;

  int row_id = blockIdx.x;
  int ite = n / blockDim.x;
  int tid = threadIdx.x;

  for(int i = 0; i < ite; ++i)
  {
    reg_bias = __ldg(&bias[i * blockDim.x + tid]);
    row_id = blockIdx.x;

    while(row_id < m)
    {
      val = out[tid + i * blockDim.x + row_id * n] + reg_bias;
      out[tid + i * blockDim.x + row_id * n] = (T)(val > 0.0f ? val : 0.0f);
      row_id += gridDim.x;
     }
  }
}

template <>
  __global__ 
void add_bias_relu(half* out, const half* bias, int m, int n)
{
  half2 val, reg_bias;
  int row_id = blockIdx.x;
  int ite = n / blockDim.x / 2;
  int tid = threadIdx.x;

  half2* out_ptr = (half2*) out;
  const half2* bias_ptr = (half2*) bias;
  for(int i = 0; i < ite; ++i)
  {
    reg_bias = __ldg(&bias_ptr[i * blockDim.x + tid]);
    row_id = blockIdx.x;

    while(row_id < m)
    {
      val = out_ptr[tid + i * blockDim.x + row_id * n / 2];
      val = __hadd2(val, reg_bias);
      val.x = val.x > (half)0.0f ? val.x : (half)0.0f;
      val.y = val.y > (half)0.0f ? val.y : (half)0.0f;
      out_ptr[tid + i * blockDim.x + row_id * n / 2] = val;
      row_id += gridDim.x;
    }
  }
}
template <typename T>
  __inline__ __device__
T warpReduceMax(T val)
{
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
//  __shared__ T shared[32]; 
  int lane = threadIdx.x & 0x1f; // in-warp idx
  int wid = threadIdx.x >> 5;  // warp idx

  val = warpReduceMax(val); // get maxx in each warp

  if(lane == 0) // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();


  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : 0;
  val = warpReduceMax(val);

  return val;
}

template <int size_per_head, int block_sz, typename T>
__global__ 
void masked_attention_kernel_opt(
  T* __restrict key_buf, T* __restrict value_buf,
  T* __restrict query_buf, const T* __restrict self_Q_bias, 
  T* __restrict key_cache, const T* __restrict self_K_bias, 
  T* __restrict value_cache, const T* __restrict self_V_bias,
  T* __restrict context_buf, int batch_size, int head_num, const int step, const T scalar)
{
  typedef Copy_t<T, size_per_head> copy_t;
  const int elems_per_thread = size_per_head / WARP_SIZE;

  union Access_t
  {
    copy_t v;
    T x[elems_per_thread]; // supported size 1,2,4
  };
  typedef struct Float_n_t
  {
    T x[elems_per_thread]; // supported size 1,2,4
  } float_n_t;

  __shared__ float_n_t sq[block_sz];

  __shared__ float logits[1024]; // only use [0 ~ step-1], the step should be smaller than 1024

  const int tid = threadIdx.x;
  const int warp_num = block_sz / WARP_SIZE;
  const int bid = blockIdx.x;
  const int head_id = blockIdx.x % head_num;
  const int warp_id = tid / WARP_SIZE; // warp_id in block
  const int lane_id = tid % WARP_SIZE; // lane_id in warp

  typedef cub::BlockReduce<float, block_sz> MaxValBlockReduce;
  typedef cub::BlockReduce<float, block_sz> BlockReduce;
  __shared__ typename MaxValBlockReduce::TempStorage max_val_block_temp_storage;
  __shared__ typename BlockReduce::TempStorage block_temp_storage;
  __shared__ typename cub::WarpReduce<float>::TempStorage temp_storage[warp_num];

  int qkv_id = bid * size_per_head;
  int qkv_bias_id = head_id * size_per_head;

  query_buf = &query_buf[qkv_id];
  key_buf = &key_buf[qkv_id];
  value_buf = &value_buf[qkv_id];
  self_K_bias = &self_K_bias[qkv_bias_id];
  key_cache = &key_cache[qkv_id];
  self_Q_bias = &self_Q_bias[qkv_bias_id];
  self_V_bias = &self_V_bias[qkv_bias_id];
  value_cache = &value_cache[qkv_id];
  context_buf = &context_buf[qkv_id];

  Access_t bias_r, query_buf_r;
  Access_t key_val_r, key_buf_r;
  Access_t value_val_r, value_buf_r;

  // each warp will have its own copy of sq
  query_buf_r.v = *((copy_t *)query_buf + lane_id);
  key_buf_r.v = *((copy_t *)key_buf + lane_id);
  bias_r.v = *((copy_t *)self_Q_bias + lane_id);
  float qb_r[elems_per_thread];
  for (int i = 0; i < elems_per_thread; ++i)
  {
    qb_r[i] =  (float)query_buf_r.x[i] + (float)bias_r.x[i];
  }

  //offset for each step
  int offset = batch_size * head_num * size_per_head;
  bias_r.v = *((copy_t *) self_K_bias + lane_id);
  for(int ite = warp_id; ite < step; ite += warp_num)
  {
    key_val_r.v = *((copy_t *)&key_cache[ite * offset] + lane_id);
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        key_val_r.x[i] = (float)key_buf_r.x[i] + (float)bias_r.x[i];
      }
      *((copy_t *)&key_cache[ite * offset] + lane_id) = key_val_r.v;
    }
    float val = 0.f;
    for (int i = 0; i < elems_per_thread; i++)
    {
      val = val +  (float)key_val_r.x[i] * qb_r[i] * (float)scalar;
    }
    float qk = cub::WarpReduce<float>(temp_storage[warp_id]).Sum(val);
    if (lane_id == 0)
    {
      logits[ite] = qk; 
    }
  }
  __syncthreads();

  __shared__ float s_max_val, s_sum;

  float local_i = -1e20f;
  for(int i = tid; i < step; i += blockDim.x)
    local_i = max(local_i, logits[i]);

  float max_val = MaxValBlockReduce(max_val_block_temp_storage).Reduce(local_i, cub::Max());
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();


  float local_o = 0.0f;
  for(int i = tid; i < step; i += blockDim.x)
  {
    logits[i] = __expf(logits[i] - s_max_val);
    local_o += logits[i];
  }
  float val = BlockReduce(block_temp_storage).Sum(local_o);

  if(tid == 0)
    s_sum = val + 1e-6;
  __syncthreads();

  float s_sum_inverse = __fdividef(1.0f, s_sum);
  for(int i = tid; i < step; i += blockDim.x)
  {
    logits[i] = logits[i] * s_sum_inverse;
  }
  __syncthreads(); 

  // This optimization introduces discrepancy because of different order in FP32 summation
  float sum_r[elems_per_thread] = {0.f};
  bias_r.v = *((copy_t *) self_V_bias + lane_id);
  value_buf_r.v = *((copy_t *)value_buf + lane_id);

  for(int ite = warp_id; ite < step; ite += warp_num)
  {
    value_val_r.v = *((copy_t *)&value_cache[ite * offset] + lane_id);
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        value_val_r.x[i] = (float)value_buf_r.x[i] + (float)bias_r.x[i];
      }
      *((copy_t *)&value_cache[ite * offset] + lane_id) = value_val_r.v;
    }
    for (int i = 0; i < elems_per_thread; ++i)
    {
      sum_r[i] += (float)value_val_r.x[i] * logits[ite]; 
    }
  }
  for (int i = 0; i < elems_per_thread; i++)
  {
    sq[warp_id * WARP_SIZE + lane_id].x[i] = sum_r[i];
  }
  __syncthreads();
  if (warp_id == 0)
  {
    #pragma unroll
    for (int j = 1; j < warp_num; j++)
    {
      for (int i = 0; i < elems_per_thread; ++i)
      {
        sum_r[i] = sum_r[i] + (float)sq[j * WARP_SIZE + tid].x[i];
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < elems_per_thread; i++)
  {
    value_val_r.x[i] = sum_r[i];
  }
  if (warp_id == 0)
  {
    *((copy_t *)context_buf + lane_id) = value_val_r.v;
  }
}

// only use for compile 
template <int size_per_head, int block_sz>
__global__ 
void masked_attention_kernel_opt_half2(
  float* __restrict key_buf, float* __restrict value_buf,
  float* __restrict query_buf, const float* __restrict self_Q_bias, 
  float* __restrict key_cache, const float* __restrict self_K_bias, 
  float* __restrict value_cache, const float* __restrict self_V_bias,
  float* __restrict context_buf, int batch_size, int head_num, const int step, const float scalar) {}

template <int size_per_head, int block_sz>
__global__ 
void masked_attention_kernel_opt_half2(
  half* __restrict key_buf, half* __restrict value_buf,
  half* __restrict query_buf, const half* __restrict self_Q_bias, 
  half* __restrict key_cache, const half* __restrict self_K_bias, 
  half* __restrict value_cache, const half* __restrict self_V_bias,
  half* __restrict context_buf, int batch_size, int head_num, const int step, const half scalar)
{
  half2* key_buf_ptr = (half2*)key_buf;
  half2* value_buf_ptr = (half2*)value_buf;
  half2* query_buf_ptr = (half2*)query_buf;
  half2* key_cache_ptr = (half2*)key_cache;
  half2* value_cache_ptr = (half2*)value_cache;
  const half2* self_Q_bias_ptr = (const half2*)self_Q_bias;
  const half2* self_K_bias_ptr = (const half2*)self_K_bias;
  const half2* self_V_bias_ptr = (const half2*)self_V_bias;
  half2* context_buf_ptr = (half2*)context_buf;

  typedef Copy_t<half2, size_per_head/2> copy_t;
  const int elems_per_thread = size_per_head / 2 / WARP_SIZE;

  union Access_t
  {
    copy_t v;
    half2 x[elems_per_thread]; // supported size 1,2,4
  };
  typedef struct Half_n_t
  {
    half2 x[elems_per_thread]; // supported size 1,2,4
  } half_n_t;

  __shared__ half_n_t sq[block_sz];

  __shared__ float logits[1024]; // only use [0 ~ step-1]

  const int tid = threadIdx.x;
  const int warp_num = block_sz / WARP_SIZE;
  const int bid = blockIdx.x;
  const int head_id = blockIdx.x % head_num;
  const int warp_id = tid / WARP_SIZE; // warp_id in block
  const int lane_id = tid % WARP_SIZE; // lane_id in warp

  typedef cub::BlockReduce<float, block_sz> MaxValBlockReduce;
  typedef cub::BlockReduce<float, block_sz> BlockReduce;
  __shared__ typename MaxValBlockReduce::TempStorage max_val_block_temp_storage;
  __shared__ typename BlockReduce::TempStorage block_temp_storage;
  __shared__ typename cub::WarpReduce<float>::TempStorage temp_storage[warp_num];

  int qkv_id = bid * size_per_head / 2;
  int qkv_bias_id = head_id * size_per_head / 2;

  query_buf_ptr = &query_buf_ptr[qkv_id];
  key_buf_ptr = &key_buf_ptr[qkv_id];
  value_buf_ptr = &value_buf_ptr[qkv_id];
  self_K_bias_ptr = &self_K_bias_ptr[qkv_bias_id];
  key_cache_ptr = &key_cache_ptr[qkv_id];
  self_Q_bias_ptr = &self_Q_bias_ptr[qkv_bias_id];
  self_V_bias_ptr = &self_V_bias_ptr[qkv_bias_id];
  value_cache_ptr = &value_cache_ptr[qkv_id];
  context_buf_ptr = &context_buf_ptr[qkv_id];

  Access_t bias_r, query_buf_r;
  Access_t key_val_r, key_buf_r;
  Access_t value_val_r, value_buf_r;

  // each warp will have its own copy of sq
  query_buf_r.v = *((copy_t *)query_buf_ptr + lane_id);
  key_buf_r.v = *((copy_t *)key_buf_ptr + lane_id);
  bias_r.v = *((copy_t *)self_Q_bias_ptr + lane_id);
  half2 qb_r[elems_per_thread];
  for (int i = 0; i < elems_per_thread; ++i)
  {
    qb_r[i] = __hadd2(query_buf_r.x[i], bias_r.x[i]);
  }

  //offset for each step
  int offset = batch_size * head_num * size_per_head / 2;
  bias_r.v = *((copy_t *) self_K_bias + lane_id);
  for(int ite = warp_id; ite < step; ite += warp_num)
  {
    key_val_r.v = *((copy_t *)&key_cache_ptr[ite * offset] + lane_id);
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        key_val_r.x[i] = __hadd2(key_buf_r.x[i], bias_r.x[i]);
      }
      *((copy_t *)&key_cache_ptr[ite * offset] + lane_id) = key_val_r.v;
    }
    float val = 0.f;
    for (int i = 0; i < elems_per_thread; i++)
    {
      half2 val2 = __hmul2(key_val_r.x[i], qb_r[i]);
      val = val + (float)((val2.x + val2.y) * scalar);
    }
    float qk = cub::WarpReduce<float>(temp_storage[warp_id]).Sum(val);
    if (lane_id == 0)
    {
      logits[ite] = qk; 
    }
  }
  __syncthreads();

  __shared__ float s_max_val, s_sum;
  float local_i = -1e20f;
  for(int i = tid; i < step; i += blockDim.x)
    local_i = max(local_i, logits[i]);

  float max_val = MaxValBlockReduce(max_val_block_temp_storage).Reduce(local_i, cub::Max());
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();

  float local_o = 0.0f;
  for(int i = tid; i < step; i += blockDim.x)
  {
    logits[i] = __expf(logits[i] - s_max_val);
    local_o += logits[i];
  }
  float val = BlockReduce(block_temp_storage).Sum(local_o);

  if(tid == 0)
    s_sum = val + 1e-6;
  __syncthreads();

  float s_sum_inverse = __fdividef(1.0f, s_sum);
  for(int i = tid; i < step; i += blockDim.x)
  {
    logits[i] = logits[i] * s_sum_inverse;
  }
  __syncthreads(); 

  // This optimization introduces discrepancy because of different order in FP32 summation
  half2 sum_r[elems_per_thread];
  for(int i = 0; i < elems_per_thread; i++)
  {
    sum_r[i].x = (half)0.f;
    sum_r[i].y = (half)0.f;
  }
  bias_r.v = *((copy_t *) self_V_bias_ptr + lane_id);
  value_buf_r.v = *((copy_t *)value_buf_ptr + lane_id);

  for(int ite = warp_id; ite < step; ite += warp_num)
  {
    value_val_r.v = *((copy_t *)&value_cache_ptr[ite * offset] + lane_id);
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        value_val_r.x[i] = __hadd2(value_buf_r.x[i], bias_r.x[i]);
      }
      *((copy_t *)&value_cache_ptr[ite * offset] + lane_id) = value_val_r.v;
    }
    for (int i = 0; i < elems_per_thread; ++i)
    {
      half2 logit2_val;
      logit2_val.x = (half)logits[ite];
      logit2_val.y = (half)logits[ite];
      sum_r[i] = __hadd2(sum_r[i], __hmul2(value_val_r.x[i], logit2_val));
    }
  }
  for (int i = 0; i < elems_per_thread; i++)
  {
    sq[warp_id * WARP_SIZE + lane_id].x[i] = sum_r[i];
  }
  __syncthreads();
  if (warp_id == 0)
  {
    #pragma unroll
    for (int j = 1; j < warp_num; j++)
    {
      for (int i = 0; i < elems_per_thread; ++i)
      {
        sum_r[i] = __hadd2(sum_r[i], sq[j * WARP_SIZE + tid].x[i]);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < elems_per_thread; i++)
  {
    value_val_r.x[i] = sum_r[i];
  }
  if (warp_id == 0)
  {
    *((copy_t *)context_buf_ptr + lane_id) = value_val_r.v;
  }
}

template <typename T>
__global__ 
void masked_attention_kernel(
  T* key_buf, T* value_buf,
  T* query_buf, const T* self_Q_bias, 
  T* key_cache, const T* self_K_bias, T* value_cache, const T* self_V_bias,
  T* context_buf, int batch_size, int head_num, int size_per_head, const int step, const T scalar)
{
  extern __shared__ __align__(sizeof(T)) unsigned s_buf[];
  T* sq = reinterpret_cast<T *>(s_buf);
  T* logits = reinterpret_cast<T *>(&sq[size_per_head]);

  int tid = threadIdx.x;
  int bid = blockIdx.x / head_num;
  int head_id = blockIdx.x % head_num;

  int qkv_id = bid * head_num * size_per_head + head_id * size_per_head + tid;
  int qkv_bias_id = head_id * size_per_head + tid;

  if(tid < size_per_head)
    sq[tid] = query_buf[qkv_id] + self_Q_bias[qkv_bias_id];
  __syncthreads();

  //offset for each step
  int offset = batch_size * head_num * size_per_head;
  for(int ite = 0; ite < step; ++ite)
  {
    T key = tid < size_per_head ? key_cache[ite * offset + qkv_id] : (T)0.0f;
    //for the last step, we should update K + bias_K to the cache
    if(ite == step - 1 && tid < size_per_head)
    {
      key = key_buf[qkv_id] + self_K_bias[qkv_bias_id];
      key_cache[ite * offset + qkv_id] = key; 
    }
    
    T val = (tid < size_per_head) ? key * sq[tid] * scalar : (T)(0.0f);
    T qk = blockReduceSum(val);
    if(threadIdx.x == 0)
      logits[ite] = qk;
    __syncthreads(); //try to remove
  }
  __syncthreads(); //try to remove

  __shared__ float s_max_val, s_sum;
  float local_i = tid < step ? (float)logits[tid] : -1e20f; 
  float max_val = blockReduceMax<float>(local_i);
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();

  local_i -= s_max_val;
  float local_o = tid < step ? __expf(local_i) : 0.0f;
  float val = blockReduceSum<float>(local_o);

  if(tid == 0)
    s_sum = val + 1e-6;
  __syncthreads();

  if(tid < step)
    logits[tid] = local_o / s_sum;
  __syncthreads();

  if(tid < size_per_head)
  {
    T sum = (T)0.0f;
    for(int ite = 0; ite < step; ++ite)
    {
      T value = value_cache[ite * offset + qkv_id];
      //for the last step, we should update K + bias_K to the cache
      if(ite == step - 1)
      {
        value = value_buf[qkv_id] + self_V_bias[qkv_bias_id];
        value_cache[ite * offset + qkv_id] = value;
      }
      sum += value * logits[ite];
    }
    context_buf[qkv_id] = sum;
  }
}

template <typename T>
__global__ 
void masked_attention_kernel_v2(T* query_buf, const T* self_Q_bias, 
  T* key_cache, const T* self_K_bias, T* value_cache, const T* self_V_bias,
  T* context_buf, int batch_size, int head_num, int size_per_head, const int step, const T scalar)
{
  extern __shared__ __align__(sizeof(T)) unsigned s_buf[];
  T* sq = reinterpret_cast<T *>(s_buf);
  T* logits = reinterpret_cast<T *>(&sq[size_per_head]);

  int tid = threadIdx.x;
  int bid = blockIdx.x / head_num;
  int head_id = blockIdx.x % head_num;

  int qkv_id = bid * head_num * size_per_head + head_id * size_per_head + tid;
  int qkv_bias_id = head_id * size_per_head + tid;

  if(tid < size_per_head)
    sq[tid] = query_buf[qkv_id] + self_Q_bias[qkv_bias_id];
  __syncthreads();

  int warp_size = 32;
  int offset = batch_size * head_num * size_per_head;
  int warp_ite = size_per_head / warp_size;

  T qk = (T)0.0f;

  //each warp process one step
  int step_id = threadIdx.x >> 5;
  if(step_id < step)
  {
    for(int wite = 0; wite < warp_ite; ++wite)
    {
      T key = key_cache[step_id * offset + bid * head_num * size_per_head + head_id * size_per_head 
        + tid % warp_size + wite * warp_size];
      //for the last step, we should update K + bias_K to the cache
      if(step_id == step - 1)
      { 
        key += self_K_bias[bid * head_num * size_per_head + head_id * size_per_head + 
          tid % warp_size + wite * warp_size];
        key_cache[step_id * offset + bid * head_num * size_per_head + head_id * size_per_head
          + tid % warp_size + wite * warp_size] = key;
      }
      qk += key * sq[tid % warp_size + wite * warp_size];
    }
  
    qk = warpReduceSum(qk * scalar);
    if(threadIdx.x % warp_size == 0)
    {
      logits[step_id] = qk;
      printf("step_id %d %f\n", step_id, qk);
    }
    
  }
  __syncthreads();

  __shared__ float s_max_val, s_sum;
  float local_i = tid < step ? (float)logits[tid] : -1e20f; 
  float max_val = blockReduceMax<float>(local_i);
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();

  local_i -= s_max_val;
  float local_o = tid < step ? __expf(local_i) : 0.0f;
  float val = blockReduceSum<float>(local_o);

  if(tid == 0)
    s_sum = val;
  __syncthreads();
  if(tid < step)
    logits[tid] = local_o / s_sum;
  __syncthreads();

  
  if(tid < size_per_head)
  {
    T sum = (T)0.0f;
    for(int ite = 0; ite < step; ++ite)
    {
      T value = value_cache[ite * offset + qkv_id];
      //for the last step, we should update K + bias_K to the cache
      if(ite == step - 1)
      {
        value += self_V_bias[qkv_bias_id];
        value_cache[ite * offset + qkv_id] = value;
      }
      sum += value * logits[ite];
    }
    context_buf[qkv_id] = sum;
  }
}

template <typename T>
void masked_attention_dispatch(
  T* key_buf, T* value_buf,
  T* query_buf, const T* self_Q_bias, 
  T* key_cache, const T* self_K_bias, T* value_cache, const T* self_V_bias,
  T* context_buf, int batch_size, int head_num, int size_per_head, const int step, cudaStream_t stream)
  {
    const int block_sz = ATTENTION_BLOCK_SIZE;
    T scalar = (T)(1.f / sqrtf(size_per_head * 1.0f));

    dim3 grid(batch_size * head_num);

    int cond = size_per_head * ((ATTENION_OPT)? 1:0);
    switch (cond)
    {
      case 32:
        masked_attention_kernel_opt<32, block_sz, T><<<grid, block_sz, 0, stream>>>(
          key_buf, value_buf,
          query_buf, self_Q_bias,  key_cache, self_K_bias, value_cache, self_V_bias, context_buf, 
          batch_size, head_num, step, scalar); 
        break;
      case 64:
        if(sizeof(T) == 2)
          masked_attention_kernel_opt_half2<64, block_sz><<<grid, block_sz, 0, stream>>>(
            key_buf, value_buf,
            query_buf, self_Q_bias,  key_cache, self_K_bias, value_cache, self_V_bias, context_buf, 
            batch_size, head_num, step, scalar);
        else
          masked_attention_kernel_opt<64, block_sz, T><<<grid, block_sz, 0, stream>>>(
            key_buf, value_buf,
            query_buf, self_Q_bias,  
            key_cache, self_K_bias, 
            value_cache, self_V_bias, 
            context_buf, 
            batch_size, head_num, step, scalar);
        break;
      case 128:
        if(sizeof(T) == 2)
          masked_attention_kernel_opt_half2<128, block_sz><<<grid, block_sz, 0, stream>>>(
            key_buf, value_buf,
            query_buf, self_Q_bias,  key_cache, self_K_bias, value_cache, self_V_bias, context_buf, 
            batch_size, head_num, step, scalar);
        else
          masked_attention_kernel_opt<128, block_sz, T><<<grid, block_sz, 0, stream>>>(
            key_buf, value_buf,
            query_buf, self_Q_bias,  key_cache, self_K_bias, value_cache, self_V_bias, context_buf, 
            batch_size, head_num, step, scalar);
        break;
      default:
        // default path
        int block_size = 128;
        
        //suppose size_per_head <= 128
        if(step <= 64)
          block_size = 64;
        else if(step <= 128 && step > size_per_head)
          block_size = 128;
        else if(step > 128 && step <= 256)
          block_size = 256;
        else if(step > 256 && step <= 512)
          block_size = 512;
        else
          block_size = 1024;
        
        if((int)block_size < size_per_head)
          block_size = size_per_head;
          
        assert(block_size <= 1024);
        dim3 block(block_size);
        T scalar = 1 / sqrtf(size_per_head * 1.0f);

        
        int shared_size = sizeof(T) * (size_per_head + step);
        masked_attention_kernel<T><<<grid, block, shared_size, stream>>>(
          key_buf, value_buf,
          query_buf, self_Q_bias, 
          key_cache, self_K_bias,
          value_cache, self_V_bias,
          context_buf, batch_size,
          head_num, size_per_head, step, scalar);
    }
  }

template<OperationType OpType_>
void OpenDecoder<OpType_>::masked_multi_head_attention(
  const DataType_* from_tensor,
  DataType_* key_cache_,
  DataType_* value_cache_,
  DataType_* decoder_output,
  const int step)
{
  int m = batch_size_;
  int n = hidden_units_;
  int k = hidden_units_;

  DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;

  if(is_fuse_QKV == true)
  {
    check_cuda_error(cublasGemmBatchedEx(param_.cublas_handle, 
      CUBLAS_OP_N, CUBLAS_OP_N, 
      n, m, k, 
      &alpha, 
      (const void* const*) qkv_kernel_, AType_, n,
      (const void* const*) qkv_input_, BType_, k,
      &beta,
      (void* const*)qkv_buf_, CType_, n,
      3, 
      computeType_,
      static_cast<cublasGemmAlgo_t>(cublasAlgo_[4])));
  }
  else
  {
    key_buf_ = key_cache_ + (step - 1) * m * n;
    value_buf_ = value_cache_ + (step - 1) * m * n;

    check_cuda_error(cublasGemmEx(param_.cublas_handle, 
      CUBLAS_OP_N, CUBLAS_OP_N, 
      n, m, k, 
      &alpha, 
      param_.self_attention.query_weight.kernel , AType_, n, 
      from_tensor, BType_, k, 
      &beta, 
      query_buf_, CType_, n, 
      computeType_, 
      static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));
  
    check_cuda_error(cublasGemmEx(param_.cublas_handle, 
      CUBLAS_OP_N, CUBLAS_OP_N, 
      n, m, k, 
      &alpha, 
      param_.self_attention.key_weight.kernel, AType_, n, 
      from_tensor, BType_, k, 
      &beta, 
      key_buf_, CType_, n, 
      computeType_, 
      static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));
  
    check_cuda_error(cublasGemmEx(param_.cublas_handle, 
      CUBLAS_OP_N, CUBLAS_OP_N, 
      n, m, k, 
      &alpha, 
      param_.self_attention.value_weight.kernel, AType_, n, 
      from_tensor, BType_, k, 
      &beta, 
      value_buf_, CType_, n, 
      computeType_, 
      static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));
  }

  masked_attention_dispatch<DataType_>(
    key_buf_, value_buf_,
    query_buf_, param_.self_attention.query_weight.bias, 
    key_cache_, param_.self_attention.key_weight.bias,
    value_cache_, param_.self_attention.value_weight.bias,
    context_buf_, batch_size_,
    head_num_, size_per_head_, step, param_.stream); 

  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &alpha, 
    param_.self_attention.attention_output_weight.kernel, AType_, n, 
    context_buf_, BType_, k, 
    &beta, 
    decoder_output, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));
} 

template <typename T, int size_per_head, int block_sz>
__global__ 
void cross_attention_kernel_opt(
  T* __restrict query_buf, const T* __restrict Q_bias, 
  T* __restrict key_cache, const T* __restrict K_bias, 
  T* __restrict value_cache, const T* __restrict V_bias,
  const int* length_per_sample, T* __restrict context_buf, 
  int batch_size, int head_num, const int step, const int seq_len, const float scalar)
{  
  typedef Copy_t<T, size_per_head> copy_t;
  const int elems_per_thread = size_per_head / WARP_SIZE;
  union Access_t
  {
    copy_t v;
    T x[elems_per_thread]; // supported size 1,2,4
  };
  typedef struct Float_n_t
  {
    float x[elems_per_thread]; // supported size 1,2,4
  } float_n_t;

  __shared__ float_n_t sq[block_sz];
  __shared__ float logits[1024];

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int warp_num = block_sz / WARP_SIZE;

  typedef cub::BlockReduce<float, block_sz> MaxValBlockReduce;
  typedef cub::BlockReduce<float, block_sz> BlockReduce;
  __shared__ typename MaxValBlockReduce::TempStorage max_val_block_temp_storage;
  __shared__ typename BlockReduce::TempStorage block_temp_storage;

  __shared__ typename cub::WarpReduce<float>::TempStorage temp_storage[warp_num];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x / head_num;
  const int head_id = blockIdx.x % head_num;

  int length = __ldg(&length_per_sample[bid]);

  const int lane_id = tid % WARP_SIZE;

  int qkv_id = bid * head_num * size_per_head + head_id * size_per_head;
  int qkv_bias_id = head_id * size_per_head;

  int key_value_id = bid * (seq_len * head_num * size_per_head) + 
  + head_id * size_per_head;

  query_buf = &query_buf[qkv_id];
  K_bias = &K_bias[qkv_bias_id];
  key_cache = &key_cache[key_value_id];
  Q_bias = &Q_bias[qkv_bias_id];
  V_bias = &V_bias[qkv_bias_id];
  value_cache = &value_cache[key_value_id];
  context_buf = &context_buf[qkv_id];

  Access_t bias_r, key_val_r, query_buf_r;

  // each warp will have its own copy of sq
  query_buf_r.v = *((copy_t *)query_buf + lane_id);
  bias_r.v = *((copy_t *)Q_bias + lane_id);
  float qb_r[elems_per_thread];
  for (int i = 0; i < elems_per_thread; ++i)
  {
    qb_r[i] =  (float)query_buf_r.x[i] + (float)bias_r.x[i];
  }

  //offset for each step
  int offset =  head_num * size_per_head;

  bias_r.v = *((copy_t *) K_bias + lane_id);
  for(int ite = warp_id; ite < length; ite += warp_num)
  {
    key_val_r.v = *((copy_t *)&key_cache[ite * offset] + lane_id);

    //For the first step, we should add bias to key memory cache.
    //The KV memory cache only need to be updated at the first step.
    if (step == 1)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        key_val_r.x[i] = (float)key_val_r.x[i] + (float)bias_r.x[i];
      }
      *((copy_t *)&key_cache[ite * offset] + lane_id) = key_val_r.v;
    }
    float val = 0.f;
    for (int i = 0; i < elems_per_thread; i++)
    {
      val = val +  (float)key_val_r.x[i] * qb_r[i] * scalar;
    }
    float qk = cub::WarpReduce<float>(temp_storage[warp_id]).Sum(val);
    if (lane_id == 0)
    {
      logits[ite] = qk; 
    }
  }
  __syncthreads();

  __shared__ float s_max_val, s_sum;
  float local_i = -1e20f;
  for(int i = tid; i < length; i += blockDim.x)
    local_i = max(local_i, logits[i]);

  float max_val = MaxValBlockReduce(max_val_block_temp_storage).Reduce(local_i, cub::Max());
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();

  float local_o = 0.0f;
  for(int i = tid; i < length; i += blockDim.x)
  {
    logits[i] = __expf(logits[i] - s_max_val);
    local_o += logits[i];
  }
  float val = BlockReduce(block_temp_storage).Sum(local_o);

  if(tid == 0)
    s_sum = val + 1e-6;
  __syncthreads();

  float s_sum_inverse = __fdividef(1.0f, s_sum);
  for(int i = tid; i < length; i += blockDim.x)
  {
    logits[i] = logits[i] * s_sum_inverse;
  }
  __syncthreads(); 

  // This optimization introduces discrepancy because of different order in FP32 summation
  float sum_r[elems_per_thread] = {0.f};
  bias_r.v = *((copy_t *) V_bias + lane_id);
  for(int ite = warp_id; ite < length; ite += warp_num)
  {
    key_val_r.v = *((copy_t *)&value_cache[ite * offset] + lane_id);

    //For the first step, we should add bias to key memory cache.
    if(step == 1)
    {
      for (int i = 0; i < elems_per_thread; i++)
      {
        key_val_r.x[i] = (float)key_val_r.x[i] + (float)bias_r.x[i];
      }
      *((copy_t *)&value_cache[ite * offset] + lane_id) = key_val_r.v;
    }
    for (int i = 0; i < elems_per_thread; ++i)
    {
      sum_r[i] += (float)key_val_r.x[i] * logits[ite]; 
    }
  }
  for (int i = 0; i < elems_per_thread; i++)
  {
    sq[warp_id * WARP_SIZE + lane_id].x[i] = sum_r[i];
  }
  __syncthreads();
  if (threadIdx.x < WARP_SIZE)
  {
    #pragma unroll
    for (int j = 1; j < warp_num; j++)
    {
      for (int i = 0; i < elems_per_thread; ++i)
      {
        sum_r[i] = sum_r[i] + (float)sq[j * WARP_SIZE + threadIdx.x].x[i];
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < elems_per_thread; i++)
  {
    key_val_r.x[i] = sum_r[i];
  }
  if (threadIdx.x  < WARP_SIZE)
  {
    *((copy_t *)context_buf + lane_id) = key_val_r.v;
  }
}

template<typename T>
__global__
void cross_attention_kernel(
  T* query_buf, const T* Q_bias,
  T* key_cache, const T* K_bias,
  T* value_cache, const T* V_bias,
  const int* length_per_sample, T* context_buf, 
  int batch_size, int head_num, int size_per_head, int step, const int seq_len, const T scalar)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x / head_num;
  int head_id = blockIdx.x % head_num;

  extern __shared__ __align__(sizeof(T)) unsigned s_buf[];
  T* sq = reinterpret_cast<T *>(s_buf);
  T* logits = reinterpret_cast<T *>(&sq[size_per_head]);

  int length = __ldg(&length_per_sample[bid]);

  int qkv_id = bid * head_num * size_per_head + head_id * size_per_head + tid;
  int qkv_bias_id = head_id * size_per_head + tid;

  if(tid < size_per_head)
    sq[tid] = query_buf[qkv_id] + Q_bias[qkv_bias_id];
  __syncthreads();

  for(int ite = 0; ite < length; ++ite)
  {
    int key_id = bid * (seq_len * head_num * size_per_head) + ite * (head_num * size_per_head)
     + head_id * size_per_head + tid;

    T key = tid < size_per_head ? key_cache[key_id] : (T)(0.0f);

    //For the first step, we should add bias to key memory cache.
    //The KV memory cache only need to be updated at the first step.
    if(step == 1 && tid < size_per_head)
    {
      key += K_bias[head_id * size_per_head + tid];
      key_cache[key_id] = key;
    }

    T val = (tid < size_per_head) ? key * sq[tid] * scalar : (T)(0.0f);
    T qk = blockReduceSum(val);
    if(threadIdx.x == 0)
      logits[ite] = qk;
    __syncthreads(); //try to remove
  }
  __syncthreads();

  __shared__ float s_max_val, s_sum;

  float local_i = tid < length ? (float)logits[tid] : -1e20f; 
  float max_val = blockReduceMax<float>(local_i);
  if(tid == 0)
    s_max_val = max_val;
  __syncthreads();

  local_i -= s_max_val;
  float local_o = tid < length ? __expf(local_i) : 0.0f;
  float val = blockReduceSum<float>(local_o);

  if(tid == 0)
    s_sum = val + 1e-6;
  __syncthreads();
  if(tid < length)
    logits[tid] = local_o / s_sum;
  __syncthreads();

  if(tid < size_per_head)
  {
    T sum = (T)0.0f;
    for(int ite = 0; ite < length; ++ite)
    {
      int value_id = bid * seq_len * head_num * size_per_head + ite * head_num * size_per_head 
        + head_id * size_per_head + tid;

      T value = value_cache[value_id];

      //for the first step, we should add bias to key memory cache
      if(step == 1)
      {
        value += V_bias[head_id * size_per_head + tid];
        value_cache[value_id] = value;
      }  
      sum += value * logits[ite];
    }
    context_buf[bid * head_num * size_per_head + head_id * size_per_head + tid] = sum;
  }
}

template <typename T>
void cross_attention_dispatch(T* query_buf, const T* Q_bias, 
  T* key_cache, const T* K_bias, T* value_cache, const T* V_bias, const int* length,
  T* context_buf, int batch_size, int head_num, int size_per_head, int step, int seq_len, cudaStream_t stream)
  {
    const int block_sz = ATTENTION_BLOCK_SIZE;
    float scalar = 1.f / sqrtf(size_per_head * 1.0f);

    dim3 grid(batch_size * head_num);

    int cond = size_per_head * ((ATTENION_OPT)? 1:0);
    switch (cond)
    {
      case 32:
        cross_attention_kernel_opt<T, 32, block_sz><<<grid, block_sz, 0, stream>>>(
          query_buf, Q_bias, key_cache, K_bias, value_cache, V_bias, length, context_buf,  
          batch_size, head_num, step, seq_len, scalar);
        break;
      case 64:
        cross_attention_kernel_opt<T, 64, block_sz><<<grid, block_sz, 0, stream>>>(
          query_buf, Q_bias, key_cache, K_bias, value_cache, V_bias, length, context_buf,  
          batch_size, head_num, step, seq_len, scalar);
        break;
      case 128:
        cross_attention_kernel_opt<T, 128, block_sz><<<grid, block_sz, 0, stream>>>(
          query_buf, Q_bias, key_cache, K_bias, value_cache, V_bias, length, context_buf,  
          batch_size, head_num, step, seq_len, scalar);
        break;
      default:
        // default path

        int block_size = 128;

        if(seq_len <= 64)
          block_size = 64;
        else if(seq_len <= 128 && seq_len > size_per_head)
          block_size = 128;
        else if(seq_len > 128 && seq_len <= 256)
          block_size = 256;
        else if(seq_len > 256 && seq_len <= 512)
          block_size = 512;
        else
          block_size = 1024;

        if(block_size < size_per_head)
          block_size = size_per_head;

        assert(block_size <= 1024);
        dim3 block(block_size);
        
        int shared_size = sizeof(T) * (size_per_head + seq_len);
        cross_attention_kernel<T><<<grid, block, shared_size, stream>>>(
          query_buf, Q_bias, 
          key_cache, K_bias,
          value_cache, V_bias,
          length, context_buf,  
          batch_size,
          head_num, size_per_head, step, seq_len, scalar);
    }
  }

/* attention with source sentence */
template<OperationType OpType_>
void OpenDecoder<OpType_>::cross_multi_head_attention(
  const DataType_* from_tensor,
  const DataType_* memory_tensor,
  DataType_* key_mem_cache,
  DataType_* value_mem_cache,
  DataType_* decoder_output,
  const int* length,
  const int seq_len,
  const int step)
{
  int m = batch_size_;
  int n = hidden_units_;
  int k = hidden_units_;

  DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;

  //reuse the query_buf 
  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &alpha, 
    param_.cross_attention.query_weight.kernel, AType_, n, 
    from_tensor, BType_, k, 
    &beta, 
    query_buf_, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

  if(step == 1)
  {
    m *= seq_len;
    k = memory_hidden_units_;
    check_cuda_error(cublasGemmEx(param_.cublas_handle, 
      CUBLAS_OP_N, CUBLAS_OP_N, 
      n, m, k, 
      &alpha, 
      param_.cross_attention.key_weight.kernel, AType_, n, 
      memory_tensor, BType_, k, 
      &beta, 
      key_mem_cache, CType_, n, 
      computeType_, 
      static_cast<cublasGemmAlgo_t>(cublasAlgo_[1])));

    check_cuda_error(cublasGemmEx(param_.cublas_handle, 
      CUBLAS_OP_N, CUBLAS_OP_N, 
      n, m, k, 
      &alpha, 
      param_.cross_attention.value_weight.kernel, AType_, n, 
      memory_tensor, BType_, k, 
      &beta, 
      value_mem_cache, CType_, n, 
      computeType_, 
      static_cast<cublasGemmAlgo_t>(cublasAlgo_[1])));
    k = hidden_units_;
  }

  cross_attention_dispatch<DataType_>(
    query_buf_, param_.cross_attention.query_weight.bias, 
    key_mem_cache, param_.cross_attention.key_weight.bias,
    value_mem_cache, param_.cross_attention.value_weight.bias,
    length, context_buf_, batch_size_,
    head_num_, size_per_head_, step, seq_len, param_.stream); 

  m = batch_size_;
  n = head_num_ * size_per_head_;
  k = n;

  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &alpha, 
    param_.cross_attention.attention_output_weight.kernel, AType_, n, 
    context_buf_, BType_, k, 
    &beta, 
    decoder_output, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));
}

template <typename T>
__global__
void decoder_norm1_kernel(const T* __restrict input, 
                          const T* __restrict gamma, 
                          const T* __restrict beta, 
                          T* output, 
                          int m, int n)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = tid < n ? (float)(__ldg(&input[blockIdx.x * n + tid])) : 0.0f;

  mean = blockReduceSum<float>(local_out);

  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum<float>(tid < n ? (local_out - s_mean) * (local_out - s_mean) : 0.0f);

  if(threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6);

  __syncthreads();

  if(tid < n)
    output[blockIdx.x * n + tid] = 
      (T)(((local_out - s_mean) * s_variance) * (float)(__ldg(&gamma[tid])) + (float)(__ldg(&beta[tid])));
}

template <>
__global__
void decoder_norm1_kernel(const half* __restrict input, 
                          const half* __restrict gamma, 
                          const half* __restrict beta, 
                          half* output, 
                          int m, int n)
{
  const int tid = threadIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;
  float2 local_out_fp2;

  const half2* input_ptr = (const half2*)input;
  const half2* gamma_ptr = (const half2*)gamma;
  const half2* beta_ptr = (const half2*)beta;
  half2* output_ptr = (half2*)output;

  float local_out = 0.0f;
  int id = blockIdx.x * blockDim.x + tid;
  if(tid < blockDim.x)
  {
    local_out_fp2 = __half22float2(__ldg(&input_ptr[id]));
    local_out += local_out_fp2.x;
    local_out += local_out_fp2.y;
  }

  mean = blockReduceSum<float>(local_out);
  if(tid == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum<float>(tid < blockDim.x ? 
    (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean) + (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean)
    : 0.0f);
  if(tid == 0)
    s_variance = rsqrtf(variance / n + 1e-6);
  __syncthreads();

  if(tid < blockDim.x)
  {
    float2 gamma_val = __half22float2(__ldg(&gamma_ptr[tid]));
    float2 beta_val = __half22float2(__ldg(&beta_ptr[tid]));
    local_out_fp2.x = (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
    local_out_fp2.y = (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
    output_ptr[id] = __float22half2_rn(local_out_fp2);
  }
}

template <typename T>
__global__
void decoder_norm2_kernel(const T* __restrict input, 
                          const T* __restrict gamma, 
                          const T* __restrict beta, 
                          const T* __restrict bias, 
                          T* output, T* norm_output, 
                          int m, int n)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;
  if(tid < n)
  {
    local_out = (float)(__ldg(&input[blockIdx.x * n + tid]));
    local_out += (float)(output[blockIdx.x * n + tid]);
    local_out += (float)(__ldg(&bias[tid]));
    output[blockIdx.x * n + tid] = (T)local_out;
  }

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum<float>(tid < n ? (local_out - s_mean) * (local_out - s_mean) : 0.0f);
  if(threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6);
  __syncthreads();

  if(tid < n)
    norm_output[blockIdx.x * n + tid] = 
      (T)((local_out - s_mean) * s_variance * (float)(__ldg(&gamma[tid])) + (float)(__ldg(&beta[tid])));
}

template <>
__global__
void decoder_norm2_kernel(const half* __restrict input, 
                          const half* __restrict gamma, 
                          const half* __restrict beta, 
                          const half* __restrict bias, 
                          half* output, half* norm_output, 
                          int m, int n)
{
  const int tid = threadIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;
  float2 local_out_fp2;

  const half2* input_ptr = (const half2*)input;
  const half2* gamma_ptr = (const half2*)gamma;
  const half2* beta_ptr = (const half2*)beta;
  const half2* bias_ptr = (const half2*)bias;
  half2* output_ptr = (half2*)output;
  half2* norm_output_ptr = (half2*)norm_output;

  float local_out = 0.0f;
  int id = blockIdx.x * blockDim.x + tid;
  if(tid < blockDim.x)
  {
    output_ptr[id] = __hadd2(__hadd2(output_ptr[id], __ldg(&input_ptr[id])), __ldg(&bias_ptr[tid]));
    local_out_fp2 = __half22float2(output_ptr[id]);
    local_out += local_out_fp2.x;
    local_out += local_out_fp2.y;
  }

  mean = blockReduceSum<float>(local_out);
  if(tid == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum<float>(tid < blockDim.x ? 
    (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean) + (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean)
    : 0.0f);
  if(tid == 0)
    s_variance = rsqrtf(variance / n + 1e-6);
  __syncthreads();

  if(tid < blockDim.x)
  {
    float2 gamma_val = __half22float2(__ldg(&gamma_ptr[tid]));
    float2 beta_val = __half22float2(__ldg(&beta_ptr[tid]));
    local_out_fp2.x = (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
    local_out_fp2.y = (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
    norm_output_ptr[id] = __float22half2_rn(local_out_fp2);
  }
}

template<OperationType OpType_>
void OpenDecoder<OpType_>::decoder_norm1(
  const DataType_* input,
  const DataType_* gamma,
  const DataType_* beta,
  DataType_* output,
  int m, int n)
{
  dim3 grid(m);
  dim3 block(min(n, 1024));

  /* For general cases, n is equal to hidden_units, e.g., 512/1024.
     Since we have warp shuffle inside the code, block.x % 32 should be 0.
  */
  if(n % 32 != 0)
    block.x = 1024;

  block.x = block.x / (4 / sizeof(DataType_)); // if using half, only need half of block.x
  assert(block.x <= 1024);

/* should pay attention to the rsqrt precision*/
  decoder_norm1_kernel<DataType_><<<grid, block, 0, param_.stream>>>(input, gamma, beta, output, m, n);
}

template<OperationType OpType_>
void OpenDecoder<OpType_>::decoder_norm2(
  const DataType_* input,
  const DataType_* gamma,
  const DataType_* beta,
  const DataType_* bias,
  DataType_* output,
  DataType_* norm_output,
  int m, int n)
{
  dim3 grid(m);
  dim3 block(min(n, 1024));

  
  /* For general cases, n is equal to hidden_units, e.g., 512/1024.
  Since we have warp shuffle inside the code, block.x % 32 should be 0.
  */
  
  if(n % 32 != 0)
  block.x = 1024;
  
  block.x = block.x / (4 / sizeof(DataType_)); // if using half, only need half of block.x
  assert(block.x <= 1024);

  /* should pay attention to the rsqrt precision*/
  decoder_norm2_kernel<DataType_><<<grid, block, 0, param_.stream>>>(input, gamma, beta, bias, output, norm_output, m, n);
}

template<OperationType OpType_>
void OpenDecoder<OpType_>::ffn(
  const DataType_* input,
  DataType_* ffn_inner,
  DataType_* output,
  const int m,
  const int inner_size,
  const int n)
{
  int m1 = m, k1 = n, n1 = inner_size;
  DataType_ alpha = (DataType_)1.0f;
  DataType_ beta = (DataType_)0.0f;

  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n1, m1, k1, 
    &alpha, 
    param_.ffn.intermediate_weight.kernel, AType_, n1, 
    input, BType_, k1, 
    &beta, 
    ffn_inner, CType_, n1, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[2])));

  dim3 grid(m1);
  dim3 block(n1 / 4);

  assert(block.x <= 1024);

  add_bias_relu<DataType_><<<grid, block, 0, param_.stream>>>(ffn_inner, param_.ffn.intermediate_weight.bias, m1, n1);

  int m2 = m, n2 = n, k2 = inner_size;
  check_cuda_error(cublasGemmEx(param_.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n2, m2, k2, 
    &alpha, 
    param_.ffn.output_weight.kernel, AType_, n2, 
    ffn_inner, BType_, k2, 
    &beta, 
    output, CType_, n2, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_[3])));
}

template <typename T>
__global__ 
void add_bias_input_kernel(T* output, const T* input, const T* bias, const int m, const int n)
{
  int id = blockIdx.x * n + threadIdx.x;
  output[id] = output[id] + input[id] + __ldg(&bias[threadIdx.x]);
}

template<OperationType OpType_>
void OpenDecoder<OpType_>::add_bias_input(DataType_* output, const DataType_* input, const int m, const int n)
{
  dim3 grid(m);
  dim3 block(n);
  assert(n <= 1024);
  add_bias_input_kernel<<<grid, block, 0, param_.stream>>>(output, input, param_.ffn.output_weight.bias, m, n);
}

template void OpenDecoder<OperationType::FP32>::masked_multi_head_attention(
  const float* from_tensor,
  float* key_cache,
  float* value_cache,
  float* decoder_output,
  const int step);

template void OpenDecoder<OperationType::FP16>::masked_multi_head_attention(
  const half* from_tensor,
  half* key_cache,
  half* value_cache,
  half* decoder_output,
  const int step);

template void OpenDecoder<OperationType::FP32>::cross_multi_head_attention(
  const float* from_tensor,
  const float* memory_tensor,
  float* key_mem_cache,
  float* value_mem_cache,
  float* decoder_output,
  const int* length,
  const int max_seq_len,
  const int step);

template void OpenDecoder<OperationType::FP16>::cross_multi_head_attention(
  const half* from_tensor,
  const half* memory_tensor,
  half* key_mem_cache,
  half* value_mem_cache,
  half* decoder_output,
  const int* length,
  const int max_seq_len,
  const int step);

template void OpenDecoder<OperationType::FP32>::ffn(
  const float* input,
  float* ffn_inner, 
  float* otuput,
  const int m,
  const int inner_size,
  const int n);

template void OpenDecoder<OperationType::FP16>::ffn(
  const half* input,
  half* ffn_inner, 
  half* otuput,
  const int m,
  const int inner_size,
  const int n);

template void OpenDecoder<OperationType::FP32>::decoder_norm1(
  const float* input,
  const float* gamma,
  const float* beta,
  float* output,
  int m, int n);

template void OpenDecoder<OperationType::FP16>::decoder_norm1(
  const half* input,
  const half* gamma,
  const half* beta,
  half* output,
  int m, int n);

template void OpenDecoder<OperationType::FP32>::decoder_norm2(
  const float* input,
  const float* gamma,
  const float* beta,
  const float* bias,
  float* output,
  float* norm_output,
  int m, int n);

template void OpenDecoder<OperationType::FP16>::decoder_norm2(
  const half* input,
  const half* gamma,
  const half* beta,
  const half* bias,
  half* output,
  half* norm_output,
  int m, int n);

template void OpenDecoder<OperationType::FP32>::add_bias_input(
  float* output,
  const float* input,
  const int m,
  const int n);

template void OpenDecoder<OperationType::FP16>::add_bias_input(
  half* output,
  const half* input,
  const int m,
  const int n);

}//namespace FasterTransformer

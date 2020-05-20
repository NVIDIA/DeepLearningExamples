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

namespace fastertransformer{

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
  //__shared__ T shared[32]; 
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
template <typename T>
__global__ 
void masked_attention_kernel(T* query_buf, const T* self_Q_bias, 
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
      key += self_K_bias[qkv_bias_id];
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
        value += self_V_bias[qkv_bias_id];
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

  DataType_* key_buf_ = key_cache_ + (step - 1) * m * n;
  DataType_* value_buf_ = value_cache_ + (step - 1) * m * n;

  DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;

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

  dim3 grid(batch_size_ * head_num_);
  dim3 block(128);

  //suppose size_per_head <= 128
  if(step <= 64)
    block.x = 64;
  else if(step <= 128 && step > size_per_head_)
    block.x = 128;
  else if(step > 128 && step <= 256)
    block.x = 256;
  else if(step > 256 && step <= 512)
    block.x = 512;
  else
    block.x = 1024;

  if(block.x < size_per_head_)
    block.x = size_per_head_;
  
  assert(block.x <= 1024);

  DataType_ scalar = 1 / sqrtf(size_per_head_ * 1.0f);

  int shared_size = sizeof(DataType_) * (size_per_head_ + step);

  masked_attention_kernel<DataType_><<<grid, block, shared_size, param_.stream>>>(
    query_buf_, param_.self_attention.query_weight.bias, 
    key_cache_, param_.self_attention.key_weight.bias,
    value_cache_, param_.self_attention.value_weight.bias,
    context_buf_, batch_size_,
    head_num_, size_per_head_, step, scalar);

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

  dim3 grid(batch_size_ * head_num_);
  dim3 block(128);

  if(seq_len <= 64)
    block.x = 64;
  else if(seq_len <= 128 && seq_len > size_per_head_)
    block.x = 128;
  else if(seq_len > 128 && seq_len <= 256)
    block.x = 256;
  else if(seq_len > 256 && seq_len <= 512)
    block.x = 512;
  else
    block.x = 1024;

  if(block.x < size_per_head_)
    block.x = size_per_head_;

  assert(block.x <= 1024);
  
  DataType_ scalar = 1 / sqrtf(size_per_head_ * 1.0f);

  int shared_size = sizeof(DataType_) * (size_per_head_ + seq_len);
  cross_attention_kernel<DataType_><<<grid, block, shared_size, param_.stream>>>(
    query_buf_, param_.cross_attention.query_weight.bias, 
    key_mem_cache, param_.cross_attention.key_weight.bias,
    value_mem_cache, param_.cross_attention.value_weight.bias,
    length, context_buf_,  
    batch_size_,
    head_num_, size_per_head_, step, seq_len, scalar);

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
void decoder_norm1_kernel(const T* input, const T* gamma, const T* beta, T* output, int m, int n)
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

template <typename T>
__global__
void decoder_norm2_kernel(const T* input, const T* gamma, const T* beta, const T* bias, T* output, T* norm_output, int m, int n)
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

  assert(n <= 1024);

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

  assert(n <= 1024);

  /* For general cases, n is equal to hidden_units, e.g., 512/1024.
     Since we have warp shuffle inside the code, block.x % 32 should be 0.
  */

  if(n % 32 != 0)
    block.x = 1024;

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

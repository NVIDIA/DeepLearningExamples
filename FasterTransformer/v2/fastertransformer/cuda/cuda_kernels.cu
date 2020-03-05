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
#include "fastertransformer/common.h"

#include "cuda_kernels.h"
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cfloat>
namespace fastertransformer{

#define FINAL_MASK 0xffffffff
#define CUDART_PI_F 3.141592654f

template <typename T>
__inline__ __device__
T gelu(T x)
{
  float cdf = 0.5f * (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
}

template <>
__inline__ __device__
half2 gelu(half2 val)
{
  half2 val_pow3 = __hmul2(val, __hmul2(val, val));
  float2 tmp_pow = __half22float2(val_pow3);
  float2 tmp =  __half22float2(val);

  tmp.x = 0.5f * (1.0f + tanhf((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
  tmp.y = 0.5f * (1.0f + tanhf((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
  return __hmul2(val, __float22half2_rn(tmp));
}

template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

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
  
  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)0.0f;
  val = warpReduceSum(val);
  return val;
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
  int lane = threadIdx.x & 0x1f; // in-warp idx
  int wid = threadIdx.x >> 5;  // warp idx

  val = warpReduceMax(val); // get maxx in each warp

  if(lane == 0) // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();


  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)-1e20f;
  val = warpReduceMax<T>(val);

  return val;
}


template <typename T>
__global__ 
void add_bias_act(T* out, const T* bias, int m, int n)
{
  T val, reg_bias;

  int row_id = blockIdx.x;
  int ite = n / blockDim.x;
  int tid = threadIdx.x;

  for(int i = 0; i < ite; ++i)
  {
    reg_bias = __ldg(&bias[i * blockDim.x + tid]);
    row_id = blockIdx.x;

    while(row_id < m){
      val = out[tid + i * blockDim.x + row_id * n]+ reg_bias;
      out[tid + i * blockDim.x + row_id * n] = gelu<T>(val);
      row_id += gridDim.x;
    }
  }
}

template <>
__global__ 
void add_bias_act(half* out, const half* bias, int m, int n)
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

    while(row_id < m){
      val = out_ptr[tid + i * blockDim.x + row_id * n / 2];
      val = __hadd2(val, reg_bias);
      out_ptr[tid + i * blockDim.x + row_id * n / 2] = gelu<half2>(val);
      row_id += gridDim.x;
    }
  }
}

template <typename T>
__global__ 
void add_bias_input_layernorm(T* out, const T* input, const T* bias, const T* gamma, const T* beta, int m, int n)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;
  for(int i = tid; i < n; i += blockDim.x)
    local_out += (float)(out[blockIdx.x * n + i] + input[blockIdx.x * n + i] + __ldg(&bias[i]));

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum<float>((local_out - s_mean) * (local_out - s_mean));
  if(threadIdx.x == 0)
    s_variance = variance / n + 1e-6f;
  __syncthreads();

  for(int i = tid; i < n; i += blockDim.x)
    out[blockIdx.x * n + i] = 
	    (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg(&gamma[i])) + (float)(__ldg(&beta[i])));
}

template <>
__global__ 
void add_bias_input_layernorm(half* out, const half* input, const half* bias, 
  const half* gamma, const half* beta, int m, int n)
{

  int tid = threadIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;
  float2 local_out_fp2;

  half2* out_ptr = (half2*)out;
  const half2* input_ptr = (const half2*)input;
  const half2* bias_ptr = (const half2*)bias;
  const half2* gamma_ptr = (const half2*)gamma;
  const half2* beta_ptr = (const half2*)beta;
 
  float local_out = 0.0f;
  int id = blockIdx.x * n / 2 + tid; 
  local_out_fp2 = __half22float2(__hadd2(__hadd2(out_ptr[id], input_ptr[id]), __ldg(&bias_ptr[tid])));
  local_out += local_out_fp2.x;
  local_out += local_out_fp2.y;

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  variance = (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
  variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
  variance = blockReduceSum<float>(variance);
  if(threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

  float2 gamma_val = __half22float2(__ldg(&gamma_ptr[tid]));
  float2 beta_val = __half22float2(__ldg(&beta_ptr[tid]));
  local_out_fp2.x = (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
  local_out_fp2.y = (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
  out_ptr[id] = __float22half2_rn(local_out_fp2);
}

template<typename T>
__global__
void broadcast_kernel(T* log_probs, T* cum_log_probs, const int batch_size, const int beam_width, const int vocab_size, const int N)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int bid = tid / vocab_size;

  if(tid < N)
    log_probs[tid] += cum_log_probs[bid];
}

template <typename T>
void add_bias_act_kernelLauncher(T* out, const T* bias, int m, int n, cudaStream_t stream)
{
//  dim3 grid(m / 64);
  dim3 grid(m / 4);
  dim3 block(n / 4);
  assert(block.x <= 1024);
//  dim3 block(n);
  add_bias_act<T><<<grid, block, 0, stream>>>(out, bias, m, n);
}

template<typename T>
void add_bias_input_layernorm_kernelLauncher(T* out, const T* input, const T* bias, 
  const T* gamma, const T* beta, int m, int n, cudaStream_t stream)
{
  dim3 grid(m);
  dim3 block(n);
  assert(n <= 1024);
  add_bias_input_layernorm<T><<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n);
}


template <>
void add_bias_input_layernorm_kernelLauncher(half* out, const half* input, const half* bias, 
  const half* gamma, const half* beta, int m, int n, cudaStream_t stream)
{
  dim3 grid(m);
  dim3 block(n / 2);
  assert(n / 2 <= 1024);
  add_bias_input_layernorm<half><<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n);
}

void broadcast_kernelLauncher(float* log_probs, float* cum_log_probs, const int batch_size, const int beam_width, 
  const int vocab_size, cudaStream_t stream)
{
  
  int N = batch_size * beam_width * vocab_size;
  dim3 block(1024);
  dim3 grid((N - 1) / block.x + 1);

  broadcast_kernel<float><<<grid, block, 0, stream>>>(log_probs, cum_log_probs, batch_size, beam_width, vocab_size, N);
}

template <typename T>
__global__
void topK_kernel(const T* log_probs, int* ids, const int batch_size, const int N, const int K)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val, max_val;
  __shared__ float s_max_val;
  for(int ite = 0; ite < batch_size; ++ite)
  {
    bool choosed = false;
    val = (tid < N ) ? (float)log_probs[ite * N + tid] : -1e20f;
    
    for(int kids = 0; kids < K; ++kids)
    {
      max_val = blockReduceMax<float>(val);
      
      if(threadIdx.x == 0)
        s_max_val = max_val;
      __syncthreads();

      if(s_max_val == val && !choosed && tid < N) 
      {
        ids[ite * gridDim.x * K + blockIdx.x * K + kids] = tid + ite * N;
        val = -1e20f;
        choosed = true;
      }
    }
  }
}

template <typename T>
__global__
void topK_kernel_2nd(const T* log_probs, int* ids, const int batch_size, const int N, const int K, const int id_offset)
{
  int tid = threadIdx.x;
  float val, max_val;
  __shared__ float s_max_val;
  __shared__ int beam_index;
  __shared__ int ids_before_sort[16];

  for(int ite = 0; ite < batch_size; ++ite)
  {
    bool choosed = false;
    const int id = (tid < N) ? ids[ite * N + tid] : -1;
    val = (tid < N) ? (float)log_probs[id] : -1e20f;

    __syncthreads();

    if(tid == 0) beam_index = 0;
    if(tid < 16) ids_before_sort[tid] = -1;
    
    __syncthreads();
    while(beam_index < K){
      int begin_beam_index = beam_index;
      max_val = blockReduceMax<float>(val);
      if(threadIdx.x == 0){
        s_max_val = max_val;
      }
      __syncthreads();
      if(s_max_val == val && !choosed && id != -1)
      {
        int id_offset_ = atomicAdd(&beam_index, 1);
        ids_before_sort[id_offset_] = id;
        val = -1e20f;
        choosed = true;
      }
      __syncthreads();
     
      // simply sort the ids
      if(threadIdx.x == 0 && beam_index - begin_beam_index > 1){
        for(int i = begin_beam_index; i < beam_index; i++){
          for(int j = i; j < beam_index; j++){
            if(ids_before_sort[j] < ids_before_sort[i]){
              int tmpid = ids_before_sort[j];
              ids_before_sort[j] = ids_before_sort[i];
              ids_before_sort[i] = tmpid;
            }
          }
        }
      }
    }
    __syncthreads();
    if(tid < K) ids[ite * K + tid] = ids_before_sort[tid];
    __syncthreads();
  }
}

void topK(const float* log_probs, int* ids, const int batch_size, const int beam_width, const int vocab_size,
  cudaStream_t stream)
{
  int N = beam_width * vocab_size;
  dim3 block(1024);
  dim3 grid((N - 1) / block.x + 1);
  /* First round topK, for each batch, get grid.x * K values */
  topK_kernel<float><<<grid, block, 0, stream>>>(log_probs, ids, batch_size, N, beam_width);
  /*Second round, for each batch, get the final TopK values out from grid.x * K values. */
  topK_kernel_2nd<float><<<1, block, 0, stream>>>(log_probs, ids, batch_size, beam_width * grid.x, beam_width, N);
}

template <typename T>
__global__
void update_kernel(T* log_probs, T* cum_log_probs, 
                  int* ids, bool* finished, 
                  int* parent_ids, int* sequence_length, 
                  int* word_ids, int* output_ids, 
                  const int batch_size, const int beam_width, 
                  const int vocab_size, const int end_id, 
                  int* finished_count)
{
  int tid = threadIdx.x;
  sequence_length[tid] = finished[tid] ? sequence_length[tid] : sequence_length[tid] + 1;

  int beam_id = ids[tid];
  beam_id /= vocab_size;
  int word_id = ids[tid];
  word_id %= vocab_size;

  cum_log_probs[tid] = log_probs[ids[tid]];
  sequence_length[tid] = sequence_length[beam_id];
  finished[tid] = word_id == end_id ? 1 : 0;
  parent_ids[tid] = beam_id;
  word_ids[tid] = word_id;
  output_ids[tid] = word_id;

  // TODO use reduce sum to compute how many sentence are finished
  // int fi = finished[tid]
  // int total_finish = reduceSum(fi);
}

template <typename T>
__global__ void embedding_lookup_kernel(const T* embedding_table, const int* word_ids,
    const int hidden_units, T* from_tensor)
{
  int write_pos = threadIdx.x + blockIdx.x * hidden_units;
  from_tensor[write_pos] = embedding_table[word_ids[blockIdx.x] * hidden_units + threadIdx.x];
}

void update(float* log_probs, float* cum_log_probs, 
            int* ids, bool* finished, 
            int* parent_ids, int* sequence_length,
            int* word_ids, int* output_ids, 
            const int batch_size, const int beam_width, 
            const int vocab_size, cudaStream_t stream, 
            const int end_id, int* finished_count)
{ 
  
  dim3 grid(1);
  dim3 block(batch_size * beam_width);

  assert(block.x <= 1024);

  update_kernel<float><<<grid, block, 0, stream>>>(log_probs, cum_log_probs, ids, 
                                                  finished, parent_ids, sequence_length,
                                                  word_ids, output_ids, batch_size, 
                                                  beam_width, vocab_size, end_id, 
                                                  finished_count);
}

template <typename T>
void embedding_lookup(const T* embedding_table, const int* word_ids, T* from_tensor,
  const int batch_size, const int beam_width, const int hidden_units, cudaStream_t stream)
{
   dim3 grid(batch_size * beam_width);
   dim3 block(hidden_units);
   assert(hidden_units <= 1024);
   embedding_lookup_kernel<<<grid, block, 0, stream>>>(embedding_table, word_ids, hidden_units, from_tensor);
}

template <typename T>
__global__ void update_logits_kernel(T* logits, const T* bias, const int end_id, const bool* finished, const int n)
{
  int bid = blockIdx.x;
  bool finish = finished[bid];
  int offset = bid * n;

  float max_val = -1 * FLT_MAX;
  __shared__ float s_max_val;
  __shared__ float s_sum_val;

  for(int tid = threadIdx.x; tid < n; tid += blockDim.x)
  {
    if(finish)
      logits[offset + tid] = (tid == end_id) ? FLT_MAX : -1 * FLT_MAX;
    else
      logits[offset + tid] += bias[tid];
    max_val = max(max_val, logits[offset + tid]);
  }

  max_val = blockReduceMax<float>((float)max_val);
  if(threadIdx.x == 0)
    s_max_val = max_val;
  __syncthreads();

  float sum_val = 0.0f;
  for(int tid = threadIdx.x; tid < n; tid += blockDim.x)
  {
    logits[offset + tid] = __expf((float)logits[offset + tid] - s_max_val);
    sum_val += (float)logits[offset + tid];
  }

  sum_val = blockReduceSum<float>(sum_val);
  if(threadIdx.x == 0)
    s_sum_val = sum_val;
  __syncthreads();

  for(int tid = threadIdx.x; tid < n; tid += blockDim.x)
  {
    logits[offset + tid] = logf((float)logits[offset + tid] / s_sum_val);
  }
}

void update_logits(float* logits, const float* bias, const int end_id, const bool* finished, 
  const int m, const int n, cudaStream_t stream)
{
  dim3 grid(m);
  dim3 block(min(n, 1024));
  /*n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big. */
  update_logits_kernel<float><<<grid, block, 0, stream>>>(logits, bias, end_id, finished, n);
}

template <typename T>
__global__ void init_kernel(bool* finished, int* sequence_length, int* word_ids, T* cum_log_probs, const int sentence_id, const int n, const int beam_width)
{
  int tid = threadIdx.x;
  finished[tid] = false;
  sequence_length[tid] = 0;
  word_ids[tid] = sentence_id;
  cum_log_probs[tid] = (T)(tid % beam_width == 0 ? 0.0f: -1e20f);
}

template <typename T>
__global__ void update_KV_cache_kernel(
  T* key_src_cache, T* key_tgt_cache,
  T* value_src_cache, T* value_tgt_cache,
  const int* beam_ids, const int batch_size, const int beam_width, const int hidden_dim, const int cache_size, const int step, const int decoder_layers)
{
  int layer_id = blockIdx.x / batch_size / beam_width / step;
  int batch_id = (blockIdx.x % (batch_size * beam_width * step)) / (beam_width * step);
  int beam_id = (blockIdx.x % (beam_width * step)) / step;
  int step_id = blockIdx.x % step;

  int hidden_id = step_id * batch_size * beam_width * hidden_dim + 
    beam_ids[batch_id * beam_width + beam_id] * hidden_dim;

  int tgt_hidden_id = step_id * batch_size * beam_width * hidden_dim + 
    batch_id * beam_width * hidden_dim + beam_id * hidden_dim;

  T* key_src_ptr = key_src_cache + layer_id * cache_size;
  T* key_tgt_ptr = key_tgt_cache + layer_id * cache_size;
  T* value_src_ptr = value_src_cache + layer_id * cache_size;
  T* value_tgt_ptr = value_tgt_cache + layer_id * cache_size;


  for(int tid = threadIdx.x; tid < hidden_dim; tid += blockDim.x)
  {
    key_tgt_ptr[tgt_hidden_id + tid] = key_src_ptr[hidden_id + tid];
    value_tgt_ptr[tgt_hidden_id + tid] = value_src_ptr[hidden_id + tid];
  }
  
}
template <typename T>
void update_KV_cache(T** key_cache, T** value_cache, const int* beam_ids, const int batch_size, const int beam_width, const int hidden_dim,
  const int step, const int cache_size, const int decoder_layers, cudaStream_t stream)
{
  dim3 grid(decoder_layers * batch_size * beam_width * step);
  dim3 block(min(1024, hidden_dim));

  int src_id = step & 0x1;
  int tgt_id = 1 - src_id;

  update_KV_cache_kernel<<<grid, block, 0, stream>>>(
    key_cache[src_id], key_cache[tgt_id],
    value_cache[src_id], value_cache[tgt_id],
    beam_ids, batch_size, beam_width, hidden_dim, cache_size, step, decoder_layers);
}

void init(bool* finished, int* sequence_length, int* word_ids, float* cum_log_probs, const int sentence_id, const int batch_size, 
  const int beam_width, cudaStream_t stream)
{
  dim3 grid(1);
  dim3 block(min(1024, batch_size * beam_width));

  assert(batch_size * beam_width <= 1024);
  
  init_kernel<float><<<grid, block, 0, stream>>>(finished, sequence_length, word_ids, cum_log_probs, sentence_id, batch_size * beam_width, beam_width);
}


template<typename T>
__global__
void sine_position_encoder_kernel(T* output, int step, int n){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  float half_n = (float)n / 2.;

  // input = input * hidden_dim**0.5
  output[bid * n + tid] = output[bid * n + tid] * (T)sqrtf(float(n));

  float log_timescale_increment = __logf(10000) / (half_n - 1.f);
  float inv_timescales = __expf( (tid % (int)half_n) * -1 * log_timescale_increment );
  float scaled_time = inv_timescales * step;
  
  T encoding_val = (tid < half_n) ? (T) __sinf(scaled_time) : (T) __cosf(scaled_time);
  output[bid * n + tid] = output[bid * n + tid]  + encoding_val;
}

template<typename T>
void sine_position_encoder(
  T* output,
  int step,
  int m, int n, cudaStream_t stream)
{
  dim3 grid(m);
  dim3 block(n);
  assert(n <= 1024);
  sine_position_encoder_kernel<T><<<grid, block, 0, stream>>>(output, step, n);
}

template void add_bias_act_kernelLauncher<float>(
  float* out, const float* bias, int m, int n, cudaStream_t stream);

template void add_bias_input_layernorm_kernelLauncher<float>(
  float* out, const float* input, const float* bias, const float* gamma, const float* beta, 
  int m, int n, cudaStream_t stream);

template void add_bias_act_kernelLauncher<half>(
  half* out, const half* bias, int m, int n, cudaStream_t stream);

template void add_bias_input_layernorm_kernelLauncher<half>(
  half* out, const half* input, const half* bias, const half* gamma, const half* beta, 
  int m, int n, cudaStream_t stream);

template void embedding_lookup(const float* embedding_table, const int* word_ids, float* from_tensor,
  const int batch_size, const int beam_width, const int hidden_units, cudaStream_t stream);

template void embedding_lookup(const half* embedding_table, const int* word_ids, half* from_tensor,
  const int batch_size, const int beam_width, const int hidden_units, cudaStream_t stream);

template void update_KV_cache(float** key_cache, float** value_cache, const int* beam_ids, const int batch_size, const int beam_width, const int hidden_dim,
  const int step, const int cache_size, const int decoder_layers, cudaStream_t stream);

template void update_KV_cache(half** key_cache, half** value_cache, const int* beam_ids, const int batch_size, const int beam_width, const int hidden_dim,
  const int step, const int cache_size, const int decoder_layers, cudaStream_t stream);

template void sine_position_encoder(
  float* output,
  int step,
  int m, int n,
  cudaStream_t stream);

template void sine_position_encoder(
  half* output,
  int step,
  int m, int n,
  cudaStream_t stream);

}//namespace 

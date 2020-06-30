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
#include "cub/cub.cuh"
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cfloat>

namespace fastertransformer
{
  /* ********************************** common kernel *********************************** */

  template <typename T>
  __global__ void init_kernel(bool* finished, 
                              int* sequence_length, 
                              int* word_ids, 
                              T* cum_log_probs, 
                              const int sentence_id, 
                              const int n, 
                              const int beam_width)
  {
    int tid = threadIdx.x;
    finished[tid] = false;
    sequence_length[tid] = 0;
    word_ids[tid] = sentence_id;
    cum_log_probs[tid] = (T)(tid % beam_width == 0 ? 0.0f: -1e20f);
  }

  void init_kernelLauncher(bool* finished, 
            int* sequence_length, 
            int* word_ids, 
            float* cum_log_probs, 
            const int sentence_id, 
            const int batch_size, 
            const int beam_width, 
            cudaStream_t stream)
  {
    dim3 grid(1);
    dim3 block(min(1024, batch_size * beam_width));
    assert(batch_size * beam_width <= 1024);
    
    init_kernel<float><<<grid, block, 0, stream>>>(finished, 
                                                  sequence_length, 
                                                  word_ids, 
                                                  cum_log_probs, 
                                                  sentence_id, 
                                                  batch_size * beam_width, 
                                                  beam_width);
  }

  template <typename T>
  __global__ void embedding_lookup_sine_position_encoding_kernel(T* from_tensor,
                                                                const T* embedding_table, 
                                                                const T* position_encoding,
                                                                const int* word_ids,
                                                                const int hidden_units)
  {
      const int tid = threadIdx.x;
      const int bid = blockIdx.x;
      const int write_pos = tid + bid * blockDim.x;
      // 1. lookup the table
      // 2. multiply hidden_dim**0.5
      // 3. add the position encoding
      from_tensor[write_pos] = embedding_table[word_ids[bid] * hidden_units + tid] * 
                                (T)sqrtf(float(hidden_units)) + position_encoding[tid];
  }

  template <typename T>
  void embedding_lookup_sine_position_encoding_kernel_launcher(T* from_tensor,
                                                              const T* embedding_table, 
                                                              const T* position_encoding,
                                                              const int* word_ids,
                                                              const int batch_size,
                                                              const int hidden_units, 
                                                              cudaStream_t stream)
  {
      assert(hidden_units <= 1024);
      dim3 grid(batch_size);
      dim3 block(hidden_units);
      embedding_lookup_sine_position_encoding_kernel<T><<<grid, block, 0, stream>>>(from_tensor,
                                                                                  embedding_table,
                                                                                  position_encoding,
                                                                                  word_ids,
                                                                                  hidden_units);
  }

  /* *************************** end of common kernel *********************************** */

  /* ********************************** BeamSearch kernel *********************************** */

  template<typename T>
  __global__
  void broadcast_kernel(T* log_probs, 
                        T* cum_log_probs, 
                        const int vocab_size, 
                        const int N)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = tid / vocab_size;

    if(tid < N)
      log_probs[tid] += cum_log_probs[bid];
}

  void broadcast_kernelLauncher(float* log_probs, 
                                float* cum_log_probs, 
                                const int batch_size, 
                                const int beam_width, 
                                const int vocab_size, 
                                cudaStream_t stream)
  {
    
    int N = batch_size * beam_width * vocab_size;
    dim3 block(1024);
    dim3 grid((N - 1) / block.x + 1);
  
    broadcast_kernel<float><<<grid, block, 0, stream>>>(log_probs, cum_log_probs, vocab_size, N);
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

    int beam_id = word_ids[tid] / vocab_size;
    int word_id = word_ids[tid] % vocab_size;

    cum_log_probs[tid] = log_probs[word_ids[tid]];
    sequence_length[tid] = sequence_length[beam_id];
    finished[tid] = word_id == end_id ? 1 : 0;
    parent_ids[tid] = beam_id;
    word_ids[tid] = word_id;
    output_ids[tid] = word_id;
  }

  void update_kernelLauncher(float* log_probs, float* cum_log_probs, 
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
  __global__
  void update_kernel_v2(bool* finished, int* parent_ids, 
                        int* sequence_length, 
                        int* word_ids, int* output_ids, 
                        const int vocab_size, const int end_id, 
                        int* finished_count)
  {
    int tid = threadIdx.x;
    sequence_length[tid] = finished[tid] ? sequence_length[tid] : sequence_length[tid] + 1;

    int beam_id = word_ids[tid] / vocab_size;
    int word_id = word_ids[tid] % vocab_size;

    sequence_length[tid] = sequence_length[beam_id];
    finished[tid] = word_id == end_id ? 1 : 0;
    parent_ids[tid] = beam_id;
    word_ids[tid] = word_id;
    output_ids[tid] = word_id;
  }

  void update_kernelLauncher_v2(bool* finished, int* parent_ids, 
                                int* sequence_length, int* word_ids, 
                                int* output_ids, 
                                int* finished_count,
                                DecodingBeamsearchArguments args,
                                cudaStream_t stream)
  {
    dim3 grid(1);
    dim3 block(args.batch_size_ * args.beam_width_);
    assert(block.x <= 1024);

    update_kernel_v2<float><<<grid, block, 0, stream>>>(finished, parent_ids, 
                                                        sequence_length, word_ids, 
                                                        output_ids, args.vocab_size_, 
                                                        args.end_id_, finished_count);
  }

  template <typename T>
  __global__ void update_KV_cache_kernel(const T* __restrict key_src_cache, 
                                        T* key_tgt_cache,
                                        const T* __restrict value_src_cache, 
                                        T* value_tgt_cache,
                                        const int* beam_ids, 
                                        const int batch_size, 
                                        const int beam_width, 
                                        const int hidden_dim, 
                                        const int cache_size, 
                                        const int step, 
                                        const int decoder_layers)
  {
    int layer_id = blockIdx.x / batch_size / beam_width / step;
    int batch_id = (blockIdx.x % (batch_size * beam_width * step)) / (beam_width * step);
    int beam_id = (blockIdx.x % (beam_width * step)) / step;
    int step_id = blockIdx.x % step;

    int hidden_id = step_id * batch_size * beam_width * hidden_dim + 
      beam_ids[batch_id * beam_width + beam_id] * hidden_dim;

    int tgt_hidden_id = step_id * batch_size * beam_width * hidden_dim + 
      batch_id * beam_width * hidden_dim + beam_id * hidden_dim;

    const T* key_src_ptr = key_src_cache + layer_id * cache_size;
    T* key_tgt_ptr = key_tgt_cache + layer_id * cache_size;
    const T* value_src_ptr = value_src_cache + layer_id * cache_size;
    T* value_tgt_ptr = value_tgt_cache + layer_id * cache_size;


    for(int tid = threadIdx.x; tid < hidden_dim; tid += blockDim.x)
    {
      key_tgt_ptr[tgt_hidden_id + tid] = key_src_ptr[hidden_id + tid];
      value_tgt_ptr[tgt_hidden_id + tid] = value_src_ptr[hidden_id + tid];
    }
    
  }

  template <>
  __global__ void update_KV_cache_kernel(const half* __restrict key_src_cache, 
                                        half* key_tgt_cache,
                                        const half* __restrict value_src_cache, 
                                        half* value_tgt_cache,
                                        const int* beam_ids, 
                                        const int batch_size, 
                                        const int beam_width, 
                                        const int hidden_dim, 
                                        const int cache_size, 
                                        const int step, 
                                        const int decoder_layers)
  {
    int layer_id = blockIdx.x / batch_size / beam_width / step;
    int batch_id = (blockIdx.x % (batch_size * beam_width * step)) / (beam_width * step);
    int beam_id = (blockIdx.x % (beam_width * step)) / step;
    int step_id = blockIdx.x % step;

    int hidden_id = (step_id * batch_size * beam_width * hidden_dim + 
      beam_ids[batch_id * beam_width + beam_id] * hidden_dim) / 2;

    int tgt_hidden_id = (step_id * batch_size * beam_width * hidden_dim + 
      batch_id * beam_width * hidden_dim + beam_id * hidden_dim) / 2;

    const half2* key_src_ptr = (const half2*)key_src_cache + layer_id * cache_size / 2;
    half2* key_tgt_ptr = (half2*)key_tgt_cache + layer_id * cache_size / 2;
    const half2* value_src_ptr = (const half2*)value_src_cache + layer_id * cache_size / 2;
    half2* value_tgt_ptr = (half2*)value_tgt_cache + layer_id * cache_size / 2;
    
    for(int tid = threadIdx.x; tid < hidden_dim / 2; tid += blockDim.x)
    {
      key_tgt_ptr[tgt_hidden_id + tid] = key_src_ptr[hidden_id + tid];
      value_tgt_ptr[tgt_hidden_id + tid] = value_src_ptr[hidden_id + tid];
    }
    
  }

  template <typename T>
  void update_KV_cache_kernelLauncher(T** key_cache, 
                                      T** value_cache, 
                                      const int* beam_ids, 
                                      const int batch_size, 
                                      const int beam_width, 
                                      const int hidden_dim,
                                      const int step, 
                                      const int cache_size, 
                                      const int decoder_layers, 
                                      cudaStream_t stream)
  {
    dim3 grid(decoder_layers * batch_size * beam_width * step);
    dim3 block(min(1024, hidden_dim));
    block.x = block.x / (4 / sizeof(T));

    int src_id = step & 0x1;
    int tgt_id = 1 - src_id;

    update_KV_cache_kernel<<<grid, block, 0, stream>>>(
      key_cache[src_id], key_cache[tgt_id],
      value_cache[src_id], value_cache[tgt_id],
      beam_ids, batch_size, beam_width, hidden_dim, cache_size, step, decoder_layers);
  }

  /* *************************** end of BeamSearch kernel *********************************** */

  /* ********************************** Sampling kernel *********************************** */
  __global__ void topp_initialization_kernel(bool* finished,
                                          int* sequence_length, 
                                          int* word_ids,
                                          int* topp_id_val_buf,
                                          int* topp_offset_buf,
                                          const int batch_size, 
                                          const int vocab_size,
                                          const int start_id)
  {
      int tid = threadIdx.x;
      int bid = blockIdx.x;
  
      if(bid == 0)
      {
          for(int i = tid; i < batch_size + 1; i+= blockDim.x)
          {
              topp_offset_buf[i] = i * vocab_size;
          }
          
          for(int i = tid; i < batch_size; i+= blockDim.x)
          {
              finished[i] = false;
              sequence_length[i] = 0;
              word_ids[i] = start_id; 
          }
      }
  
      int index = tid + bid * blockDim.x;
      while(index < batch_size * vocab_size)
      {
          topp_id_val_buf[index] = index % vocab_size;
          index += blockDim.x * gridDim.x;
      }
  }

  void topp_initialization_kernelLauncher(bool* finished,
                                          int* sequence_length, 
                                          int* word_ids,
                                          int* topp_id_val_buf,
                                          int* topp_offset_buf,
                                          DecodingSamplingArguments args,
                                          cudaStream_t stream)
  {
      topp_initialization_kernel<<<32, 512, 0, stream>>>(finished,
                                                      sequence_length,
                                                      word_ids,
                                                      topp_id_val_buf,
                                                      topp_offset_buf,
                                                      args.batch_size_, 
                                                      args.vocab_size_,
                                                      args.start_id_);
  }

  size_t get_topp_sort_temp_storage_size(const float* log_probs,
                                          const int* id_vals,
                                          float* sorted_log_probs,
                                          int* sorted_id_vals, 
                                          int* topp_offset_buf,
                                          const int batch_size,
                                          const int vocab_size)
  {
      void *d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;
      
      cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, 
                                                      temp_storage_bytes,
                                                      log_probs, 
                                                      sorted_log_probs,
                                                      id_vals, 
                                                      sorted_id_vals, 
                                                      vocab_size * batch_size,
                                                      batch_size, 
                                                      topp_offset_buf, topp_offset_buf + 1);
      return temp_storage_bytes;
  }
  /* *************************** end of Sampling kernel *********************************** */

  /* ********************************** Instantiation *********************************** */
  template 
  void embedding_lookup_sine_position_encoding_kernel_launcher(float* from_tensor,
                                                              const float* embedding_table, 
                                                              const float* position_encoding, 
                                                              const int* word_ids,
                                                              const int batch_size,
                                                              const int hidden_units, 
                                                              cudaStream_t stream);

  template 
  void embedding_lookup_sine_position_encoding_kernel_launcher(half* from_tensor,
                                                              const half* embedding_table, 
                                                              const half* position_encoding, 
                                                              const int* word_ids,
                                                              const int batch_size,
                                                              const int hidden_units, 
                                                              cudaStream_t stream);

  template void update_KV_cache_kernelLauncher(float** key_cache, 
                                              float** value_cache, 
                                              const int* beam_ids, 
                                              const int batch_size, 
                                              const int beam_width, 
                                              const int hidden_dim,
                                              const int step, 
                                              const int cache_size, 
                                              const int decoder_layers, 
                                              cudaStream_t stream);
  
  template void update_KV_cache_kernelLauncher(half** key_cache, 
                                              half** value_cache, 
                                              const int* beam_ids, 
                                              const int batch_size, 
                                              const int beam_width, 
                                              const int hidden_dim,
                                              const int step, 
                                              const int cache_size, 
                                              const int decoder_layers, 
                                              cudaStream_t stream);
    
  /* *************************** end of Instantiation *********************************** */

} // end of name space fastertransformer
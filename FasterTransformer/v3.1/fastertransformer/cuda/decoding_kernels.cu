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
#include <vector>
#include <type_traits>

namespace fastertransformer
{
  /* ********************************** common kernel *********************************** */

  template <typename T>
  __global__ void init_kernel(bool* finished, 
                              int* sequence_length, 
                              int* word_ids, 
                              T* cum_log_probs, 
                              const int sentence_id, 
                              const int beam_width)
  {
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16)? HALF_FLT_MAX : 1e20f;
    int tid = threadIdx.x;
    finished[tid] = false;
    sequence_length[tid] = 0;
    word_ids[tid] = sentence_id;
    cum_log_probs[tid] = (tid % beam_width == 0) ? (T)0.0f: -MAX_T_VAL;
  }

  template <typename T>
  void init_kernelLauncher(bool* finished, 
            int* sequence_length, 
            int* word_ids, 
            T* cum_log_probs, 
            const int sentence_id, 
            const int batch_size, 
            const int beam_width, 
            cudaStream_t stream)
  {
    dim3 grid(1);
    dim3 block(min(1024, batch_size * beam_width));
    assert(batch_size * beam_width <= 1024);
    
    init_kernel<T><<<grid, block, 0, stream>>>(finished,
                                               sequence_length,
                                               word_ids,
                                               cum_log_probs,
                                               sentence_id,
                                               beam_width);
  }

  __global__ void sampling_init_kernel(bool* finished, 
                                       int* sequence_length, 
                                       int* word_ids, 
                                       const int start_id)
  {
    const int tid = threadIdx.x;
    finished[tid] = false;
    sequence_length[tid] = 0;
    word_ids[tid] = start_id;
  }

  void sampling_init_kernelLauncher(bool* finished, 
                                    int* sequence_length, 
                                    int* word_ids, 
                                    const int start_id, 
                                    const int batch_size, 
                                    cudaStream_t stream)
  {
    dim3 grid(1);
    dim3 block(min(1024, batch_size));
    assert(batch_size <= 1024);
    
    sampling_init_kernel<<<grid, block, 0, stream>>>(finished,
                                                     sequence_length,
                                                     word_ids,
                                                     start_id);
  }

  template <typename T>
  __global__ void embedding_lookup_sine_position_encoding_kernel(T* from_tensor,
                                                                const T* embedding_table, 
                                                                const T* position_encoding,
                                                                const int* word_ids,
                                                                const int batch_size,
                                                                const int hidden_units)
  {
      // 1. lookup from embedding table
      // 2. multiply hidden_dim**0.5
      // 3. add the position encoding
      T scale = (T)sqrtf(float(hidden_units));
      for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * hidden_units; index += blockDim.x * gridDim.x)
      {
        const int row_index = index / hidden_units; 
        const int col_index = index % hidden_units; 
        from_tensor[index] = embedding_table[word_ids[row_index] * hidden_units + col_index] * scale + position_encoding[col_index];
      }
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
      dim3 grid(min(batch_size, 65536));
      dim3 block(min(hidden_units, 1024));
      embedding_lookup_sine_position_encoding_kernel<T><<<grid, block, 0, stream>>>(from_tensor,
                                                                                  embedding_table,
                                                                                  position_encoding,
                                                                                  word_ids,
                                                                                  batch_size, 
                                                                                  hidden_units);
  }



  template <typename T>
  __global__ void embedding_position_lookups_kernel(T* from_tensor,
                                                    const T* embedding_table,
                                                    const T* pos_table,
                                                    const int* word_ids,
                                                    const int batch_size,
                                                    const int hidden_units,
                                                    int step)
  {
      for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * hidden_units; index += blockDim.x * gridDim.x)
      {
          const int row_index = index / hidden_units;
          const int col_index = index % hidden_units;
          from_tensor[index] = embedding_table[word_ids[row_index] * hidden_units + col_index]
                              + pos_table[(step - 1) * hidden_units + col_index];
      }
  }


  template <typename T>
  void embedding_position_lookups_kernel_launcher(T* from_tensor,
                                                  const T* embedding_table, 
                                                  const T* pos_table, 
                                                  const int* word_ids,
                                                  const int batch_size,
                                                  const int hidden_units, 
                                                  int step, 
                                                  cudaStream_t stream)
  {
      dim3 grid(min(batch_size, 65536));
      dim3 block(min(hidden_units, 1024));
      embedding_position_lookups_kernel<T><<<grid, block, 0, stream>>>(from_tensor,
                                                                       embedding_table,
                                                                       pos_table,
                                                                       word_ids,
                                                                       batch_size,
                                                                       hidden_units,
                                                                       step);
  }

  template <typename T>
  __global__ void apply_temperature_penalty_kernel(T* logits,
                                                   const T temperature_inverse,
                                                   const int m,
                                                   const int n)
  {
      for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < m * n; index += blockDim.x * gridDim.x)
      {
          logits[index] = logits[index] * temperature_inverse;
      }
  }

  template <typename T>
  void apply_temperature_penalty_kernelLauncher(T* logits,
                                                const T temperature,
                                                const int m,
                                                const int n,
                                                cudaStream_t stream)
  {
      dim3 grid(min(m, 65536));
      dim3 block(min(n, 1024));
      const T temperature_inverse = (T)(1.f / (float) temperature);
      apply_temperature_penalty_kernel<T><<<grid, block, 0, stream>>>(logits,
                                                                      temperature_inverse,
                                                                      m,
                                                                      n);
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
                    bool* finished, 
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
    bool* finished, 
    int* parent_ids, int* sequence_length,
    int* word_ids, int* output_ids, 
    const int batch_size, const int beam_width, 
    const int vocab_size, cudaStream_t stream, 
    const int end_id, int* finished_count)
  { 
    dim3 grid(1);
    dim3 block(batch_size * beam_width);

    assert(block.x <= 1024);

    update_kernel<float><<<grid, block, 0, stream>>>(log_probs, cum_log_probs,
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

  template <typename T>
  __global__
  void apply_logit_penalties_kernel(int step,
      int vocab_size, 
      int beam_width,
      T* log_probs, 
      int* current_ids,
      int* previous_ids,
      int* parent_ids,
      int  end_id,
      float inv_temp,
      float len_penalty,
      float repeat_penalty,
      int* vocab_mask) {

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bbid = blockIdx.y;
    int bbsize = gridDim.y;
    int batchid = bbid / beam_width;
    // int beamid = bbid % beam_width;

    for (int i = tid + bid*blockDim.x; i < vocab_size; i +=  blockDim.x*gridDim.x) {
      log_probs[i+bbid*vocab_size] *= inv_temp;
    }
    if (tid == 0 && bid == 0) {
      // apply repetition penalty (this can apply the penalty multiple times to a repeated word).
      int prev_id = current_ids[bbid];
      if (log_probs[prev_id+bbid*vocab_size] > T(0)) {
        log_probs[prev_id+bbid*vocab_size] = float(log_probs[prev_id+bbid*vocab_size]) / repeat_penalty;
      } else {
        log_probs[prev_id+bbid*vocab_size] = float(log_probs[prev_id+bbid*vocab_size]) * repeat_penalty;
      }
      if (step > 1) {
        int parent_beamid = parent_ids[bbsize*(step-2) + bbid];
        for (int i = step-2; i > 0; --i) {
          prev_id = previous_ids[bbsize*i+batchid*beam_width+parent_beamid];
          if (log_probs[prev_id+bbid*vocab_size] > T(0)) {
            log_probs[prev_id+bbid*vocab_size] = float(log_probs[prev_id+bbid*vocab_size]) / repeat_penalty;
          } else {
            log_probs[prev_id+bbid*vocab_size] = float(log_probs[prev_id+bbid*vocab_size]) * repeat_penalty;
          }
          //if (i > 0) parent_beamid = parent_ids[bbsize*(i-1)+parent_beamid];
          parent_beamid = parent_ids[bbsize*(i-1)+parent_beamid];
        }
      }
      prev_id = previous_ids[batchid*beam_width];
      if (log_probs[prev_id+bbid*vocab_size] > T(0)) {
        log_probs[prev_id+bbid*vocab_size] = float(log_probs[prev_id+bbid*vocab_size]) / repeat_penalty;
      } else {
        log_probs[prev_id+bbid*vocab_size] = float(log_probs[prev_id+bbid*vocab_size]) * repeat_penalty;
      }
      // apply length penalty
      if (log_probs[end_id+bbid*vocab_size] > T(0))  {
        log_probs[end_id+bbid*vocab_size] = float(log_probs[end_id+bbid*vocab_size]) / len_penalty;
      } else {
        log_probs[end_id+bbid*vocab_size] = float(log_probs[end_id+bbid*vocab_size]) * len_penalty;
      }
    }
  }

  template <typename T>
  void apply_logit_penalties(int step, 
                            T* log_probs, 
                            int* current_ids,
                            int* previous_ids, 
                            int* parent_ids,
                            Gpt2Arguments args,
                            cudaStream_t stream) {

    int vocab_size = args.vocab_size_;
    int beam_width = 1;
    int batch_size = args.batch_size_;
    dim3 block(256);
    dim3 grid((vocab_size + block.x - 1)/block.x, beam_width*batch_size);
    apply_logit_penalties_kernel<T><<<grid, block, 0, stream>>> (step, 
        vocab_size, 
        beam_width, 
        log_probs, 
        current_ids,
        previous_ids, 
        parent_ids,
        args.end_id_, 
        1.f/args.temperature_, 
        args.len_penalty,
        args.repeat_penalty, 
        args.vocab_mask);
  }

  extern __shared__ char transposeTileBuf_g[];

  template <typename data_type>
  __global__ void transpose_kernel(data_type * __restrict__ out, const data_type *__restrict__ in, int height, int width, int tH, int tW, int stride)
  // int tH, int tW should be template parameters for the best performance, we do not do that sine the task is tiny.
  // batch  stride (blockIdx.z dimension) for fully packed tensor ==  height * width
  {
      data_type *tile = (data_type *)transposeTileBuf_g;

      int tidx = threadIdx.x % tW;
      int tidy = threadIdx.x / tW;

      int xIndex = blockIdx.x * tW + tidx;
      int yIndex = blockIdx.y * tH + tidy;
      int indexIn = xIndex + yIndex * width;

      if ((xIndex < width) && (yIndex < height))
      {
          tile[tidy * tW + tidx] = in[blockIdx.z * stride + indexIn];
      }

      tidx = threadIdx.x % tH;
      tidy = threadIdx.x / tH;

      xIndex = blockIdx.y * tH + tidx;
      yIndex = blockIdx.x * tW + tidy;
      int indexOut = xIndex + yIndex * height;

      __syncthreads();

      if ((xIndex < height) &&  (yIndex < width))
      {
          out[blockIdx.z * stride + indexOut] = tile[tidx * tW + tidy];
      }
  }

  template <typename data_type>
  void transpose(data_type *out, const data_type *in, int batch, int height, int width, int stride, cudaStream_t stream)
  {
      int tW, tH;

      if ((width <= 1) || (height <= 1) )
      {
          assert(0);
      }

      if (height <= width)
      {
          tH = std::min((height / 2) * 2, 16);
          tW = std::min(256 / tH, width);
      }
      else
      {
          tW = std::min((width / 2) * 2, 16);
          tH = std::min(256 / tW, height);
      }
      assert(tW <= width);
      assert(tH <= height);

      dim3 grid((width + tW - 1) / tW, (height + tH - 1) / tH, batch);
      transpose_kernel<data_type><<<grid, tW * tH, tH * tW * sizeof(data_type), stream>>>(out, in, height, width, tH, tW, stride);
  }

  /* *************************** end of BeamSearch kernel *********************************** */

  /* ********************************** Sampling kernel *********************************** */
  __global__ void topp_initialization_kernel(bool* finished,
                                             int* sequence_length, 
                                             int* word_ids,
                                             int* topp_id_val_buf,
                                             int* topp_offset_buf,
                                             const int batch_size, 
                                             const int n,
                                             const int start_id)
  {
      int tid = threadIdx.x;
      int bid = blockIdx.x;
  
      if(bid == 0)
      {
          for(int i = tid; i < batch_size + 1; i+= blockDim.x)
          {
              topp_offset_buf[i] = i * n;
          }
          
          for(int i = tid; i < batch_size; i+= blockDim.x)
          {
              if(finished != nullptr) finished[i] = false;
              if(sequence_length != nullptr) sequence_length[i] = 0;
              if(word_ids != nullptr) word_ids[i] = start_id; 
          }
      }
  
      int index = tid + bid * blockDim.x;
      while(index < batch_size * n)
      {
          topp_id_val_buf[index] = index % n;
          index += blockDim.x * gridDim.x;
      }
  }

  void topp_initialization_kernelLauncher(bool* finished,
                                          int* sequence_length, 
                                          int* word_ids,
                                          int* topp_id_val_buf,
                                          int* topp_offset_buf,
                                          const int n,
                                          DecodingSamplingArguments args,
                                          cudaStream_t stream)
  {
      // n: the coloumn number of logits_buffer for top_p sampling
      topp_initialization_kernel<<<32, 512, 0, stream>>>(finished,
                                                         sequence_length,
                                                         word_ids,
                                                         topp_id_val_buf,
                                                         topp_offset_buf,
                                                         args.batch_size_, 
                                                         n,
                                                         args.start_id_);
  }

  template <typename T>
  size_t get_topp_sort_temp_storage_size(const T* log_probs,
                                         const int* id_vals,
                                         T* sorted_log_probs,
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

  // TODO Remove the gather_tree_kernel of th_op/utils.cu
  // modified from TensorFlow's implementation of tf.contrib.seq2seq.gather_tree
  __global__ void gather_tree_kernel(const int batch_size, const int max_time, const int beam_width, const int end_token,
                                    const int* step_ids, const int* parent_ids, int* max_sequence_lengths, int* beams) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch_size * beam_width; i += gridDim.x * blockDim.x) {
      const int batch = i / beam_width;
      const int beam = i % beam_width;

      const int max_seq_len_b = min(max_time, __ldg(max_sequence_lengths + batch));
      if (max_seq_len_b <= 0) {
        continue;
      }

  #define GET_IX(time_ix, beam_ix) (batch_size * beam_width * (time_ix) + beam_width * batch + (beam_ix))

      const int initial_beam_ix = GET_IX(max_seq_len_b - 1, beam);
      beams[initial_beam_ix] = __ldg(step_ids + initial_beam_ix);
      int parent = __ldg(parent_ids + initial_beam_ix) % beam_width;
      bool found_bad = false;
      for (int level = max_seq_len_b - 2; level >= 0; --level) {
        const int level_beam_ix = GET_IX(level, beam);
        const int level_parent_ix = GET_IX(level, parent);
        if (parent < 0 || parent > beam_width) {
          beams[level_beam_ix] = -1;
          parent = -1;
          found_bad = true;
        } else {
          beams[level_beam_ix] = __ldg(step_ids + level_parent_ix);
          parent = __ldg(parent_ids + level_parent_ix) % beam_width;
        }
      }
  // Not necessary when using a BeamSearchDecoder, but necessary
  // when a user feeds in possibly broken trajectory (i.e., non-eos
  // entries in a beam following eos entries).
      if (!found_bad) {
        bool finished = false;
        for (int time = 0; time < max_seq_len_b; ++time) {
          const int level_beam_ix = GET_IX(time, beam);
          if (finished) {
            beams[level_beam_ix] = end_token;
          } else if (beams[level_beam_ix] == end_token) {
            finished = true;
          }
        }
      }
  #undef GET_IX
    }
  }


  void gather_tree_kernel_launcher(int max_time, int batch_size, int beam_width,
                                  int* step_ids, int* parent_ids, int* max_sequence_lengths,
                                  int end_token, int* beams, cudaStream_t stream) {
    int batchbeam = batch_size * beam_width;
    dim3 grid(1), block(batchbeam);
    // though decoder do not support > 1024 for now
    if (batchbeam > 1024) {
      grid.x = ceil(batch_size * beam_width / 1024.);
      block.x = 1024;
    }
    gather_tree_kernel<<<grid, block, 0, stream>>>(batch_size, max_time, beam_width, end_token,
                                                  step_ids, parent_ids, max_sequence_lengths, beams);
  }



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

  template 
  void embedding_position_lookups_kernel_launcher(float* from_tensor,
                                                  const float* embedding_table,
                                                  const float* pos_table,
                                                  const int* word_ids,
                                                  const int batch_size,
                                                  const int hidden_units,
                                                  int step,
                                                  cudaStream_t stream);

  template 
  void embedding_position_lookups_kernel_launcher(half* from_tensor,
                                                  const half* embedding_table,
                                                  const half* pos_table,
                                                  const int* word_ids,
                                                  const int batch_size,
                                                  const int hidden_units,
                                                  int step,
                                                  cudaStream_t stream);

  template void apply_temperature_penalty_kernelLauncher(float* logits,
                                                         const float temperature,
                                                         const int m,
                                                         const int n,
                                                         cudaStream_t stream);

  template void apply_temperature_penalty_kernelLauncher(half* logits,
                                                         const half temperature,
                                                         const int m,
                                                         const int n,
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

  template void apply_logit_penalties(int step,
                                      float* log_probs,
                                      int* current_ids,
                                      int* previous_ids,
                                      int* parent_ids,
                                      Gpt2Arguments args,
                                      cudaStream_t stream);

  template void apply_logit_penalties(int step,
                                      half* log_probs,
                                      int* current_ids,
                                      int* previous_ids,
                                      int* parent_ids,
                                      Gpt2Arguments args,
                                      cudaStream_t stream);

  template size_t get_topp_sort_temp_storage_size(const float* log_probs,
                                                  const int* id_vals,
                                                  float* sorted_log_probs,
                                                  int* sorted_id_vals,
                                                  int* topp_offset_buf,
                                                  const int batch_size,
                                                  const int vocab_size);

  template size_t get_topp_sort_temp_storage_size(const half* log_probs,
                                                  const int* id_vals,
                                                  half* sorted_log_probs,
                                                  int* sorted_id_vals,
                                                  int* topp_offset_buf,
                                                  const int batch_size,
                                                  const int vocab_size);

  template void transpose(float *out,
                          const float *in,
                          int batch,int height,
                          int width,int stride,
                          cudaStream_t stream);
  template void transpose(half *out,
                          const half *in,
                          int batch,int height,
                          int width,int stride,
                          cudaStream_t stream);

  template void init_kernelLauncher(bool* finished,
                                    int* sequence_length,
                                    int* word_ids,
                                    float* cum_log_probs,
                                    const int sentence_id,
                                    const int batch_size,
                                    const int beam_width,
                                    cudaStream_t stream);

  template void init_kernelLauncher(bool* finished,
                                   int* sequence_length,
                                   int* word_ids,
                                   half* cum_log_probs,
                                   const int sentence_id,
                                   const int batch_size,
                                   const int beam_width,
                                   cudaStream_t stream);
  /* *************************** end of Instantiation *********************************** */

} // end of name space fastertransformer

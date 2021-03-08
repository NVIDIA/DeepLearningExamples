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
 * BeamSearch OpenNMT
 **/

#pragma once

#include <cuda_runtime.h>
#include "fastertransformer/allocator.h"
#include "fastertransformer/cuda/cuda_kernels.h"
#include "fastertransformer/cuda/open_attention.h"
#include "fastertransformer/cuda/decoding_kernel_check.h"

namespace fastertransformer
{

template <typename T>
void BeamSearch_OpenNMT(
    float *log_probs, float *cum_log_probs, bool *finished,
    T **key_cache, T **value_cache,
    int *parent_ids,
    int *sequence_length,
    int *word_ids,
    int *ids,
    int *output_ids,
    const int batch_size, const int beam_width,
    const int vocab_size, const int hidden_dim, const int step,
    const int cache_size, const int decoder_layers, cudaStream_t stream,
    const int end_id, 
    int *finished_count)
{
#ifdef NDEBUG
  /* adding cum_log_probs to log_probs */
  broadcast_kernelLauncher(log_probs, cum_log_probs, batch_size, beam_width, vocab_size, stream);
#else
  broadcast_kernelLauncher(log_probs, cum_log_probs, batch_size, beam_width, vocab_size, stream);
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());

  /*
    User can check the broadcast_kernel by broadcast_kernel_check.
    broadcast_kernel_check will compare the results of GPU and CPU.
    Note that broadcast_kernel_check contains broadcast_kernelLauncher and uses do not need to call it again. 
  */
  // broadcast_kernel_check(log_probs, cum_log_probs, batch_size, beam_width, vocab_size, stream);
#endif

#ifdef NDEBUG
  /*Use two round kernels to pick the topK values for each batch */
  topK(log_probs, ids, batch_size, beam_width, vocab_size, stream);
#else
  topK(log_probs, ids, batch_size, beam_width, vocab_size, stream);
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());

  /*
    User can check the topK by topK_check.
    topK_check will compare the results of GPU and CPU.
    Note that topK_check contains topK and uses do not need to call it again. 
  */
  // topK_kernel_check(log_probs, ids, batch_size, beam_width, vocab_size, stream);
#endif

#ifdef NDEBUG
  update(log_probs, cum_log_probs, ids, finished, 
        parent_ids, sequence_length, word_ids, output_ids,
        batch_size, beam_width, vocab_size, stream, 
        end_id, finished_count);
#else
  update(log_probs, cum_log_probs, ids, finished, 
        parent_ids, sequence_length, word_ids, output_ids,
        batch_size, beam_width, vocab_size, stream, 
        end_id, finished_count);
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());

  /*
    User can check the update by update_kernel_check.
    update_kernel_check will compare the results of GPU and CPU.
    Note that update_kernel_check contains update and uses do not need to call it again. 
  */
  // update_kernel_check(log_probs, cum_log_probs, ids, finished, parent_ids, sequence_length, word_ids, output_ids,
  //                     batch_size, beam_width, vocab_size, stream, end_id, finished_count);
#endif

#ifdef NDEBUG
  update_KV_cache<T>(key_cache, value_cache, parent_ids, batch_size, 
                    beam_width, hidden_dim, step, cache_size, 
                    decoder_layers, stream);
#else
  update_KV_cache<T>(key_cache, value_cache, parent_ids, batch_size, 
                    beam_width, hidden_dim, step, cache_size, 
                    decoder_layers, stream);
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());

  /*
    User can check the update_KV_cache by update_KV_cache_kernel_check.
    update_KV_cache_kernel_check will compare the results of GPU and CPU.
    Note that update_KV_cache_kernel_check contains update_KV_cache and uses do not need to call it again. 
  */
  // update_KV_cache_kernel_check(key_cache, value_cache, parent_ids, batch_size, beam_width, hidden_dim, step, cache_size, decoder_layers, stream);
#endif
}

} // namespace fastertransformer

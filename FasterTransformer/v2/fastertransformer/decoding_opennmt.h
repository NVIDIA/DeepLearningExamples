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
 * Decoder transformer
 **/

#pragma once

#include "fastertransformer/common.h"
#include "fastertransformer/allocator.h"
#include "fastertransformer/open_decoder.h"
#include "fastertransformer/cuda/cuda_kernels.h"
#include "fastertransformer/beamsearch_opennmt.h"
#include <cuda_runtime.h>

namespace fastertransformer
{

template <typename T>
class DecodingInitParam
{
public:
  /* weights for masked_multi_head_attention */
  const T *embedding_table;
  const T *embedding_kernel;
  const float *embedding_bias;

  const T *memory_tensor;
  const int *memory_sequence_length;

  LayerNormWeight<T> layernorm;

  int *output_ids;
  int *parent_ids;
  int *sequence_length;
  cublasHandle_t cublas_handle;
  cudaStream_t stream;
};

template <OperationType OpType_>
class DecodingOpenNMT
{
private:
  typedef DecoderTransformerTraits<OpType_> Traits_;
  typedef typename Traits_::DataType DataType_;
  const IAllocator &allocator_;

  const cudaDataType_t computeType_ = Traits_::computeType;
  const cudaDataType_t AType_ = Traits_::AType;
  const cudaDataType_t BType_ = Traits_::BType;
  const cudaDataType_t CType_ = Traits_::CType;
  int cublasAlgo_[1] = {20};

  int batch_size_;
  int beam_width_;
  int seq_len_;
  int head_num_;
  int size_per_head_;
  int hidden_units_;
  int decoder_layers_;
  int vocab_size_;
  OpenDecoder<OpType_> *decoder_;
  DataType_ **K_cache_;
  DataType_ **V_cache_;
  DataType_ **K_mem_cache_;
  DataType_ **V_mem_cache_;
  DataType_ *from_tensor_[2];
  DataType_ *decoder_buf_;
  DataType_ *decoder_normed_result_buf_;
  float *logits_buf_;
  float *cum_log_buf_;
  int *word_ids_buf_;
  int *topk_ids_buf_;
  bool *finished_buf_;
  void *buf_;
  int start_id_;
  int end_id_;
  int *finished_count_buf_;
  bool *h_finished_buf_;

public:
  DecodingOpenNMT(const IAllocator &allocator, const int batch_size,
                  const int beam_width, const int seq_len,
                  const int head_num, const int size_per_head,
                  const int vocab_size, const int decoder_layers,
                  const int memory_hidden_units, const int memory_max_seq_len,
                  const int start_id, const int end_id) : allocator_(allocator), batch_size_(batch_size), beam_width_(beam_width),
                                                          seq_len_(seq_len), head_num_(head_num), size_per_head_(size_per_head),
                                                          vocab_size_(vocab_size), decoder_layers_(decoder_layers),
                                                          start_id_(start_id), end_id_(end_id)
  {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    K_cache_ = new DataType_ *[2];
    V_cache_ = new DataType_ *[2];

    K_mem_cache_ = new DataType_ *[decoder_layers_];
    V_mem_cache_ = new DataType_ *[decoder_layers_];

    hidden_units_ = head_num_ * size_per_head_;
    decoder_ = new OpenDecoder<OpType_>(allocator, batch_size * beam_width, memory_max_seq_len, 
                                        head_num, size_per_head, memory_hidden_units);

    int from_tensor_size = batch_size_ * beam_width_ * hidden_units_;      // type T
    int decoder_workspace_size = decoder_->getWorkspaceSize();             // type T
    int decoder_normed_result_buffer_size = batch_size_ * beam_width_ * hidden_units_; // type T
    int cache_size = batch_size_ * beam_width_ * seq_len_ * hidden_units_; // type T

    int logits_buf_size = batch_size_ * beam_width_ * vocab_size_;         // type float
    int cum_log_buf_size = batch_size_ * beam_width_;  // type float
    int word_ids_buf_size = batch_size_ * beam_width_; //type int
    int finished_buf_size = batch_size_ * beam_width_; //type bool
    int finished_count_size = (int)(ceil(1 / 4.)) * 4; // type int

    //type int
    int topk_ids_buf_size = batch_size_ * beam_width_ * (ceil)((beam_width_ * vocab_size_ * 1.0) / 1024.0);
    // prevent memory misalinged address
    cum_log_buf_size = (int)(ceil(cum_log_buf_size / 4.)) * 4;
    word_ids_buf_size = (int)(ceil(word_ids_buf_size / 4.)) * 4;
    finished_buf_size = (int)(ceil(finished_buf_size / 32.)) * 32;
    topk_ids_buf_size = (int)(ceil(topk_ids_buf_size / 4.)) * 4;
    

    int datatype_buf_size = from_tensor_size * 2 + decoder_workspace_size +
                            cache_size * 6 * decoder_layers_ + decoder_normed_result_buffer_size;

    buf_ = reinterpret_cast<void *>(allocator_.malloc(
        sizeof(DataType_) * datatype_buf_size +
        sizeof(float) * (logits_buf_size + cum_log_buf_size) +
        sizeof(int) * word_ids_buf_size +
        sizeof(bool) * finished_buf_size +
        sizeof(int) * topk_ids_buf_size + 
        sizeof(int) * finished_count_size ));

    from_tensor_[0] = (DataType_ *)buf_;
    from_tensor_[1] = (DataType_ *)(from_tensor_[0] + from_tensor_size);

    for (int i = 0; i < decoder_layers_; ++i)
    {
      K_mem_cache_[i] = from_tensor_[1] + from_tensor_size + i * cache_size * 2;
      V_mem_cache_[i] = from_tensor_[1] + from_tensor_size + i * cache_size * 2 + cache_size;
    }

    /* We use two-way buffer since we have to update KV buf at the end of each step. */
    K_cache_[0] = V_mem_cache_[decoder_layers - 1] + cache_size + 0 * cache_size * decoder_layers_;
    K_cache_[1] = V_mem_cache_[decoder_layers - 1] + cache_size + 1 * cache_size * decoder_layers_;
    V_cache_[0] = V_mem_cache_[decoder_layers - 1] + cache_size + 2 * cache_size * decoder_layers_;
    V_cache_[1] = V_mem_cache_[decoder_layers - 1] + cache_size + 3 * cache_size * decoder_layers_;

    decoder_buf_ = V_cache_[1] + cache_size * decoder_layers_;
    decoder_normed_result_buf_ = (decoder_buf_ + decoder_workspace_size);
    logits_buf_ = (float *)(decoder_normed_result_buf_ + decoder_normed_result_buffer_size);
    cum_log_buf_ = (float *)(logits_buf_ + logits_buf_size);
    word_ids_buf_ = (int *)(cum_log_buf_ + cum_log_buf_size);
    finished_buf_ = (bool *)(word_ids_buf_ + word_ids_buf_size);
    topk_ids_buf_ = (int *)(finished_buf_ + finished_buf_size);
    finished_count_buf_ = (int *)(topk_ids_buf_ + topk_ids_buf_size);

    h_finished_buf_ = new bool[finished_buf_size];

    FILE *fd = fopen("decoding_gemm_config.in", "r");
    int err = 0;
    if (fd == NULL)
      printf("[WARNING] decoding_gemm_config.in is not found\n");
    else
    {
      err = fscanf(fd, "%d%*d%*d", &cublasAlgo_[0]);
      fclose(fd);
    }
    if (err != 1)
    {
      printf("[WARNING] decoding loading GEMM algorithms error, using default GEMM algorithms!\n");
      if (Traits_::OpType == OperationType::FP32)
      {
        cublasAlgo_[0] = CUBLAS_GEMM_DEFAULT;
      }
      else
      {
        cublasAlgo_[0] = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
      }
    }
    else
    {
      // check that the gemm_config setting is runnable
      if (Traits_::OpType == OperationType::FP32)
      {
        if (cublasAlgo_[0] > CUBLAS_GEMM_ALGO23 || cublasAlgo_[0] < CUBLAS_GEMM_DEFAULT)
        {
          // the algorithm is not for FP32
          printf("[ERROR] cuBLAS Algorithm %d is not used in FP32. \n", (int)cublasAlgo_[0]);
          exit(-1);
        }
      }
      else
      {
        if (cublasAlgo_[0] > CUBLAS_GEMM_ALGO15_TENSOR_OP || cublasAlgo_[0] < CUBLAS_GEMM_DEFAULT_TENSOR_OP)
        {
          // the algorithm is not for FP16
          printf("[ERROR] cuBLAS Algorithm %d is not used in FP16. \n", (int)cublasAlgo_[0]);
          exit(-1);
        }
      }
    }
  }

  void forward(const DecoderInitParam<DataType_> *param,
               DecodingInitParam<DataType_> decoding_params)
  {

#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    int m = batch_size_ * beam_width_;
    int k = hidden_units_;
    int n = vocab_size_;

    /*
      sequence_length initialize to 0
      finished: false
      word_ids: start_id_
      cum_log_probs (for eacm beam, the first element is 0). e.g., [0 -inf -inf -inf][0 -inf -inf -inf]
    */

#ifdef NDEBUG
    init(finished_buf_, decoding_params.sequence_length, word_ids_buf_, cum_log_buf_,
         start_id_, batch_size_, beam_width_, decoding_params.stream);
#else
    init(finished_buf_, decoding_params.sequence_length, word_ids_buf_, cum_log_buf_,
         start_id_, batch_size_, beam_width_, decoding_params.stream);

    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());

    /*
      User can check the init by init_kernel_check.
      init_kernel_check will compare the results of GPU and CPU.
      Note that init_kernel_check contains init and uses do not need to call it again. 
    */
    // init_kernel_check(finished_buf_, decoding_params.sequence_length, word_ids_buf_, cum_log_buf_,
    //                   start_id_, batch_size_, beam_width_, decoding_params.stream);
#endif

    int cache_size = batch_size_ * beam_width_ * seq_len_ * hidden_units_; // type T

    for (int step = 1; step <= seq_len_; ++step)
    {
      //we use two-way buffer
      int kv_cache_id = step & 0x1;

#ifdef NDEBUG
      embedding_lookup(decoding_params.embedding_table, word_ids_buf_, from_tensor_[0],
                       batch_size_, beam_width_, hidden_units_, decoding_params.stream);
#else
      embedding_lookup(decoding_params.embedding_table, word_ids_buf_, from_tensor_[0],
                       batch_size_, beam_width_, hidden_units_, decoding_params.stream);
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());

      /*
        User can check the embedding_lookup by embedding_lookup_kernel_check.
        embedding_lookup_kernel_check will compare the results of GPU and CPU.
        Note that embedding_lookup_kernel_check contains embedding_lookup and uses do not need to call it again. 
      */
      // embedding_lookup_kernel_check(decoding_params.embedding_table, word_ids_buf_, from_tensor_[0],
      //                               batch_size_, beam_width_, hidden_units_, vocab_size_, decoding_params.stream);
#endif

      sine_position_encoder<DataType_>(from_tensor_[0], step, m, hidden_units_, decoding_params.stream);

      int from_id, out_id;
      for (int layer = 0; layer < decoder_layers_; ++layer)
      {
        /*
          For the first layer (layer-0), from_id is 0. We also stored the embedding lookup 
          result in from_tensor_[0]
        */
        from_id = layer & 0x1;
        out_id = 1 - from_id;

        /*
          We use one decoder_ object to process multiple decoder layers. 
        
          At the beginning of each decoder layer, we initialize the decoder object 
          with corresponding weights and decoder_buf_.

          The decoder_buf_ is reused.
        */
        decoder_->initialize(param[layer], decoder_buf_);

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
        decoder_->forward(from_tensor_[from_id], decoding_params.memory_tensor,
                          K_cache_[kv_cache_id] + layer * cache_size,
                          V_cache_[kv_cache_id] + layer * cache_size,
                          K_mem_cache_[layer], V_mem_cache_[layer],
                          decoding_params.memory_sequence_length, from_tensor_[out_id], step);

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
      }
      decoder_->decoder_norm1(from_tensor_[out_id], decoding_params.layernorm.gamma,
                  decoding_params.layernorm.beta, decoder_normed_result_buf_, m, k);

      float alpha = (float)1.0f;
      float beta = (float)0.0f;

      check_cuda_error(cublasGemmEx(decoding_params.cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    n, m, k,
                                    &alpha,
                                    decoding_params.embedding_kernel, AType_, n,
                                    decoder_normed_result_buf_, BType_, k,
                                    &beta,
                                    logits_buf_, CUDA_R_32F, n,
                                    CUDA_R_32F,
                                    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

#ifdef NDEBUG
      update_logits(logits_buf_, decoding_params.embedding_bias, end_id_, finished_buf_, m, n, decoding_params.stream);
#else
      update_logits(logits_buf_, decoding_params.embedding_bias, end_id_, finished_buf_, m, n, decoding_params.stream);
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());

      /*
        User can check the update_logits by update_logits_kernel_check.
        update_logits_kernel_check will compare the results of GPU and CPU.
        Note that update_logits_kernel_check contains update_logits and uses do not need to call it again. 
      */
      // update_logits_kernel_check(logits_buf_, decoding_params.embedding_bias, end_id_, finished_buf_, m, n, decoding_params.stream);
#endif
      BeamSearch_OpenNMT(
          logits_buf_, cum_log_buf_, finished_buf_,
          K_cache_,
          V_cache_,
          decoding_params.parent_ids + (step - 1) * batch_size_ * beam_width_,
          decoding_params.sequence_length,
          word_ids_buf_,
          topk_ids_buf_,
          decoding_params.output_ids + (step - 1) * batch_size_ * beam_width_,
          batch_size_, beam_width_, vocab_size_, hidden_units_, step, cache_size, decoder_layers_, decoding_params.stream,
          end_id_, 
          finished_count_buf_);

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

      // TODO 
      // Find a better method to check the is_finished
      cudaMemcpy(h_finished_buf_, finished_buf_, sizeof(bool) * batch_size_ * beam_width_, cudaMemcpyDeviceToHost);
      int sum = 0;
      for(int i = 0; i < batch_size_ * beam_width_; i++){
        sum += (int)h_finished_buf_[i];
      }
      if(sum == batch_size_ * beam_width_) break;
    }
  }

  virtual ~DecodingOpenNMT() {}
};

} //namespace fastertransformer

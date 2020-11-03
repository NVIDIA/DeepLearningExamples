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
#include "fastertransformer/arguments.h"
#include <cuda_runtime.h>

namespace fastertransformer
{

template <OperationType OpType_>
class DecodingSampling
{
private:
  typedef DecoderTransformerTraits<OpType_> Traits_;
  typedef typename Traits_::DataType DataType_;
  const IAllocator &allocator_;
  struct DecodingSamplingArguments args_;

  const cudaDataType_t computeType_ = Traits_::computeType;
  const cudaDataType_t AType_ = Traits_::AType;
  const cudaDataType_t BType_ = Traits_::BType;
  const cudaDataType_t CType_ = Traits_::CType;
  int cublasAlgo_[1] = {20};

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
  bool *finished_buf_;
  int *topk_ids_buf_;
  float *topk_val_buf_;
  void *buf_;
  // int start_id_;
  // int end_id_;
  int *finished_count_buf_;
  bool *h_finished_buf_;

  int *topp_id_vals_buf_;
  float *topp_sorted_log_prob_buf_;
  int *topp_sorted_id_vals_buf_;
  int *topp_offset_buf_;

  void *temp_storage_;
  // size_t temp_storage_size_;
  

public:
  DecodingSampling(const IAllocator &allocator, const int batch_size,
                  const int seq_len,
                  const int head_num, const int size_per_head,
                  const int vocab_size, const int decoder_layers,
                  const int memory_hidden_units, const int memory_max_seq_len,
                  const int start_id, const int end_id, 
                  const int candidate_num=0,
                  const float probability_threshold=0.0) :  allocator_(allocator)
  {
    args_.batch_size_ = batch_size;
    args_.seq_len_ = seq_len;
    args_.head_num_ = head_num;
    args_.size_per_head_ = size_per_head;
    args_.hidden_units_ = head_num * size_per_head;
    args_.decoder_layers_ = decoder_layers;
    args_.vocab_size_ = vocab_size;
    args_.candidate_num_ = candidate_num;
    args_.probability_threshold_ = probability_threshold;
    args_.start_id_ = start_id;
    args_.end_id_ = end_id;

    if(args_.candidate_num_ == 0 && args_.probability_threshold_ == 0.0)
    {
      printf("[ERROR] Candidate_num for topk is 0 and probability threshold for top p is 0.0 \n");
      exit(-1);
    }
    else if(args_.candidate_num_ != 0 && args_.probability_threshold_ != 0.0)
    {
      printf("[ERROR] Candidate_num for topk is not 0 and probability threshold for top p is not 0.0 \n");
      exit(-1);
    }
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    K_cache_ = new DataType_ *[1];
    V_cache_ = new DataType_ *[1];

    K_mem_cache_ = new DataType_ *[args_.decoder_layers_];
    V_mem_cache_ = new DataType_ *[args_.decoder_layers_];

    decoder_ = new OpenDecoder<OpType_>(batch_size, memory_max_seq_len, 
                                        head_num, size_per_head, memory_hidden_units);

    int from_tensor_size = args_.batch_size_ * args_.hidden_units_;      // type T
    int decoder_workspace_size = decoder_->getWorkspaceSize();             // type T
    int decoder_normed_result_buffer_size = args_.batch_size_ * args_.hidden_units_; // type T
    int cache_size = args_.batch_size_ * args_.seq_len_ * args_.hidden_units_; // type T
    int mem_cache_size = args_.batch_size_ * memory_max_seq_len * args_.hidden_units_; // type T

    int logits_buf_size = args_.batch_size_ * args_.vocab_size_;         // type float
    int cum_log_buf_size = args_.batch_size_;  // type float
    int word_ids_buf_size = args_.batch_size_; //type int
    int finished_buf_size = args_.batch_size_;  //type bool
    int finished_count_size = (int)(ceil(1 / 32.)) * 32; // type int

    int topk_ids_buf_size = args_.batch_size_ * args_.candidate_num_; // type int
    int topk_val_buf_size = args_.batch_size_ * args_.candidate_num_;  // type float
    int topp_id_vals_buf_size = args_.batch_size_ * args_.vocab_size_;
    int topp_sorted_log_prob_buf_size = args_.batch_size_ * args_.vocab_size_;
    int topp_sorted_id_vals_buf_size = args_.batch_size_ * args_.vocab_size_;

    // prevent memory misalinged address
    logits_buf_size = (int)(ceil(logits_buf_size / 4.)) * 4;
    cum_log_buf_size = (int)(ceil(cum_log_buf_size / 4.)) * 4;
    word_ids_buf_size = (int)(ceil(word_ids_buf_size / 4.)) * 4;
    finished_buf_size = (int)(ceil(finished_buf_size / 32.)) * 32;
    topk_ids_buf_size = (int)(ceil(topk_ids_buf_size / 4.)) * 4;
    topk_val_buf_size = (int)(ceil(topk_val_buf_size / 4.)) * 4;
    topp_id_vals_buf_size = (int)(ceil(topp_id_vals_buf_size / 4.)) * 4;
    topp_sorted_log_prob_buf_size = (int)(ceil(topp_sorted_log_prob_buf_size / 4.)) * 4;
    topp_sorted_id_vals_buf_size = (int)(ceil(topp_sorted_id_vals_buf_size / 4.)) * 4;

    args_.temp_storage_size_ = get_topp_sort_temp_storage_size(logits_buf_, 
                                                                            topp_id_vals_buf_,
                                                                            topp_sorted_log_prob_buf_,
                                                                            topp_sorted_id_vals_buf_,
                                                                            topp_offset_buf_,
                                                                            args_.batch_size_,
                                                                            args_.vocab_size_);
    
    int topp_offset_buf_size = args_.batch_size_ + 1;
    args_.temp_storage_size_ = (int)(ceil(args_.temp_storage_size_ / 4.)) * 4;
    topp_offset_buf_size = (int)(ceil(topp_offset_buf_size / 4.)) * 4;

    int datatype_buf_size = from_tensor_size * 2 + decoder_workspace_size +
                            (cache_size * 4 + mem_cache_size * 2) * args_.decoder_layers_ + decoder_normed_result_buffer_size;

    buf_ = reinterpret_cast<void *>(allocator_.malloc(
        sizeof(DataType_) * datatype_buf_size +
        sizeof(float) * (logits_buf_size + cum_log_buf_size) +
        sizeof(int) * word_ids_buf_size +
        sizeof(bool) * finished_buf_size +
        sizeof(int) * finished_count_size +
        sizeof(int) * topk_ids_buf_size + 
        sizeof(float) * topk_val_buf_size +
        sizeof(int) * (topp_id_vals_buf_size + topp_sorted_id_vals_buf_size + topp_offset_buf_size) + 
        sizeof(float) * topp_sorted_log_prob_buf_size + 
        args_.temp_storage_size_ ));

    from_tensor_[0] = (DataType_ *)buf_;
    from_tensor_[1] = (DataType_ *)(from_tensor_[0] + from_tensor_size);

    for (int i = 0; i < args_.decoder_layers_; ++i)
    {
      K_mem_cache_[i] = from_tensor_[1] + from_tensor_size + i * mem_cache_size * 2;
      V_mem_cache_[i] = from_tensor_[1] + from_tensor_size + i * mem_cache_size * 2 + mem_cache_size;
    }

    /* We use two-way buffer since we have to update KV buf at the end of each step. */
    K_cache_[0] = V_mem_cache_[args_.decoder_layers_ - 1] + mem_cache_size + 0 * cache_size * args_.decoder_layers_;
    V_cache_[0] = V_mem_cache_[args_.decoder_layers_ - 1] + mem_cache_size + 1 * cache_size * args_.decoder_layers_;

    decoder_buf_ = V_cache_[0] + cache_size * args_.decoder_layers_;
    decoder_normed_result_buf_ = (decoder_buf_ + decoder_workspace_size);
    logits_buf_ = (float *)(decoder_normed_result_buf_ + decoder_normed_result_buffer_size);
    cum_log_buf_ = (float *)(logits_buf_ + logits_buf_size);
    word_ids_buf_ = (int *)(cum_log_buf_ + cum_log_buf_size);
    finished_buf_ = (bool *)(word_ids_buf_ + word_ids_buf_size);
    topk_ids_buf_ = (int *)(finished_buf_ + finished_buf_size);
    topk_val_buf_ = (float*)(topk_ids_buf_ + topk_ids_buf_size);
    finished_count_buf_ = (int *)(topk_val_buf_ + topk_val_buf_size);
    topp_id_vals_buf_ = (int*)(finished_count_buf_ + finished_count_size);
    topp_sorted_id_vals_buf_ = (int*)(topp_id_vals_buf_ + topp_id_vals_buf_size);
    topp_offset_buf_ = (int*)(topp_sorted_id_vals_buf_ + topp_sorted_id_vals_buf_size);
    topp_sorted_log_prob_buf_ = (float*)(topp_offset_buf_ + topp_offset_buf_size);
    temp_storage_ = (void*)(topp_sorted_log_prob_buf_ + topp_sorted_log_prob_buf_size);

    h_finished_buf_ = new bool[finished_buf_size];

    FILE *fd = fopen("decoding_gemm_config.in", "r");
    int err = 0;
    if (fd == NULL)
      printf("[WARNING] decoding_gemm_config.in is not found\n");
    else
    {
      err = fscanf(fd, "%d", &cublasAlgo_[0]);
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
    const int m = args_.batch_size_;
    const int k = args_.hidden_units_;
    const int n = args_.vocab_size_;

    /*
      sequence_length initialize to 0
      finished: false
      word_ids: start_id_
      cum_log_buf_: useless, keep it to reuse the kernel of decoding_opennmt.h
    */

    if(args_.candidate_num_ != 0)
    {
      init_kernelLauncher(finished_buf_, decoding_params.sequence_length, word_ids_buf_, cum_log_buf_,
         args_.start_id_, args_.batch_size_, 1, decoding_params.stream);
    }
    else if(args_.probability_threshold_ != 0.0)
    {
      topp_initialization_kernelLauncher(finished_buf_,
                                          decoding_params.sequence_length, 
                                          word_ids_buf_,
                                          topp_id_vals_buf_,
                                          topp_offset_buf_,
                                          args_,
                                          decoding_params.stream);
    }

#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif

    int cache_size = args_.batch_size_ * args_.seq_len_ * args_.hidden_units_; // type T

    for (int step = 1; step <= args_.seq_len_; ++step)
    {
      embedding_lookup_sine_position_encoding_kernel_launcher(from_tensor_[0],
                                                            decoding_params.embedding_table, 
                                                            decoding_params.position_encoding_table + (step - 1) * args_.hidden_units_,
                                                            word_ids_buf_,
                                                            args_.batch_size_,
                                                            args_.hidden_units_, 
                                                            decoding_params.stream);

      int from_id, out_id;
      for (int layer = 0; layer < args_.decoder_layers_; ++layer)
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
                          K_cache_[0] + layer * cache_size,
                          V_cache_[0] + layer * cache_size,
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

      if(args_.candidate_num_ != 0)
      {
        // top k sampling
        update_logits_without_softmax(logits_buf_, decoding_params.embedding_bias, args_.end_id_, finished_buf_, m, n, decoding_params.stream);
        topK_sampling_kernel_kernelLauncher(logits_buf_, 
                                            topk_ids_buf_,
                                            topk_val_buf_,
                                            decoding_params.output_ids + (step - 1) * args_.batch_size_,
                                            decoding_params.sequence_length,
                                            finished_buf_,
                                            step, // used as random number
                                            args_, 
                                            decoding_params.stream);
      }
      else if(args_.probability_threshold_ != 0.0)
      {
        // top p sampling
        update_logits_without_log(logits_buf_, decoding_params.embedding_bias, args_.end_id_, finished_buf_, m, n, decoding_params.stream);
        topP_sampling_kernel_kernelLauncher(logits_buf_,
                                          topp_id_vals_buf_,
                                          topp_sorted_log_prob_buf_,
                                          topp_sorted_id_vals_buf_, 
                                          topp_offset_buf_,
                                          temp_storage_,
                                          finished_buf_,
                                          step,
                                          args_,
                                          decoding_params.output_ids,
                                          decoding_params.sequence_length,
                                          decoding_params.stream);
      }


      word_ids_buf_ = decoding_params.output_ids + (step - 1) * args_.batch_size_;

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

      // TODO 
      // Find a better method to check the is_finished
      cudaMemcpy(h_finished_buf_, finished_buf_, sizeof(bool) * args_.batch_size_ , cudaMemcpyDeviceToHost);
      int sum = 0;
      for(int i = 0; i < args_.batch_size_ ; i++){
        sum += (int)h_finished_buf_[i];
      }
      if(sum == args_.batch_size_ ) break;
    }
  }

  virtual ~DecodingSampling() 
  {
    delete [] K_cache_;
    delete [] V_cache_;
    delete [] K_mem_cache_;
    delete [] V_mem_cache_;
    delete [] h_finished_buf_;
    delete decoder_;
    allocator_.free(buf_);
  }
};

} //namespace fastertransformer

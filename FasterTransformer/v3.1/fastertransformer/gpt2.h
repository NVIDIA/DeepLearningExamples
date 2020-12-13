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
#include <stdlib.h>

#define EMBEDDING_TRANSPOSE_OPT 0 // TODO This feature has bug.

namespace fastertransformer
{

template <OperationType OpType_>
class DecodingGpt2
{
private:
    typedef DecoderTransformerTraits<OpType_> Traits_;
    typedef typename Traits_::DataType DataType_;
    const IAllocator &allocator_;
    struct Gpt2Arguments args_;

    const cudaDataType_t computeType_ = Traits_::computeType;
    const cudaDataType_t AType_ = Traits_::AType;
    const cudaDataType_t BType_ = Traits_::BType;
    const cudaDataType_t CType_ = Traits_::CType;
    int cublasAlgo_[1] = {20};

    DataType_ *embedding_kernel_transposed_padded_;

    OpenDecoder<OpType_> *decoder_;
    DataType_ **K_cache_;
    DataType_ **V_cache_;
    DataType_ *from_tensor_[2];
    DataType_ *decoder_buf_;
    DataType_ *decoder_normed_result_buf_;
    DataType_ *logits_buf_;
    void *buf_;
    
    void *topk_workspace_ = nullptr;
    size_t topk_workspace_size_ = 0;
    void *topp_workspace_ = nullptr;
    size_t topp_workspace_size_ = 0;
    void *topk_topp_workspace_ = nullptr;
    size_t topk_topp_workspace_size_ = 0;
    int *topp_id_vals_buf_;
    int *topp_offset_buf_;

public:
    DecodingGpt2(const IAllocator &allocator, const int batch_size,
                 const int seq_len,
                 const int head_num, const int size_per_head,
                 const int vocab_size, const int decoder_layers,
                 const int start_id, const int end_id,
                 const int *start_ids = nullptr, const int start_len = -1,
                 const int candidate_num = 1,
                 const float probability_threshold = 0.0,
                 const float temperature = 1.0) : allocator_(allocator)
    {
#ifndef NDEBUG
        PRINT_FUNC_NAME_();
#endif
        assert(temperature != 0.0);
        assert(candidate_num > 0 || probability_threshold > 0.0);

        args_.batch_size_ = batch_size;
        args_.seq_len_ = seq_len;
        args_.head_num_ = head_num;
        args_.size_per_head_ = size_per_head;
        args_.hidden_units_ = head_num * size_per_head;
        args_.decoder_layers_ = decoder_layers;
        args_.vocab_size_ = vocab_size;
        args_.start_id_ = start_id;
        args_.end_id_ = end_id;
        args_.candidate_num_ = candidate_num;
        args_.probability_threshold_ = probability_threshold;
        args_.temperature_ = temperature;

        // Convert the start_ids to 2D and transpose the
        // start_ids from [batch_size, start_len] to [start_len, batch_size]
        if (start_ids != nullptr && start_len > 0)
        {
            args_.start_len_ = start_len;
            args_.start_ids_ = new int*[start_len];
            for(int i = 0; i < start_len; i++)
            {
                args_.start_ids_[i] = new int[batch_size];
                for(int j = 0; j < batch_size; j++)
                {
                    args_.start_ids_[i][j] = start_ids[j * start_len + i];
                }
            }
        }
        else
        {
            // fill the start_ids by start_id
            args_.start_len_ = 1;
            args_.start_ids_ = new int*[start_len];
            args_.start_ids_[0] = new int[batch_size];
            for(int j = 0; j < batch_size; j++)
            {
                args_.start_ids_[0][j] = args_.start_id_;
            }
        }

        K_cache_ = new DataType_ *[1];
        V_cache_ = new DataType_ *[1];

        decoder_ = new OpenDecoder<OpType_>(batch_size * 1, 0 /* memory_max_seq_len */,
                                            head_num, size_per_head, 0 /* memory_hidden_units */ );

#if EMBEDDING_TRANSPOSE_OPT == 1
        args_.vocab_size_padded_ = div_up(args_.vocab_size_, 8) * 8;
#else  
        args_.vocab_size_padded_ = args_.vocab_size_;
#endif

        int from_tensor_size = args_.batch_size_ * args_.hidden_units_;                    // type T
        int decoder_workspace_size = decoder_->getWorkspaceSize();                                             // type T
        int decoder_normed_result_buffer_size = args_.batch_size_ * args_.hidden_units_;   // type T
        int cache_size = args_.batch_size_ * args_.seq_len_ * args_.hidden_units_;         // type T
        int logits_buf_size = args_.batch_size_ * args_.vocab_size_padded_; // type T

        int topp_id_vals_buf_size = args_.batch_size_ * args_.vocab_size_; // type int
        int topp_offset_buf_size = args_.batch_size_ + 1;

        const int MEM_C = 128;
        /*from_tensor_size = div_up(from_tensor_size, MEM_C) * MEM_C;
        decoder_workspace_size = div_up(decoder_workspace_size, MEM_C) * MEM_C;
        decoder_normed_result_buffer_size = div_up(decoder_normed_result_buffer_size, MEM_C) * MEM_C;
        cache_size =  div_up(cache_size, MEM_C) * MEM_C;

        logits_buf_size = div_up(logits_buf_size, MEM_C) * MEM_C;
        cum_log_buf_size = div_up(cum_log_buf_size, MEM_C) * MEM_C;
        finished_buf_size = div_up(finished_buf_size, MEM_C) * MEM_C;
        
        topk_ids_buf_size = div_up(topk_ids_buf_size, MEM_C) * MEM_C;
        topk_val_buf_size = div_up(topk_val_buf_size, MEM_C) * MEM_C;
        args_.temp_storage_size_ = div_up(args_.temp_storage_size_, MEM_C) * MEM_C; */

        int embedding_kernel_transposed_padded_size = args_.hidden_units_ * args_.vocab_size_padded_;
        embedding_kernel_transposed_padded_size = div_up(embedding_kernel_transposed_padded_size, MEM_C) * MEM_C;

        // prevent memory misalinged address
        logits_buf_size = (int)(ceil(logits_buf_size / 4.)) * 4;
        
        topp_id_vals_buf_size = (int)(ceil(topp_id_vals_buf_size / 4.)) * 4;
        topp_offset_buf_size = (int)(ceil(topp_offset_buf_size / 4.)) * 4;

        topP_sampling_kernel_kernelLauncher(topp_workspace_,
                                            topp_workspace_size_,
                                            logits_buf_,
                                            topp_id_vals_buf_,
                                            topp_offset_buf_,
                                            nullptr,
                                            0,
                                            args_,
                                            nullptr, 
                                            nullptr, 
                                            args_.vocab_size_,
                                            0);
        topK_sampling_kernel_kernelLauncher(topk_workspace_,
                                            topk_workspace_size_,
                                            logits_buf_,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            0,
                                            args_,
                                            0);
        topK_topP_sampling_kernel_kernelLauncher(topk_topp_workspace_,
                                                 topk_topp_workspace_size_,
                                                 nullptr,
                                                 logits_buf_,
                                                 0,
                                                 args_,
                                                 0);

        int datatype_buf_size = from_tensor_size * 2 + decoder_workspace_size +
                                cache_size * 2 * args_.decoder_layers_ + decoder_normed_result_buffer_size;

        buf_ = reinterpret_cast<void *>(allocator_.malloc(
#if EMBEDDING_TRANSPOSE_OPT == 1
            sizeof(DataType_) * embedding_kernel_transposed_padded_size +
#endif
            sizeof(DataType_) * (datatype_buf_size + logits_buf_size) + 
            sizeof(int) * (topp_id_vals_buf_size + topp_offset_buf_size) +
            topp_workspace_size_ + topk_workspace_size_ + topk_topp_workspace_size_));

#if EMBEDDING_TRANSPOSE_OPT == 1
        embedding_kernel_transposed_padded_ = (DataType_ *)buf_;
        from_tensor_[0] = (DataType_ *)(embedding_kernel_transposed_padded_ + embedding_kernel_transposed_padded_size);
#else
        from_tensor_[0] = (DataType_ *)buf_;
#endif
        from_tensor_[1] = (DataType_ *)(from_tensor_[0] + from_tensor_size);

        /* We use two-way buffer since we have to update KV buf at the end of each step. */
        K_cache_[0] = from_tensor_[1] + from_tensor_size + 0 * cache_size * args_.decoder_layers_;
        V_cache_[0] = from_tensor_[1] + from_tensor_size + 1 * cache_size * args_.decoder_layers_;

        decoder_buf_ = V_cache_[0] + cache_size * args_.decoder_layers_;
        decoder_normed_result_buf_ = (decoder_buf_ + decoder_workspace_size);
        logits_buf_ = decoder_normed_result_buf_ + decoder_normed_result_buffer_size;
        topp_id_vals_buf_ = (int *)(logits_buf_ + logits_buf_size);
        topp_offset_buf_ = (int *)(topp_id_vals_buf_ + topp_id_vals_buf_size);
        topp_workspace_ = (void *)(topp_offset_buf_ + topp_offset_buf_size);
        topk_workspace_ = (void *)(topp_workspace_ + topp_workspace_size_);
        topk_topp_workspace_ = (void *)(topk_workspace_ + topk_workspace_size_);

#if EMBEDDING_TRANSPOSE_OPT == 1
        cudaMemset(embedding_kernel_transposed_padded_, 0, embedding_kernel_transposed_padded_size * sizeof(DataType_));
#endif

        cudaDeviceSynchronize();

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
            cum_log_probs (for eacm beam, the first element is 0). e.g., [0 -inf -inf -inf][0 -inf -inf -inf]
        */

        /* Initialize the first output_ids */

        check_cuda_error(cudaMemcpyAsync(decoding_params.output_ids, args_.start_ids_[0], m*sizeof(int), cudaMemcpyHostToDevice, decoding_params.stream));
        if (args_.probability_threshold_ != 0.0)
        {
            topp_initialization_kernelLauncher(nullptr,
                                               nullptr,
                                               nullptr,
                                               topp_id_vals_buf_,
                                               topp_offset_buf_,
                                               args_.candidate_num_ > 0 ? args_.candidate_num_ : args_.vocab_size_, 
                                               args_,
                                               decoding_params.stream);
        }

#if EMBEDDING_TRANSPOSE_OPT == 1
        transpose(embedding_kernel_transposed_padded_, decoding_params.embedding_kernel, 1,
                  args_.vocab_size_, args_.hidden_units_, 0, decoding_params.stream);
#endif
#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        int cache_size = m * args_.seq_len_ * args_.hidden_units_; // type T

        bool do_beamsearch = false;
        for (int step = 1; step < args_.seq_len_; ++step)
        {
            int *word_ids_buf_ = decoding_params.output_ids + (step - 1) * m;
            do_beamsearch = step >= args_.start_len_;
            //we use two-way buffer
            embedding_position_lookups_kernel_launcher(from_tensor_[0],
                                                       decoding_params.embedding_table,
                                                       decoding_params.position_encoding_table,
                                                       word_ids_buf_,
                                                       m,
                                                       args_.hidden_units_,
                                                       step,
                                                       decoding_params.stream);
#ifndef NDEBUG
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif
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
                decoder_->forward(from_tensor_[from_id], 
                                  nullptr, // memory_tensor should be nullptr
                                  K_cache_[0] + layer * cache_size,
                                  V_cache_[0] + layer * cache_size,
                                  nullptr, nullptr, // key_mem_cache_ and value_mem_cache_ should be nullptr
                                  nullptr, // memory_sequence_length should be nullptr
                                  from_tensor_[out_id], step,
                                  false);

#ifndef NDEBUG
                cudaDeviceSynchronize();
                check_cuda_error(cudaGetLastError());
#endif
            }
            decoder_->decoder_norm1(from_tensor_[out_id], decoding_params.layernorm.gamma,
                                    decoding_params.layernorm.beta, decoder_normed_result_buf_, m, k);

#ifndef NDEBUG
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif

            DataType_ alpha = DataType_(1.0f);
            DataType_ beta = DataType_(0.0f);

            cublasGemmAlgo_t cublasAlgo = static_cast<cublasGemmAlgo_t>(cublasAlgo_[0]);
            check_cuda_error(cublasGemmEx(decoding_params.cublas_handle,
#if EMBEDDING_TRANSPOSE_OPT == 1
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          args_.vocab_size_padded_, m, k,
                                          &alpha,
                                          embedding_kernel_transposed_padded_,
                                          AType_, args_.vocab_size_padded_, //n
#else
                                          CUBLAS_OP_T, CUBLAS_OP_N,
                                          n, m, k,
                                          &alpha,
                                          decoding_params.embedding_kernel,
                                          AType_, k,
#endif
                                          decoder_normed_result_buf_, BType_, k,
                                          &beta,
                                          logits_buf_, CType_,
#if EMBEDDING_TRANSPOSE_OPT == 1
                                          args_.vocab_size_padded_,
#else
                                          n,
#endif
                                          computeType_,
                                          cublasAlgo));

#ifndef NDEBUG
            cudaDeviceSynchronize();
            check_cuda_error(cudaGetLastError());
#endif

            apply_temperature_penalty_kernelLauncher(logits_buf_,
                                                     (DataType_) args_.temperature_,
                                                     m,
                                                     n,
                                                     decoding_params.stream);
            int random_num = rand();
            if (do_beamsearch)
            {
                // Sampling
                if(args_.candidate_num_ > 0 && args_.probability_threshold_ == 0.0)
                {
                    // top k sampling
                    topK_sampling_kernel_kernelLauncher(topk_workspace_,
                                                        topk_workspace_size_,
                                                        logits_buf_,
                                                        decoding_params.output_ids + step * m,
                                                        nullptr,
                                                        nullptr,
                                                        random_num,
                                                        args_,
                                                        decoding_params.stream);
                }
                else if(args_.candidate_num_ == 0 && args_.probability_threshold_ > 0.0f)
                {
                    // top p sampling
                    softmax_kernelLauncher(logits_buf_,
                                           (DataType_*) nullptr,
                                           args_.end_id_,
                                           nullptr,
                                           m,
                                           n,
                                           decoding_params.stream);
#ifndef NDEBUG
                    cudaDeviceSynchronize();
                    check_cuda_error(cudaGetLastError());
#endif
                    topP_sampling_kernel_kernelLauncher(topp_workspace_,
                                                        topp_workspace_size_,
                                                        logits_buf_,
                                                        topp_id_vals_buf_,
                                                        topp_offset_buf_,
                                                        nullptr,
                                                        random_num,
                                                        args_,
                                                        decoding_params.output_ids + step * m,
                                                        nullptr,
                                                        n,
                                                        decoding_params.stream);
                }
                else if(args_.candidate_num_ > 0 && args_.probability_threshold_ > 0.0f)
                {
                    topK_topP_sampling_kernel_kernelLauncher(topk_topp_workspace_,
                                                             topk_topp_workspace_size_,
                                                             decoding_params.output_ids + step * m,
                                                             logits_buf_,
                                                             random_num,
                                                             args_,
                                                             decoding_params.stream);
                }
#ifndef NDEBUG
                cudaDeviceSynchronize();
                check_cuda_error(cudaGetLastError());
#endif
            }
            else
            {
                // else of do_beamsearch (set pre-determined word ids)
                check_cuda_error(cudaMemcpyAsync(decoding_params.output_ids + step*m, args_.start_ids_[step], 
                                m*sizeof(int), cudaMemcpyHostToDevice, decoding_params.stream));
            }
        } // end for decoding step for llop
    } // end of forward

    virtual ~DecodingGpt2()
    {
        delete[] K_cache_;
        delete[] V_cache_;
        delete decoder_;
        allocator_.free(buf_);
        for(int i = 0; i < args_.start_len_; i++)
        {
            delete [] args_.start_ids_[i];
        }
        delete [] args_.start_ids_;
    }
};

} //namespace fastertransformer

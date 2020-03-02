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
 * BERT Encoder transformer
 **/

#pragma once

#include <cuda_runtime.h>
#include "fastertransformer/allocator.h"
#include "fastertransformer/cuda/cuda_kernels.h"
#include "fastertransformer/cuda/open_attention.h"
#include "fastertransformer/common_structure.h"

namespace fastertransformer
{

template <typename T>
class EncoderInitParam
{
public:
  const T *from_tensor;
  const T *to_tensor;

  AttentionWeight<T> self_attention;
  const T *attr_mask;
  LayerNormWeight<T> self_layernorm;

  FFNWeight<T> ffn;
  LayerNormWeight<T> ffn_layernorm;

  T *transformer_out;
  cublasHandle_t cublas_handle;
  cudaStream_t stream;
};

template <OperationType OpType_, template <OperationType> class MultiHeadAttention_>
class BertEncoderTransformerTraits;

template <template <OperationType> class MultiHeadAttention_>
class BertEncoderTransformerTraits<OperationType::FP32, MultiHeadAttention_>
    : public TransformerTraits<OperationType::FP32>
{
public:
  typedef MultiHeadAttention_<OpType> MultiHeadAttention;
};

template <template <OperationType> class MultiHeadAttention_>
class BertEncoderTransformerTraits<OperationType::FP16, MultiHeadAttention_>
    : public TransformerTraits<OperationType::FP16>
{
public:
  typedef MultiHeadAttention_<OpType> MultiHeadAttention;
};

template <class Traits_>
class BertEncoderTransformer
{
  const IAllocator &allocator_;
  typename Traits_::MultiHeadAttention *attention_;
  typedef typename Traits_::DataType DataType_;
  EncoderInitParam<DataType_> param_;

  const cudaDataType_t computeType_ = Traits_::computeType;
  const cudaDataType_t AType_ = Traits_::AType;
  const cudaDataType_t BType_ = Traits_::BType;
  const cudaDataType_t CType_ = Traits_::CType;
  int cublasAlgo_[3];

  DataType_ *buf_;
  DataType_ *attr_out_buf_;
  DataType_ *attr_matmul_buf_;
  DataType_ *inter_matmul_buf_;

  int batch_size_;
  int from_seq_len_;
  int to_seq_len_;
  int head_num_;
  int size_per_head_;

public:
  BertEncoderTransformer(const IAllocator &allocator, int batch_size, int from_seq_len,
                         int to_seq_len, int head_num, int size_per_head) : allocator_(allocator), batch_size_(batch_size), from_seq_len_(from_seq_len),
                                                                            to_seq_len_(to_seq_len), head_num_(head_num), size_per_head_(size_per_head)
  {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif

    int m = batch_size_ * from_seq_len_;
    int k = head_num_ * size_per_head_;
    int n = k;

    int buf_size = m * n;

    try
    {
      buf_ = reinterpret_cast<DataType_ *>(allocator_.malloc(sizeof(DataType_) * buf_size * 6));
      if (buf_ == nullptr)
        throw std::runtime_error(std::string("Tensorflow Allocator failed to allocate internal buffer."));

      attr_out_buf_ = buf_;
      attr_matmul_buf_ = attr_out_buf_ + buf_size;
      inter_matmul_buf_ = attr_matmul_buf_ + buf_size;

      attention_ = new typename Traits_::MultiHeadAttention(allocator_, batch_size_, from_seq_len_, to_seq_len_, head_num_, size_per_head_);
      FILE *fd = fopen("gemm_config.in", "r");
      int err = 0;
      if (fd == NULL)
        printf("gemm_config.in is not found\n");
      else
      {
        err = fscanf(fd, "%d%d%d%*d%*d", &cublasAlgo_[0], &cublasAlgo_[1], &cublasAlgo_[2]);
        fclose(fd);
      }
      if (err != 3)
      {
        printf("loading GEMM algorithms error, using default GEMM algorithms!\n");
        if (Traits_::OpType == OperationType::FP32)
        {
          cublasAlgo_[0] = -1;
          cublasAlgo_[1] = -1;
          cublasAlgo_[2] = -1;
        }
        else
        {
          cublasAlgo_[0] = 99;
          cublasAlgo_[1] = 99;
          cublasAlgo_[2] = 99;
        }
      }
    }
    catch (std::runtime_error &error)
    {
      throw error;
    }
  }
  /**
   * Initialize the parameters in class
   * We will keep the Ctor empty to ensure the sub classes follow the same init routine.
   * Please be aware that no dynamic memory allocation should be placed
   **/
  void initialize(EncoderInitParam<DataType_> param)
  {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    param_ = param;
    cuda::MultiHeadInitParam<DataType_> multi_head_init_param;

    multi_head_init_param.from_tensor = param.from_tensor;
    multi_head_init_param.to_tensor = param.to_tensor;
    multi_head_init_param.self_attention = param.self_attention;
    multi_head_init_param.attr_mask = param.attr_mask;
    multi_head_init_param.stream = param.stream;
    multi_head_init_param.cublas_handle = param.cublas_handle;
    multi_head_init_param.attr_out = attr_out_buf_;

    attention_->initialize(multi_head_init_param);
  }

  /**
   * do forward 
   **/
  void forward()
  {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    try
    {
      attention_->forward();

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

      DataType_ alpha = (DataType_)1.0f;
      DataType_ beta = (DataType_)0.0f;
      int m = batch_size_ * from_seq_len_;
      int k = head_num_ * size_per_head_;
      int n = k;

      check_cuda_error(cublasGemmEx(param_.cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    n, m, k,
                                    &alpha,
                                    param_.self_attention.attention_output_weight.kernel, AType_, n,
                                    attr_out_buf_, BType_, k,
                                    &beta,
                                    attr_matmul_buf_, CType_, n,
                                    computeType_,
                                    static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

      add_bias_input_layernorm_kernelLauncher<DataType_>(attr_matmul_buf_,
                                                         param_.from_tensor, param_.self_attention.attention_output_weight.bias,
                                                         param_.self_layernorm.gamma,
                                                         param_.self_layernorm.beta, m, n, param_.stream);

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

      n *= 4;
      check_cuda_error(cublasGemmEx(param_.cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    n, m, k,
                                    &alpha,
                                    param_.ffn.intermediate_weight.kernel, AType_, n,
                                    attr_matmul_buf_, BType_, k,
                                    &beta,
                                    inter_matmul_buf_, CType_, n,
                                    computeType_,
                                    static_cast<cublasGemmAlgo_t>(cublasAlgo_[1])));

      add_bias_act_kernelLauncher<DataType_>(inter_matmul_buf_, param_.ffn.intermediate_weight.bias, m, n, param_.stream);

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

      n = k;
      k *= 4;
      check_cuda_error(cublasGemmEx(param_.cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    n, m, k,
                                    &alpha,
                                    param_.ffn.output_weight.kernel, AType_, n,
                                    inter_matmul_buf_, BType_, k,
                                    &beta,
                                    param_.transformer_out, CType_, n,
                                    computeType_,
                                    static_cast<cublasGemmAlgo_t>(cublasAlgo_[2])));
      add_bias_input_layernorm_kernelLauncher<DataType_>(param_.transformer_out, attr_matmul_buf_,
                                                         param_.ffn.output_weight.bias,
                                                         param_.ffn_layernorm.gamma,
                                                         param_.ffn_layernorm.beta,
                                                         m, n, param_.stream);
#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif
    }
    catch (std::runtime_error &error)
    {
      throw error;
    }
  }

  void trt_initialize(DataType_ *from_tensor, DataType_ *to_tensor, DataType_ *attr_mask, DataType_ *out, cudaStream_t stream, cublasHandle_t cublas_handle)
  {
    param_.from_tensor = from_tensor;
    param_.to_tensor = to_tensor;
    param_.stream = stream;
    param_.transformer_out = out;
    param_.cublas_handle = cublas_handle;
    attention_->trt_initialize(from_tensor, to_tensor, attr_mask, stream, param_.cublas_handle);
  }

  ~BertEncoderTransformer()
  {
    delete attention_;
    allocator_.free(buf_);
  }
};

} // namespace fastertransformer

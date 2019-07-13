/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "fastertransformer/allocator.h"
#include "fastertransformer/cuda/multi_head_attention.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
namespace fastertransformer{
namespace cuda{


template<OperationType OpType_>
class OpenMultiHeadAttentionTraits;

template<>
class OpenMultiHeadAttentionTraits<OperationType::FP32>
{
 public:
  typedef float DataType;
  static cudaDataType_t const computeType = CUDA_R_32F;
  static cudaDataType_t const AType = CUDA_R_32F;
  static cudaDataType_t const BType = CUDA_R_32F;
  static cudaDataType_t const CType = CUDA_R_32F;
  //others
};

template<>
class OpenMultiHeadAttentionTraits<OperationType::HALF>
{
 public:
  typedef __half DataType;
  static cudaDataType_t const computeType = CUDA_R_16F;
  static cudaDataType_t const AType = CUDA_R_16F;
  static cudaDataType_t const BType = CUDA_R_16F;
  static cudaDataType_t const CType = CUDA_R_16F;
  //others
};

/**
 * Multi-head attetion open sourced
 */
template<OperationType OpType_>
class OpenMultiHeadAttention: IMultiHeadAttention<OpType_>
{
 private:
  typedef OpenMultiHeadAttentionTraits<OpType_> Traits_;
  typedef typename Traits_::DataType DataType_;
  const cudaDataType_t computeType_ = Traits_::computeType;
  const cudaDataType_t AType_ = Traits_::AType;
  const cudaDataType_t BType_ = Traits_::BType;
  const cudaDataType_t CType_ = Traits_::CType;
  const IAllocator& allocator_;
  MultiHeadInitParam<DataType_> param_;

  int cublasAlgo_[3];

  DataType_* buf_;
  DataType_* query_buf_;
  DataType_* key_buf_;
  DataType_* value_buf_;
  DataType_* q_buf_;
  DataType_* k_buf_;
  DataType_* v_buf_;
  DataType_* qk_buf_;
  DataType_* transpose_dst_;


  int batch_size_;
  int from_seq_len_;
  int to_seq_len_;
  int head_num_;
  int size_per_head_;
 public:
  //Ctor
  OpenMultiHeadAttention(const IAllocator& allocator, int batch_size, int from_seq_len, 
      int to_seq_len, int head_num, int size_per_head): 
    allocator_(allocator), batch_size_(batch_size), from_seq_len_(from_seq_len), to_seq_len_(to_seq_len), 
    head_num_(head_num), size_per_head_(size_per_head)
   {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif

    int buf_size = batch_size_ * head_num_ * from_seq_len_ * size_per_head_;
    int qk_buf_size = batch_size_ * head_num_ * from_seq_len_ * from_seq_len_;
    try
    {
      buf_ = (DataType_*) allocator_.malloc(sizeof(DataType_) * (buf_size * 7 + qk_buf_size));
      query_buf_ = buf_;
      key_buf_ = buf_ + buf_size;
      value_buf_ = buf_ + 2 * buf_size;
      q_buf_ = buf_ + 3 * buf_size;
      k_buf_ = buf_ + 4 * buf_size;
      v_buf_ = buf_ + 5 * buf_size;
      qk_buf_ = buf_ + 6 * buf_size;
      transpose_dst_ = qk_buf_ + qk_buf_size;

      FILE* fd = fopen("gemm_config.in", "r");
      int err = 0;
      if(fd == NULL)
        printf("gemm_config.in is not found\n");
      else
      {
        err = fscanf(fd, "%d%*d%*d%d%d", &cublasAlgo_[0], &cublasAlgo_[1], &cublasAlgo_[2]);
        fclose(fd);
      }
      if(err != 3)
      {
	 printf("loading GEMM algorithms error, using default GEMM algorithms\n");
         if(OpType_ == OperationType::FP32)
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
    catch(std::runtime_error& error)
    {
      throw error;
    }
  }

  void forward()
  {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    int m = batch_size_ * from_seq_len_;
    int k = head_num_ * size_per_head_;
    int n = k;

    DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;

    try
    {
      check_cuda_error(cublasGemmEx(param_.cublas_handle, 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        n, m, k, 
        &alpha, 
        param_.attr_kernel_Q, AType_, n, 
        param_.from_tensor, BType_, k, 
        &beta, 
        query_buf_, CType_, n, 
        computeType_, 
        static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

      check_cuda_error(cublasGemmEx(param_.cublas_handle, 
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k, 
        &alpha, 
        param_.attr_kernel_K, AType_, n, 
        param_.to_tensor, BType_, k, 
        &beta, 
        key_buf_, CType_, n, 
        computeType_, 
        static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

      check_cuda_error(cublasGemmEx(param_.cublas_handle, 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        n, m, k,
        &alpha,
        param_.attr_kernel_V, AType_, n, 
        param_.to_tensor, BType_, k, 
        &beta, 
        value_buf_, CType_, n, 
        computeType_, 
        static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

      DataType_ scaler = 1 / sqrtf(size_per_head_ * 1.0f);
      multiHeadAttr_nofuse_kernelLauncher(
        param_.stream,
        param_.cublas_handle,
        query_buf_,
        param_.attr_bias_Q,
        key_buf_,
        param_.attr_bias_K,
        value_buf_,
        param_.attr_bias_V,
        param_.attr_mask,
        param_.attr_out,
        batch_size_,
        from_seq_len_,
        head_num_,
        size_per_head_,
        scaler);
    }
    catch(std::runtime_error& error)
    {
      throw error;
    }
  }

  void multiHeadAttr_kernelLauncher(
      cudaStream_t stream,
      const DataType_* Q,
      const DataType_* bias_Q,
      const DataType_* K,
      const DataType_* bias_K,
      const DataType_* V,
      const DataType_* bias_V,
      const DataType_* attr_mask,
      DataType_* dst,
      const int batch_size,
      const int seq_len,
      const int head_num,
      const int size_per_head,
      const DataType_ scaler);

  void multiHeadAttr_nofuse_kernelLauncher(
      cudaStream_t stream,
      cublasHandle_t handle,
      DataType_* Q,
      const DataType_* bias_Q,
      DataType_* K,
      const DataType_* bias_K,
      DataType_* V,
      const DataType_* bias_V,
      const DataType_* attr_mask,
      DataType_* dst,
      const int batch_size,
      const int seq_len,
      const int head_num,
      const int size_per_head,
      const DataType_ scaler);

  void initialize(MultiHeadInitParam<DataType_> param)
  {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    //Do all the malloc here
    param_ = param;
  }
  void trt_initialize(DataType_* from_tensor, DataType_* to_tensor, DataType_* attr_mask, cudaStream_t stream, 
    cublasHandle_t cublas_handle)
  {
    param_.from_tensor = from_tensor;
    param_.to_tensor = to_tensor;
    param_.attr_mask = attr_mask;
    param_.stream = stream;
    param_.cublas_handle = cublas_handle;
  }

  ~OpenMultiHeadAttention() override
  {
    allocator_.free(buf_);
  }
};

}//namespace cuda
}//namespace fastertransformer

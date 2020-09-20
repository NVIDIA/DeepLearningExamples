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

#pragma once

#include "fastertransformer/allocator.h"
#include "fastertransformer/cuda/multi_head_attention.h"
#include "fastertransformer/cuda/cuda_kernels.h"
#include "fastertransformer/cuda/cuda_int8_kernels.h"
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
class OpenMultiHeadAttentionTraits<OperationType::FP16>
{
 public:
  typedef half DataType;
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

  int cublasAlgo_[4];
  std::map<std::string, cublasLtMatmulAlgo_info> cublasLtAlgoMap;
  bool is_fuse_QKV;

  DataType_* buf_;
  DataType_* query_buf_;
  DataType_* key_buf_;
  DataType_* value_buf_;
  DataType_* q_buf_;
  DataType_* k_buf_;
  DataType_* v_buf_;
  DataType_* qk_buf_;
  DataType_* transpose_dst_;
  
  DataType_** qkv_kernel_;
  DataType_** qkv_input_;
  DataType_** qkv_buf_;

  const float *query_weight_amax_list, *key_weight_amax_list, *value_weight_amax_list;

  int batch_size_;
  int from_seq_len_;
  int to_seq_len_;
  int head_num_;
  int size_per_head_;
  //int8_mode == 0 -- not use int8
  //int8_mode == 1 -- use int8 without quantized residual
  //int8_mode == 2 -- use int8 with quantized residual
  int int8_mode_;
  int* sequence_id_map_;
  int* Q_int_buf_;
  int* K_int_buf_;
  int* V_int_buf_;
  int* qk_int_buf_;
  int* transpose_dst_int_buf_;
 public:
  //Ctor
  OpenMultiHeadAttention(const IAllocator& allocator, int batch_size, int from_seq_len, 
      int to_seq_len, int head_num, int size_per_head, int int8_mode): 
    allocator_(allocator), batch_size_(batch_size), from_seq_len_(from_seq_len), to_seq_len_(to_seq_len), 
    head_num_(head_num), size_per_head_(size_per_head), int8_mode_(int8_mode)
   {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif

    int buf_size = batch_size_ * head_num_ * from_seq_len_ * size_per_head_;
    int qk_buf_size = batch_size_ * head_num_ * from_seq_len_ * from_seq_len_;
    try
    {
      if (int8_mode_ != 0){
        buf_ = (DataType_*) allocator_.malloc(
                                             //query_buf_(Q_int_buf_) key_buf_(K_int_buf_) value_buf_(V_int_buf_) qk_int_buf_ transpose_dst_(transpose_dst_int_buf_)
                                             sizeof(int) * (4*buf_size + qk_buf_size) +
                                             //q_buf_ k_buf_ v_buf_
                                             sizeof(DataType_) * (3*buf_size + qk_buf_size) +
                                             //for fused qkv pointer
                                             sizeof(DataType_*) * 9 +
                                             //sequence_id_map 
                                             (batch_size_*from_seq_len_)*sizeof(int));

        Q_int_buf_ = (int *)(buf_);
        K_int_buf_ = Q_int_buf_ + buf_size;
        V_int_buf_ = K_int_buf_ + buf_size;
        transpose_dst_int_buf_ = V_int_buf_ + buf_size;
        qk_int_buf_ = transpose_dst_int_buf_ + buf_size;
        q_buf_ = (DataType_*)(qk_int_buf_ + qk_buf_size);
        k_buf_ = q_buf_ + buf_size;
        v_buf_ = k_buf_ + buf_size;
        qk_buf_ = v_buf_ + buf_size;
        qkv_kernel_ = (DataType_**)(qk_buf_ + qk_buf_size);
        qkv_input_ = qkv_kernel_ + 3;
        qkv_buf_ = qkv_input_ + 3;
        sequence_id_map_ = (int*)(qkv_buf_ + 3);
        FILE* fd = fopen("igemm_config.in", "r");

        if (fd == NULL)
          printf("igemm_config.in is not found\n");
        else
        {
          int batchCount2, m2, n2, k2, algoId, customOption, tile, splitK_val, swizzle, reductionScheme, workspaceSize;
          while(fscanf(fd,"%d %d %d %d %d %d %d %d %d %d %d\n", &batchCount2, &m2, &n2, &k2, &algoId, &customOption, &tile, &splitK_val, &swizzle, &reductionScheme, &workspaceSize)!=EOF){
            char mark[256];
            sprintf(mark, "%d_%d_%d_%d", batchCount2, m2, n2, k2);
            std::string markStr(mark);
            //workspaceSize should be 0
            if (cublasLtAlgoMap.find(markStr) == cublasLtAlgoMap.end() && workspaceSize == 0){
              cublasLtAlgoMap[markStr].algoId = algoId;
              cublasLtAlgoMap[markStr].customOption = customOption;
              cublasLtAlgoMap[markStr].tile = tile;
              cublasLtAlgoMap[markStr].splitK_val = splitK_val;
              cublasLtAlgoMap[markStr].swizzle = swizzle;
              cublasLtAlgoMap[markStr].reductionScheme = reductionScheme;
              cublasLtAlgoMap[markStr].workspaceSize = workspaceSize;
            }
          }
          fclose(fd);
        }
      }
      else{
        buf_ = (DataType_*) allocator_.malloc(sizeof(DataType_) * (buf_size * 7 + qk_buf_size) + sizeof(DataType_*) * 9 + (batch_size_*from_seq_len_)*sizeof(int));
        query_buf_ = buf_;
        key_buf_ = buf_ + buf_size;
        value_buf_ = buf_ + 2 * buf_size;
        q_buf_ = buf_ + 3 * buf_size;
        k_buf_ = buf_ + 4 * buf_size;
        v_buf_ = buf_ + 5 * buf_size;
        qk_buf_ = buf_ + 6 * buf_size;
        transpose_dst_ = qk_buf_ + qk_buf_size;
        qkv_kernel_ = (DataType_**)(transpose_dst_ + buf_size);
        qkv_input_ = qkv_kernel_ + 3;
        qkv_buf_ = qkv_input_ + 3;
        FILE* fd = fopen("gemm_config.in", "r");
        int err = 0;
        if(fd == NULL)
          printf("gemm_config.in is not found\n");
        else
        {
          float split_time, fused_time;
          err = fscanf(fd, "%d %f %*d %*f %*d %*f %d %*f %d %*f %d %f", 
                      &cublasAlgo_[0], &split_time, &cublasAlgo_[1], &cublasAlgo_[2], &cublasAlgo_[3], &fused_time);
          is_fuse_QKV = fused_time < split_time * 3 ? true : false;
          fclose(fd);
        }
        if(err != 6)
        {
          printf("loading GEMM algorithms error, using default GEMM algorithms\n");
          if(OpType_ == OperationType::FP32)
          {
            cublasAlgo_[0] = -1;
            cublasAlgo_[1] = -1;
            cublasAlgo_[2] = -1;
            cublasAlgo_[3] = -1;
          }
          else
          {
            cublasAlgo_[0] = 99;
            cublasAlgo_[1] = 99;
            cublasAlgo_[2] = 99;
            cublasAlgo_[3] = 99;
          }
          is_fuse_QKV = false;
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
    const int m = param_.sequence_id_offset == nullptr ? batch_size_ * from_seq_len_ : param_.valid_word_num;
    const int k = head_num_ * size_per_head_;
    const int n = k;
    const DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;

    try
    { 
      if (int8_mode_ != 0){
        int fusedINT8QKV = 0;
        const int8_t* Q_weight = (const int8_t*)(param_.self_attention.query_weight.kernel);
        const int8_t* K_weight = (const int8_t*)(param_.self_attention.key_weight.kernel);
        const int8_t* V_weight = (const int8_t*)(param_.self_attention.value_weight.kernel);
        //for QKV weight are DataType_ & continue
        if ((param_.self_attention.query_weight.kernel + n*k == param_.self_attention.key_weight.kernel) &&
            (param_.self_attention.key_weight.kernel + n*k == param_.self_attention.value_weight.kernel))
          fusedINT8QKV = 1;
          //for QVK weight are int8 & continue
          else if ((Q_weight + n*k == K_weight) && (K_weight + n*k == V_weight))
            fusedINT8QKV = 2;

          if (fusedINT8QKV == 0){
            cublasLtMM_withAlgo(Q_int_buf_, 1, m, n, k, 0, 0, 0, 
                                param_.int8_from_tensor, Q_weight, 
                                param_.cublaslt_handle, param_.stream, 
                                cublasLtAlgoMap);
            cublasLtMM_withAlgo(K_int_buf_, 1, m, n, k, 0, 0, 0, 
                                param_.int8_from_tensor, K_weight, 
                                param_.cublaslt_handle, param_.stream, 
                                cublasLtAlgoMap);
            cublasLtMM_withAlgo(V_int_buf_, 1, m, n, k, 0, 0, 0, 
                                param_.int8_from_tensor, V_weight, 
                                param_.cublaslt_handle, param_.stream, 
                                cublasLtAlgoMap);
        }
        else{
          int strideFactor = (fusedINT8QKV == 1) ? (sizeof(DataType_)/sizeof(int8_t)) : 1; 
          cublasLtMM_withAlgo(Q_int_buf_, 3, m, n, k, 0, n*k*strideFactor, 
                              n*m, param_.int8_from_tensor, Q_weight, 
                              param_.cublaslt_handle, param_.stream, cublasLtAlgoMap);
        }

        DataType_ scalar = 1 / sqrtf(size_per_head_ * 1.0f);
        multiHeadAttr_nofuse_kernelLauncher(
                param_.stream,
                param_.cublas_handle,
                param_.cublaslt_handle,
                (DataType_*)Q_int_buf_,
                param_.self_attention.query_weight.bias,
                (DataType_*)(K_int_buf_),
                param_.self_attention.key_weight.bias,
                (DataType_*)(V_int_buf_),
                param_.self_attention.value_weight.bias,
                param_.attr_mask,
                param_.attr_out,
                batch_size_,
                from_seq_len_,
                head_num_,
                size_per_head_,
                int8_mode_,
                scalar);
      }
      else{      
        if(is_fuse_QKV == true)
        {
          check_cuda_error(cublasGemmBatchedEx(param_.cublas_handle, 
                           CUBLAS_OP_N, CUBLAS_OP_N, 
                           n, m, k, 
                           &alpha, 
                           (const void* const*) qkv_kernel_, AType_, n,
                           (const void* const*) qkv_input_, BType_, k,
                           &beta, 
                           (void* const*)qkv_buf_, CType_, n,
                           3,
                           computeType_, 
                           static_cast<cublasGemmAlgo_t>(cublasAlgo_[3])));
        }
        else
        {
          check_cuda_error(cublasGemmEx(param_.cublas_handle, 
            CUBLAS_OP_N, CUBLAS_OP_N, 
            n, m, k, 
            &alpha, 
            param_.self_attention.query_weight.kernel, AType_, n, 
            param_.from_tensor, BType_, k, 
            &beta, 
            query_buf_, CType_, n, 
            computeType_, 
            static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

          check_cuda_error(cublasGemmEx(param_.cublas_handle, 
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k, 
            &alpha, 
            param_.self_attention.key_weight.kernel, AType_, n, 
            param_.to_tensor, BType_, k, 
            &beta, 
            key_buf_, CType_, n, 
            computeType_, 
            static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

          check_cuda_error(cublasGemmEx(param_.cublas_handle, 
            CUBLAS_OP_N, CUBLAS_OP_N, 
            n, m, k,
            &alpha,
            param_.self_attention.value_weight.kernel, AType_, n, 
            param_.to_tensor, BType_, k, 
            &beta, 
            value_buf_, CType_, n, 
            computeType_, 
            static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));
        }
     
#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        DataType_ scalar = 1 / sqrtf(size_per_head_ * 1.0f);
        multiHeadAttr_nofuse_kernelLauncher(
          param_.stream,
          param_.cublas_handle,
          param_.cublaslt_handle,
          query_buf_,
          param_.self_attention.query_weight.bias,
          key_buf_,
          param_.self_attention.key_weight.bias,
          value_buf_,
          param_.self_attention.value_weight.bias,
          param_.attr_mask,
          param_.attr_out,
          batch_size_,
          from_seq_len_,
          head_num_,
          size_per_head_,
          int8_mode_,
          scalar);
      }
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
      const DataType_ scalar);

  void multiHeadAttr_nofuse_kernelLauncher(
      cudaStream_t stream,
      cublasHandle_t handle,
      cublasLtHandle_t cublaslt_handle,
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
      const int int8_mode_,
      const DataType_ scalar);

  void initialize(MultiHeadInitParam<DataType_> param)
  {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    param_ = param;
    if (int8_mode_ != 0){
      int hidden_dim = head_num_ * size_per_head_;
      query_weight_amax_list = param_.amaxList + ACTIVATION_AMAX_NUM;
      key_weight_amax_list = query_weight_amax_list + hidden_dim;
      value_weight_amax_list = key_weight_amax_list + hidden_dim;
    } 
    if(is_fuse_QKV == true && param_.from_tensor != nullptr)
    {
      // For tensorrt, we cannot get the pointer of from tensor until enqueue
      const DataType_* hA[] {param_.self_attention.query_weight.kernel, 
                            param_.self_attention.key_weight.kernel, 
                            param_.self_attention.value_weight.kernel,
                            param_.from_tensor, param_.from_tensor, param_.from_tensor,
                            query_buf_, key_buf_, value_buf_};
      cudaMemcpyAsync((void*)qkv_kernel_, hA, sizeof(DataType_*) * 9, cudaMemcpyHostToDevice, param_.stream);
    }
  }
  void trt_initialize(DataType_* from_tensor, DataType_* to_tensor, DataType_* attr_mask, cudaStream_t stream, 
    cublasHandle_t cublas_handle)
  {
    param_.from_tensor = from_tensor;
    param_.to_tensor = to_tensor;
    param_.attr_mask = attr_mask;
    param_.stream = stream;
    param_.cublas_handle = cublas_handle;
    if(is_fuse_QKV == true)
    {
      const DataType_* hA[] {param_.self_attention.query_weight.kernel, 
                            param_.self_attention.key_weight.kernel, 
                            param_.self_attention.value_weight.kernel,
                            param_.from_tensor, param_.from_tensor, param_.from_tensor,
                            query_buf_, key_buf_, value_buf_};
      cudaMemcpyAsync((void*)qkv_kernel_, hA, sizeof(DataType_*) * 9, cudaMemcpyHostToDevice, param_.stream);
    }
  }

  ~OpenMultiHeadAttention() override
  {
    allocator_.free(buf_);
  }
};

}//namespace cuda
}//namespace fastertransformer

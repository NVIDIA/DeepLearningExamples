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
#include "fastertransformer/cuda/cuda_int8_kernels.h"
#include "fastertransformer/cuda/open_attention.h"
#include "fastertransformer/common_structure.h"

namespace fastertransformer
{

template <typename T>
class EncoderInitParam
{
public:
  const T *from_tensor = nullptr;
  const T *to_tensor = nullptr;

  AttentionWeight<T> self_attention;
  const T *attr_mask = nullptr;
  LayerNormWeight<T> self_layernorm;

  FFNWeight<T> ffn;
  LayerNormWeight<T> ffn_layernorm;

  T *transformer_out;
  cublasHandle_t cublas_handle = nullptr;
  cublasLtHandle_t cublaslt_handle = nullptr;
  cudaStream_t stream = 0;

  const int* sequence_id_offset = nullptr;
  int valid_word_num = -1;
  int layer_idx = 0;
  int layer_num = 12;
  
  //First 80 are for activation amaxs. 
  //For each activation amax, there are 4 values: amax, amax/127.0f, amax/127.0f/127.0f, 127.0f/amax -- input_amax 0-3 , Qbias_amax 4-7, Kbias_amax 8-11, Vbias_amax 12-15, Softmax_amax 16-19, bmm2_amax 20-23, ProjBiasNorm_amax 24-27, F1Bias_amax 28-31, F2BiasNorm_amax 32-35, reserve 36-80
  //following by kernel amaxs : query_weight_amax_list, key_weight_amax_list, value_weight_amax_list, proj_weight_amax_list, FC1_weight_amax_list, FC2_weight_amax_list
  const float *amaxList = nullptr;
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
  std::map<std::string, cublasLtMatmulAlgo_info> cublasLtAlgoMap;

  DataType_ *buf_;
  DataType_ *attr_out_buf_;
  DataType_ *attr_matmul_buf_;
  DataType_ *inter_matmul_buf_;
  DataType_ *attr_out_tmp_buf_;
  DataType_ *out_tmp_buf_;
  DataType_ *from_tensor_tmp_buf_;

  int batch_size_;
  int from_seq_len_;
  int to_seq_len_;
  int head_num_;
  int size_per_head_;


  //for int8 quantization
  const float *FC0_weight_amax_list, *FC1_weight_amax_list, *FC2_weight_amax_list;
  const float *qkv_amax_ptr, *FC0_addBias_layernorm_amax_ptr, *FC1_addBias_amax_ptr, *FC2_addBias_layernorm_amax_ptr, *to_tensor_amax_ptr;
  //int8_mode == 0 -- not use int8
  //int8_mode == 1 -- use int8 without quantized residual
  //int8_mode == 2 -- use int8 with quantized residual
  int int8_mode_;
  int layer_idx_;
  int layer_num_;
  const int8_t *int8_from_tensor_;
  const DataType_ * transA_from_tensor_;
  int32_t *int_buf_;
  DataType_ *tmp_DataType_, *transA_from_tensor_tmp_, *transformer_out_tmp_DataType_;
  int8_t *tmp_int8_, *int8_from_tensor_tmp_, *attr_matmul_buf_tmp_, *transformer_out_tmp_int8_;

public:

 void setLayerIdx(int layer_idx){
   layer_idx_ = layer_idx;
 }

  BertEncoderTransformer(const IAllocator &allocator, int batch_size, int from_seq_len,
                         int to_seq_len, int head_num, int size_per_head, int int8_mode=0) : allocator_(allocator), batch_size_(batch_size), from_seq_len_(from_seq_len),
                                                                            to_seq_len_(to_seq_len), head_num_(head_num), size_per_head_(size_per_head), int8_mode_(int8_mode)
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
      if (int8_mode_ != 0){
        buf_ = reinterpret_cast<DataType_ *>(allocator_.malloc(
                                               //transA_from_tensor & transformer_out_tmp_DataType
                                               m*k*sizeof(DataType_) + 
                                               //int8_from_tensor & attr_matmul_buf_tmp & transformer_out_tmp_int8
                                               m*k*sizeof(int8_t) +
                                               //int8 qkv weight
                                               3*n*k*sizeof(int8_t) + 
                                               //FC0 & FC1 & FC2 for m*k(4k)*sizeof(DataType)
                                               4*m*k * sizeof(int) +
                                               //attr_out_buf_ & attr_matmul_buf_ & inter_matmul_buf_
                                               6*m*n*sizeof(DataType_) +
                                               //temp buf
                                               m*n*sizeof(DataType_)
                                               )
                                              );
        if (buf_ == nullptr)
          throw std::runtime_error(std::string("Allocator failed to allocate internal buffer."));

        attr_out_buf_ = (DataType_*)(((void*)buf_) + m*k*sizeof(DataType_) + m*k*sizeof(int8_t) + 3*n*k*sizeof(int8_t) + 4*m*k * sizeof(int));
        attr_matmul_buf_ = attr_out_buf_ + buf_size;
        inter_matmul_buf_ = attr_matmul_buf_ + buf_size;

        int8_from_tensor_tmp_ = (int8_t *)(((void*)buf_) + m*k*(sizeof(DataType_)));
        attr_matmul_buf_tmp_ = int8_from_tensor_tmp_;
        transformer_out_tmp_int8_ = int8_from_tensor_tmp_;
        transA_from_tensor_tmp_ = (DataType_*)buf_;
        transformer_out_tmp_DataType_ = transA_from_tensor_tmp_;

        int_buf_ = (int32_t*)(((void*)buf_) + (m * k) * (sizeof(DataType_) + sizeof(int8_t)) + 3*n*k*sizeof(int8_t));

        tmp_DataType_ = (DataType_*)(((void*)buf_) + m*k*(sizeof(DataType_)+sizeof(int8_t)) + 3*n*k*sizeof(int8_t) + 4*m*k * sizeof(int32_t) + 6*m*n*sizeof(DataType_));
        tmp_int8_ = (int8_t*)tmp_DataType_;
      
        FILE *fd = fopen("igemm_config.in", "r");
 
        if (fd == NULL)
          printf("igemm_config.in is not found\n");
        else
        {
          int batchCount2, m2, n2, k2, algoId, customOption, tile, splitK_val, swizzle, reductionScheme, workspaceSize;
          while(fscanf(fd,"%d %d %d %d %d %d %d %d %d %d %d\n", &batchCount2, &m2, &n2, &k2, &algoId, &customOption, &tile, &splitK_val, &swizzle, &reductionScheme, &workspaceSize)!=EOF){
            char mark[256];
            sprintf(mark, "%d_%d_%d_%d", batchCount2, m2, n2, k2);
            std::string markStr(mark);
            //workspaceSize should be zero
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
        buf_ = reinterpret_cast<DataType_ *>(allocator_.malloc(sizeof(DataType_) * buf_size * (6 + 3)));
        if (buf_ == nullptr)
          throw std::runtime_error(std::string("Allocator failed to allocate internal buffer."));

        attr_out_buf_ = buf_;
        attr_matmul_buf_ = attr_out_buf_ + buf_size;
        inter_matmul_buf_ = attr_matmul_buf_ + buf_size;

        attr_out_tmp_buf_ = inter_matmul_buf_ + 4 * buf_size;
        out_tmp_buf_ = attr_out_tmp_buf_ + buf_size;
        from_tensor_tmp_buf_ = out_tmp_buf_ + buf_size;
        FILE *fd = fopen("gemm_config.in", "r");
        int err = 0;
        if (fd == NULL)
          printf("gemm_config.in is not found\n");
        else
        {
          err = fscanf(fd, "%d %*f %d %*f %d %*f %*d %*f %*d %*f %*d %*f", &cublasAlgo_[0], &cublasAlgo_[1], &cublasAlgo_[2]);
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
      attention_ = new typename Traits_::MultiHeadAttention(allocator_, batch_size_, from_seq_len_, to_seq_len_, head_num_, size_per_head_, int8_mode_);
    }
    catch (std::runtime_error &error)
    {
      throw error;
    }
  }

  void genTransATensorAndInt8TensorForFirstLayer(){
    const int m = param_.sequence_id_offset == nullptr ? batch_size_ * from_seq_len_ : param_.valid_word_num;
    const int k = head_num_ * size_per_head_;
    const int n = k;
    transposeMatrix_kernelLauncher(tmp_DataType_, param_.from_tensor, k, m, param_.stream);
    FT_transformA_kernelLauncher(transA_from_tensor_tmp_, tmp_DataType_, m, k, param_.stream);
    transA_from_tensor_ = (const DataType_*)transA_from_tensor_tmp_;
    quantized_kernelLauncher(int8_from_tensor_tmp_, transA_from_tensor_, m*k, to_tensor_amax_ptr+3, param_.stream);
    int8_from_tensor_ = (const int8_t*)(int8_from_tensor_tmp_);
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

    if (int8_mode_ != 0){
      int hidden_dim = size_per_head_*head_num_;
      layer_idx_ = param_.layer_idx;
      layer_num_ = param_.layer_num;

      qkv_amax_ptr = param_.amaxList + 20;
      FC0_addBias_layernorm_amax_ptr = param_.amaxList + 24; 
      FC1_addBias_amax_ptr = param_.amaxList + 28;
      FC2_addBias_layernorm_amax_ptr = param_.amaxList + 32; 
      to_tensor_amax_ptr = param_.amaxList;

      FC0_weight_amax_list = param_.amaxList + ACTIVATION_AMAX_NUM + 3*hidden_dim;
      FC1_weight_amax_list = FC0_weight_amax_list + hidden_dim;
      FC2_weight_amax_list = FC1_weight_amax_list + 4*hidden_dim;

      int n = hidden_dim;
      int k = hidden_dim;

      const int m = param_.sequence_id_offset == nullptr ? batch_size_ * from_seq_len_ : param_.valid_word_num;
      if (layer_idx_ == 0){
        genTransATensorAndInt8TensorForFirstLayer();
      }
      else
      {
        transA_from_tensor_ = param_.from_tensor;
        if (int8_mode_ == 2){
          int8_from_tensor_ = (const int8_t*)transA_from_tensor_;
        }
        else{
          quantized_kernelLauncher(int8_from_tensor_tmp_, transA_from_tensor_, m*k, to_tensor_amax_ptr + 3, param_.stream);
          int8_from_tensor_ = (const int8_t*)(int8_from_tensor_tmp_);
        } 
      }

      multi_head_init_param.int8_from_tensor = int8_from_tensor_;
      
      multi_head_init_param.cublaslt_handle = param_.cublaslt_handle;

      multi_head_init_param.amaxList = param_.amaxList;
    }


    multi_head_init_param.from_tensor = param.from_tensor;
    multi_head_init_param.to_tensor = param.to_tensor;
    multi_head_init_param.self_attention = param.self_attention;
    multi_head_init_param.attr_mask = param.attr_mask;
    multi_head_init_param.stream = param.stream;
    multi_head_init_param.cublas_handle = param.cublas_handle;
    multi_head_init_param.attr_out = attr_out_buf_;
    multi_head_init_param.valid_word_num = param.valid_word_num;
    multi_head_init_param.sequence_id_offset = param.sequence_id_offset;

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
      const int m = param_.sequence_id_offset == nullptr ? batch_size_ * from_seq_len_ : param_.valid_word_num;
      int k = head_num_ * size_per_head_;
      int n = k;

      if (int8_mode_ != 0){
        cublasLtMM_withAlgo(int_buf_, 1, m, n, k, m*k, n*k, m*n, 
                            (int8_t*)attr_out_buf_, (int8_t*)(param_.self_attention.attention_output_weight.kernel), 
                            param_.cublaslt_handle, param_.stream, cublasLtAlgoMap);
        if (int8_mode_ == 1){
          add_bias_input_layernorm_COL32_int32I_DataTypeO_kernelLauncher(attr_matmul_buf_, int_buf_, transA_from_tensor_, param_.self_attention.attention_output_weight.bias, 
                                                                         param_.self_layernorm.gamma, param_.self_layernorm.beta, m, n, param_.stream, 
                                                                         FC0_weight_amax_list, qkv_amax_ptr);
        }
        else{
          add_bias_input_layernorm_COL32_mixIntI_int8O_kernelLauncher((int8_t*)attr_matmul_buf_, int_buf_, int8_from_tensor_, 
                                                                      param_.self_attention.attention_output_weight.bias, 
                                                                      param_.self_layernorm.gamma, param_.self_layernorm.beta, 
                                                                      m, n, param_.stream, FC0_weight_amax_list, qkv_amax_ptr+2, 
                                                                      to_tensor_amax_ptr+1, FC0_addBias_layernorm_amax_ptr+3);
        }
        
#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        n *= 4;
        
        if (int8_mode_ == 1){
          quantized_kernelLauncher(attr_matmul_buf_tmp_, attr_matmul_buf_, k*m, FC0_addBias_layernorm_amax_ptr + 3, param_.stream);
          cublasLtMM_withAlgo(int_buf_, 1, m, n, k, m*k, n*k, m*n, 
                              attr_matmul_buf_tmp_, (int8_t*)(param_.ffn.intermediate_weight.kernel), 
                              param_.cublaslt_handle, param_.stream, cublasLtAlgoMap);
        }
        else{
          cublasLtMM_withAlgo(int_buf_, 1, m, n, k, m*k, n*k, m*n, 
                              (int8_t*)attr_matmul_buf_, (int8_t*)(param_.ffn.intermediate_weight.kernel), 
                              param_.cublaslt_handle, param_.stream, cublasLtAlgoMap);
        }
        
        add_bias_act_COL32_int32I_int8O_kernelLauncher((int8_t*)inter_matmul_buf_, int_buf_, param_.ffn.intermediate_weight.bias, 
                                                       m, n, param_.stream, FC1_weight_amax_list, FC0_addBias_layernorm_amax_ptr+2, 
                                                       FC1_addBias_amax_ptr+3);
      
#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        n = k;
        k *= 4;
        
        cublasLtMM_withAlgo(int_buf_, 1, m, n, k, m*k, n*k, m*n, 
                            (int8_t*)inter_matmul_buf_, (int8_t*)(param_.ffn.output_weight.kernel), 
                            param_.cublaslt_handle, param_.stream, cublasLtAlgoMap);
      
        if (int8_mode_ == 1){
          add_bias_input_layernorm_COL32_int32I_DataTypeO_kernelLauncher(param_.transformer_out, int_buf_, attr_matmul_buf_, 
                                                                         param_.ffn.output_weight.bias, param_.ffn_layernorm.gamma, 
                                                                         param_.ffn_layernorm.beta, m, n, param_.stream, FC2_weight_amax_list, 
                                                                         FC1_addBias_amax_ptr);
          if (layer_idx_ == layer_num_ - 1){
            FT_transformC_kernelLauncher(transformer_out_tmp_DataType_, param_.transformer_out, m, n, param_.stream);
            transposeMatrix_kernelLauncher(param_.transformer_out, transformer_out_tmp_DataType_, m, n, param_.stream);
          }
        }
        else{
          add_bias_input_layernorm_COL32_mixIntI_int8O_kernelLauncher((int8_t*)param_.transformer_out, int_buf_, (int8_t*)attr_matmul_buf_, 
                                                                      param_.ffn.output_weight.bias, param_.ffn_layernorm.gamma,
                                                                      param_.ffn_layernorm.beta, m, n, param_.stream, FC2_weight_amax_list, 
                                                                      FC1_addBias_amax_ptr+2, FC0_addBias_layernorm_amax_ptr+1, 
                                                                      FC2_addBias_layernorm_amax_ptr+3);
          if (layer_idx_ == layer_num_ - 1){
            FT_transformC_kernelLauncher(transformer_out_tmp_int8_, (int8_t*)param_.transformer_out, m, n, param_.stream);
            transposeMatrix_kernelLauncher(tmp_int8_, transformer_out_tmp_int8_, m, n, param_.stream);
            dequantized_kernelLauncher(param_.transformer_out, tmp_int8_, m*n, FC2_addBias_layernorm_amax_ptr+1, param_.stream);
          }
        }
        
#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif  
      }
      else{
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

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
 * Multi-head attention interface
 */

#pragma once

#include "fastertransformer/common.h"
#include "fastertransformer/common_structure.h"

namespace fastertransformer{
namespace cuda{

template<typename T>
class MultiHeadInitParam{
 public:
   const T* from_tensor;
   const int8_t* int8_from_tensor;
   const T* to_tensor;
   AttentionWeight<T> self_attention;
   const T* attr_mask;
   T* attr_out;

   const int* sequence_id_offset;
   int valid_word_num;
   cublasHandle_t cublas_handle;
   cublasLtHandle_t cublaslt_handle;
   cudaStream_t stream;

   //First 80 are for activation amaxs. 
   //For each activation amax, there are 4 values: amax, amax/127.0f, amax/127.0f/127.0f, 127.0f/amax -- input_amax 0-3 , Q_aftergemm_amax 4-7, Qbias_amax 8-11, K_aftergemm_amax 12-15, Kbias_amax 16-19, V_aftergemm_amax 20-23, Vbias_amax 24-27, bmm1_amax 28-31, Softmax_amax 32-35, bmm2_amax 36-39, Proj_aftergemm_scale 40-43, ProjBiasNorm_amax 44-47, FC1_aftergemm_amax 48-51, F1Bias_amax 52-55, FC2_aftergemm_amax 56-59, F2BiasNorm_amax 60-63, reserve 64-79
   //following by kernel amaxs : query_weight_amax_list, key_weight_amax_list, value_weight_amax_list, proj_weight_amax_list, FC1_weight_amax_list, FC2_weight_amax_list
   //following by int8 gemm deQ scale list: Q_deQ_scale, K_deQ_scale, V_deQ_scale, bmm1_deQ_scale, bmm2_deQ_scale, proj_deQ_scale, FC1_deQ_scale, FC2_deQ_scale
   const float *amaxList;
   const float *int8O_gemm_deQ_scale_list;
   const int *trt_seqlen_offset;
   int trt_seqlen_size;

   MultiHeadInitParam(){
     from_tensor = nullptr;
     to_tensor = nullptr;
     attr_mask = nullptr;
     attr_out = nullptr;
     cublas_handle = nullptr;
     sequence_id_offset = nullptr;
     cublaslt_handle = nullptr;
     int8_from_tensor = nullptr;
     amaxList = nullptr;
     int8O_gemm_deQ_scale_list = nullptr;
     
     stream = 0;
     trt_seqlen_offset = nullptr;
     trt_seqlen_size = -1;
   }
};


/**
 * Interface of attention operation
 */
template<OperationType OpType_>
class IMultiHeadAttention{
 public:
//  typedef MultiHeadInitParam<OpType_> InitParam;
  /**
   * do forward 
   **/
  virtual void forward() = 0;

  /**
   * Initialize the parameters in class
   * We will keep the Ctor empty to ensure the sub classes follow the same init routine.
   * Please be aware that no dynamic memory allocation should be placed
   **/
//  virtual void free() = 0;

  virtual ~IMultiHeadAttention(){}

};


}//namespace cuda
}//namespace fastertransformer

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
   const T* to_tensor;
   AttentionWeight<T> self_attention;
   const T* attr_mask;
   T* attr_out;

   const int* sequence_id_offset;
   int valid_word_num;
   cublasHandle_t cublas_handle;
   cudaStream_t stream;
   MultiHeadInitParam(){
     from_tensor = nullptr;
     to_tensor = nullptr;
     attr_mask = nullptr;
     attr_out = nullptr;
     cublas_handle = nullptr;
     sequence_id_offset = nullptr;
     stream = 0;
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

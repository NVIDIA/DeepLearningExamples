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
#pragma once

#ifndef TENSORFLOW_CORE_KERNELS_MULTIHEADATTR_OP_H_
#define TENSORFLOW_CORE_KERNELS_MULTIHEADATTR_OP_H_

#include "fastertransformer/common.h"
#include "fastertransformer/faster_transformer.h"
#include "fastertransformer/tf_op/tf_traits.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <cublas_v2.h>
using namespace fastertransformer;
namespace tensorflow
{
  namespace functor
  {
    template <typename Device, typename T>
    struct BertTransformerOpFunctor
    {
      typedef typename TFTraits<T>::DataType DataType_;
      static Status Compute(OpKernelContext *context,
        EncoderInitParam<DataType_ > param,
        BertEncoderTransformer<BertEncoderTransformerTraits< TFTraits<T>::OpType, 
          cuda::OpenMultiHeadAttention > > *encoder_transformer);
    };
  } //namespace functor
} //namespace tensorflow
#endif

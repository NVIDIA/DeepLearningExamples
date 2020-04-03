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
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "fastertransformer/tf_op/decoding_op.h"
#include "fastertransformer/decoding_opennmt.h"
#include "fastertransformer/common.h"
#include "fastertransformer/open_decoder.h"
#include "tensorflow/core/framework/op.h"
#include <cuda_runtime.h>
#include <string>
namespace tensorflow
{
using GPUDevice = Eigen::GpuDevice;
using namespace fastertransformer;

namespace functor
{
template <typename T>
struct DecodingOpFunctor<GPUDevice, T>
{
  typedef typename TFTraits<T>::DataType DataType_;
  static Status DynamicDecode(OpKernelContext *context,
      const int num_layers,
      DecoderInitParam<DataType_ > *params, 
      DecodingOpenNMT<TFTraits<T>::OpType> *decoding_opennmt,
      const int max_seq_len,
      DecodingInitParam<DataType_> decoding_params)
  {
    const cudaStream_t &stream = context->eigen_device<GPUDevice>().stream();
    try
    {
      decoding_params.stream = stream;
      for(int i = 0; i < num_layers; ++i)
      {
        params[i].stream = stream;
        check_cuda_error(cublasSetStream(params[i].cublas_handle, stream));
      }
      decoding_opennmt->forward(params, decoding_params);

      return Status::OK();
    }
    catch(std::runtime_error& error)
    {
      return errors::Internal(error.what());
    }
    catch(...)
    {
      return errors::Internal("Runtime error");
    }
  }
};
} //namespace functor

template struct functor::DecodingOpFunctor<GPUDevice, float>;
template struct functor::DecodingOpFunctor<GPUDevice, Eigen::half>;
} //namespace tensorflow
#endif

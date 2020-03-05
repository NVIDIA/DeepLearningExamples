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
#include "fastertransformer/tf_op/decoder_op.h"
#include "fastertransformer/beamsearch_opennmt.h"
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
struct DecoderOpFunctor<GPUDevice, T>
{
  typedef typename TFTraits<T>::DataType DataType_;
  static Status DynamicDecode(OpKernelContext *context,
      DecoderInitParam<DataType_ > params, 
      OpenDecoder<TFTraits<T>::OpType> *decoder, DataType_ *decoder_buffer,
      const DataType_ *from_tensor, const DataType_ *memory_tensor, 
      DataType_ *key_cache, DataType_ *value_cache,
      DataType_ *key_mem_cache, DataType_ *value_mem_cache,
      const int* memory_sequence_length,
      DataType_ *decoder_output, const int step)
  {
    const cudaStream_t &stream = context->eigen_device<GPUDevice>().stream();
    params.stream = stream;
    try
    {
      check_cuda_error(cublasSetStream(params.cublas_handle, stream));
      decoder->initialize(params, decoder_buffer);
      decoder->forward(from_tensor, memory_tensor, 
        key_cache, value_cache, 
        key_mem_cache, value_mem_cache, 
        memory_sequence_length, decoder_output, step);

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

template struct functor::DecoderOpFunctor<GPUDevice, float>;
template struct functor::DecoderOpFunctor<GPUDevice, Eigen::half>;
} //namespace tensorflow
#endif

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
#include "fastertransformer/open_decoder.h"
#include "fastertransformer/tf_op/decoder_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include <cuda_fp16.h>

namespace tensorflow
{
namespace
{
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <typename T>
class CommonOp : public OpKernel
{
public:
  explicit CommonOp(OpKernelConstruction *context) : OpKernel(context) {
    try
    {
    check_cuda_error(cublasCreate(&cublas_handle_));
    }
    catch(std::runtime_error& error)
    {
    OP_REQUIRES(context, false, errors::Internal(error.what()));
    }
  };

  template<typename DataType_>
  void get_tensor(OpKernelContext *context, int tensor_id, const DataType_** tensor_ptr, int off_set = 0){
    *tensor_ptr = reinterpret_cast<const DataType_ *>(context->input(tensor_id).flat<T>().data()) + off_set;
    OP_REQUIRES(context, *tensor_ptr != nullptr, errors::InvalidArgument("tensor %d is null", tensor_id));
  }

  cublasHandle_t get_cublas_handler() {return cublas_handle_; }

  ~CommonOp() { cublasDestroy(cublas_handle_); }
private:
  cublasHandle_t cublas_handle_;

};

} //namespace
} //namespace tensorflow

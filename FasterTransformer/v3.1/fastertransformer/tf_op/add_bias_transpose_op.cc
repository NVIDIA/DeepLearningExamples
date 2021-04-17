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

#define EIGEN_USE_GPU

#include "fastertransformer/faster_transformer.h"
#include "fastertransformer/tf_op/common_op.h"
#include "fastertransformer/cuda/open_attention.h"

namespace tensorflow
{
namespace
{
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("AddBiasTranspose")
    .Input("q_tensor: T")
    .Input("q_bias: T")
    .Input("k_tensor: T")
    .Input("k_bias: T")
    .Input("v_tensor: T")
    .Input("v_bias: T")
    .Output("output: T")
    .Attr("T: {float, half}")
    .Attr("head_num: int >= 1")
    .Attr("size_per_head: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        shape_inference::DimensionHandle dim0 = c->Dim(c->input(0), 0);
        int head_num_, size_per_head_;
        c->GetAttr("head_num", &head_num_);
        c->GetAttr("size_per_head", &size_per_head_);
        c->set_output(0, c->MakeShape({dim0, head_num_, 3, size_per_head_}));
        return Status::OK();
    });
template <typename Device, typename T>
class AddBiasTransposeOp : public CommonOp<T>
{
public:
  explicit AddBiasTransposeOp(OpKernelConstruction *context) : CommonOp<T>(context)
  {
      OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
      OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));
  }

  void Compute(OpKernelContext *context) override
  {
    OP_REQUIRES(context, context->input(0).dims()==2,
                errors::InvalidArgument("Invalid rank. The rank of from tensor should be 3\
                                        ([varSeqlen, 3 * hidden_dimension])"));
    OP_REQUIRES(context, context->input(1).dims() == 1,
                errors::InvalidArgument("Invalid rank. The rank of sequence length should be 1 " \
                                        "([batch_size + 1])"));

    const int valid_word_num_ = (int)context->input(0).dim_size(0);

    const cudaStream_t &stream = context->eigen_device<Device>().stream();

    const T* q = reinterpret_cast<const T *>(context->input(0).flat<T>().data());
    const T* q_b = reinterpret_cast<const T *>(context->input(1).flat<T>().data());
    const T* k = reinterpret_cast<const T *>(context->input(2).flat<T>().data());
    const T* k_b = reinterpret_cast<const T *>(context->input(3).flat<T>().data());
    const T* v = reinterpret_cast<const T *>(context->input(4).flat<T>().data());
    const T* v_b = reinterpret_cast<const T *>(context->input(5).flat<T>().data());

    Tensor *output = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(0, {valid_word_num_, head_num_, 3, size_per_head_}, &output));
    DataType_* output_ptr = reinterpret_cast<DataType_*>(output->flat<T>().data());

    try
    {
        cuda::add_QKV_bias_transpose_kernelLauncher((const half*)q, (const half*)q_b, 
                                                    (const half*)k, (const half*)k_b, 
                                                    (const half*)v, (const half*)v_b, 
                                                    (half*)output_ptr, 
                                                    valid_word_num_, 
                                                    head_num_, size_per_head_, 
                                                    stream);

        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
    }
    catch(std::runtime_error& error)
    {
        std::cout << errors::Internal(error.what());
        exit(-1);
    }
    catch(...)
    {
        std::cout << errors::Internal("Runtime error");
        exit(-1);
    }
  }

private:
    int head_num_, size_per_head_;
    typedef TFTraits<T> traits_;
    typedef typename traits_::DataType DataType_;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                                                  \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("AddBiasTranspose").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      AddBiasTransposeOp<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU

#endif

} //namespace
} //namespace tensorflow

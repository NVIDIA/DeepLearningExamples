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
#include "fastertransformer/faster_transformer.h"
#include "fastertransformer/tf_op/bert_transformer_op.h"
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

REGISTER_OP("BertTransformer")
  .Input("from_tensor: T")
  .Input("to_tensor: T")
  .Input("attr_kernel_q: T")
  .Input("attr_kernel_k: T")
  .Input("attr_kernel_v: T")
  .Input("attr_bias_q: T")
  .Input("attr_bias_k: T")
  .Input("attr_bias_v: T")
  .Input("attr_mask: T")
  .Input("attr_output_kernel: T")
  .Input("attr_output_bias: T")
  .Input("attr_output_layernorm_beta: T")
  .Input("attr_output_layernorm_gamma: T")
  .Input("inter_kernel: T")
  .Input("inter_bias: T")
  .Input("output_kernel: T")
  .Input("output_bias: T")
  .Input("output_layernorm_beta: T")
  .Input("output_layernorm_gamma: T")
  .Output("output: T")
  .Attr("T: {float, half}")
  .Attr("batch_size: int >= 1")
  .Attr("from_seq_len: int >= 1")
  .Attr("to_seq_len: int >= 1")
  .Attr("head_num: int >= 1")
  .Attr("size_per_head: int >= 1")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
      int batch_size, from_seq_len, to_seq_len, head_num, size_per_head;
      c->GetAttr("batch_size", &batch_size);
      c->GetAttr("from_seq_len", &from_seq_len);
      c->GetAttr("to_seq_len", &to_seq_len);
      c->GetAttr("head_num", &head_num);
      c->GetAttr("size_per_head", &size_per_head);
      c->set_output(0, c->MakeShape({batch_size * from_seq_len, head_num * size_per_head}));
      return Status::OK();
      });
template <typename Device, typename T>
class BertTransformerOp : public OpKernel
{
  public:
    explicit BertTransformerOp(OpKernelConstruction *context) : OpKernel(context)
    {
      OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batch_size_));
      OP_REQUIRES_OK(context, context->GetAttr("from_seq_len", &from_seq_len_));
      OP_REQUIRES_OK(context, context->GetAttr("to_seq_len", &to_seq_len_));
      OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
      OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));

      OP_REQUIRES(context, (from_seq_len_ == to_seq_len_),
          errors::InvalidArgument("Only support from_seq_len == to_seq_len"));

      try
      {
        check_cuda_error(cublasCreate(&cublas_handle_));
      }
      catch(std::runtime_error& error)
      {
        OP_REQUIRES(context, false, errors::Internal(error.what()));
      }
    }

    void Compute(OpKernelContext *context) override
    {
      typedef BertEncoderTransformerTraits<traits_::OpType, cuda::OpenMultiHeadAttention> EncoderTraits_;
      BertEncoderTransformer<EncoderTraits_> *encoder_transformer_;
      try
      {
        fastertransformer::Allocator<AllocatorType::TF> allocator_(context);
        encoder_transformer_ = new BertEncoderTransformer<EncoderTraits_>(allocator_, 
          batch_size_, from_seq_len_, to_seq_len_, head_num_, size_per_head_);
      }
      catch(std::runtime_error& error)
      {
        OP_REQUIRES(context, false, errors::Internal(error.what()));
      }
      
      OP_REQUIRES(context, context->num_inputs() == 19, errors::InvalidArgument("Less input arguments"));

      EncoderInitParam<DataType_> param; //init param here
      param.cublas_handle = cublas_handle_;
      param.from_tensor = reinterpret_cast<const DataType_ *>(context->input(0).flat<T>().data());
      param.to_tensor = reinterpret_cast<const DataType_ *>(context->input(1).flat<T>().data());
      param.attr_kernel_Q = reinterpret_cast<const DataType_ *>(context->input(2).flat<T>().data());
      param.attr_kernel_K = reinterpret_cast<const DataType_ *>(context->input(3).flat<T>().data());
      param.attr_kernel_V = reinterpret_cast<const DataType_ *>(context->input(4).flat<T>().data());
      param.attr_bias_Q = reinterpret_cast<const DataType_ *>(context->input(5).flat<T>().data());
      param.attr_bias_K = reinterpret_cast<const DataType_ *>(context->input(6).flat<T>().data());
      param.attr_bias_V = reinterpret_cast<const DataType_ *>(context->input(7).flat<T>().data());
      param.attr_mask = reinterpret_cast<const DataType_ *>(context->input(8).flat<T>().data());
      param.attr_output_kernel = reinterpret_cast<const DataType_ *>(context->input(9).flat<T>().data());
      param.attr_output_bias = reinterpret_cast<const DataType_ *>(context->input(10).flat<T>().data());
      param.attr_output_layernorm_beta = reinterpret_cast<const DataType_ *>(context->input(11).flat<T>().data());
      param.attr_output_layernorm_gamma = reinterpret_cast<const DataType_ *>(context->input(12).flat<T>().data());
      param.inter_kernel = reinterpret_cast<const DataType_ *>(context->input(13).flat<T>().data());
      param.inter_bias = reinterpret_cast<const DataType_ *>(context->input(14).flat<T>().data());
      param.output_kernel = reinterpret_cast<const DataType_ *>(context->input(15).flat<T>().data());
      param.output_bias = reinterpret_cast<const DataType_ *>(context->input(16).flat<T>().data());
      param.output_layernorm_beta = reinterpret_cast<const DataType_ *>(context->input(17).flat<T>().data());
      param.output_layernorm_gamma = reinterpret_cast<const DataType_ *>(context->input(18).flat<T>().data());

      OP_REQUIRES(context, param.from_tensor != nullptr, errors::InvalidArgument("from tensor is null"));
      OP_REQUIRES(context, param.to_tensor != nullptr, errors::InvalidArgument("to tensor is null"));
      OP_REQUIRES(context, param.attr_kernel_Q != nullptr, errors::InvalidArgument("attr_kernel_Q is null"));
      OP_REQUIRES(context, param.attr_kernel_K != nullptr, errors::InvalidArgument("attr_kernel_K is null"));
      OP_REQUIRES(context, param.attr_kernel_V != nullptr, errors::InvalidArgument("attr_kernel_V is null"));
      OP_REQUIRES(context, param.attr_bias_Q != nullptr, errors::InvalidArgument("attr_bias_Q is null"));
      OP_REQUIRES(context, param.attr_bias_K != nullptr, errors::InvalidArgument("attr_bias_K is null"));
      OP_REQUIRES(context, param.attr_bias_V != nullptr, errors::InvalidArgument("attr_bias_V is null"));
      OP_REQUIRES(context, param.attr_mask != nullptr, errors::InvalidArgument("attr_mask is null"));
      OP_REQUIRES(context, param.attr_output_kernel != nullptr, errors::InvalidArgument("attr_output_kernel is null"));
      OP_REQUIRES(context, param.attr_output_bias != nullptr, errors::InvalidArgument("attr_output_bias is null"));
      OP_REQUIRES(context, param.attr_output_layernorm_beta != nullptr, errors::InvalidArgument("attr_output_layernorm_beta is null"));
      OP_REQUIRES(context, param.attr_output_layernorm_gamma != nullptr, errors::InvalidArgument("attr_output_layernorm_gamma is null"));
      OP_REQUIRES(context, param.inter_kernel != nullptr, errors::InvalidArgument("inter_kernel is null"));
      OP_REQUIRES(context, param.inter_bias != nullptr, errors::InvalidArgument("inter_bias is null"));
      OP_REQUIRES(context, param.output_kernel != nullptr, errors::InvalidArgument("output_kernel is null"));
      OP_REQUIRES(context, param.output_bias != nullptr, errors::InvalidArgument("output_bias is null"));
      OP_REQUIRES(context, param.output_layernorm_beta != nullptr, errors::InvalidArgument("output_layernorm_beta is null"));
      OP_REQUIRES(context, param.output_layernorm_gamma != nullptr, errors::InvalidArgument("output_layernorm_gamma is null"));

      Tensor *output = nullptr;

      OP_REQUIRES_OK(
          context,
          context->allocate_output(0, {batch_size_ * from_seq_len_, head_num_ * size_per_head_}, &output));

      param.transformer_out = reinterpret_cast<DataType_ *>(output->flat<T>().data());

      OP_REQUIRES_OK(
          context,
          functor::BertTransformerOpFunctor<Device, T>::Compute(
            context,
            param,
            encoder_transformer_));
    }
    private:
    int batch_size_, from_seq_len_, to_seq_len_, head_num_, size_per_head_;
    typedef TransformerTFTraits<T> traits_;
    typedef typename traits_::DataType DataType_;
    cublasHandle_t cublas_handle_;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                                                                       \
    REGISTER_KERNEL_BUILDER(                                                                  \
        Name("BertTransformer").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
        BertTransformerOp<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU

#endif
} //namespace
} //namespace tensorflow

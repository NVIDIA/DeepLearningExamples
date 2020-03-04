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
#include "fastertransformer/faster_transformer.h"
#include "fastertransformer/tf_op/bert_transformer_op.h"
#include "fastertransformer/tf_op/common_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/logging.h"
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
    .Attr("from_seq_len: int >= 1")
    .Attr("to_seq_len: int >= 1")
    .Attr("head_num: int >= 1")
    .Attr("size_per_head: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      int from_seq_len, to_seq_len, head_num, size_per_head;
      c->GetAttr("from_seq_len", &from_seq_len);
      c->GetAttr("to_seq_len", &to_seq_len);
      c->GetAttr("head_num", &head_num);
      c->GetAttr("size_per_head", &size_per_head);
      int rank = c->Rank(c->input(0));
      if (rank != 2 && rank != 3)
      {
        return errors::InvalidArgument("[@BertTransformer::ShapeInference] "
                                       "invalid rank (from_tensor@input[0]): ",
                                       rank,
                                       ", should be 2 or 3");
      }
      // calculate batch size
      shape_inference::DimensionOrConstant from_len_dim((int64)from_seq_len);
      shape_inference::DimensionHandle output_dim1;
      shape_inference::DimensionHandle batch_dim;
      shape_inference::ShapeHandle input0;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), rank, &input0));
      if (rank == 3)
      { // embedding_output, [batch_size, seq_len, hidden_size]
        batch_dim = c->Dim(c->input(0), 0);
      }
      else
      { // should be 2, transformer's output, [batch_size*seq_len, hidden_size]
        shape_inference::DimensionHandle tmp;
        TF_RETURN_IF_ERROR(c->Divide(c->Dim(c->input(0), 0), from_len_dim,
                                     true, &tmp));
        batch_dim = tmp;
      }

      TF_RETURN_IF_ERROR(c->Multiply(batch_dim, from_len_dim, &output_dim1));

      VLOG(2) << "[@BertTransformer::ShapeInference] batch_size: "
              << c->Value(shape_inference::DimensionOrConstant(batch_dim))
              << ", output shape: [" << c->Value(shape_inference::DimensionOrConstant(output_dim1))
              << "," << head_num * size_per_head << "]\n";

      c->set_output(0, c->MakeShape({output_dim1, head_num * size_per_head}));
      return Status::OK();
    });
template <typename Device, typename T>
class BertTransformerOp : public CommonOp<T>
{
public:
  explicit BertTransformerOp(OpKernelConstruction *context) : CommonOp<T>(context)
  {
    OP_REQUIRES_OK(context, context->GetAttr("from_seq_len", &from_seq_len_));
    OP_REQUIRES_OK(context, context->GetAttr("to_seq_len", &to_seq_len_));
    OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
    OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));
    OP_REQUIRES(context, (from_seq_len_ == to_seq_len_),
                errors::InvalidArgument("Only support from_seq_len == to_seq_len"));
  }

  void Compute(OpKernelContext *context) override
  {
    int rank = (int)context->input(0).dims();
    if (rank != 2 && rank != 3)
    {
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("[@BertTransformer::Compute] "
                                          "invalid rank (from_tensor@input[0]): ",
                                          rank,
                                          ", should be 2 or 3"));
    }
    else if (rank == 3)
    { // [batch_size, from_seq_len, hidden_size]
      batch_size_ = (int)context->input(0).dim_size(0);
    }
    else
    { // [batch_size * from_seq_len, hidden_size]
      batch_size_ = (int)context->input(0).dim_size(0) / from_seq_len_;
    }

    VLOG(2) << "[@BertTransformer::Compute] getting batch size: "
            << batch_size_ << "\n";

    typedef BertEncoderTransformerTraits<traits_::OpType, cuda::OpenMultiHeadAttention> EncoderTraits_;
    BertEncoderTransformer<EncoderTraits_> *encoder_transformer_;
    fastertransformer::Allocator<AllocatorType::TF> allocator_(context);
    try
    {
      encoder_transformer_ = new BertEncoderTransformer<EncoderTraits_>(allocator_,
                                                                        batch_size_,
                                                                        from_seq_len_, 
                                                                        to_seq_len_, 
                                                                        head_num_, 
                                                                        size_per_head_);
    }
    catch (std::runtime_error &error)
    {
      OP_REQUIRES(context, false, errors::Internal(error.what()));
    }

    OP_REQUIRES(context, context->num_inputs() == 19, errors::InvalidArgument("Less input arguments"));

    EncoderInitParam<DataType_> param; //init param here
    param.cublas_handle = this->get_cublas_handler();
    this->get_tensor(context, 0, &param.from_tensor);
    this->get_tensor(context, 1, &param.to_tensor);
    this->get_tensor(context, 2, &param.self_attention.query_weight.kernel);
    this->get_tensor(context, 3, &param.self_attention.key_weight.kernel);
    this->get_tensor(context, 4, &param.self_attention.value_weight.kernel);
    this->get_tensor(context, 5, &param.self_attention.query_weight.bias);
    this->get_tensor(context, 6, &param.self_attention.key_weight.bias);
    this->get_tensor(context, 7, &param.self_attention.value_weight.bias);
    this->get_tensor(context, 8, &param.attr_mask);
    this->get_tensor(context, 9, &param.self_attention.attention_output_weight.kernel);
    this->get_tensor(context, 10, &param.self_attention.attention_output_weight.bias);
    this->get_tensor(context, 11, &param.self_layernorm.beta);
    this->get_tensor(context, 12, &param.self_layernorm.gamma);
    this->get_tensor(context, 13, &param.ffn.intermediate_weight.kernel);
    this->get_tensor(context, 14, &param.ffn.intermediate_weight.bias);
    this->get_tensor(context, 15, &param.ffn.output_weight.kernel);
    this->get_tensor(context, 16, &param.ffn.output_weight.bias);
    this->get_tensor(context, 17, &param.ffn_layernorm.beta);
    this->get_tensor(context, 18, &param.ffn_layernorm.gamma);

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
    delete encoder_transformer_;
  }

private:
  int batch_size_, from_seq_len_, to_seq_len_, head_num_, size_per_head_;
  typedef TFTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                                                  \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("BertTransformer").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      BertTransformerOp<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU

#endif
} //namespace
} //namespace tensorflow

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
#include "fastertransformer/tf_op/common_op.h"
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

REGISTER_OP("Decoder")
    .Input("from_tensor: T")
    .Input("memory_tensor: T")
    .Input("memory_sequence_length: int32")
    .Input("self_beta: T")
    .Input("self_gamma: T")
    .Input("self_q_kernel: T")
    .Input("self_q_bias: T")
    .Input("self_k_kernel: T")
    .Input("self_k_bias: T")
    .Input("self_v_kernel: T")
    .Input("self_v_bias: T")
    .Input("self_output_kernel: T")
    .Input("self_output_bias: T")
    .Input("cross_beta: T")
    .Input("cross_gamma: T")
    .Input("cross_q_kernel: T")
    .Input("cross_q_bias: T")
    .Input("cross_k_kernel: T")
    .Input("cross_k_bias: T")
    .Input("cross_v_kernel: T")
    .Input("cross_v_bias: T")
    .Input("cross_output_kernel: T")
    .Input("cross_output_bias: T")
    .Input("ffn_beta: T")
    .Input("ffn_gamma: T")
    .Input("ffn_kernel1: T")
    .Input("ffn_bias1: T")
    .Input("ffn_kernel2: T")
    .Input("ffn_bias2: T")
    .Input("old_self_cache: T")
    .Input("old_mem_cache: T")
    .Input("pseudo_input: T") // pseudo input, used to prevent the parallel execution for OP and TF
    .Output("decoder_output: T")
    .Output("new_self_cache: T")
    .Output("new_mem_cache: T")
    .Attr("T: {float, half}")
    .Attr("head_num: int >= 1")
    .Attr("size_per_head: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(29));
      c->set_output(2, c->input(30));
      return Status::OK();
    });
template <typename Device, typename T>
class DecoderOp : public CommonOp<T>
{
public:
  explicit DecoderOp(OpKernelConstruction *context) : CommonOp<T>(context)
  {
    OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
    OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));
  }

  void Compute(OpKernelContext *context) override
  {
    // input(1): memory_tensor: [batch_size, memory_max_seq_len, memory_hidden_dim]
    assert((int)(context->input(1).dims()) == 3);
    const int batch_size_ = (int)context->input(1).dim_size(0);
    const int max_seq_len_ = (int)context->input(1).dim_size(1);
    const int memory_hidden_dim_ = (int)context->input(1).dim_size(2);

    typedef DecoderTransformerTraits<traits_::OpType> DecoderTraits_;
    OpenDecoder<DecoderTraits_::OpType> *decoder_;
    fastertransformer::Allocator<AllocatorType::TF> allocator_(context);
    try
    {
      decoder_ = new OpenDecoder<DecoderTraits_::OpType>(allocator_, batch_size_,
                                                         max_seq_len_, head_num_, size_per_head_, memory_hidden_dim_);
    }
    catch (std::runtime_error &error)
    {
      OP_REQUIRES(context, false, errors::Internal(error.what()));
    }
    OP_REQUIRES(context, context->num_inputs() == 32, errors::InvalidArgument("[ERROR] More or Less input arguments"));

    Tensor *decoder_output_tensor = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(0, {batch_size_, 1, head_num_ * size_per_head_}, &decoder_output_tensor));
    DataType_ *decoder_output = reinterpret_cast<DataType_ *>(decoder_output_tensor->flat<T>().data());

    Tensor self_cache_tensor = context->mutable_input(29, true);
    context->set_output(1, self_cache_tensor);
    DataType_ *self_cache;
    self_cache = reinterpret_cast<DataType_ *>(self_cache_tensor.flat<T>().data());

    Tensor memory_cache_tensor = context->mutable_input(30, true);
    context->set_output(2, memory_cache_tensor);
    DataType_ *memory_cache;
    memory_cache = reinterpret_cast<DataType_ *>(memory_cache_tensor.flat<T>().data());

    const DataType_ *from_tensor = reinterpret_cast<const DataType_ *>(context->input(0).flat<T>().data());
    const DataType_ *memory_tensor = reinterpret_cast<const DataType_ *>(context->input(1).flat<T>().data());
    const int *memory_sequence_length = reinterpret_cast<const int *>(context->input(2).flat<int>().data());

    OP_REQUIRES(context, from_tensor != nullptr, errors::InvalidArgument("from_tensor"));
    OP_REQUIRES(context, memory_tensor != nullptr, errors::InvalidArgument("memory_tensor"));
    OP_REQUIRES(context, memory_sequence_length != nullptr, errors::InvalidArgument("memory_sequence_length"));

    DecoderInitParam<DataType_> params;
    params.cublas_handle = this->get_cublas_handler();

    this->get_tensor(context, 3, &params.self_layernorm.beta);
    this->get_tensor(context, 4, &params.self_layernorm.gamma);

    this->get_tensor(context, 5, &params.self_attention.query_weight.kernel);
    this->get_tensor(context, 6, &params.self_attention.query_weight.bias);
    this->get_tensor(context, 7, &params.self_attention.key_weight.kernel);
    this->get_tensor(context, 8, &params.self_attention.key_weight.bias);
    this->get_tensor(context, 9, &params.self_attention.value_weight.kernel);
    this->get_tensor(context, 10, &params.self_attention.value_weight.bias);
    this->get_tensor(context, 11, &params.self_attention.attention_output_weight.kernel);
    this->get_tensor(context, 12, &params.self_attention.attention_output_weight.bias);

    this->get_tensor(context, 13, &params.cross_layernorm.beta);
    this->get_tensor(context, 14, &params.cross_layernorm.gamma);
    this->get_tensor(context, 15, &params.cross_attention.query_weight.kernel);
    this->get_tensor(context, 16, &params.cross_attention.query_weight.bias);
    this->get_tensor(context, 17, &params.cross_attention.key_weight.kernel);
    this->get_tensor(context, 18, &params.cross_attention.key_weight.bias);
    this->get_tensor(context, 19, &params.cross_attention.value_weight.kernel);
    this->get_tensor(context, 20, &params.cross_attention.value_weight.bias);
    this->get_tensor(context, 21, &params.cross_attention.attention_output_weight.kernel);
    this->get_tensor(context, 22, &params.cross_attention.attention_output_weight.bias);

    this->get_tensor(context, 23, &params.ffn_layernorm.beta);
    this->get_tensor(context, 24, &params.ffn_layernorm.gamma);
    this->get_tensor(context, 25, &params.ffn.intermediate_weight.kernel);
    this->get_tensor(context, 26, &params.ffn.intermediate_weight.bias);
    this->get_tensor(context, 27, &params.ffn.output_weight.kernel);
    this->get_tensor(context, 28, &params.ffn.output_weight.bias);

    const int step = (int)context->input(29).dim_size(1);
    DataType_ *K_cache = self_cache;
    DataType_ *V_cache = self_cache + batch_size_ * step * head_num_ * size_per_head_;
    DataType_ *K_mem_cache = memory_cache;
    DataType_ *V_mem_cache = memory_cache + batch_size_ * max_seq_len_ * head_num_ * size_per_head_;
    const int decoder_buffer_size = decoder_->getWorkspaceSize() * sizeof(DataType_);
    DataType_ *decoder_buffer = (DataType_ *)allocator_.malloc(decoder_buffer_size);

    OP_REQUIRES_OK(
        context,
        functor::DecoderOpFunctor<Device, T>::DynamicDecode(
            context,
            params,
            decoder_, decoder_buffer,
            from_tensor, memory_tensor,
            K_cache, V_cache,
            K_mem_cache, V_mem_cache,
            memory_sequence_length,
            decoder_output, step));

    allocator_.free(decoder_buffer);
    delete decoder_;
  }

private:
  int head_num_, size_per_head_;
  typedef TFTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Decoder").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DecoderOp<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU

#endif
} //namespace
} //namespace tensorflow

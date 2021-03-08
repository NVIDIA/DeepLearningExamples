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
#include "fastertransformer/tf_op/decoding_op.h"
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

REGISTER_OP("Decoding")
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
    .Input("embedding_table: T")
    .Input("decoding_beta: T")
    .Input("decoding_gamma: T")
    .Input("embedding_kernel: T")
    .Input("embedding_bias: float32")
    .Output("output_ids: int32")
    .Output("parent_ids: int32")
    .Output("sequence_lengths: int32")
    .Attr("T: {float, half}")
    .Attr("batch_size: int >= 1")
    .Attr("beam_width: int >= 1")
    .Attr("max_seq_len: int >= 1")
    .Attr("head_num: int >= 1")
    .Attr("size_per_head: int >= 1")
    .Attr("num_layer: int >= 1")
    .Attr("memory_hidden_dim: int >= 1")
    .Attr("vocab_size: int >= 1")
    .Attr("start_id: int >= 0")
    .Attr("end_id: int >= 0")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      int batch_size, beam_width, max_seq_len;
      c->GetAttr("batch_size", &batch_size);
      c->GetAttr("beam_width", &beam_width);
      c->GetAttr("max_seq_len", &max_seq_len);
      c->set_output(0, c->MakeShape({batch_size * beam_width * max_seq_len}));
      c->set_output(1, c->MakeShape({batch_size * beam_width * max_seq_len}));
      c->set_output(2, c->MakeShape({batch_size * beam_width}));
      return Status::OK();
    });
template <typename Device, typename T>
class DecodingOp : public CommonOp<T>
{
public:
  explicit DecodingOp(OpKernelConstruction *context) : CommonOp<T>(context)
  {
    OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batch_size_));
    OP_REQUIRES_OK(context, context->GetAttr("beam_width", &beam_width_));
    OP_REQUIRES_OK(context, context->GetAttr("max_seq_len", &max_seq_len_));
    OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
    OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));
    OP_REQUIRES_OK(context, context->GetAttr("num_layer", &num_layer_));
    OP_REQUIRES_OK(context, context->GetAttr("vocab_size", &vocab_size_));
    OP_REQUIRES_OK(context, context->GetAttr("start_id", &start_id_));
    OP_REQUIRES_OK(context, context->GetAttr("end_id", &end_id_));
  }

  void Compute(OpKernelContext *context) override
  {
    // input(0): memory_tensor: [batch_size * beam_width, memory_max_seq_len, memory_hidden_dim]
    assert((int)(context->input(0).dims()) == 3);
    const int memory_max_seq_len = (int)context->input(0).dim_size(1);
    const int memory_hidden_dim_ = (int)context->input(0).dim_size(2);

    DecodingInitParam<DataType_> decoding_params;
    decoding_params.cublas_handle = this->get_cublas_handler();
    Tensor *output_ids = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(0, {max_seq_len_, batch_size_ * beam_width_}, &output_ids));

    Tensor *parent_ids = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(1, {max_seq_len_, batch_size_ * beam_width_}, &parent_ids));

    Tensor *sequence_length = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(2, {batch_size_ * beam_width_}, &sequence_length));

    decoding_params.output_ids = reinterpret_cast<int *>(output_ids->flat<int>().data());
    decoding_params.parent_ids = reinterpret_cast<int *>(parent_ids->flat<int>().data());
    decoding_params.sequence_length = reinterpret_cast<int *>(sequence_length->flat<int>().data());

    check_cuda_error(cudaMemset(decoding_params.output_ids, 0, sizeof(int) * max_seq_len_ * batch_size_ * beam_width_));
    check_cuda_error(cudaMemset(decoding_params.parent_ids, 0, sizeof(int) * max_seq_len_ * batch_size_ * beam_width_));
    check_cuda_error(cudaMemset(decoding_params.sequence_length, 0, sizeof(int) * batch_size_ * beam_width_));

    typedef DecoderTransformerTraits<traits_::OpType> DecodingTraits_;
    DecodingOpenNMT<DecodingTraits_::OpType> *decoding_opennmt_;
    fastertransformer::Allocator<AllocatorType::TF> allocator_(context);
    try
    {
      decoding_opennmt_ = new DecodingOpenNMT<DecodingTraits_::OpType>(
          allocator_, batch_size_, beam_width_,
          max_seq_len_, head_num_, size_per_head_,
          vocab_size_, num_layer_,
          memory_hidden_dim_, memory_max_seq_len,
          start_id_, end_id_);
    }
    catch (std::runtime_error &error)
    {
      OP_REQUIRES(context, false, errors::Internal(error.what()));
    }

    OP_REQUIRES(context, context->num_inputs() == 33, errors::InvalidArgument("[ERROR] Less or more input arguments"));

    this->get_tensor(context, 0, &decoding_params.memory_tensor);
    decoding_params.memory_sequence_length = reinterpret_cast<const int *>(context->input(1).flat<int>().data());
    OP_REQUIRES(context, decoding_params.memory_sequence_length != nullptr, errors::InvalidArgument("memory_sequence_length"));

    DecoderInitParam<DataType_> *params = new DecoderInitParam<DataType_>[num_layer_];
    const int hidden_unit = size_per_head_ * head_num_;
    for (int i = 0; i < num_layer_; i++)
    {
      params[i].cublas_handle = this->get_cublas_handler();
      this->get_tensor(context, 2, &params[i].self_layernorm.beta, i * hidden_unit);
      this->get_tensor(context, 3, &params[i].self_layernorm.gamma, i * hidden_unit);
      this->get_tensor(context, 4, &params[i].self_attention.query_weight.kernel, i * hidden_unit * hidden_unit);
      this->get_tensor(context, 5, &params[i].self_attention.query_weight.bias, i * hidden_unit);
      this->get_tensor(context, 6, &params[i].self_attention.key_weight.kernel, i * hidden_unit * hidden_unit);
      this->get_tensor(context, 7, &params[i].self_attention.key_weight.bias, i * hidden_unit);
      this->get_tensor(context, 8, &params[i].self_attention.value_weight.kernel, i * hidden_unit * hidden_unit);
      this->get_tensor(context, 9, &params[i].self_attention.value_weight.bias, i * hidden_unit);
      this->get_tensor(context, 10, &params[i].self_attention.attention_output_weight.kernel, i * hidden_unit * hidden_unit);
      this->get_tensor(context, 11, &params[i].self_attention.attention_output_weight.bias, i * hidden_unit);
      this->get_tensor(context, 12, &params[i].cross_layernorm.beta, i * hidden_unit);
      this->get_tensor(context, 13, &params[i].cross_layernorm.gamma, i * hidden_unit);
      this->get_tensor(context, 14, &params[i].cross_attention.query_weight.kernel, i * hidden_unit * hidden_unit);
      this->get_tensor(context, 15, &params[i].cross_attention.query_weight.bias, i * hidden_unit);
      this->get_tensor(context, 16, &params[i].cross_attention.key_weight.kernel, i * memory_hidden_dim_ * hidden_unit);
      this->get_tensor(context, 17, &params[i].cross_attention.key_weight.bias, i * hidden_unit);
      this->get_tensor(context, 18, &params[i].cross_attention.value_weight.kernel, i * memory_hidden_dim_ * hidden_unit);
      this->get_tensor(context, 19, &params[i].cross_attention.value_weight.bias, i * hidden_unit);
      this->get_tensor(context, 20, &params[i].cross_attention.attention_output_weight.kernel, i * hidden_unit * hidden_unit);
      this->get_tensor(context, 21, &params[i].cross_attention.attention_output_weight.bias, i * hidden_unit);
      this->get_tensor(context, 22, &params[i].ffn_layernorm.beta, i * hidden_unit);
      this->get_tensor(context, 23, &params[i].ffn_layernorm.gamma, i * hidden_unit);
      this->get_tensor(context, 24, &params[i].ffn.intermediate_weight.kernel, i * hidden_unit * hidden_unit * 4);
      this->get_tensor(context, 25, &params[i].ffn.intermediate_weight.bias, i * hidden_unit * 4);
      this->get_tensor(context, 26, &params[i].ffn.output_weight.kernel, i * hidden_unit * hidden_unit * 4);
      this->get_tensor(context, 27, &params[i].ffn.output_weight.bias, i * hidden_unit);
    }

    this->get_tensor(context, 28, &decoding_params.layernorm.beta);
    this->get_tensor(context, 29, &decoding_params.layernorm.gamma);
    this->get_tensor(context, 30, &decoding_params.embedding_table);
    this->get_tensor(context, 31, &decoding_params.embedding_kernel);

    decoding_params.embedding_bias = reinterpret_cast<const float *>(context->input(32).flat<float>().data());
    OP_REQUIRES(context, decoding_params.embedding_bias != nullptr, errors::InvalidArgument("memory_sequence_length"));

    OP_REQUIRES_OK(
        context,
        functor::DecodingOpFunctor<Device, T>::DynamicDecode(
            context,
            num_layer_,
            params,
            decoding_opennmt_,
            max_seq_len_,
            decoding_params));

    delete decoding_opennmt_;
    delete params;
  }

private:
  int batch_size_, beam_width_, max_seq_len_;
  int head_num_, size_per_head_, num_layer_;
  int memory_hidden_dim_, vocab_size_, start_id_, end_id_;
  typedef TFTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                                           \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("Decoding").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DecodingOp<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU

#endif
} //namespace
} //namespace tensorflow

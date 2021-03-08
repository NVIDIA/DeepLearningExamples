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

#include "fastertransformer/open_decoder.h"
#include "fastertransformer/decoding_sampling.h"
#include "fastertransformer/tf_op/common_op.h"

namespace tensorflow
{
namespace
{
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("DecodingSampling")
    .Input("memory_tensor: T") // 0
    .Input("memory_sequence_length: int32") // 1
    .Input("self_beta: T") // 2
    .Input("self_gamma: T") // 3
    .Input("self_q_kernel: T") // 4
    .Input("self_q_bias: T") // 5
    .Input("self_k_kernel: T") // 6
    .Input("self_k_bias: T") // 7
    .Input("self_v_kernel: T") // 8
    .Input("self_v_bias: T") // 9
    .Input("self_output_kernel: T") // 10
    .Input("self_output_bias: T") // 11
    .Input("cross_beta: T") // 12
    .Input("cross_gamma: T") // 13
    .Input("cross_q_kernel: T") // 14
    .Input("cross_q_bias: T") // 15
    .Input("cross_k_kernel: T") // 16
    .Input("cross_k_bias: T") // 17
    .Input("cross_v_kernel: T") // 18
    .Input("cross_v_bias: T") // 19
    .Input("cross_output_kernel: T") // 20
    .Input("cross_output_bias: T") // 21
    .Input("ffn_beta: T") // 22
    .Input("ffn_gamma: T") // 23
    .Input("ffn_kernel1: T") // 24
    .Input("ffn_bias1: T") // 25
    .Input("ffn_kernel2: T") // 26
    .Input("ffn_bias2: T") // 27
    .Input("decoding_beta: T") // 28
    .Input("decoding_gamma: T") // 29
    .Input("embedding_table: T") // 30
    .Input("embedding_kernel: T") // 31
    .Input("embedding_bias: T") // 32
    .Input("position_encoding_table: T") // 33
    .Output("output_ids: int32")
    .Output("sequence_lengths: int32")
    .Attr("T: {float, half}")
    .Attr("max_seq_len: int >= 1")
    .Attr("candidate_num: int >= 0")
    .Attr("probability_threshold: float = 0.0")
    .Attr("head_num: int >= 1")
    .Attr("size_per_head: int >= 1")
    .Attr("num_layer: int >= 1")
    .Attr("start_id: int >= 0")
    .Attr("end_id: int >= 0")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        int max_seq_len;
        c->GetAttr("max_seq_len", &max_seq_len);

        int rank = c->Rank(c->input(0));
        assert(rank == 3);

        shape_inference::DimensionOrConstant max_seq_dim((int64)max_seq_len);
        shape_inference::DimensionHandle output_dim;
        shape_inference::DimensionHandle batch_dim;

        batch_dim = c->Dim(c->input(0), 0);
        TF_RETURN_IF_ERROR(c->Multiply(batch_dim, max_seq_dim, &output_dim));

        c->set_output(0, c->MakeShape({output_dim}));
        c->set_output(1, c->MakeShape({batch_dim}));
        return Status::OK();

    });
template <typename Device, typename T>
class DecodingSamplingOp : public CommonOp<T>
{
public:
  explicit DecodingSamplingOp(OpKernelConstruction *context) : CommonOp<T>(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("max_seq_len", &max_seq_len_));
        OP_REQUIRES_OK(context, context->GetAttr("candidate_num", &candidate_num_));
        OP_REQUIRES_OK(context, context->GetAttr("probability_threshold", &probability_threshold_));
        OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
        OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));
        OP_REQUIRES_OK(context, context->GetAttr("num_layer", &num_layer_));
        OP_REQUIRES_OK(context, context->GetAttr("start_id", &start_id_));
        OP_REQUIRES_OK(context, context->GetAttr("end_id", &end_id_));
    }

  void Compute(OpKernelContext *context) override
    {
        // input(0): memory_tensor: [batch_size * memory_max_seq_len, memory_hidden_dim]
        assert((int)(context->input(0).dims()) == 3);
        batch_size_ = (int)context->input(0).dim_size(0);
        const int memory_max_seq_len = (int)context->input(0).dim_size(1);
        const int memory_hidden_dim = (int)context->input(0).dim_size(2);
        const int vocab_size = (int)context->input(30).dim_size(0);

        DecodingInitParam<DataType_> decoding_params;
        decoding_params.cublas_handle = this->get_cublas_handler();
        Tensor *output_ids = nullptr;
        OP_REQUIRES_OK(
            context,
            context->allocate_output(0, {max_seq_len_, batch_size_}, &output_ids));

        Tensor *sequence_length = nullptr;
        OP_REQUIRES_OK(
            context,
            context->allocate_output(1, {batch_size_}, &sequence_length));

        decoding_params.output_ids = reinterpret_cast<int *>(output_ids->flat<int>().data());
        decoding_params.sequence_length = reinterpret_cast<int *>(sequence_length->flat<int>().data());

        check_cuda_error(cudaMemset(decoding_params.output_ids, 0, sizeof(int) * max_seq_len_ * batch_size_));
        check_cuda_error(cudaMemset(decoding_params.sequence_length, 0, sizeof(int) * batch_size_));

        typedef DecoderTransformerTraits<traits_::OpType> DecodingTraits_;
        DecodingSampling<DecodingTraits_::OpType> *decoding_sampling_;
        const cudaStream_t &stream = context->eigen_device<Device>().stream();
        decoding_params.stream = stream;
        fastertransformer::Allocator<AllocatorType::TF> allocator_(context, stream);
        try
        {
            decoding_sampling_ = new DecodingSampling<DecodingTraits_::OpType>(
                allocator_, batch_size_, 
                max_seq_len_, head_num_, size_per_head_,
                vocab_size, num_layer_,
                memory_hidden_dim, memory_max_seq_len,
                start_id_, end_id_, 
                candidate_num_, probability_threshold_);
        }
        catch (std::runtime_error &error)
        {
        OP_REQUIRES(context, false, errors::Internal(error.what()));
        }

        OP_REQUIRES(context, context->num_inputs() == 34, errors::InvalidArgument("[ERROR] Less or more input arguments"));

        this->get_tensor(context, 0, &decoding_params.memory_tensor);
        decoding_params.memory_sequence_length = reinterpret_cast<const int *>(context->input(1).flat<int>().data());
        OP_REQUIRES(context, decoding_params.memory_sequence_length != nullptr, errors::InvalidArgument("memory_sequence_length"));

        DecoderInitParam<DataType_> *params = new DecoderInitParam<DataType_>[num_layer_];
        const int hidden_unit = size_per_head_ * head_num_;
        for (int i = 0; i < num_layer_; i++)
        {
            params[i].stream = stream;
            params[i].cublas_handle = this->get_cublas_handler();
            check_cuda_error(cublasSetStream(params[i].cublas_handle, params[i].stream));

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
            this->get_tensor(context, 16, &params[i].cross_attention.key_weight.kernel, i * memory_hidden_dim * hidden_unit);
            this->get_tensor(context, 17, &params[i].cross_attention.key_weight.bias, i * hidden_unit);
            this->get_tensor(context, 18, &params[i].cross_attention.value_weight.kernel, i * memory_hidden_dim * hidden_unit);
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
        this->get_tensor(context, 32, &decoding_params.embedding_bias_T);
        this->get_tensor(context, 33, &decoding_params.position_encoding_table);

        try
        {
            decoding_sampling_->forward(params, decoding_params);
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

        delete decoding_sampling_;
        delete params;
    }

private:
    int batch_size_, max_seq_len_, candidate_num_;
    int head_num_, size_per_head_, num_layer_;
    int start_id_, end_id_;
    float probability_threshold_;
    typedef TFTraits<T> traits_;
    typedef typename traits_::DataType DataType_;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                                           \
    REGISTER_KERNEL_BUILDER(                                        \
        Name("DecodingSampling").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
        DecodingSamplingOp<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU

#endif
} //namespace
} //namespace tensorflow

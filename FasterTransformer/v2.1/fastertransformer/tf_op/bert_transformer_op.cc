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
#include "fastertransformer/cuda/cuda_kernels.h"

namespace tensorflow
{
namespace
{
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("BertTransformer")
    .Input("from_tensor: T")
    .Input("to_tensor: T")
    .Input("attr_q_kernel: T")
    .Input("attr_q_bias: T")
    .Input("attr_k_kernel: T")
    .Input("attr_k_bias: T")
    .Input("attr_v_kernel: T")
    .Input("attr_v_bias: T")
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
    .Input("sequence_id_offset: int32") // shape: [valid_word_num]
    .Output("output: T")
    .Attr("T: {float, half}")
    .Attr("head_num: int >= 1")
    .Attr("size_per_head: int >= 1")
    .Attr("remove_padding: bool = true")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
template <typename Device, typename T>
class BertTransformerOp : public CommonOp<T>
{
public:
  explicit BertTransformerOp(OpKernelConstruction *context) : CommonOp<T>(context)
  {
    OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
    OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));
    OP_REQUIRES_OK(context, context->GetAttr("remove_padding", &remove_padding_));
  }

  void Compute(OpKernelContext *context) override
  {
    int rank = (int)context->input(0).dims();
    OP_REQUIRES(context, rank==3 || rank == 2,
                errors::InvalidArgument("Invalid rank. The rank of from tensor should be 3 or 2 \
                                        ([batch size, sequence length, hidden dimension] or [valid_word_num, hidden_dimension])"));
    OP_REQUIRES(context, context->input(8).dims() == 4,
                errors::InvalidArgument("Invalid rank. The rank of attention mask should be 4 " \
                                        "([batch_size, 1, seq_len, seq_len])"));

    batch_size_ = (int)context->input(8).dim_size(0);
    from_seq_len_ = (int)context->input(8).dim_size(2);
    to_seq_len_ = (int)context->input(8).dim_size(3);
    OP_REQUIRES(context, (from_seq_len_ == to_seq_len_),
                errors::InvalidArgument("Only support from_seq_len == to_seq_len"));

    typedef BertEncoderTransformerTraits<traits_::OpType, cuda::OpenMultiHeadAttention> EncoderTraits_;
    BertEncoderTransformer<EncoderTraits_> *encoder_transformer_;
    const cudaStream_t &stream = context->eigen_device<Device>().stream();
    fastertransformer::Allocator<AllocatorType::TF> allocator_(context, stream);
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
    OP_REQUIRES(context, context->num_inputs() == 20, errors::InvalidArgument("Less input arguments"));
    const int hidden_units = head_num_ * size_per_head_;
    EncoderInitParam<DataType_> param; //init param here
    param.stream = stream;
    param.cublas_handle = this->get_cublas_handler();
    check_cuda_error(cublasSetStream(param.cublas_handle, param.stream));
    this->get_tensor(context, 0, &param.from_tensor);
    this->get_tensor(context, 1, &param.to_tensor);
    this->get_tensor(context, 2, &param.self_attention.query_weight.kernel);
    this->get_tensor(context, 3, &param.self_attention.query_weight.bias);
    this->get_tensor(context, 4, &param.self_attention.key_weight.kernel);
    this->get_tensor(context, 5, &param.self_attention.key_weight.bias);
    this->get_tensor(context, 6, &param.self_attention.value_weight.kernel);
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

    int valid_word_num;
    if(remove_padding_ == true)
    {
      valid_word_num = (int)context->input(19).dim_size(0);
      param.sequence_id_offset = reinterpret_cast<const int *>(context->input(19).flat<int>().data());
      OP_REQUIRES(context, param.sequence_id_offset != nullptr, errors::InvalidArgument("sequence_id_offset is null"));
    }
    else
    {
      param.sequence_id_offset = nullptr;
      valid_word_num = batch_size_ * from_seq_len_;
    }
    param.valid_word_num = valid_word_num;
    
    Tensor *output = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(0, context->input(0).shape(), &output));
    param.transformer_out = reinterpret_cast<DataType_ *>(output->flat<T>().data());

    try
    {
      encoder_transformer_->initialize(param);
      encoder_transformer_->forward();
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

    delete encoder_transformer_;
  }

private:
  int batch_size_, from_seq_len_, to_seq_len_, head_num_, size_per_head_;
  bool remove_padding_;
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

/* ******************************** Build mask and Remove padding *************************** */

REGISTER_OP("BuildMaskRemovePadding")
    .Input("from_tensor: T") // shape: [batch_size, max_seq_len, hidden_dim]
    .Input("sequence_length: int32") // shape: [batch_size]
    .Output("output: T") // shpae: [valid_word_num, hidden_dim]
    .Output("sequence_id_offset: int32") // shape: [valid_word_num]
    .Attr("T: {float, half}")
    .SetShapeFn([](shape_inference::InferenceContext *c) {

      assert(c->Rank(c->input(0)) == 3);
      assert(c->Rank(c->input(1)) == 1);
      c->set_output(0, c->MakeShape({shape_inference::InferenceContext::kUnknownDim, c->Dim(c->input(0), 2)}));
      c->set_output(1, c->MakeShape({shape_inference::InferenceContext::kUnknownDim}));
      
      return Status::OK();
    });

template <typename Device, typename T>
class BuildMaskRemovePaddingOp : public CommonOp<T>
{
public:
  explicit BuildMaskRemovePaddingOp(OpKernelConstruction *context) : CommonOp<T>(context)
  {
  }

  void Compute(OpKernelContext *context) override
  {
    OP_REQUIRES(context, context->num_inputs() == 2, errors::InvalidArgument("Less input arguments"));
    OP_REQUIRES(context, context->input(0).dims()==3,
                errors::InvalidArgument("Invalid rank. The rank of from tensor should be 3 \
                                        ([batch size, sequence length, hidden dimension])"));
    OP_REQUIRES(context, context->input(1).dims()==1,
                errors::InvalidArgument("Invalid rank. The rank of sequence_id_offset should be 1 \
                                        ([batch_size])"));
    const int batch_size = (int)context->input(0).dim_size(0);
    const int max_seq_len = (int)context->input(0).dim_size(1);
    const int hidden_dim = (int)context->input(0).dim_size(2);

    const DataType_* input_ptr = reinterpret_cast<const DataType_ *>(context->input(0).flat<T>().data());
    const int* sequence_length = reinterpret_cast<const int *>(context->input(1).flat<int>().data());
    OP_REQUIRES(context, input_ptr != nullptr, errors::InvalidArgument("input_ptr is null"));
    OP_REQUIRES(context, sequence_length != nullptr, errors::InvalidArgument("sequence_length is null"));
    
    Tensor buf;
    long long int buf_size = (long long int)(ceil((batch_size * max_seq_len + 1) * sizeof(int) / 4.) * 4);
    tensorflow::Status status = context->allocate_temp(DT_UINT8, TensorShape{buf_size}, &buf);
    if (status != tensorflow::Status::OK())
      throw std::runtime_error("TF error: context->allocate_temp failed");

    int* tmp_sequence_id_offset = (int*)buf.flat<uint8>().data();
    int* d_valid_word_num = tmp_sequence_id_offset + batch_size * max_seq_len;
    
    const cudaStream_t &stream = context->eigen_device<Device>().stream();

    try
    {
      build_sequence_length_padding_offset_kernelLauncher(sequence_length, 
          batch_size, max_seq_len, d_valid_word_num, tmp_sequence_id_offset, stream);
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

    int* h_valid_word_num = new int[1];
    cudaMemcpyAsync(h_valid_word_num, d_valid_word_num, sizeof(int), cudaMemcpyDeviceToHost, stream);
    const int valid_word_num = h_valid_word_num[0];
    delete h_valid_word_num;

    Tensor *output = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(0, {valid_word_num, hidden_dim}, &output));
    DataType_* output_ptr = reinterpret_cast<DataType_ *>(output->flat<T>().data());

    Tensor *sequence_id_offset_buf = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(1, {valid_word_num}, &sequence_id_offset_buf));
    int* sequence_id_offset = reinterpret_cast<int *>(sequence_id_offset_buf->flat<int>().data());

    try
    {
      remove_sequence_length_padding_kernelLauncher(input_ptr, output_ptr, 
                                                    tmp_sequence_id_offset,
                                                    sequence_id_offset, 
                                                    valid_word_num, hidden_dim,
                                                    stream);
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
  typedef TFTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                                                  \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("BuildMaskRemovePadding").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      BuildMaskRemovePaddingOp<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU

#endif

/* ******************************** Rebuild padding *************************** */

REGISTER_OP("RebuildPadding")
    .Input("from_tensor: T") // shape: [valid_word_num, hidden_dim]
    .Input("sequence_id_offset: int32") // shape: [valid_word_num]
    .Input("atten_mask: T") // shape: [batch_size, 1, seq_len, seq_len]
    .Output("output: T") // shape: [batch_size, seq_len, hidden_dim] 
    .Attr("T: {float, half}")
    .SetShapeFn([](shape_inference::InferenceContext *c) {

      assert(c->Rank(c->input(0)) == 2);
      assert(c->Rank(c->input(1)) == 1);
      assert(c->Rank(c->input(2)) == 4);
      shape_inference::DimensionHandle batch_size = c->Dim(c->input(2), 0);
      shape_inference::DimensionHandle seq_len = c->Dim(c->input(2), 2);
      shape_inference::DimensionHandle hidden_dim = c->Dim(c->input(0), 1);
      c->set_output(0, c->MakeShape({batch_size, seq_len, hidden_dim}));
      return Status::OK();
    });

template <typename Device, typename T>
class RebuildPaddingOp : public CommonOp<T>
{
public:
  explicit RebuildPaddingOp(OpKernelConstruction *context) : CommonOp<T>(context)
  {
  }

  void Compute(OpKernelContext *context) override
  {
    OP_REQUIRES(context, context->num_inputs() == 3, errors::InvalidArgument("Less input arguments"));

    OP_REQUIRES(context, context->input(0).dims()==2,
                errors::InvalidArgument("Invalid rank. The rank of from tensor should be 2 " \
                                        "([valid_word_num, hidden_dimension])"));
    OP_REQUIRES(context, context->input(1).dims()==1,
                errors::InvalidArgument("Invalid rank. The rank of sequence_id_offset should be 1 " \
                                        "([vaoid_word_num])"));
    OP_REQUIRES(context, context->input(2).dims()==4,
                errors::InvalidArgument("Invalid rank. The rank of attention mask should be 4 " \
                                        "([batch_size, 1, seq_len, seq_len])"));

    const int batch_size = (int)context->input(2).dim_size(0);
    const int seq_len = (int)context->input(2).dim_size(2);
    const int hidden_dim = (int)context->input(0).dim_size(1);
    const int valid_word_num = (int)context->input(1).dim_size(0);

    const DataType_* input_ptr = reinterpret_cast<const DataType_ *>(context->input(0).flat<T>().data());
    const int* sequence_id_offset = reinterpret_cast<const int *>(context->input(1).flat<int>().data());
    OP_REQUIRES(context, input_ptr != nullptr, errors::InvalidArgument("input_ptr is null"));
    OP_REQUIRES(context, sequence_id_offset != nullptr, errors::InvalidArgument("sequence_id_offset is null"));

    Tensor *output = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(0, {batch_size, seq_len, hidden_dim}, &output));
    DataType_* output_ptr = reinterpret_cast<DataType_ *>(output->flat<T>().data());

    const cudaStream_t &stream = context->eigen_device<Device>().stream();
    cudaMemsetAsync(output_ptr, 0, sizeof(DataType_) * batch_size * seq_len * hidden_dim, stream);
    try
    {
      rebuild_sequence_length_padding_kernelLauncher(input_ptr, output_ptr, 
                                                    sequence_id_offset, 
                                                    valid_word_num, hidden_dim,
                                                    stream);
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
  typedef TFTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                                                  \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("RebuildPadding").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      RebuildPaddingOp<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#undef REGISTER_GPU

#endif

} //namespace
} //namespace tensorflow


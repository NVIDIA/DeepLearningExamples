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

#include <vector>
#include <iostream>
#include <cuda_fp16.h>
#include <nvToolsExt.h> 

#include "torch/extension.h"
#include "torch/csrc/cuda/Stream.h"

#include "fastertransformer/common.h"
#include "fastertransformer/faster_transformer.h"
#include "fastertransformer/cuda/cuda_kernels.h"
#include "fastertransformer/th_op/th_traits.h"
#include "fastertransformer/th_op/utils.h"


namespace torch_ext {
using namespace fastertransformer;
using torch::Tensor;

class IFTEncoder {
public:
  virtual ~IFTEncoder() {}
  virtual void forward(int batch_size,
                       int seq_len,
                       Tensor& input,
                       Tensor& attr_mask,
                       Tensor& output,
                       Tensor& trt_seqlen_offset,
                       Tensor& sequence_id_offset,
                       bool removing_padding) = 0;
};

template <typename T>
class FTEncoder : public IFTEncoder {
public:
  FTEncoder(int head_num, int head_size,
            int int8_mode, int layer_num, int layer_idx, bool allow_gemm_test, bool use_trt_kernel,
            const std::vector<Tensor>& w) : _head_num(head_num), _head_size(head_size), _use_trt_kernel(use_trt_kernel), _weights(w) {
    int hidden_dim = _head_num * _head_size;
    check_cuda_error(cublasCreate(&_cublasHandle));
    check_cuda_error(cublasLtCreate(&_cublasltHandle));
    encoder_param.self_attention.query_weight.kernel = get_ptr<T>(_weights[0]);
    encoder_param.self_attention.query_weight.bias = get_ptr<T>(_weights[1]);
    encoder_param.self_attention.key_weight.kernel = get_ptr<T>(_weights[2]);
    encoder_param.self_attention.key_weight.bias = get_ptr<T>(_weights[3]);
    encoder_param.self_attention.value_weight.kernel = get_ptr<T>(_weights[4]);
    encoder_param.self_attention.value_weight.bias = get_ptr<T>(_weights[5]);
    encoder_param.self_attention.attention_output_weight.kernel = get_ptr<T>(_weights[6]);
    encoder_param.self_attention.attention_output_weight.bias = get_ptr<T>(_weights[7]);
    encoder_param.self_layernorm.gamma = get_ptr<T>(_weights[8]);
    encoder_param.self_layernorm.beta = get_ptr<T>(_weights[9]);
    encoder_param.ffn.intermediate_weight.kernel = get_ptr<T>(_weights[10]);
    encoder_param.ffn.intermediate_weight.bias = get_ptr<T>(_weights[11]);
    encoder_param.ffn.output_weight.kernel = get_ptr<T>(_weights[12]);
    encoder_param.ffn.output_weight.bias = get_ptr<T>(_weights[13]);
    encoder_param.ffn_layernorm.gamma = get_ptr<T>(_weights[14]);
    encoder_param.ffn_layernorm.beta = get_ptr<T>(_weights[15]);
    if (int8_mode) {
      encoder_param.amaxList = get_ptr<float>(_weights[16]);
      encoder_param.layer_num = layer_num;
      encoder_param.layer_idx = layer_idx;
    } else {
      encoder_param.amaxList = nullptr;
    }
    encoder_param.cublas_handle = _cublasHandle;
    encoder_param.cublaslt_handle = _cublasltHandle;
    encoder = new BertEncoderTransformer<EncoderTraits_>(int8_mode, allow_gemm_test);
  }

  ~FTEncoder() override {
    cublasDestroy(_cublasHandle);
    cublasLtDestroy(_cublasltHandle);
    if (encoder != nullptr) {
      delete encoder;
    }
  }

  void forward(int batch_size,
               int seq_len,
               Tensor& input,
               Tensor& attr_mask,
               Tensor& output,
               Tensor& trt_seqlen_offset,
               Tensor& sequence_id_offset,
               bool removing_padding) override {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    encoder_param.stream = stream;
    int hidden_dim = _head_num * _head_size;

    if (removing_padding) {
      encoder_param.sequence_id_offset = get_ptr<int>(sequence_id_offset);
      encoder_param.valid_word_num = sequence_id_offset.size(0);
    } else {
      encoder_param.sequence_id_offset = nullptr;
      encoder_param.valid_word_num = batch_size * seq_len;
    }

    encoder_param.from_tensor = get_ptr<T>(input);
    encoder_param.to_tensor = get_ptr<T>(input);
    encoder_param.transformer_out = get_ptr<T>(output);
    encoder_param.attr_mask = get_ptr<T>(attr_mask);
    encoder_param.trt_seqlen_offset = get_ptr<int>(trt_seqlen_offset);
    encoder_param.trt_seqlen_size = (int)trt_seqlen_offset.size(0);
    check_cuda_error(cublasSetStream(encoder_param.cublas_handle, encoder_param.stream));
    fastertransformer::Allocator<AllocatorType::TH>* allocator = new fastertransformer::Allocator<AllocatorType::TH>();
    encoder->allocateBuffer(allocator, batch_size, seq_len, seq_len, _head_num, _head_size, _use_trt_kernel);
    encoder->initialize(encoder_param);
    encoder->forward();
    encoder->freeBuffer();
    delete allocator;
  }

private:
  typedef BertEncoderTransformerTraits<THTraits<T>::OpType, cuda::OpenMultiHeadAttention> EncoderTraits_;
  const int _head_num;
  const int _head_size;
  std::vector<Tensor> _weights;
  cublasHandle_t _cublasHandle;
  cublasLtHandle_t _cublasltHandle;
  EncoderInitParam<T> encoder_param;
  BertEncoderTransformer<EncoderTraits_>* encoder = nullptr;
  bool _use_trt_kernel;
};

template <typename T>
std::vector<Tensor> build_mask_remove_padding_impl(Tensor input, Tensor sequence_lengths) {
  const int batch_size = input.size(0);
  const int seq_len = input.size(1);
  const int hidden_dim = input.size(2);
  const T* input_ptr = get_ptr<T>(input);
  const int* sequence_lengths_ptr = get_ptr<int>(sequence_lengths);

  auto buf = torch::empty({batch_size * seq_len + 1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  int* tmp_sequence_id_offset = get_ptr<int>(buf);
  int* d_valid_word_num = tmp_sequence_id_offset + batch_size * seq_len;

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  build_sequence_length_padding_offset_kernelLauncher(sequence_lengths_ptr, batch_size, seq_len,
                                                      d_valid_word_num, tmp_sequence_id_offset, stream);

  int* h_valid_word_num = new int[1];
  cudaMemcpyAsync(h_valid_word_num, d_valid_word_num, sizeof(int), cudaMemcpyDeviceToHost, stream);
  const int valid_word_num = h_valid_word_num[0];
  delete h_valid_word_num;

  auto output =
      torch::empty({valid_word_num, hidden_dim}, torch::dtype(input.dtype()).device(torch::kCUDA).requires_grad(false));
  T* output_ptr = get_ptr<T>(output);
  auto sequence_id_offset =
      torch::empty({valid_word_num}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
  int* sequence_id_offset_ptr = get_ptr<int>(sequence_id_offset);

  remove_sequence_length_padding_kernelLauncher(input_ptr, output_ptr, 
                                                tmp_sequence_id_offset, sequence_id_offset_ptr, 
                                                valid_word_num, hidden_dim, stream);

  return std::vector<Tensor>{output, sequence_id_offset};
}

template <typename T>
Tensor rebuild_padding_impl(Tensor input, Tensor sequence_id_offset, Tensor attention_mask, int int8_mode) {
  const int batch_size = attention_mask.size(0);
  const int seq_len = attention_mask.size(2);
  const int hidden_dim = input.size(1);
  const int valid_word_num = input.size(0);
  const T* input_ptr = get_ptr<T>(input);
  const int* sequence_id_offset_ptr = get_ptr<int>(sequence_id_offset);

  auto output =
      torch::zeros({batch_size, seq_len, hidden_dim}, torch::dtype(input.dtype()).device(torch::kCUDA).requires_grad(false));
  T* output_ptr = get_ptr<T>(output);
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  if (int8_mode == 0) {
    rebuild_sequence_length_padding_kernelLauncher(input_ptr, output_ptr, sequence_id_offset_ptr, valid_word_num, hidden_dim, stream);
  } else if (int8_mode == 1) {
    rebuild_sequence_length_padding_COL32_kernelLauncher(input_ptr, output_ptr, sequence_id_offset_ptr,
                                                         valid_word_num, hidden_dim, batch_size * seq_len, stream);
  } else if (int8_mode == 2) {
    rebuild_sequence_length_padding_COL32_kernelLauncher((const int8_t*)input_ptr, (int8_t*)output_ptr, sequence_id_offset_ptr,
                                                         valid_word_num, hidden_dim, batch_size * seq_len, stream);
  }
  return output;
}

class FasterTransformerEncoder {
public:
  FasterTransformerEncoder(
    Tensor q_kernel,
    Tensor q_bias,
    Tensor k_kernel,
    Tensor k_bias,
    Tensor v_kernel,
    Tensor v_bias,
    Tensor attr_output_kernel,
    Tensor attr_output_bias,
    Tensor attr_output_layernorm_gamma,
    Tensor attr_output_layernorm_beta,
    Tensor inter_kernel,
    Tensor inter_bias,
    Tensor output_kernel,
    Tensor output_bias,
    Tensor output_layernorm_gamma,
    Tensor output_layernorm_beta,
    Tensor amax_list,
    int head_num,
    int head_size,
    bool remove_padding,
    int int8_mode,
    int layer_num,
    int layer_idx,
    bool allow_gemm_test,
    bool use_trt_kernel);

  ~FasterTransformerEncoder();
  
  Tensor forward(Tensor input, Tensor attr_mask, Tensor trt_seqlen_offset, Tensor sequence_id_offset);

private:
  const at::ScalarType _st;
  bool _remove_padding;
  IFTEncoder* ftencoder;
};

std::vector<Tensor> build_mask_remove_padding(Tensor input, Tensor sequence_lengths);

Tensor rebuild_padding(Tensor input, Tensor sequence_id_offset, Tensor attention_mask, int int8_mode);

} // namespace torch_ext

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

#include <torch/script.h>
#include "torch/csrc/cuda/Stream.h"
#include "ATen/cuda/CUDAContext.h"

#include "fastertransformer/faster_transformer.h"
#include "fastertransformer/cuda/cuda_kernels.h"
#include "fastertransformer/th_op/th_traits.h"
#include "fastertransformer/th_op/utils.h"

namespace torch_ths {
using namespace fastertransformer;
using torch::Tensor;
using torch_ext::THTraits;
using torch_ext::get_ptr;

template <typename T>
void  ftencoder(int head_num, int head_size, std::vector<Tensor>& weights,
                int batch_size, int seq_len, bool removing_padding,
                Tensor& input, Tensor& attr_mask, Tensor& sequence_lengths, Tensor& output) {
  typedef BertEncoderTransformerTraits<THTraits<T>::OpType, cuda::OpenMultiHeadAttention> EncoderTraits_;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  cublasHandle_t cublasHandle = at::cuda::getCurrentCUDABlasHandle();
  EncoderInitParam<T> encoder_param;
  int hidden_dim = head_num * head_size;
  encoder_param.self_attention.query_weight.kernel = get_ptr<T>(weights[0]);
  encoder_param.self_attention.query_weight.bias = get_ptr<T>(weights[1]);
  encoder_param.self_attention.key_weight.kernel = get_ptr<T>(weights[2]);
  encoder_param.self_attention.key_weight.bias = get_ptr<T>(weights[3]);
  encoder_param.self_attention.value_weight.kernel = get_ptr<T>(weights[4]);
  encoder_param.self_attention.value_weight.bias = get_ptr<T>(weights[5]);
  encoder_param.self_attention.attention_output_weight.kernel = get_ptr<T>(weights[6]);
  encoder_param.self_attention.attention_output_weight.bias = get_ptr<T>(weights[7]);
  encoder_param.self_layernorm.gamma = get_ptr<T>(weights[8]);
  encoder_param.self_layernorm.beta = get_ptr<T>(weights[9]);
  encoder_param.ffn.intermediate_weight.kernel = get_ptr<T>(weights[10]);
  encoder_param.ffn.intermediate_weight.bias = get_ptr<T>(weights[11]);
  encoder_param.ffn.output_weight.kernel = get_ptr<T>(weights[12]);
  encoder_param.ffn.output_weight.bias = get_ptr<T>(weights[13]);
  encoder_param.ffn_layernorm.gamma = get_ptr<T>(weights[14]);
  encoder_param.ffn_layernorm.beta = get_ptr<T>(weights[15]);
  encoder_param.attr_mask = get_ptr<T>(attr_mask);
  encoder_param.stream = stream;
  encoder_param.cublas_handle = cublasHandle;

  std::vector<Tensor> buf_vector;
  if (removing_padding) {
    const T* input_ptr = get_ptr<T>(input);
    const int* sequence_lengths_ptr = get_ptr<int>(sequence_lengths);
    auto buf = torch::empty({batch_size * seq_len + 1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    int* tmp_sequence_id_offset = get_ptr<int>(buf);
    int* d_valid_word_num = tmp_sequence_id_offset + batch_size * seq_len;
    build_sequence_length_padding_offset_kernelLauncher(sequence_lengths_ptr, batch_size, seq_len,
                                                        d_valid_word_num, tmp_sequence_id_offset, stream);
    int* h_valid_word_num = new int[1];
    cudaMemcpyAsync(h_valid_word_num, d_valid_word_num, sizeof(int), cudaMemcpyDeviceToHost, stream);
    const int valid_word_num = h_valid_word_num[0];
    delete h_valid_word_num;
    auto intermediate_input =
        torch::empty({valid_word_num, hidden_dim}, torch::dtype(input.dtype()).device(torch::kCUDA).requires_grad(false));
    buf_vector.push_back(intermediate_input);
    T* intermediate_input_ptr = get_ptr<T>(intermediate_input);
    auto sequence_id_offset =
        torch::empty({valid_word_num}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    buf_vector.push_back(sequence_id_offset);
    int* sequence_id_offset_ptr = get_ptr<int>(sequence_id_offset);
    remove_sequence_length_padding_kernelLauncher(input_ptr, intermediate_input_ptr, 
                                                  tmp_sequence_id_offset, sequence_id_offset_ptr, 
                                                  valid_word_num, hidden_dim, stream);
    auto intermediate_output = torch::empty_like(intermediate_input);
    buf_vector.push_back(intermediate_output);
    encoder_param.from_tensor = intermediate_input_ptr;
    encoder_param.to_tensor = intermediate_input_ptr;
    encoder_param.sequence_id_offset = sequence_id_offset_ptr;
    encoder_param.valid_word_num = valid_word_num;
    encoder_param.transformer_out = get_ptr<T>(intermediate_output);
  } else {
    encoder_param.from_tensor = get_ptr<T>(input);
    encoder_param.to_tensor = get_ptr<T>(input);
    encoder_param.sequence_id_offset = nullptr;
    encoder_param.valid_word_num = batch_size * seq_len;
    encoder_param.transformer_out = get_ptr<T>(output);
  }

  fastertransformer::Allocator<AllocatorType::TH> allocator;
  BertEncoderTransformer<EncoderTraits_>* encoder = 
      new BertEncoderTransformer<EncoderTraits_>(allocator, batch_size, seq_len, seq_len, head_num, head_size);
  encoder->initialize(encoder_param);
  encoder->forward();
  delete encoder;

  if (removing_padding) {
    rebuild_sequence_length_padding_kernelLauncher(encoder_param.transformer_out, get_ptr<T>(output), 
                                                  encoder_param.sequence_id_offset, encoder_param.valid_word_num,
                                                  hidden_dim, stream);
  }
}

Tensor  fastertransformerthsencoder(
  int64_t head_num,
  int64_t head_size,
  bool remove_padding,
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
  Tensor input,
  Tensor attr_mask,
  Tensor sequence_lengths);

} // namespace torch_ths
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

#include "fastertransformer/th_op/encoder_ths_op_f.h"

namespace torch_ths {
using torch::Tensor;

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
  Tensor sequence_lengths)
{
  const at::ScalarType _st = q_kernel.scalar_type();
  CHECK_INPUT(q_kernel, _st);  // hidden_dim, hidden_dim
  CHECK_INPUT(q_bias, _st);  // hidden_dim
  CHECK_INPUT(k_kernel, _st);  // hidden_dim, hidden_dim
  CHECK_INPUT(k_bias, _st);  // hidden_dim
  CHECK_INPUT(v_kernel, _st);  // hidden_dim, hidden_dim
  CHECK_INPUT(v_bias, _st);  // hidden_dim
  CHECK_INPUT(attr_output_kernel, _st);  // hidden_dim, hidden_dim
  CHECK_INPUT(attr_output_bias, _st);  // hidden_dim
  CHECK_INPUT(attr_output_layernorm_gamma, _st);  // hidden_dim
  CHECK_INPUT(attr_output_layernorm_beta, _st);  // hidden_dim
  CHECK_INPUT(inter_kernel, _st);  // 4 * hidden_dim, hidden_dim
  CHECK_INPUT(inter_bias, _st);  // 4 * hidden_dim
  CHECK_INPUT(output_kernel, _st);  // hidden_dim, 4 * hidden_dim
  CHECK_INPUT(output_bias, _st);  // hidden_dim
  CHECK_INPUT(output_layernorm_gamma, _st);  // hidden_dim
  CHECK_INPUT(output_layernorm_beta, _st);  // hidden_dim
  CHECK_INPUT(input, _st);
  CHECK_INPUT(attr_mask, _st);
  auto input_size = input.sizes();
  int batch_size = input_size[0];
  int seq_len = input_size[1];
  if (remove_padding) {
    CHECK_CUDA(sequence_lengths); CHECK_CONTIGUOUS(sequence_lengths);
    TORCH_CHECK(sequence_lengths.dtype()==torch::kInt32, "sequence_length dtype should be int32");
    TORCH_CHECK(sequence_lengths.numel()!=0, "sequence_length should not be empty tensor");
    TORCH_CHECK(sequence_lengths.size(0)==batch_size, "wrong sequence_length shape");
  }
  std::vector<Tensor> weights{q_kernel, q_bias, k_kernel, k_bias, v_kernel, v_bias,
                              attr_output_kernel, attr_output_bias, attr_output_layernorm_gamma, attr_output_layernorm_beta,
                              inter_kernel, inter_bias, output_kernel, output_bias, output_layernorm_gamma, output_layernorm_beta};
  auto output = torch::empty_like(input);
  switch (_st) {
    case at::ScalarType::Float:
      ftencoder<float>(head_num, head_size, weights, batch_size, seq_len, remove_padding, input, attr_mask, sequence_lengths, output);
      break;
    case at::ScalarType::Half:
      ftencoder<half>(head_num, head_size, weights, batch_size, seq_len, remove_padding, input, attr_mask, sequence_lengths, output);
      break;
    default:
      throw std::runtime_error("Wrong Tensor type.");
  }
  return output;
}
} // namespace torch_ths

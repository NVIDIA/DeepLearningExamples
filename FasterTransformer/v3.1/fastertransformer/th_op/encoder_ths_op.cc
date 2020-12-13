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

#include "fastertransformer/th_op/encoder_ths_op.h"

namespace torch_ths {
using torch::Tensor;

FasterTransformerEncoder::FasterTransformerEncoder(
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
  int64_t head_num,
  int64_t head_size,
  bool remove_padding,
  int64_t int8_mode,
  int64_t layer_num,
  int64_t layer_idx,
  bool allow_gemm_test,
  bool use_trt_kernel)
: _st(q_kernel.scalar_type()), _remove_padding(remove_padding),
  weights{q_kernel, q_bias, k_kernel, k_bias, v_kernel, v_bias,
          attr_output_kernel, attr_output_bias, attr_output_layernorm_gamma, attr_output_layernorm_beta,
          inter_kernel, inter_bias,
          output_kernel, output_bias, output_layernorm_gamma, output_layernorm_beta,
          amax_list}
{
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
  if (int8_mode != 0) {
    CHECK_CUDA(amax_list); CHECK_CONTIGUOUS(amax_list);
    TORCH_CHECK(amax_list.dtype()==torch::kFloat32, "amax_list dtype should be float32");
    TORCH_CHECK(amax_list.numel()!=0, "amax_list should not be empty tensor");
  }
  switch (_st) {
    case at::ScalarType::Float:
      ftencoder = new torch_ext::FTEncoder<float>(head_num, head_size,
                                                  int8_mode, layer_num, layer_idx,
                                                  allow_gemm_test, use_trt_kernel, weights);
      break;
    case at::ScalarType::Half:
      ftencoder = new torch_ext::FTEncoder<half>(head_num, head_size,
                                                 int8_mode, layer_num, layer_idx,
                                                 allow_gemm_test, use_trt_kernel, weights);
      break;
    default:
      throw std::runtime_error("Wrong Tensor type.");
  }
  head_info = torch::empty({8}, torch::dtype(torch::kInt64));
  head_info[0] = head_num;
  head_info[1] = head_size;
  head_info[2] = (int64_t)remove_padding;
  head_info[3] = int8_mode;
  head_info[4] = layer_num;
  head_info[5] = layer_idx;
  head_info[6] = (int64_t)allow_gemm_test;
  head_info[7] = (int64_t)use_trt_kernel;
}

FasterTransformerEncoder::~FasterTransformerEncoder() {
  delete ftencoder;
}

Tensor FasterTransformerEncoder::forward(Tensor input, Tensor attr_mask, Tensor trt_seqlen_offset, Tensor sequence_id_offset) {
  CHECK_INPUT(input, _st);
  CHECK_INPUT(attr_mask, _st);
  TORCH_CHECK(attr_mask.dim()==4, "Invalid rank. The rank of attention mask should be 4 ([batch_size, 1, seq_len, seq_len])");
  TORCH_CHECK(attr_mask.size(2)==attr_mask.size(3), "Wrong attr_mask size");
  CHECK_CUDA(trt_seqlen_offset); CHECK_CONTIGUOUS(trt_seqlen_offset);
  TORCH_CHECK(trt_seqlen_offset.dtype()==torch::kInt32, "trt_seqlen_offset dtype should be int32");
  int batch_size = attr_mask.size(0);
  int seq_len = attr_mask.size(2);
  if (_remove_padding) {
    CHECK_CUDA(sequence_id_offset); CHECK_CONTIGUOUS(sequence_id_offset);
    TORCH_CHECK(sequence_id_offset.dtype()==torch::kInt32, "v dtype should be int32");
    TORCH_CHECK(sequence_id_offset.numel()!=0, "sequence_id_offset should not be empty tensor");
  }
  auto output = torch::empty_like(input);
  ftencoder->forward(batch_size, seq_len, input, attr_mask, output, trt_seqlen_offset, sequence_id_offset, _remove_padding);
  return output;
}

std::vector<Tensor> FasterTransformerEncoder::get_pickle_info() const {
  std::vector<Tensor> tmp(weights);
  tmp.push_back(head_info);
  return tmp;
}

std::vector<Tensor> build_mask_remove_padding(Tensor input, Tensor sequence_lengths) {
  const at::ScalarType _st = input.scalar_type();
  CHECK_INPUT(input, _st);
  CHECK_CUDA(sequence_lengths); CHECK_CONTIGUOUS(sequence_lengths);
  TORCH_CHECK(sequence_lengths.dtype()==torch::kInt32, "sequence_length dtype should be int32");
  switch (_st) {
    case at::ScalarType::Float:
      return torch_ext::build_mask_remove_padding_impl<float>(input, sequence_lengths);
    case at::ScalarType::Half:
      return torch_ext::build_mask_remove_padding_impl<half>(input, sequence_lengths);
    default:
      throw std::runtime_error("Wrong Tensor type.");
  }
}

Tensor rebuild_padding(Tensor input, Tensor sequence_id_offset, Tensor attention_mask, int64_t int8_mode) {
  TORCH_CHECK(int8_mode==0||int8_mode==1||int8_mode==2, "int8_mode can only be one of [0, 1, 2]");
  const at::ScalarType _st = input.scalar_type();
  CHECK_INPUT(input, _st);
  CHECK_INPUT(attention_mask, _st);
  CHECK_CUDA(sequence_id_offset); CHECK_CONTIGUOUS(sequence_id_offset);
  TORCH_CHECK(sequence_id_offset.dtype()==torch::kInt32, "sequence_id_offset dtype should be int32");
  switch (_st) {
    case at::ScalarType::Float:
      return torch_ext::rebuild_padding_impl<float>(input, sequence_id_offset, attention_mask, (int)int8_mode);
    case at::ScalarType::Half:
      return torch_ext::rebuild_padding_impl<half>(input, sequence_id_offset, attention_mask, (int)int8_mode);
    default:
      throw std::runtime_error("Wrong Tensor type.");
  }
}

} // namespace torch_ths

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

#include "fastertransformer/th_op/decoding_ths_op.h"

namespace torch_ths {
using torch::Tensor;

FasterTransformerDecoding::FasterTransformerDecoding(
  int64_t head_num,
  int64_t head_size,
  int64_t mem_hidden_dim,
  int64_t layer_num,
  int64_t vocab_size,
  int64_t start_id,
  int64_t end_id,
  double beam_search_diversity_rate,
  Tensor self_layernorm_gamma,
  Tensor self_layernorm_beta,
  Tensor self_kernel_q,
  Tensor self_kernel_k,
  Tensor self_kernel_v,
  Tensor self_bias_q,
  Tensor self_bias_k,
  Tensor self_bias_v,
  Tensor self_output_kernel,
  Tensor self_output_bias,
  Tensor cross_layernorm_gamma,
  Tensor cross_layernorm_beta,
  Tensor cross_kernel_q,
  Tensor cross_kernel_k,
  Tensor cross_kernel_v,
  Tensor cross_bias_q,
  Tensor cross_bias_k,
  Tensor cross_bias_v,
  Tensor cross_output_kernel,
  Tensor cross_output_bias,
  Tensor ffn_layernorm_gamma,
  Tensor ffn_layernorm_beta,
  Tensor inter_kernel,
  Tensor inter_bias,
  Tensor output_kernel,
  Tensor output_bias,
  Tensor decoding_gamma,
  Tensor decoding_beta,
  Tensor embedding_table,
  Tensor position_encoding_table,
  Tensor embedding_kernel,
  Tensor embedding_bias)
: _st(self_layernorm_gamma.scalar_type()),
  weights{self_layernorm_gamma, self_layernorm_beta,
          self_kernel_q, self_kernel_k, self_kernel_v, self_bias_q, self_bias_k, self_bias_v,
          self_output_kernel, self_output_bias,
          cross_layernorm_gamma, cross_layernorm_beta,
          cross_kernel_q, cross_kernel_k, cross_kernel_v, cross_bias_q, cross_bias_k, cross_bias_v,
          cross_output_kernel, cross_output_bias,
          ffn_layernorm_gamma, ffn_layernorm_beta, inter_kernel, inter_bias, output_kernel, output_bias,
          decoding_gamma, decoding_beta, embedding_table, position_encoding_table,
          embedding_kernel, embedding_bias}
{
  CHECK_INPUT(self_layernorm_gamma, _st);  // layer_num, hidden_dim
  CHECK_INPUT(self_layernorm_beta, _st);  // layer_num, hidden_dim
  CHECK_INPUT(self_kernel_q, _st);  // hidden_dim, hidden_dim
  CHECK_INPUT(self_kernel_k, _st);  // hidden_dim, hidden_dim
  CHECK_INPUT(self_kernel_v, _st);  // hidden_dim, hidden_dim
  CHECK_INPUT(self_bias_q, _st);  // layer_num, hidden_dim
  CHECK_INPUT(self_bias_k, _st);  // layer_num, hidden_dim
  CHECK_INPUT(self_bias_v, _st);  // layer_num, hidden_dim
  CHECK_INPUT(self_output_kernel, _st);  // layer_num, hidden_dim, hidden_dim
  CHECK_INPUT(self_output_bias, _st);  // layer_num, hidden_dim
  CHECK_INPUT(cross_layernorm_gamma, _st);  // layer_num, hidden_dim
  CHECK_INPUT(cross_layernorm_beta, _st);  // layer_num, hidden_dim
  CHECK_INPUT(cross_kernel_q, _st);  // layer_num, hidden_dim, hidden_dim
  CHECK_INPUT(cross_kernel_k, _st);  // layer_num, mem_hidden_dim, hidden_dim
  CHECK_INPUT(cross_kernel_v, _st);  // layer_num, mem_hidden_dim, hidden_dim
  CHECK_INPUT(cross_bias_q, _st);  // layer_num, hidden_dim
  CHECK_INPUT(cross_bias_k, _st);  // layer_num, hidden_dim
  CHECK_INPUT(cross_bias_v, _st);  // layer_num, hidden_dim
  CHECK_INPUT(cross_output_kernel, _st);  // layer_num, hidden_dim, hidden_dim
  CHECK_INPUT(cross_output_bias, _st);  // layer_num, hidden_dim
  CHECK_INPUT(ffn_layernorm_gamma, _st);  // layer_num, hidden_dim
  CHECK_INPUT(ffn_layernorm_beta, _st);  // layer_num, hidden_dim
  CHECK_INPUT(inter_kernel, _st);  // layer_num, hidden_dim, 4 * hidden_dim
  CHECK_INPUT(inter_bias, _st);  // layer_num, 4 * hidden_dim
  CHECK_INPUT(output_kernel, _st);  // layer_num, 4 * hidden_dim, hidden_dim
  CHECK_INPUT(output_bias, _st);  // layer_num, hidden_dim
  CHECK_INPUT(decoding_gamma, _st); // hidden_dim
  CHECK_INPUT(decoding_beta, _st); // hidden_dim
  CHECK_INPUT(embedding_table, _st); // vocab_size, hidden_dim
  CHECK_INPUT(position_encoding_table, _st); // max_step, hidden_dim
  CHECK_INPUT(embedding_kernel, _st); // hidden_dim, vocab_size
  CHECK_INPUT(embedding_bias, at::ScalarType::Float); // vocab_size
  switch (_st) {
    case at::ScalarType::Float:
      ftdecoding = new torch_ext::FTDecoding<float>(head_num, head_size, mem_hidden_dim, layer_num, vocab_size,
                                         start_id, end_id, (float)beam_search_diversity_rate, weights);
      break;
    case at::ScalarType::Half:
      ftdecoding = new torch_ext::FTDecoding<half>(head_num, head_size, mem_hidden_dim, layer_num, vocab_size,
                                        start_id, end_id, (float)beam_search_diversity_rate, weights);
      break;
    default:
      throw std::runtime_error("Wrong Tensor type.");
  }
  decoding_info = torch::empty({7}, torch::dtype(torch::kInt64));
  decoding_info[0] = head_num;
  decoding_info[1] = head_size;
  decoding_info[2] = mem_hidden_dim;
  decoding_info[3] = layer_num;
  decoding_info[4] = vocab_size;
  decoding_info[5] = start_id;
  decoding_info[6] = end_id;
  beam_search_diversity_rate_info = torch::empty({1}, torch::dtype(torch::kFloat64));
  beam_search_diversity_rate_info[0] = beam_search_diversity_rate;
}

FasterTransformerDecoding::~FasterTransformerDecoding() {
  delete ftdecoding;
}
  
std::vector<Tensor> FasterTransformerDecoding::forward(int64_t batch_size, int64_t beam_size, int64_t max_seq_len, Tensor memory, Tensor memory_seq_lens) {
  CHECK_INPUT(memory, _st);
  CHECK_CUDA(memory_seq_lens); CHECK_CONTIGUOUS(memory_seq_lens); TORCH_CHECK(memory_seq_lens.dtype()==torch::kInt32, "mem_seq_lens dtype should be int32");
  int mem_max_seq_len = memory.size(1);
  auto output_ids = torch::empty({batch_size * beam_size * max_seq_len}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
  auto parent_ids = torch::empty({batch_size * beam_size * max_seq_len}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
  auto out_seq_lens = torch::empty({batch_size * beam_size}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
  ftdecoding->forward(batch_size, beam_size, max_seq_len, mem_max_seq_len,
                      memory, memory_seq_lens, output_ids, parent_ids, out_seq_lens);
  return std::vector<Tensor>{output_ids, parent_ids, out_seq_lens};
}

std::vector<Tensor> FasterTransformerDecoding::get_pickle_info() const {
  std::vector<Tensor> tmp(weights);
  tmp.push_back(decoding_info);
  tmp.push_back(beam_search_diversity_rate_info);
  return tmp;
}

Tensor gather_tree(Tensor step_ids, Tensor parent_ids, Tensor max_sequence_lengths, int64_t end_token) {
  CHECK_CUDA(step_ids); CHECK_CONTIGUOUS(step_ids); TORCH_CHECK(step_ids.dtype()==torch::kInt32, "step_ids dtype should be int32");
  CHECK_CUDA(parent_ids); CHECK_CONTIGUOUS(parent_ids); TORCH_CHECK(parent_ids.dtype()==torch::kInt32, "parent_ids dtype should be int32");
  CHECK_CUDA(max_sequence_lengths); CHECK_CONTIGUOUS(max_sequence_lengths); TORCH_CHECK(max_sequence_lengths.dtype()==torch::kInt32, "max_sequence_lengths dtype should be int32");
  int max_step = step_ids.size(0);
  int batch_size = step_ids.size(1);
  int beam_width = step_ids.size(2);
  auto beams = torch::empty_like(step_ids);
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  torch_ext::gather_tree_kernel_launcher(max_step, batch_size, beam_width,
                                         torch_ext::get_ptr<int>(step_ids),
                                         torch_ext::get_ptr<int>(parent_ids),
                                         torch_ext::get_ptr<int>(max_sequence_lengths), 
                                         end_token,
                                         torch_ext::get_ptr<int>(beams),
                                         stream);
  return beams;
}

} // namespace torch_ths
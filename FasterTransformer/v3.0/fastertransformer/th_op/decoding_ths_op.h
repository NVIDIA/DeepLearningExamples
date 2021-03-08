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

#include <torch/script.h>
#include <torch/custom_class.h>

#include "fastertransformer/open_decoder.h"
#include "fastertransformer/decoding_beamsearch.h"
#include "fastertransformer/th_op/decoding_ext.h"

namespace torch_ths {
using namespace fastertransformer;
using torch::Tensor;

class FasterTransformerDecoding : public torch::jit::CustomClassHolder {
public:
  FasterTransformerDecoding(
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
    Tensor embedding_bias);

  ~FasterTransformerDecoding();
  
  std::vector<Tensor> forward(int64_t batch_size, int64_t beam_size, int64_t max_seq_len, Tensor memory, Tensor memory_seq_lens);

  std::vector<Tensor> get_pickle_info() const;

private:
  const at::ScalarType _st;
  torch_ext::IFTDecoding* ftdecoding;
  Tensor decoding_info;
  Tensor beam_search_diversity_rate_info;
  std::vector<Tensor> weights;
};

Tensor gather_tree(Tensor step_ids, Tensor parent_ids, Tensor max_sequence_lengths, int64_t end_token);

} // namespace torch_ths
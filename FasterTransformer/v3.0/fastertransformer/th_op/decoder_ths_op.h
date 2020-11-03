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
#include "fastertransformer/th_op/decoder_ext.h"

namespace torch_ths {
using namespace fastertransformer;
using torch::Tensor;

class FasterTransformerDecoder : public torch::jit::CustomClassHolder {
public:
  FasterTransformerDecoder(
    int64_t head_num,
    int64_t head_size,
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
    Tensor output_bias);

  ~FasterTransformerDecoder();
  
  Tensor forward(Tensor input, Tensor memory, Tensor memory_seq_lens, Tensor self_cache, Tensor mem_cache);

  std::vector<Tensor> get_pickle_info() const;

private:
  const at::ScalarType _st;
  torch_ext::IFTDecoder* ftdecoder;
  Tensor head_info;
  std::vector<Tensor> weights;
};

} // namespace torch_ths
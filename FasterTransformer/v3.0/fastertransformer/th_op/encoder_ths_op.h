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

#include "fastertransformer/faster_transformer.h"
#include "fastertransformer/th_op/encoder_ext.h"

namespace torch_ths {
using namespace fastertransformer;
using torch::Tensor;

class FasterTransformerEncoder : public torch::jit::CustomClassHolder {
public:
  FasterTransformerEncoder(
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
    Tensor output_layernorm_beta);

  ~FasterTransformerEncoder();
  
  Tensor forward(Tensor input, Tensor attr_mask, Tensor sequence_lengths);

  std::vector<Tensor> get_pickle_info() const;

private:
  const at::ScalarType _st;
  bool _remove_padding;
  torch_ext::IFTEncoder* ftencoder;
  Tensor head_info;
  std::vector<Tensor> weights;
};
} // namespace torch_ths
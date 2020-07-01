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

#include "fastertransformer/th_op/encoder_ths_op.h"
#include "fastertransformer/th_op/decoder_ths_op.h"
#include "fastertransformer/th_op/decoding_ths_op.h"

using torch::Tensor;

static auto fasterTransformerEncoderTHS = 
  torch::jit::class_<torch_ths::FasterTransformerEncoder>("FasterTransformerEncoder")
  .def(torch::jit::init<int64_t, int64_t, bool,
                        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                        Tensor, Tensor, Tensor, Tensor>())
  .def("forward", &torch_ths::FasterTransformerEncoder::forward)
  .def_pickle(
    [](const c10::intrusive_ptr<torch_ths::FasterTransformerEncoder>& self) -> std::vector<Tensor> {
      return self->get_pickle_info();
    },
    [](std::vector<Tensor> state) -> c10::intrusive_ptr<torch_ths::FasterTransformerEncoder> {
      int head_num = state[16][0].item().to<int>();
      int head_size = state[16][1].item().to<int>();
      bool remove_padding = (bool)(state[16][2].item().to<int>());
      return c10::make_intrusive<torch_ths::FasterTransformerEncoder>(head_num, head_size, remove_padding,
        state[0], state[1], state[2], state[3], state[4], state[5],
        state[6], state[7], state[8], state[9], state[10], state[11],
        state[12], state[13], state[14], state[15]);
    }
  );

static auto fasterTransformerDecoderTHS = 
  torch::jit::class_<torch_ths::FasterTransformerDecoder>("FasterTransformerDecoder")
  .def(torch::jit::init<int64_t, int64_t,
                        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>())
  .def("forward", &torch_ths::FasterTransformerDecoder::forward)
  .def_pickle(
    [](const c10::intrusive_ptr<torch_ths::FasterTransformerDecoder>& self) -> std::vector<Tensor> {
      return self->get_pickle_info();
    },
    [](std::vector<Tensor> state) -> c10::intrusive_ptr<torch_ths::FasterTransformerDecoder> {
      int head_num = state[26][0].item().to<int>();
      int head_size = state[26][1].item().to<int>();
      return c10::make_intrusive<torch_ths::FasterTransformerDecoder>(head_num, head_size,
        state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
        state[8], state[9], state[10], state[11], state[12], state[13], state[14], state[15],
        state[16], state[17], state[18], state[19], state[20], state[21], state[22], state[23],
        state[24], state[25]);
    }
  );

static auto fasterTransformerDecodingTHS = 
  torch::jit::class_<torch_ths::FasterTransformerDecoding>("FasterTransformerDecoding")
  .def(torch::jit::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, double,
                        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>())
  .def("forward", &torch_ths::FasterTransformerDecoding::forward)
  .def_pickle(
    [](const c10::intrusive_ptr<torch_ths::FasterTransformerDecoding>& self) -> std::vector<Tensor> {
      return self->get_pickle_info();
    },
    [](std::vector<Tensor> state) -> c10::intrusive_ptr<torch_ths::FasterTransformerDecoding> {
      int head_num = state[32][0].item().to<int>();
      int head_size = state[32][1].item().to<int>();
      int mem_hidden_dim = state[32][2].item().to<int>();
      int layer_num = state[32][3].item().to<int>();
      int vocab_size = state[32][4].item().to<int>();
      int start_id = state[32][5].item().to<int>();
      int end_id = state[32][6].item().to<int>();
      double beam_search_diversity_rate = state[31][0].item().to<double>();
      return c10::make_intrusive<torch_ths::FasterTransformerDecoding>(head_num, head_size,
        mem_hidden_dim, layer_num, vocab_size, start_id, end_id, beam_search_diversity_rate,
        state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
        state[8], state[9], state[10], state[11], state[12], state[13], state[14], state[15],
        state[16], state[17], state[18], state[19], state[20], state[21], state[22], state[23],
        state[24], state[25], state[26], state[27], state[28], state[29], state[30], state[31]);
    }
  );

static auto gather_tree =
  torch::RegisterOperators("fastertransformer::gather_tree", &torch_ths::gather_tree);

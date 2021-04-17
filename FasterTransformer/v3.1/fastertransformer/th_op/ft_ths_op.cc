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
#include "fastertransformer/th_op/weight_quantize_op.h"

using torch::Tensor;

static auto fasterTransformerEncoderTHS = 
#ifdef LEGACY_THS
  torch::jit::class_<torch_ths::FasterTransformerEncoder>("FasterTransformerEncoder")
#else
  torch::jit::class_<torch_ths::FasterTransformerEncoder>("FasterTransformer", "Encoder")
#endif
  .def(torch::jit::init<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                        Tensor, Tensor, Tensor, Tensor, Tensor,
                        int64_t, int64_t, bool, int64_t, int64_t, int64_t, bool, bool>())
  .def("forward", &torch_ths::FasterTransformerEncoder::forward)
  .def_pickle(
    [](const c10::intrusive_ptr<torch_ths::FasterTransformerEncoder>& self) -> std::vector<Tensor> {
      return self->get_pickle_info();
    },
    [](std::vector<Tensor> state) -> c10::intrusive_ptr<torch_ths::FasterTransformerEncoder> {
      int64_t head_num = state[17][0].item().to<int>();
      int64_t head_size = state[17][1].item().to<int>();
      bool remove_padding = (bool)(state[17][2].item().to<int>());
      int64_t int8_mode = state[17][3].item().to<int>();
      int64_t layer_num = state[17][4].item().to<int>();
      int64_t layer_idx = state[17][5].item().to<int>();
      bool allow_gemm_test = (bool)(state[17][6].item().to<int>());
      bool use_trt_kernel = (bool)(state[17][7].item().to<int>());
      return c10::make_intrusive<torch_ths::FasterTransformerEncoder>(
        state[0], state[1], state[2], state[3], state[4], state[5],
        state[6], state[7], state[8], state[9], state[10], state[11],
        state[12], state[13], state[14], state[15], state[16],
        head_num, head_size, remove_padding, int8_mode, layer_num, layer_idx, allow_gemm_test, use_trt_kernel);
    }
  );

static auto fasterTransformerDecoderTHS = 
#ifdef LEGACY_THS
  torch::jit::class_<torch_ths::FasterTransformerDecoder>("FasterTransformerDecoder")
#else
  torch::jit::class_<torch_ths::FasterTransformerDecoder>("FasterTransformer", "Decoder")
#endif
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
#ifdef LEGACY_THS
  torch::jit::class_<torch_ths::FasterTransformerDecoding>("FasterTransformerDecoding")
#else
  torch::jit::class_<torch_ths::FasterTransformerDecoding>("FasterTransformer", "Decoding")
#endif
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

static auto build_mask_remove_padding =
  torch::RegisterOperators("fastertransformer::build_mask_remove_padding", &torch_ths::build_mask_remove_padding);

static auto rebuild_padding =
  torch::RegisterOperators("fastertransformer::rebuild_padding", &torch_ths::rebuild_padding);

static auto gather_tree =
  torch::RegisterOperators("fastertransformer::gather_tree", &torch_ths::gather_tree);

static auto weight_quantize =
  torch::RegisterOperators("fastertransformer::weight_quantize", &torch_ext::weight_quantize);

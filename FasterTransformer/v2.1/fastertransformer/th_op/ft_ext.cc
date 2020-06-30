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

#include "torch/extension.h"

#include "fastertransformer/th_op/encoder_ext.h"
#include "fastertransformer/th_op/decoder_ext.h"
#include "fastertransformer/th_op/decoding_ext.h"

using torch::Tensor;
namespace py = pybind11;

PYBIND11_MODULE(th_fastertransformer, m) {
  py::class_<torch_ext::FasterTransformerEncoder>(m, "FasterTransformerEncoder")
    .def(py::init<int, int, bool,
                  Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                  Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                  Tensor, Tensor, Tensor, Tensor>())
    .def("forward", &torch_ext::FasterTransformerEncoder::forward);

  py::class_<torch_ext::FasterTransformerDecoder>(m, "FasterTransformerDecoder")
    .def(py::init<int, int,
                  Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                  Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                  Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>())
    .def("forward", &torch_ext::FasterTransformerDecoder::forward);

  py::class_<torch_ext::FasterTransformerDecoding>(m, "FasterTransformerDecoding")
    .def(py::init<int, int, int, int, int, int, int, float,
                  Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                  Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                  Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                  Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>())
    .def("forward", &torch_ext::FasterTransformerDecoding::forward);

  m.def("gather_tree", &torch_ext::gather_tree);
}

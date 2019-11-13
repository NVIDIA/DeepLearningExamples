/******************************************************************************
*
* Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
*

 ******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <torch/extension.h>
#include <ATen/ATen.h>


namespace py = pybind11;

// Box encoder
std::vector<at::Tensor> box_encoder(const int N_img,
                                    const at::Tensor& bbox_input,
                                    const at::Tensor& bbox_offsets,
                                    const at::Tensor& labels_input,
                                    const at::Tensor& dbox,
                                    const float criteria = 0.5);

std::vector<at::Tensor> random_horiz_flip(
                             at::Tensor& img,
                             at::Tensor& bboxes,
                             const at::Tensor& bbox_offsets,
                             const float p,
                             const bool nhwc);

// Fused color jitter application
// ctm [4,4], img [H, W, C]
py::array_t<float> apply_transform(int H, int W, int C, py::array_t<float> img, py::array_t<float> ctm) {
  auto img_buf = img.request();
  auto ctm_buf = ctm.request();

  // printf("H: %d, W: %d, C: %d\n", H, W, C);
  py::array_t<float> result{img_buf.size};
  auto res_buf = result.request();

  float *img_ptr = (float *)img_buf.ptr;
  float *ctm_ptr = (float *)ctm_buf.ptr;
  float *res_ptr = (float *)res_buf.ptr;

  for (int h = 0; h < H; ++h) {
    for (int w = 0; w < W; ++w) {
      float *ptr = &img_ptr[h * W * C + w * C];
      float *out_ptr = &res_ptr[h * W * C + w * C];
      // manually unroll over C
      out_ptr[0] = ctm_ptr[0] * ptr[0] + ctm_ptr[1] * ptr[1] + ctm_ptr[2] * ptr[2] + ctm_ptr[3];
      out_ptr[1] = ctm_ptr[4] * ptr[0] + ctm_ptr[5] * ptr[1] + ctm_ptr[6] * ptr[2] + ctm_ptr[7];
      out_ptr[2] = ctm_ptr[8] * ptr[0] + ctm_ptr[9] * ptr[1] + ctm_ptr[10] * ptr[2] + ctm_ptr[11];
    }
  }

  result.resize({H, W, C});

  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // batched box encoder
  m.def("box_encoder", &box_encoder, "box_encoder");
  m.def("random_horiz_flip", &random_horiz_flip, "random_horiz_flip");
  // Apply fused color jitter
  m.def("apply_transform", &apply_transform, "apply_transform");
}

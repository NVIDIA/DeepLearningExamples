/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include <ATen/ATen.h>
#include <THC/THC.h>

#include <vector>

namespace rpn {
/**
 * Generate boxes associated to topN pre-NMS scores
 */
std::vector<at::Tensor> GeneratePreNMSUprightBoxes(
    const int num_images,
    const int A,
    const int H,
    const int W,
    at::Tensor& sorted_indices, // topK sorted pre_nms_topn indices
    at::Tensor& sorted_scores,  // topK sorted pre_nms_topn scores [N, A, H, W]
    at::Tensor& bbox_deltas,    // [N, A*4, H, W] (full, unsorted / sliced)
    at::Tensor& anchors,        // input (full, unsorted, unsliced)
    at::Tensor& image_shapes,   // (h, w) of images
    const int pre_nms_nboxes,
    const int rpn_min_size,
    const float bbox_xform_clip_default,
    const bool correct_transform_coords,
    const bool is_channels_last);
} // namespace rpn

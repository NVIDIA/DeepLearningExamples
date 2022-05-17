/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include <torch/extension.h>


at::Tensor ROIAlign_forward_cuda(const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int sampling_ratio,
                                 const bool is_nhwc);

at::Tensor ROIAlign_backward_cuda(const at::Tensor& grad,
                                  const at::Tensor& rois,
                                  const float spatial_scale,
                                  const int pooled_height,
                                  const int pooled_width,
                                  const int batch_size,
                                  const int channels,
                                  const int height,
                                  const int width,
                                  const int sampling_ratio,
                                  const bool is_nhwc);


std::tuple<at::Tensor, at::Tensor> ROIPool_forward_cuda(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width);

at::Tensor ROIPool_backward_cuda(const at::Tensor& grad,
                                 const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const at::Tensor& argmax,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int batch_size,
                                 const int channels,
                                 const int height,
                                 const int width);

at::Tensor nms_cuda(const at::Tensor boxes, float nms_overlap_thresh);


at::Tensor compute_flow_cuda(const at::Tensor& boxes,
                             const int height,
                             const int width);
                            
at::Tensor generate_mask_targets_cuda(at::Tensor dense_vector, 
                                      const std::vector<std::vector<at::Tensor>> polygons, 
                                      const at::Tensor anchors, 
                                      const int mask_size);                             

at::Tensor box_iou_cuda(at::Tensor box1, at::Tensor box2);
                                      
std::vector<at::Tensor> box_encode_cuda(at::Tensor boxes, 
                                     at::Tensor anchors, 
                                     float wx, 
                                     float wy, 
                                     float ww, 
                                     float wh);                                       

at::Tensor match_proposals_cuda(at::Tensor match_quality_matrix,
                                bool include_low_quality_matches, 
                                float low_th, 
                                float high_th);                                      

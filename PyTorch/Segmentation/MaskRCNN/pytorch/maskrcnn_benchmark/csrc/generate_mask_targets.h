/**
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#pragma once
#include "cpu/vision.h"
#ifndef _generate_mask_targets_h_
#define _generate_mask_targets_h_ 

at::Tensor generate_mask_targets( at::Tensor dense_vector, const std::vector<std::vector<at::Tensor>> polygons, const at::Tensor anchors, const int mask_size){
  at::Tensor result = generate_mask_targets_cuda(dense_vector, polygons,anchors, mask_size);
  return result;
}

#endif


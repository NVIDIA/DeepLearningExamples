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

#include <vector>
#include <cuda_fp16.h>
#include "fastertransformer/th_op/utils.h"
#include "fastertransformer/faster_transformer.h"


namespace torch_ext
{
using torch::Tensor;

std::vector<Tensor> weight_quantize(Tensor weight, Tensor quant_max, Tensor quant_min, bool if_per_channel);
} //namespace torch_ext

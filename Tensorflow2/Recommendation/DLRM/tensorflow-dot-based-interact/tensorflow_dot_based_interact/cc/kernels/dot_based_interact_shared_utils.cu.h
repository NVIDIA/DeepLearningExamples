// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#ifndef KERNEL_DOT_BASED_INTERACT_SHARED_UTILS_H_
#define KERNEL_DOT_BASED_INTERACT_SHARED_UTILS_H_

#include <math.h>

#define CHK_CUDA(expression)                                                                                      \
{                                                                                                                 \
  cudaError_t status = (expression);                                                                              \
  if (status != cudaSuccess) {                                                                                    \
    std::cerr << "Error in file: " << __FILE__ << ", on line: " << __LINE__ << ": " << cudaGetErrorString(status) \
              << std::endl;                                                                                       \
    std::exit(EXIT_FAILURE);                                                                                      \
  }                                                                                                               \
}

template <uint x>
struct Log2 {
  static constexpr uint value = 1 + Log2<x / 2>::value;
};

template <>
struct Log2<1> {
  static constexpr uint value = 0;
};

#endif //KERNEL_DOT_BASED_INTERACT_SHARED_UTILS_H_

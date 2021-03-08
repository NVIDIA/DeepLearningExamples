# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required(VERSION 3.8)

set(encoder_kernel_files
  open_attention.cu
)

set(decoder_kernel_files
  open_decoder.cu
)

set(online_softmax_beamsearch_kernel_files
  online_softmax_beamsearch_kernels.cu
)

set(topk_kernel_files
  topk_kernels.cu
)

set(decoding_kernel_files
  decoding_kernels.cu
)

add_library(cuda_kernels STATIC cuda_kernels.cu)
target_link_libraries(cuda_kernels PUBLIC -lcublas -lcudart -lcurand)
set_property(TARGET cuda_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(encoder STATIC ${encoder_kernel_files})
target_link_libraries(encoder PUBLIC -lcublas -lcudart -lcurand cuda_kernels)
set_property(TARGET encoder PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(decoder STATIC ${decoder_kernel_files})
target_link_libraries(decoder PUBLIC -lcublas -lcudart -lcurand)
set_property(TARGET decoder PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(online_softmax_beamsearch STATIC ${online_softmax_beamsearch_kernel_files})
target_link_libraries(online_softmax_beamsearch PUBLIC -lcublas -lcudart -lcurand)
set_property(TARGET online_softmax_beamsearch PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(topk STATIC ${topk_kernel_files})
target_link_libraries(topk PUBLIC -lcublas -lcudart -lcurand)
set_property(TARGET topk PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(decoding STATIC ${decoding_kernel_files})
target_link_libraries(decoding PUBLIC -lcublas -lcudart -lcurand topk online_softmax_beamsearch cuda_kernels)
set_property(TARGET decoding PROPERTY POSITION_INDEPENDENT_CODE ON)

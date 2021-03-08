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

set(encoder_gemm_func_files
  encoder_gemm_func.cc
)

set(encoder_igemm_func_files
  encoder_igemm_func.cc
)

add_library(encoder_gemm_func STATIC ${encoder_gemm_func_files})
target_link_libraries(encoder_gemm_func PUBLIC -lcublas -lcudart)
set_property(TARGET encoder_gemm_func PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET encoder_gemm_func PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
add_library(encoder_igemm_func STATIC ${encoder_igemm_func_files})
target_link_libraries(encoder_igemm_func PUBLIC -lcublas -lcublasLt -lcudart)
set_property(TARGET encoder_igemm_func PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET encoder_gemm_func PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

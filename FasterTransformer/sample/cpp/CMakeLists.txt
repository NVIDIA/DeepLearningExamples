# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

set(transformer_fp32_files
  transformer_fp32.cc
)

set(transformer_fp16_files
  transformer_fp16.cc
)

add_executable(transformer_fp32 ${transformer_fp32_files})
target_link_libraries(transformer_fp32 PUBLIC -lcublas -lcudart fastertransformer ${CMAKE_THREAD_LIBS_INIT})

add_executable(transformer_fp16 ${transformer_fp16_files})
target_link_libraries(transformer_fp16 PUBLIC -lcublas -lcudart fastertransformer ${CMAKE_THREAD_LIBS_INIT})

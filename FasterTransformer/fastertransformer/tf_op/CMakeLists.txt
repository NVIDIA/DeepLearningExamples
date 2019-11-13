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

set(tf_bert_transformer_files
  bert_transformer_op.cc
  bert_transformer_op.cu.cc
  ../cuda/open_attention.cu
  ../cuda/cuda_kernels.cu
)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-DGOOGLE_CUDA=1)
add_definitions(-DNDEBUG)

add_library(tf_fastertransformer SHARED ${tf_bert_transformer_files})
set_target_properties(tf_fastertransformer PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(tf_fastertransformer PRIVATE -lcublas -lcudart -ltensorflow_framework ${CMAKE_THREAD_LIBS_INIT})

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

set(tf_bert_transformer_files
  bert_transformer_op.cc
  bert_transformer_op.cu.cc
  ../cuda/open_attention.cu
  ../cuda/cuda_kernels.cu
)

set(tf_decoder_files
  decoder_op.cc
  decoder_op.cu.cc
  ../cuda/open_decoder.cu
  ../cuda/cuda_kernels.cu
)

set(tf_decoding_files
  decoding_op.cc
  decoding_op.cu.cc
  ../cuda/open_decoder.cu
  ../cuda/cuda_kernels.cu
  ../cuda/open_attention.cu
  ../cuda/decoding_kernel_check.cpp
)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-DGOOGLE_CUDA=1)

add_library(tf_fastertransformer SHARED ${tf_bert_transformer_files})
target_link_libraries(tf_fastertransformer PRIVATE -lcublas -lcudart -ltensorflow_framework)
add_library(tf_decoder SHARED ${tf_decoder_files})
target_link_libraries(tf_decoder PRIVATE -lcublas -lcudart -ltensorflow_framework)
add_library(tf_decoding SHARED ${tf_decoding_files})
target_link_libraries(tf_decoding PRIVATE -lcublas -lcudart -ltensorflow_framework)

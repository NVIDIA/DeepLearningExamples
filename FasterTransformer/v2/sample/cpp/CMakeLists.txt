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

set(encoder_sample_files
  encoder_sample.cc
)

set(decoding_sample_files
  decoding_sample.cc
)

add_executable(encoder_sample ${encoder_sample_files})
target_link_libraries(encoder_sample PUBLIC -lcublas -lcudart fastertransformer)

add_executable(decoding_sample ${decoding_sample_files})
target_link_libraries(decoding_sample PUBLIC -lcublas -lcudart fastertransformer)
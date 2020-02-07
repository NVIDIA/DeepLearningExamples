# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

cmake_minimum_required (VERSION 3.5)

add_executable(kaldi_asr_parallel_client kaldi_asr_parallel_client.cc asr_client_imp.cc) 

target_link_libraries(
   kaldi_asr_parallel_client
   PRIVATE request
)

target_link_libraries(
  kaldi_asr_parallel_client
  PRIVATE protobuf::libprotobuf
)

target_include_directories(
  kaldi_asr_parallel_client 
  PRIVATE
  /opt/kaldi/src/
)

target_include_directories(
  kaldi_asr_parallel_client 
  PRIVATE
  /opt/kaldi/tools/openfst-1.6.7/include/
)

target_link_libraries(
  kaldi_asr_parallel_client
  PRIVATE /opt/kaldi/src/lib/libkaldi-feat.so
)

target_link_libraries(
  kaldi_asr_parallel_client
  PRIVATE /opt/kaldi/src/lib/libkaldi-util.so
)

target_link_libraries(
   kaldi_asr_parallel_client
   PRIVATE /opt/kaldi/src/lib/libkaldi-matrix.so
)

target_link_libraries(
  kaldi_asr_parallel_client
  PRIVATE /opt/kaldi/src/lib/libkaldi-base.so
)


install(
  TARGETS kaldi_asr_parallel_client
  RUNTIME DESTINATION bin
)

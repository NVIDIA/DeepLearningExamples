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
cmake_minimum_required(VERSION 3.13)

set(th_fastertransformer_ext_files
  ft_ext.cc
  encoder_ext.cc
  decoder_ext.cc
  decoding_ext.cc
  utils.cu
)

set(th_fastertransformer_ths_files
  ft_ths_op.cc
  encoder_ths_op.cc
  decoder_ths_op.cc
  decoding_ths_op.cc
  utils.cu
)

set(th_fastertransformer_ths_f_files
  ft_ths_op_f.cc
  encoder_ths_op_f.cc
)

add_definitions(-DTORCH_CUDA=1)

# hack for bugs in torch
if(TARGET torch_cpu)
  set_target_properties(torch_cpu PROPERTIES
                        INTERFACE_COMPILE_OPTIONS "")
endif()
if(TARGET torch_cuda)
  set_target_properties(torch_cuda PROPERTIES
                        INTERFACE_COMPILE_OPTIONS "")
  set(NEW_TORCH_CUDA_LINK_VAR)
  get_target_property(OLD_TORCH_CUDA_LINK_VAR torch_cuda INTERFACE_LINK_LIBRARIES)
  foreach (TMPVAR ${OLD_TORCH_CUDA_LINK_VAR})
    string(REPLACE "/usr/local/cuda" "${CUDA_TOOLKIT_ROOT_DIR}" TMPVAR ${TMPVAR})
    list(APPEND NEW_TORCH_CUDA_LINK_VAR ${TMPVAR})
  endforeach(TMPVAR)
  set_target_properties(torch_cuda PROPERTIES
                        INTERFACE_LINK_LIBRARIES "${NEW_TORCH_CUDA_LINK_VAR}")
endif()
if(TARGET torch)
  set(NEW_TORCH_LINK_VAR)
  get_target_property(OLD_TORCH_LINK_VAR torch INTERFACE_LINK_LIBRARIES)
  foreach (TMPVAR ${OLD_TORCH_LINK_VAR})
    string(REPLACE "/usr/local/cuda" "${CUDA_TOOLKIT_ROOT_DIR}" TMPVAR ${TMPVAR})
    list(APPEND NEW_TORCH_LINK_VAR ${TMPVAR})
  endforeach(TMPVAR)
  set_target_properties(torch PROPERTIES
                        INTERFACE_LINK_LIBRARIES "${NEW_TORCH_LINK_VAR}")
endif()


if(BUILD_THE)
  set(LIB_NAME_1 "th_fastertransformer")
  add_library(${LIB_NAME_1} SHARED ${th_fastertransformer_ext_files})
  set_target_properties(${LIB_NAME_1} PROPERTIES
                        PREFIX ""
                        SUFFIX ${PY_SUFFIX}
                        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
  target_link_libraries(${LIB_NAME_1} "${TORCH_LIBRARIES}" "${TORCH_LINK}" encoder decoder decoding)
endif()

if(BUILD_THS)
  set(LIB_NAME_2 "ths_fastertransformer")
  add_library(${LIB_NAME_2} SHARED ${th_fastertransformer_ths_files})
  target_link_libraries(${LIB_NAME_2} "${TORCH_LIBRARIES}" encoder decoder decoding)
endif()

if(BUILD_THSOP)
  set(LIB_NAME_3 "ths_fastertransformer_op")
  add_library(${LIB_NAME_3} SHARED ${th_fastertransformer_ths_f_files})
  target_link_libraries(${LIB_NAME_3} "${TORCH_LIBRARIES}" encoder)
endif()

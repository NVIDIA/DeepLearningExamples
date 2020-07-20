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
cmake_minimum_required(VERSION 3.8 FATAL_ERROR) # for PyTorch extensions, version should be greater than 3.13
project(FasterTransformer LANGUAGES CXX CUDA)

find_package(CUDA 10.0 REQUIRED)

option(BUILD_TRT "Build in TensorRT mode" OFF)
option(BUILD_TF "Build in TensorFlow mode" OFF)
option(BUILD_THE "Build in PyTorch eager mode" OFF)
option(BUILD_THS "Build in TorchScript class mode" OFF)
option(BUILD_THSOP "Build in TorchScript OP mode" OFF)

set(CXX_STD "11" CACHE STRING "C++ standard")

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

set(TF_PATH "" CACHE STRING "TensorFlow path")
#set(TF_PATH "/usr/local/lib/python3.5/dist-packages/tensorflow")

if(BUILD_TF AND NOT TF_PATH)
  message(FATAL_ERROR "TF_PATH must be set if BUILD_TF(=TensorFlow mode) is on.")
endif()

set(TRT_PATH "" CACHE STRING "TensorRT path")
#set(TRT_PATH "/myspace/TensorRT-5.1.5.0")

if(BUILD_TRT AND NOT TRT_PATH)
  message(FATAL_ERROR "TRT_PATH must be set if BUILD_TRT(=TensorRT mode) is on.")
endif()

list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)
find_package(CUDA REQUIRED)

# setting compiler flags
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")	
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")

if (SM STREQUAL 70 OR
    SM STREQUAL 75 OR
    SM STREQUAL 61 OR
    SM STREQUAL 60)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${SM},code=\\\"sm_${SM},compute_${SM}\\\" -rdc=true")
  if (SM STREQUAL 70 OR SM STREQUAL 75)
    set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
  endif()
if(BUILD_THE OR BUILD_THS OR BUILD_THSOP)
  string(SUBSTRING ${SM} 0 1 SM_MAJOR)
  string(SUBSTRING ${SM} 1 1 SM_MINOR)
  set(ENV{TORCH_CUDA_ARCH_LIST} "${SM_MAJOR}.${SM_MINOR}")
endif()
message("-- Assign GPU architecture (sm=${SM})")

else()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                      -gencode=arch=compute_60,code=\\\"sm_60,compute_60\\\" \
                      -gencode=arch=compute_61,code=\\\"sm_61,compute_61\\\" \
                      -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                      -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                      -rdc=true")
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
if(BUILD_THE OR BUILD_THS OR BUILD_THSOP)
  set(ENV{TORCH_CUDA_ARCH_LIST} "6.0;6.1;7.0;7.5")
endif()
message("-- Assign GPU architecture (sm=60,61,70,75)")
endif()

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall  --ptxas-options=-v --resource-usage")

set(CMAKE_CXX_STANDARD "${CXX_STD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++${CXX_STD}")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CUDA_PATH}/include
)

set(COMMON_LIB_DIRS
  ${CUDA_PATH}/lib64
)

if(BUILD_TF)
  list(APPEND COMMON_HEADER_DIRS ${TF_PATH}/include)
  list(APPEND COMMON_LIB_DIRS ${TF_PATH})
endif()

if(BUILD_TRT)
  list(APPEND COMMON_HEADER_DIRS ${TRT_PATH}/include)
  list(APPEND COMMON_LIB_DIRS ${TRT_PATH}/lib)
endif()

set(PYTHON_PATH "python" CACHE STRING "Python path")
if(BUILD_THE OR BUILD_THS OR BUILD_THSOP)
  execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import os; import torch;
print(os.path.dirname(torch.__file__),end='');"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE TORCH_DIR)
  if (NOT _PYTHON_SUCCESS MATCHES 0)
      message(FATAL_ERROR "Torch config Error.")
  endif()
  list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
  find_package(Torch REQUIRED)

  execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; from distutils import sysconfig;
print(sysconfig.get_python_inc());
print(sysconfig.get_config_var('SO'));"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE _PYTHON_VALUES)
  if (NOT _PYTHON_SUCCESS MATCHES 0)
      message(FATAL_ERROR "Python config Error.")
  endif()
  string(REGEX REPLACE ";" "\\\\;" _PYTHON_VALUES ${_PYTHON_VALUES})
  string(REGEX REPLACE "\n" ";" _PYTHON_VALUES ${_PYTHON_VALUES})
  list(GET _PYTHON_VALUES 0 PY_INCLUDE_DIR)
  list(GET _PYTHON_VALUES 1 PY_SUFFIX)
  list(APPEND COMMON_HEADER_DIRS ${PY_INCLUDE_DIR})

  execute_process(COMMAND ${PYTHON_PATH} "-c" "from torch.utils import cpp_extension; print(' '.join(cpp_extension._prepare_ldflags([],True,False)),end='');"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE TORCH_LINK)
  if (NOT _PYTHON_SUCCESS MATCHES 0)
      message(FATAL_ERROR "PyTorch link config Error.")
  endif()
endif()


include_directories(
  ${COMMON_HEADER_DIRS}
)

link_directories(
  ${COMMON_LIB_DIRS}
)

add_subdirectory(tools)
add_subdirectory(fastertransformer)
add_subdirectory(sample)

if(BUILD_TF)
  add_custom_target(copy ALL COMMENT "Copying tensorflow test scripts")
  add_custom_command(TARGET copy
      POST_BUILD
      COMMAND cp ${PROJECT_SOURCE_DIR}/sample/tensorflow/ ${PROJECT_BINARY_DIR} -r
      COMMAND cp ${PROJECT_SOURCE_DIR}/sample/tensorflow_bert ${PROJECT_BINARY_DIR}/tensorflow -r
 )
endif()

if(BUILD_THE OR BUILD_THS OR BUILD_THSOP)
  add_custom_target(copy ALL COMMENT "Copying pytorch test scripts")
  add_custom_command(TARGET copy
      POST_BUILD
      COMMAND cp ${PROJECT_SOURCE_DIR}/sample/pytorch/ ${PROJECT_BINARY_DIR} -r
      COMMAND mkdir -p ${PROJECT_BINARY_DIR}/pytorch/translation/data/
      COMMAND cp ${PROJECT_SOURCE_DIR}/sample/tensorflow/utils/translation/test.* ${PROJECT_BINARY_DIR}/pytorch/translation/data/
 )
endif()

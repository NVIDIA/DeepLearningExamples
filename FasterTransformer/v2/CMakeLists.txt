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
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(FasterTransformer LANGUAGES CXX CUDA)

find_package(CUDA 10.0 REQUIRED)

option(BUILD_TRT "Build in TensorRT mode" OFF)
option(BUILD_TF "Build in TensorFlow mode" OFF)

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
message("-- Assign GPU architecture (sm=60,61,70,75)")
endif()

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_STANDARD STREQUAL "11")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++11")
endif()

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
      COMMAND cp ${PROJECT_SOURCE_DIR}/sample/tensorflow/*.py ${PROJECT_SOURCE_DIR}/build/
      COMMAND cp ${PROJECT_SOURCE_DIR}/sample/tensorflow/utils ${PROJECT_SOURCE_DIR}/build/ -r
      COMMAND cp ${PROJECT_SOURCE_DIR}/sample/tensorflow/scripts ${PROJECT_SOURCE_DIR}/build/ -r
      COMMAND cp ${PROJECT_SOURCE_DIR}/sample/tensorflow_bert ${PROJECT_SOURCE_DIR}/build/ -r 
 )
endif()


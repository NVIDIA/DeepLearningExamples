##
# Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(tacotron2_inference LANGUAGES CXX CUDA)

if (DEFINED DEVEL AND NOT DEVEL EQUAL 0) 
  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    # g++ warnings
    set(CPP_DEVEL_FLAGS "${CPP_DEVEL_FLAGS} -Wall")
    set(CPP_DEVEL_FLAGS "${CPP_DEVEL_FLAGS} -Werror")
    set(CPP_DEVEL_FLAGS "${CPP_DEVEL_FLAGS} -Wpedantic")
    set(CPP_DEVEL_FLAGS "${CPP_DEVEL_FLAGS} -Weffc++")
    set(CPP_DEVEL_FLAGS "${CPP_DEVEL_FLAGS} -Wextra")
    set(CPP_DEVEL_FLAGS "${CPP_DEVEL_FLAGS} -DDEVEL=1")

    # nvcc warnings
    set(CUDA_DEVEL_FLAGS "${CUDA_DEVEL_FLAGS} -Xcompiler=-Wall")
    set(CUDA_DEVEL_FLAGS "${CUDA_DEVEL_FLAGS} -Xcompiler=-Werror")
    set(CUDA_DEVEL_FLAGS "${CUDA_DEVEL_FLAGS} -Xcompiler=-Weffc++")
    set(CUDA_DEVEL_FLAGS "${CUDA_DEVEL_FLAGS} -Xcompiler=-Wextra")
    set(CUDA_DEVEL_FLAGS "${CUDA_DEVEL_FLAGS} -Xcompiler=-DDEVEL=1")
  endif()
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CPP_DEVEL_FLAGS} -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_DEVEL_FLAGS} -Xcompiler=-fPIC")

enable_testing()

add_subdirectory("src")

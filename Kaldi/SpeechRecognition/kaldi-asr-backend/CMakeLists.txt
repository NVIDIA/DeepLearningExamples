# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.17..3.20)

project(TritonKaldiBackend LANGUAGES C CXX)

#
# Options
#
# Must include options required for this project as well as any
# projects included in this one by FetchContent.
#
# GPU support is enabled by default because the Kaldi backend requires GPUs.
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

set(TRITON_COMMON_REPO_TAG "r21.05" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "r21.05" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "r21.05" CACHE STRING "Tag for triton-inference-server/backend repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
# FetchContent's composibility isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)
FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# Shared library implementing the Triton Backend API
#
add_library(triton-kaldi-backend SHARED)
add_library(TritonKaldiBackend::triton-kaldi-backend ALIAS triton-kaldi-backend)
target_sources(triton-kaldi-backend
  PRIVATE
    triton-kaldi-backend.cc
    kaldi-backend-utils.cc 
    kaldi-backend-utils.h
)
target_include_directories(triton-kaldi-backend SYSTEM
  PRIVATE
    $<$<BOOL:${TRITON_ENABLE_GPU}>:${CUDA_INCLUDE_DIRS}>
    /opt/kaldi/src
    /opt/kaldi/tools/openfst/include
)
target_include_directories(triton-kaldi-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(triton-kaldi-backend PRIVATE cxx_std_17)
target_compile_options(triton-kaldi-backend
  PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:-Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
)
target_link_directories(triton-kaldi-backend
  PRIVATE
    /opt/kaldi/src/lib
)
target_link_libraries(triton-kaldi-backend
  PRIVATE
    TritonCore::triton-core-serverapi    # from repo-core
    TritonCore::triton-core-backendapi   # from repo-core
    TritonCore::triton-core-serverstub   # from repo-core
    TritonBackend::triton-backend-utils  # from repo-backend
    -lkaldi-cudadecoder
)
set_target_properties(triton-kaldi-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME triton_kaldi
)

#
# Install
#
include(GNUInstallDirs)

install(
  TARGETS
    triton-kaldi-backend
  EXPORT
    triton-kaldi-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/kaldi
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/kaldi
)

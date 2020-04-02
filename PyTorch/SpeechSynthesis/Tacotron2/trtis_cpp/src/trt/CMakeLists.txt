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

file(GLOB TACOTRON2WAVEGLOW_SOURCES 
    speechSynthesizer.cpp
    denoiser/*.cpp
    layers/*.cpp
    tacotron2/*.cpp
    tacotron2/*.cu
    util/*.cpp
    util/*.cu
    waveglow/*.cpp
    waveglow/*.cu
    plugins/taco2AttentionPlugin/*.cpp
    plugins/taco2AttentionPlugin/*.cu
    plugins/taco2DenoiseTransformPlugin/*.cpp
    plugins/taco2DenoiseTransformPlugin/*.cu
    plugins/taco2LSTMCellPlugin/*.cpp
    plugins/taco2LSTMCellPlugin/*.cu
    plugins/taco2ModulationRemovalPlugin/*.cpp
    plugins/taco2ModulationRemovalPlugin/*.cu
    plugins/taco2PrenetPlugin/*.cpp
    plugins/taco2PrenetPlugin/*.cu
    plugins/taco2ProjectionPlugin/*.cpp
    plugins/taco2ProjectionPlugin/*.cu
)

set(TARGET_NAME "tt2i")

## library
add_library(${TARGET_NAME} ${TACOTRON2WAVEGLOW_SOURCES})
target_link_libraries(${TARGET_NAME}
    cublas
    nvinfer
    nvonnxparser
)
set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET ${TARGET_NAME} PROPERTY ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include_directories(
    ./
    ./denoiser/
    ./layers/
    ./tacotron2/
    ./util/
    ./waveglow/
    ./common/
    ./plugins/taco2AttentionPlugin/
    ./plugins/taco2DenoiseTransformPlugin/
    ./plugins/taco2LSTMCellPlugin/
    ./plugins/taco2ModulationRemovalPlugin/
    ./plugins/taco2PrenetPlugin/
    ./plugins/taco2ProjectionPlugin/
)

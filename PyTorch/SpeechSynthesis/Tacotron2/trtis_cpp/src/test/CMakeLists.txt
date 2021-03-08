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

function(add_unit_test test_file)
  get_filename_component(test_name "${test_file}" NAME_WE)
  add_executable(${test_name} ${test_file} UnitTest.cpp)
  target_link_libraries(${test_name} tt2i)
  add_test(NAME ${test_name} 
           COMMAND "${CMAKE_CURRENT_BINARY_DIR}/${test_name}"
           WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endfunction()

include_directories(
    ../extra
    ../trt/plugins/taco2AttentionPlugin/
    ../trt/plugins/taco2DenoiseTransformPlugin/
    ../trt/plugins/taco2LSTMCellPlugin/
    ../trt/plugins/taco2ModulationRemovalPlugin/
    ../trt/plugins/taco2PrenetPlugin/
    ../trt/plugins/taco2ProjectionPlugin/
    ../trt/plugins/common/
    ../trt/
    ../trt/util
    ../trt/tacotron2
    ../trt/waveglow
    ../trt/denoiser
    ../trt/common
)

file(GLOB tests *_test.cpp)

foreach (file ${tests})
  add_unit_test(${file})
endforeach()

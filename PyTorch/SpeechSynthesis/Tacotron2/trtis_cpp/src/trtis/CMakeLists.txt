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

include_directories(".")

include_directories("${CMAKE_SOURCE_DIR}/trtis_sdk/include")
link_directories("${CMAKE_SOURCE_DIR}/trtis_sdk/lib")

file(GLOB trtis_sources *.cpp)


configure_file(tt2i_trtis.ldscript tt2i_trtis.ldscript COPYONLY)

add_library(tt2i_trtis SHARED ${trtis_sources})
target_include_directories(tt2i_trtis PRIVATE )
target_include_directories(tt2i_trtis PRIVATE
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    ../trt/
    ../trt/util
    ../trt/tacotron2
    ../trt/waveglow
    ../trt/denoiser
    ../trt/common
    ../extra
)

target_link_libraries(tt2i_trtis tt2i "${LIBS}" custombackend)
set_property(TARGET tt2i_trtis PROPERTY LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set_target_properties(
  tt2i_trtis
  PROPERTIES LINK_FLAGS "-Wl,--version-script tt2i_trtis.ldscript"
)



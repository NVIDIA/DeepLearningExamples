#!/bin/bash
##
# Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     # Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     # Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     # Neither the name of the NVIDIA CORPORATION nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 



MODEL_DIR="/models/"
ENGINE_DIR="/engines/"

TACOTRON2_JSON="${MODEL_DIR}/tacotron2.json"
WAVEGLOW_ONNX="${MODEL_DIR}/waveglow.onnx"
DENOISER_JSON="${MODEL_DIR}/denoiser.json"

TACOTRON2_ENG="${ENGINE_DIR}/tacotron2.eng"
WAVEGLOW_ENG="${ENGINE_DIR}/waveglow_chunk160_fp16.eng"
DENOISER_ENG="${ENGINE_DIR}/denoiser.eng"

BIN_DIR="./build/bin"
BENCHMARK_BIN="${BIN_DIR}/benchmark"
BUILD_TACOTRON2_BIN="${BIN_DIR}/build_tacotron2"
BUILD_WAVEGLOW_BIN="${BIN_DIR}/build_waveglow"

MAX_BATCH_SIZE=32

die() {
  echo "ERROR: ${@}" 1>&2
  exit 1
}

AMP="amp"

if [[ "${#}" == "1" ]]; then
  if [[ "${1}" == "0" || "${1}" == "no" ]]; then
    AMP="fp32"
  elif [[ "${1}" == "1" || "${1}" == "yes" ]]; then
    AMP="amp"
  else
    echo "Invalid arguments."
    exit 1
  fi 
fi

echo
echo "Building with -F${AMP}"
echo

## build tacotron2 engine

./build/bin/build_tacotron2 "${TACOTRON2_JSON}" "${TACOTRON2_ENG}" -B ${MAX_BATCH_SIZE} -I 400 -F${AMP} || die "Failed to build tacotron2 engine."

rm -v "${TACOTRON2_JSON}"


## build wave glow engine

./build/bin/build_waveglow "${WAVEGLOW_ONNX}" "${WAVEGLOW_ENG}" -B ${MAX_BATCH_SIZE} -F${AMP} || die "Failed to build waveglow engine."

rm -v "${WAVEGLOW_ONNX}"

## build denoiser engine

./build/bin/build_denoiser "${DENOISER_JSON}" "${DENOISER_ENG}" -B ${MAX_BATCH_SIZE} -F${AMP} || die "Failed to build waveglow engine."

rm -v "${DENOISER_JSON}"

ls "${TACOTRON2_ENG}" "${WAVEGLOW_ENG}" "${DENOISER_ENG}" || die "Unable to access built engines."

echo "Successfully built '${TACOTRON2_ENG}', '${WAVEGLOW_ENG}', and '${DENOISER_ENG}'"

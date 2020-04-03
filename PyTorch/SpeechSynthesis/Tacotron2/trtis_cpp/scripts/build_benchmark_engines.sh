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

TACOTRON2_ID="1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA"
WAVEGLOW_ID="1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx"

TACOTRON2_PT="${MODEL_DIR}/tacotron2.pt"
WAVEGLOW_PT="${MODEL_DIR}/waveglow.pt"
TACOTRON2_JSON="${MODEL_DIR}/tacotron2.json"
WAVEGLOW_ONNX="${MODEL_DIR}/waveglow.onnx"
DENOISER_JSON="${MODEL_DIR}/denoiser.json"

HELPER_DIR="src/trt/helpers"

BIN_DIR="./build/bin"
BENCHMARK_BIN="${BIN_DIR}/benchmark"

MAX_BATCH_SIZE=32

SCRIPT_DIR="$(dirname "${0}")"
ENGINE_BUILD_SCRIPT="${SCRIPT_DIR}/build_engines.sh"

die() {
  echo "ERROR: ${@}" 1>&2
  exit 1
}

download_gfile() {
  which curl &> /dev/null || die "Failed to find 'curl'."

  # download file from google drive
  local GOID="${1}"
  local filename="${2}"
  local GURL='https://drive.google.com/uc?export=download'
  local cookie="$(mktemp)"
  curl -sc "${cookie}" "${GURL}&id=${GOID}"
  local getcode="$(awk '/_warning_/ {print $NF}' "${cookie}")"
  curl -Lb "${cookie}" "${GURL}&confirm=${getcode}&id=${GOID}" -o "${filename}"
  rm "${cookie}"
}

mkdir -p "${ENGINE_DIR}" "${MODEL_DIR}"

apt-get update -qy
apt-get install -y libsndfile1 || die "Failed to install libsndfile"
apt-get clean

git clone --depth=1 https://github.com/NVIDIA/DeepLearningExamples
TACO2_DIR="./DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/"

# install required packages
pip3 install "torch==1.3" onnx scipy librosa || die "Failed while installing python packages."

# test packages
python3 -c "import torch; import onnx; import scipy; import numpy; import librosa" || die "Python packages fail to import" 

## build tacotron2 engine

# download model
download_gfile "${TACOTRON2_ID}" "${TACOTRON2_PT}" || die "Failed to get tacotron2.pt"

# convert model to importable format
${HELPER_DIR}/tacotron2_to_json.py "${TACOTRON2_PT}" "${TACOTRON2_JSON}" || die "Failed to export tacotron2 to json."

rm -v "${TACOTRON2_PT}"


## build wave glow engine

# download model
download_gfile "${WAVEGLOW_ID}" "${WAVEGLOW_PT}" || die "Failed to get waveglow.pt"

# convert model to importable format
${HELPER_DIR}/waveglow_to_onnx.py \
      -w "${WAVEGLOW_PT}" \
      -W "${TACO2_DIR}" \
      -o "${WAVEGLOW_ONNX}" \
      --length_mels=160 || die "Failed to export waveglow to onnx."


## build denoiser engine

${HELPER_DIR}/denoiser_to_json.py "${TACO2_DIR}" "${WAVEGLOW_PT}" "${DENOISER_JSON}" || die "Failed to export denoiser to json."

# wait to remove wave glow until after denoiser is finished
rm -v "${WAVEGLOW_PT}"
rm -rvf "./DeepLearningExamples"

pip3 uninstall -qy torch onnx scipy

apt-get purge -y libsndfile1

"${ENGINE_BUILD_SCRIPT}" || die "Failed to build engines"


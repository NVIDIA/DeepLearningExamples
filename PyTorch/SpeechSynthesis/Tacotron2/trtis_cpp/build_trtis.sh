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



NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
IMAGE_NAME="trt-tacotron2-waveglow.trtis"
CONTAINER_NAME="trt-tacotron2-waveglow.trtis.container"

die() {
  echo "ERROR: ${@}" 1>&2
  exit 1
}

if [[ $# != 4 && $# != 3 ]]; then
  echo "Unexpected number of arguments: $#"
  echo "USAGE:"
  echo "\t${0} <tacotron2 model> <waveglow model> <denoiser model> [use amp 0/1]"
  exit 1
fi


# remove container if it exists
if [[ "$(docker ps -f "name=${CONTAINER_NAME}" -qa | wc -l)" != "0" ]]; then
  docker rm "${CONTAINER_NAME}"
fi


TACOTRON2_MODEL="${1}"
WAVEGLOW_MODEL="${2}"
DENOISER_MODEL="${3}"
AMP="${4:-1}"

# copy models to build context
mkdir -p tmp/

cp -v "${TACOTRON2_MODEL}" tmp/tacotron2.json && TACOTRON2_MODEL="tmp/tacotron2.json" || die "Failed to copy ${TACOTRON2_MODEL}"
cp -v "${WAVEGLOW_MODEL}" tmp/waveglow.onnx && WAVEGLOW_MODEL="tmp/waveglow.onnx" || die "Failed to copy ${WAVEGLOW_MODEL}"
cp -v "${DENOISER_MODEL}" tmp/denoiser.json && DENOISER_MODEL="tmp/denoiser.json" || die "Failed to copy ${DENOISER_MODEL}"



docker build \
    --build-arg TACOTRON2_MODEL="${TACOTRON2_MODEL}" \
    --build-arg WAVEGLOW_MODEL="${WAVEGLOW_MODEL}" \
    --build-arg DENOISER_MODEL="${DENOISER_MODEL}" \
    -f Dockerfile.trtis . -t "${IMAGE_NAME}" || die "Failed to build docker container."

nvidia-docker run \
              -e "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}" \
              --name "${CONTAINER_NAME}" \
              "${IMAGE_NAME}" "./scripts/build_engines.sh" "${AMP}" || die "Failed to build engines."

docker commit "${CONTAINER_NAME}" "${IMAGE_NAME}" || die "Failed commit changes."
docker rm "${CONTAINER_NAME}"

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



NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-0}"
DOCKER_FILE="$(realpath Dockerfile.export_weights)"
IMAGE_NAME="trt-tacotron2-waveglow.weight_export"
CONTAINER_NAME="trt-tacotron2-waveglow.weight_export.container"

die() {
  echo "ERROR: ${@}" 1>&2
  exit 1
}

die_and_remove_image() {
  #docker rmi "${IMAGE_NAME}"
  die "${@}"
}

if [[ "${#}" != 3 ]]; then
  echo "Invalid arguments: ${@}"
  echo "USAGE:"
  echo "    ${0} <tacotron2 checkpoint> <waveglow checkpoint> <output directory>"
  exit 1
fi

TACOTRON2_PT="${1}"
WAVEGLOW_PT="${2}"
MODEL_DIR="$(realpath ${3})"

TACOTRON2_DIR="$(dirname $(realpath ${TACOTRON2_PT}))"
TACOTRON2_NAME="$(basename ${TACOTRON2_PT})"
WAVEGLOW_DIR="$(dirname $(realpath ${WAVEGLOW_PT}))"
WAVEGLOW_NAME="$(basename ${WAVEGLOW_PT})"

DLE_DIR="../"

# remove docker container if it exists
docker rm "${CONTAINER_NAME}" &> /dev/null

pushd "${DLE_DIR}"

docker build . -f "${DOCKER_FILE}" -t "${IMAGE_NAME}" || die "Failed to build container"

# export taoctron2
nvidia-docker run \
              --rm \
              -e "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}" \
              --name "${CONTAINER_NAME}" \
              -v "${TACOTRON2_DIR}:/checkpoints" \
              -v "${MODEL_DIR}:/models" \
              "${IMAGE_NAME}" "./scripts/tacotron2_to_json.py \"/checkpoints/${TACOTRON2_NAME}\" /models/tacotron2.json" || \
              die_and_remove_image "Failed to export tacotron2."

# export waveglow 
nvidia-docker run \
              --rm \
              -e "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}" \
              --name "${CONTAINER_NAME}" \
              -v "${WAVEGLOW_DIR}:/checkpoints" \
              -v "${MODEL_DIR}:/models" \
              "${IMAGE_NAME}" \
              "./scripts/waveglow_to_onnx.py -W \"${DLE_DIR}\" -w \"/checkpoints/${WAVEGLOW_NAME}\" -o /models/waveglow.onnx" || \
              die_and_remove_image "Failed to export waveglow."

# export denoiser
nvidia-docker run \
              --rm \
              -e "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}" \
              --name "${CONTAINER_NAME}" \
              -v "${WAVEGLOW_DIR}:/checkpoints" \
              -v "${MODEL_DIR}:/models" \
              "${IMAGE_NAME}" \
              "./scripts/denoiser_to_json.py \"${DLE_DIR}\" \"/checkpoints/${WAVEGLOW_NAME}\" /models/denoiser.json" || \
              die_and_remove_image "Failed to export the denoiser."


docker rmi "${IMAGE_NAME}"


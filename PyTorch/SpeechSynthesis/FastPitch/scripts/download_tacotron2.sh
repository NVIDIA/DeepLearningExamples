#!/usr/bin/env bash

set -e

MODEL_DIR=${MODEL_DIR:-"pretrained_models"}
TACO_CH="nvidia_tacotron2pyt_fp32_20190427.pt"
TACO_URL="https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2pyt_fp32/versions/2/files/nvidia_tacotron2pyt_fp32_20190427"

if [ ! -f "${MODEL_DIR}/tacotron2/${TACO_CH}" ]; then
  echo "Downloading ${TACO_CH} ..."
  mkdir -p "$MODEL_DIR"/tacotron2
  wget -qO ${MODEL_DIR}/tacotron2/${TACO_CH} ${TACO_URL}
fi

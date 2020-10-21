#!/usr/bin/env bash

set -e

MODEL_DIR=${MODEL_DIR:-"pretrained_models"}
WAVEG="waveglow_1076430_14000_amp"
WAVEG_URL="https://api.ngc.nvidia.com/v2/models/nvidia/waveglow256pyt_fp16/versions/2/zip"

mkdir -p "$MODEL_DIR"/waveglow

if [ ! -f "${MODEL_DIR}/waveglow/${WAVEG}.zip" ]; then
  echo "Downloading ${WAVEG}.zip ..."
  wget -qO "${MODEL_DIR}/waveglow/${WAVEG}.zip" ${WAVEG_URL}
fi

if [ ! -f "${MODEL_DIR}/waveglow/${WAVEG}.pt" ]; then
  echo "Extracting ${WAVEG} ..."
  unzip -qo "${MODEL_DIR}/waveglow/${WAVEG}.zip" -d ${MODEL_DIR}/waveglow/
  mv "${MODEL_DIR}/waveglow/${WAVEG}" "${MODEL_DIR}/waveglow/${WAVEG}.pt"
fi

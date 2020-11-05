#!/usr/bin/env bash

set -e

: ${MODEL_DIR:="pretrained_models/tacotron2"}
MODEL="nvidia_tacotron2pyt_fp16.pt"
MODEL_URL="https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2_pyt_ckpt_amp/versions/19.12.0/files/nvidia_tacotron2pyt_fp16.pt"

mkdir -p "$MODEL_DIR"

if [ ! -f "${MODEL_DIR}/${MODEL}" ]; then
  echo "Downloading ${MODEL} ..."
  wget --content-disposition -qO ${MODEL_DIR}/${MODEL} ${MODEL_URL} \
       || { echo "ERROR: Failed to download ${MODEL} from NGC"; exit 1; }
  echo "OK"

else
  echo "${MODEL} already downloaded."
fi

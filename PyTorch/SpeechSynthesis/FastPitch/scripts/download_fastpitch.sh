#!/usr/bin/env bash

set -e

MODEL_DIR=${MODEL_DIR:-"pretrained_models"}
FASTP_ZIP="nvidia_fastpitch_200518.zip"
FASTP_CH="nvidia_fastpitch_200518.pt"
FASTP_URL="https://api.ngc.nvidia.com/v2/models/nvidia/fastpitch_pyt_amp_ckpt_v1/versions/20.02.0/zip"

mkdir -p "$MODEL_DIR"/fastpitch

if [ ! -f "${MODEL_DIR}/fastpitch/${FASTP_ZIP}" ]; then
  echo "Downloading ${FASTP_ZIP} ..."
  wget -qO ${MODEL_DIR}/fastpitch/${FASTP_ZIP} ${FASTP_URL}
fi

if [ ! -f "${MODEL_DIR}/fastpitch/${FASTP_CH}" ]; then
  echo "Extracting ${FASTP_CH} ..."
  unzip -qo ${MODEL_DIR}/fastpitch/${FASTP_ZIP} -d ${MODEL_DIR}/fastpitch/
fi

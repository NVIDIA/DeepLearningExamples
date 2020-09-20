#!/usr/bin/env bash

set -e

DATA_DIR="LJSpeech-1.1"
MODEL_DIR="pretrained_models"

LJS_ARCH="LJSpeech-1.1.tar.bz2"
LJS_URL="http://data.keithito.com/data/speech/${LJS_ARCH}"

TACO_CH="nvidia_tacotron2pyt_fp32_20190427.pt"
TACO_URL="https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2pyt_fp32/versions/2/files/nvidia_tacotron2pyt_fp32_20190427"

WAVEG_CH="waveglow_1076430_14000_amp.pt"
WAVEG_URL="https://api.ngc.nvidia.com/v2/models/nvidia/waveglow256pyt_fp16/versions/2/files/waveglow_1076430_14000_amp"


if [ ! -f ${LJS_ARCH} ]; then
  echo "Downloading ${LJS_ARCH} ..."
  wget -q ${LJS_URL}
fi

if [ ! -d ${DATA_DIR} ]; then
  echo "Extracting ${LJS_ARCH} ..."
  tar jxvf ${LJS_ARCH}
  rm -f ${LJS_ARCH}
fi

if [ ! -f "${MODEL_DIR}/tacotron2/${TACO_CH}" ]; then
  echo "Downloading ${TACO_CH} ..."
  mkdir -p "$MODEL_DIR"/tacotron2
  wget -qO ${MODEL_DIR}/tacotron2/${TACO_CH} ${TACO_URL}
fi

if [ ! -f "${MODEL_DIR}/waveglow/${WAVEG_CH}" ]; then
  echo "Downloading ${WAVEG_CH} ..."
  mkdir -p ${MODEL_DIR}/waveglow
  wget -qO ${MODEL_DIR}/waveglow/${WAVEG_CH} ${WAVEG_URL}
fi

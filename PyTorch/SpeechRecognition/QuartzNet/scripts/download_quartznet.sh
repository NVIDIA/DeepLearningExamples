#!/usr/bin/env bash

set -e

: ${LANGUAGE:=${1:-en}}
: ${MODEL_DIR:="pretrained_models/quartznet_${LANGUAGE}"}

case $LANGUAGE in
  en)
    MODEL="nvidia_quartznet_210504.pt"
    MODEL_ZIP="quartznet_pyt_ckpt_amp_21.03.0.zip"
    MODEL_URL="https://api.ngc.nvidia.com/v2/models/nvidia/quartznet_pyt_ckpt_amp/versions/21.03.0/zip"
    ;;
  ca|de|es|fr|it|pl|ru|zh)
    MODEL="stt_${LANGUAGE}_quartznet15x5.nemo"
    MODEL_URL="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_${LANGUAGE}_quartznet15x5/versions/1.0.0rc1/zip"
    MODEL_ZIP="stt_${LANGUAGE}_quartznet15x5_1.0.0rc1.zip"
    ;;
  *)
    echo "Unsupported language $LANGUAGE"
    exit 1
    ;;
esac

mkdir -p "$MODEL_DIR"

if [ ! -f "${MODEL_DIR}/${MODEL_ZIP}" ]; then
  echo "Downloading ${MODEL_ZIP} ..."
  wget -O ${MODEL_DIR}/${MODEL_ZIP} ${MODEL_URL} \
       || { echo "ERROR: Failed to download ${MODEL_ZIP} from NGC"; exit 1; }
fi

if [ ! -f "${MODEL_DIR}/${MODEL}" ]; then
  echo "Extracting ${MODEL} ..."
  unzip -o ${MODEL_DIR}/${MODEL_ZIP} -d ${MODEL_DIR} \
        || { echo "ERROR: Failed to extract ${MODEL_ZIP}"; exit 1; }

  echo "OK"

else
  echo "${MODEL} already downloaded."
fi

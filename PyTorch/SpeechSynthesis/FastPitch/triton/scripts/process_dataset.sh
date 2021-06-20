#!/usr/bin/env bash
#!/usr/bin/env bash
# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

DATASET_DIR="${DATASETS_DIR}/LJSpeech-1.1/LJSpeech-1.1_fastpitch"

TACO_DIR="${DATASETS_DIR}/tacotron2"
TACO_MODEL="nvidia_tacotron2pyt_fp16.pt"
TACO_MODEL_URL="https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2_pyt_ckpt_amp/versions/19.12.0/files/nvidia_tacotron2pyt_fp16.pt"

if [ -f "${TACO_DIR}/${TACO_MODEL}" ]; then
  echo "${TACO_MODEL} already downloaded."
elif [ -f "${WORKDIR}/pretrained_models/tacotron2/${TACO_MODEL}" ]; then
  echo "Linking existing model from ${WORKDIR}/pretrained_models/tacotron2/${TACO_MODEL}"
  mkdir -p ${TACO_DIR}
  ln -s "${WORKDIR}/pretrained_models/tacotron2/${TACO_MODEL}" "${TACO_DIR}/${TACO_MODEL}"
elif [ -f "${PWD}/pretrained_models/tacotron2/${TACO_MODEL}" ]; then
  echo "Linking existing model from ${PWD}/pretrained_models/tacotron2/${TACO_MODEL}"
  mkdir -p ${TACO_DIR}
  ln -s "${PWD}/pretrained_models/tacotron2/${TACO_MODEL}" "${TACO_DIR}/${TACO_MODEL}"
else
  echo "Downloading ${TACO_MODEL} ..."
  mkdir -p ${TACO_DIR}
  wget --content-disposition -qO ${TACO_DIR}/${TACO_MODEL} ${TACO_MODEL_URL} ||
    {
      echo "ERROR: Failed to download ${TACO_MODEL} from NGC"
      exit 1
    }
  echo "OK"
fi

if [ ! -d "${DATASET_DIR}/mels" ]; then

  for FILELIST in ljs_audio_text_train_filelist.txt \
    ljs_audio_text_val_filelist.txt \
    ljs_audio_text_test_filelist.txt; do
    python extract_mels.py \
      --cuda \
      --dataset-path ${DATASET_DIR} \
      --wav-text-filelist filelists/${FILELIST} \
      --batch-size 256 \
      --extract-mels \
      --extract-durations \
      --extract-pitch-char \
      --tacotron2-checkpoint ${TACO_DIR}/${TACO_MODEL}
  done
fi

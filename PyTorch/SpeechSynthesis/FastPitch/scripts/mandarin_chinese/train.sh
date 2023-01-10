#!/usr/bin/env bash

set -a

PYTHONIOENCODING=utf-8

# Mandarin & English bilingual
ARGS+=" --symbol-set english_mandarin_basic"

# Initialize weights with a pre-trained English model
bash scripts/download_models.sh fastpitch
ARGS+=" --init-from-checkpoint pretrained_models/fastpitch/nvidia_fastpitch_210824.pt"

AMP=false  # FP32 training for better stability

: ${DATASET_PATH:=data/SF_bilingual}
: ${TRAIN_FILELIST:=filelists/sf_audio_pitch_text_train.txt}
: ${VAL_FILELIST:=filelists/sf_audio_pitch_text_val.txt}
: ${OUTPUT_DIR:=./output_sf}

bash scripts/train.sh $ARGS "$@"

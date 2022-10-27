#!/usr/bin/env bash

set -e

: ${DATASET_PATH:=data/LJSpeech-1.1}

export DATASET_PATH
bash scripts/generate_filelists.sh

# Generate mel-spectrograms
python prepare_dataset.py \
    --wav-text-filelists data/filelists/ljs_audio_text.txt \
    --n-workers 16 \
    --batch-size 1 \
    --dataset-path $DATASET_PATH \
    --extract-mels \
    "$@"

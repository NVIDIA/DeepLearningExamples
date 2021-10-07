#!/usr/bin/env bash

set -e

: ${DATA_DIR:=LJSpeech-1.1}
: ${ARGS="--extract-mels"}

python prepare_dataset.py \
    --wav-text-filelists filelists/ljs_audio_text.txt \
    --n-workers 16 \
    --batch-size 1 \
    --dataset-path $DATA_DIR \
    --extract-pitch \
    --f0-method pyin \
    $ARGS

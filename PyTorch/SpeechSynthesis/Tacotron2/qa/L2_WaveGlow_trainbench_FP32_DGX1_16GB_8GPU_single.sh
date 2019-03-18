#!/usr/bin/env bash

set -e

bash scripts/prepare_dataset.sh

python -m multiproc train.py \
    -m WaveGlow \
    -o ./ \
    -lr 1e-4 \
    --epochs 10 \
    -bs 4 \
    --segment-length 8000 \
    --weight-decay 0 \
    --grad-clip-thresh 65504.0 \
    --log-file ./nvlog.json \
    --epochs-per-checkpoint 250 \
    --cudnn-benchmark=True \
    --training-files filelists/ljs_audio_text_train_subset_1250_filelist.txt \
    --dataset-path ./

python qa/check_curves.py \
    -b qa/waveglow_fp32-perf.json \
    -g "epoch" \
    -k "train_epoch_items/sec" \
    --skip 9 \
    --damping 1 \
    --eps 0 \
    ./nvlog.json \
    --fail low

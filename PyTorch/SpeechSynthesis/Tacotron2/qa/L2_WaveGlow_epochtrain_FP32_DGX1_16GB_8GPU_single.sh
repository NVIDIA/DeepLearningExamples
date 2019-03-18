#!/usr/bin/env bash

set -e

bash scripts/prepare_dataset.sh

python -m multiproc train.py \
    -m WaveGlow \
    -o ./ \
    -lr 1e-4 \
    --epochs 2 \
    -bs 4 \
    --segment-length 8000 \
    --weight-decay 0 \
    --grad-clip-thresh 65504.0 \
    --cudnn-benchmark=True \
    --log-file ./nvlog.json \
    --epochs-per-checkpoint 250

python qa/check_curves.py \
    -b qa/waveglow_fp32-short-loss.json \
    -p ./waveglow_fp32-short-loss.png \
    -g iter \
    -k "train_iteration_loss" \
    ./nvlog.json \
    --fail high

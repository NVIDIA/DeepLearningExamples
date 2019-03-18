#!/usr/bin/env bash

set -e

bash scripts/prepare_dataset.sh

python -m multiproc train.py \
    -m WaveGlow \
    -o ./ \
    -lr 1e-4 \
    --epochs 1001 \
    -bs 8 \
    --segment-length 8000 \
    --weight-decay 0 \
    --grad-clip-thresh 65504.0 \
    --cudnn-benchmark=True \
    --log-file ./nvlog.json \
    --epochs-per-checkpoint 250 \
    --fp16-run

python qa/check_curves.py \
    -b qa/waveglow_fp16-long-loss.json \
    -p ./waveglow_fp16-long-loss.png \
    -g iter \
    -k "train_iteration_loss" \
    ./nvlog.json \
    --fail high

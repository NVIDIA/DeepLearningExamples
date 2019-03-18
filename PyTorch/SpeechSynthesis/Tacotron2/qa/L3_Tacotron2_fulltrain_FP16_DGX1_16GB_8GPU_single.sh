#!/usr/bin/env bash

set -e

bash scripts/prepare_dataset.sh

python -m multiproc train.py \
    -m Tacotron2 \
    -o ./ \
    -lr 1e-3 \
    --epochs 1001 \
    -bs 80 \
    --weight-decay 1e-6 \
    --grad-clip-thresh 1.0 \
    --cudnn-benchmark=True \
    --log-file ./nvlog.json \
    --anneal-steps 500 1000 1500 \
    --anneal-factor 0.1 \
    --epochs-per-checkpoint 250 \
    --fp16-run

python qa/check_curves.py \
    -b qa/tacotron2_fp16-long-loss.json \
    -p ./tacotron2_fp16-long-loss.png \
    -g iter \
    -k "train_iteration_loss" \
    ./nvlog.json \
    --fail high

#!/usr/bin/env bash

set -e

bash scripts/prepare_dataset.sh

python -m multiproc train.py \
    -m Tacotron2 \
    -o ./ \
    -lr 1e-3 \
    --epochs 10 \
    -bs 80 \
    --weight-decay 1e-6 \
    --grad-clip-thresh 1.0 \
    --cudnn-benchmark=True \
    --log-file ./nvlog.json \
    --anneal-steps 500 1000 1500 \
    --anneal-factor 0.3 \
    --epochs-per-checkpoint 250 \
    --training-files filelists/ljs_audio_text_train_subset_2500_filelist.txt \
    --dataset-path ./ \
    --fp16-run

python qa/check_curves.py \
    -b qa/tacotron2_fp16-perf.json \
    -g "epoch" \
    -k "train_epoch_items/sec" \
    --skip 9 \
    --damping 1 \
    --eps 0 \
    ./nvlog.json \
    --fail low

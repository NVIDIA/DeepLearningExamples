#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ../data/text8/ \
        --dataset text8 \
        --n_layer 24 \
        --d_model 1024 \
        --n_head 8 \
        --d_head 128 \
        --d_inner 3072 \
        --dropout 0.15 \
        --dropatt 0.15 \
        --optim adam \
        --lr 0.00025 \
        --tgt_len 768 \
        --mem_len 768 \
        --eval_tgt_len 128 \
        --batch_size 64 \
        --max_step 400000 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../data/text8/ \
        --dataset text8 \
        --tgt_len 128 \
        --mem_len 3800 \
        --clamp_len 1000 \
        --same_length \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi

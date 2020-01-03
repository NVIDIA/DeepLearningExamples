#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ../data/one-billion-words/ \
        --dataset lm1b \
        --adaptive \
        --div_val 4 \
        --n_layer 24 \
        --d_model 1280 \
        --n_head 16 \
        --d_head 80 \
        --d_inner 8192 \
        --dropout 0.05 \
        --dropatt 0.05 \
        --optim adam \
        --warmup_step 30000 \
        --max_step 1200000 \
        --lr 0.00025 \
        --tgt_len 32 \
        --mem_len 32 \
        --eval_tgt_len 32 \
        --batch_size 512 \
        --multi_gpu \
        --gpu0_bsz 0 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../data/one-billion-words/ \
        --dataset lm1b \
        --batch_size 8 \
        --tgt_len 32 \
        --mem_len 128 \
        --split test \
        --same_length \
        ${@:2}
else
    echo 'unknown argment 1'
fi

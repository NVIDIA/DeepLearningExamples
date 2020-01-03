#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python -m torch.distributed.launch --nproc_per_node=$2 train.py \
        --cuda \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --n_layer 18 \
        --d_model 1024 \
        --n_head 16 \
        --d_head 64 \
        --d_inner 4096 \
        --dropout 0.2 \
        --dropatt 0.2 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 16000 \
        --max_step 4000000 \
        --tgt_len 256 \
        --mem_len 256 \
        --eval_tgt_len 128 \
        --batch_size 128 \
        --multi_gpu ddp \
        ${@:3}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python -m torch.distributed.launch --nproc_per_node=$2 eval.py \
        --cuda \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --tgt_len 128 \
        --mem_len 1600 \
        --clamp_len 1000 \
        --same_length \
        --split test \
        ${@:3}
else
    echo 'unknown argment 1'
fi

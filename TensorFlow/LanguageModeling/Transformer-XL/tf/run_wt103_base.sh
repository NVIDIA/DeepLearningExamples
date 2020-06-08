#!/bin/bash

# Data
DATA_ROOT=../data/wikitext-103/

# Model
DIV_VAL=1
N_LAYER=16
D_MODEL=512
D_EMBED=512
N_HEAD=8
D_HEAD=64
D_INNER=2048

# Training
TGT_LEN=192
MEM_LEN=192

NUM_CORE=${2:-"8"}

# Testing
TEST_TGT_LEN=64
TEST_MEM_LEN=640
TEST_CLAMP_LEN=400

TEST_NUM_CORE=1


if [[ $1 == 'train_data' ]]; then
    python data_utils.py \
        --data_dir=${DATA_ROOT}/ \
        --dataset=wt103 \
        --tgt_len=${TGT_LEN} \
        --num_passes=2 \
        --use_tpu=False \
        --eval_batch_size=0 \
        ${@:2}
elif [[ $1 == 'test_data' ]]; then
    python data_utils.py \
        --data_dir=${DATA_ROOT}/ \
        --dataset=enwik8 \
        --tgt_len=${TEST_TGT_LEN} \
        --num_passes=1 \
        --use_tpu=False \
        ${@:2}
elif [[ $1 == 'train' ]]; then
    echo 'Run training...'
    horovodrun -np ${NUM_CORE} -H localhost:${NUM_CORE} python main.py \
        --data_dir=${DATA_ROOT}/tfrecords \
        --record_info_dir=${DATA_ROOT}/tfrecords/ \
        --corpus_info_path=${DATA_ROOT}/corpus-info.json \
        --div_val=${DIV_VAL} \
        --untie_r=True \
        --proj_share_all_but_first=True \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.1 \
        --dropatt=0.0 \
        --learning_rate=0.01 \
        --warmup_steps=1000 \
        --tgt_len=${TGT_LEN} \
        --mem_len=${MEM_LEN} \
        --num_core_per_host=${NUM_CORE} \
        ${@:3}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python main.py \
        --data_dir=${DATA_ROOT}/tfrecords \
        --record_info_dir=${DATA_ROOT}/tfrecords/ \
        --corpus_info_path=${DATA_ROOT}/corpus-info.json \
        --div_val=${DIV_VAL} \
        --untie_r=True \
        --proj_share_all_but_first=True \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.0 \
        --dropatt=0.0 \
        --tgt_len=${TEST_TGT_LEN} \
        --mem_len=${TEST_MEM_LEN} \
        --clamp_len=${TEST_CLAMP_LEN} \
        --same_length=True \
        --num_core_per_host=${TEST_NUM_CORE} \
        --do_train=False \
        --do_eval=True \
        --horovod=False \
        --eval_split=test \
        ${@:2}
else
    echo 'unknown argment 1'
fi

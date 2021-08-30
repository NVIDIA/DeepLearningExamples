#!/bin/bash
set -xe
export LD_LIBRARY_PATH=/usr/lib/libibverbs/:/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH

PADDLE_TRAINER_ENDPOINTS=`echo $PADDLE_TRAINER_ENDPOINTS | tr ',' '\n' | head -n 1`

batch_size=${1:-"96"}  # batch size per gpu
num_gpus=${2:-"8"}    # number of gpu
precision=${3:-"fp16"}   # fp32 | fp16
gradient_accumulation_steps=$(expr 67584 \/ $batch_size \/ $num_gpus)
train_batch_size=$(expr 67584 \/ $num_gpus)   # total batch_size per gpu
train_steps=${4:-4}    # max train steps

export NODE_RANK=`python get_mpi_rank.py`

cd ${HOME_WORK_DIR}/workspace/env_run/zengjinle/DeepLearningExamples/PyTorch/LanguageModeling/BERT

rm -rf ./results/checkpoints
# run pre-training
bash scripts/run_pretraining.sh $train_batch_size 6e-3 $precision $num_gpus 0.2843 $train_steps 200 false true true $gradient_accumulation_steps


#!/bin/bash

set -x

batch_size=$1
num_gpus=$2
precision=$3
num_accumulation_steps_phase1=$(expr 65536 \/ $batch_size \/ $num_gpus)
train_steps=${4:-200}
bert_model=${5:-"base"}

# run pre-training
bash scripts/run_pretraining_lamb.sh $batch_size 64 8 7.5e-4 5e-4 $precision true $num_gpus 2000 200 $train_steps 200 $num_accumulation_steps_phase1 512 $bert_model
#!/bin/bash

set -x

batch_size=$1
num_gpus=$2
precision=$3
gradient_accumulation_steps=$(expr 65536 \/ $batch_size \/ $num_gpus)
train_batch_size=$(expr 65536 \/ $num_gpus)
train_steps=${4:-250}

# run pre-training
rm results/checkpoints/*.pt
bash scripts/run_pretraining.sh $train_batch_size 6e-3 $precision $num_gpus 0.2843 $train_steps 200 false true true $gradient_accumulation_steps
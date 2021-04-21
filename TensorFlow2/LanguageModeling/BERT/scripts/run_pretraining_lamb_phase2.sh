#! /bin/bash

# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

echo "Container nvidia build = " $NVIDIA_BUILD_ID

train_batch_size_phase1=${1:-60}
train_batch_size_phase2=${2:-10}
eval_batch_size=${3:-8}
learning_rate_phase1=${4:-"7.5e-4"}
learning_rate_phase2=${5:-"5e-4"}
precision=${6:-"fp16"}
use_xla=${7:-"true"}
num_gpus=${8:-8}
warmup_steps_phase1=${9:-"2133"}
warmup_steps_phase2=${10:-"213"}
train_steps=${11:-8341}
save_checkpoints_steps=${12:-100}
num_accumulation_steps_phase1=${13:-128}
num_accumulation_steps_phase2=${14:-384}
bert_model=${15:-"large"}

DATA_DIR=${DATA_DIR:-data}
#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=${RESULTS_DIR:-/results}

if [ "$bert_model" = "large" ] ; then
    export BERT_CONFIG=data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/bert_config.json
else
    export BERT_CONFIG=data/download/google_pretrained_weights/uncased_L-12_H-768_A-12/bert_config.json
fi

echo "Container nvidia build = " $NVIDIA_BUILD_ID

PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--use_fp16"
elif [ "$precision" = "fp32" ] || [ "$precision" = "tf32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

if [ "$use_xla" = "true" ] ; then
    PREC="$PREC --enable_xla"
    echo "XLA activated"
fi

mpi=""
if [ $num_gpus -gt 1 ] ; then
   mpi="mpiexec --allow-run-as-root -np $num_gpus"
   horovod="--use_horovod"
fi

#PHASE 1 Config

train_steps_phase1=$(expr $train_steps \* 9 \/ 10) #Phase 1 is 10% of training
gbs_phase1=$(expr $train_batch_size_phase1 \* $num_accumulation_steps_phase1)
PHASE1_CKPT=${RESULTS_DIR}/phase_1/pretrained/bert_model.ckpt-1

#PHASE 2

seq_len=512
max_pred_per_seq=80
train_steps_phase2=$(expr $train_steps \* 1 \/ 10) #Phase 2 is 10% of training
gbs_phase2=$(expr $train_batch_size_phase2 \* $num_accumulation_steps_phase2)
train_steps_phase2=$(expr $train_steps_phase2 \* $gbs_phase1 \/ $gbs_phase2) # Adjust for batch size

RESULTS_DIR_PHASE2=${RESULTS_DIR}/phase_2
mkdir -m 777 -p $RESULTS_DIR_PHASE2

INPUT_FILES="$DATA_DIR/tfrecord/lower_case_1_seq_len_${seq_len}_max_pred_${max_pred_per_seq}_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/training/*"
EVAL_FILES="$DATA_DIR/tfrecord/lower_case_1_seq_len_${seq_len}_max_pred_${max_pred_per_seq}_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/test"

$mpi python /workspace/bert_tf2/run_pretraining.py \
    --input_files=$INPUT_FILES \
    --init_checkpoint=$PHASE1_CKPT \
    --model_dir=$RESULTS_DIR_PHASE2 \
    --bert_config_file=$BERT_CONFIG \
    --train_batch_size=$train_batch_size_phase2 \
    --max_seq_length=$seq_len \
    --max_predictions_per_seq=$max_pred_per_seq \
    --num_steps_per_epoch=$train_steps_phase2 --num_train_epochs=1 \
    --steps_per_loop=$save_checkpoints_steps \
    --save_checkpoint_steps=$save_checkpoints_steps \
    --warmup_steps=$warmup_steps_phase2 \
    --num_accumulation_steps=$num_accumulation_steps_phase2 \
    --learning_rate=$learning_rate_phase2 \
    --optimizer_type=LAMB \
    $horovod $PREC


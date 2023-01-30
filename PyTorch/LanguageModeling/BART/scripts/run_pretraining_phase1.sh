#!/usr/bin/env bash

# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

train_batch_size_phase1=${1:-200}
train_batch_size_phase2=${2:-32}
learning_rate_phase1=${3:-"5e-3"}
learning_rate_phase2=${4:-"4e-3"}
precision=${5:-"bf16"}
use_preln=${6:-"true"}
num_gpus=${7:-8}
warmup_steps_phase1=${8:-"2166"}
warmup_steps_phase2=${9:-"200"}
train_steps_phase1=${10:-95040}
train_steps_phase2=${11:-7560}
save_checkpoints_steps=${12:-100}
num_accumulation_steps_phase1=${13:-40}
num_accumulation_steps_phase2=${14:-120}
config_path=${15:-"configs/config.json"}

DATA_DIR=${DATA_DIR:-data}
RESULTS_DIR=${RESULTS_DIR:-results}

RESULTS_DIR_PHASE1=${RESULTS_DIR}/phase_1
mkdir -m 777 -p $RESULTS_DIR_PHASE1

DATESTAMP=`date +'%y%m%d%H%M%S'`
LOGFILE=$RESULTS_DIR_PHASE1/$DATESTAMP.log
printf "Logs written to %s\n" "$LOGFILE"

SOURCE_LEN=128

if [ "$precision" = "fp16" ] ; then
    echo "fp16 activated!"
    USE_FP16="--fp16"
elif [ "$precision" = "bf16" ] ; then
    echo "bf16 activated!"
    USE_FP16="--bf16"
else
    echo "fp32/tf32 activated!"
    USE_FP16=""
fi

if [ "$use_preln" = "true" ] ; then
    echo "Trained with PreLN"
    USE_FP16="--pre_ln $USE_FP16"
else
    echo "Trained with PostLN"
fi

export TOKENIZERS_PARALLELISM=true;
python -m torch.distributed.launch --nproc_per_node=${num_gpus} pretrain.py \
--data_dir=${DATA_DIR}/pretrain_lddl_${SOURCE_LEN} \
--config_path=${config_path} \
--output_dir=${RESULTS_DIR_PHASE1} \
--num_workers 4 \
--learning_rate=${learning_rate_phase1} \
${USE_FP16} \
--do_train \
--train_batch_size=${train_batch_size_phase1} --gradient_accumulation_steps=${num_accumulation_steps_phase1} \
--max_steps=${train_steps_phase1} --warmup_steps=${warmup_steps_phase1} \
--max_source_length=${SOURCE_LEN} \
--lr_scheduler polynomial \
--label_smoothing 0 \
--weight_decay 0.1 \
--dropout 0.1 --attention_dropout 0.1 --gradient_clip_val=0.1 \
--resume_from_checkpoint=True \
--save_checkpoint_steps=${save_checkpoints_steps} --log_freq=${save_checkpoints_steps} \
--allreduce_post_accumulation_half_precision \
--seed $RANDOM --lamb |& tee -a ${LOGFILE}

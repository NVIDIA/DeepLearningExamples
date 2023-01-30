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

DATA_DIR=data
export DATA_DIR=$DATA_DIR

printf -v TAG "bart_pyt_pretraining"
RESULTS_DIR=${RESULTS_DIR:-results/${TAG}}

printf "Saving checkpoints to %s\n" "$RESULTS_DIR"
export RESULTS_DIR=$RESULTS_DIR

printf -v SCRIPT_ARGS "%d %d %e %e %s %s %d %d %d %d %d %d %d %d %s" \
                      $train_batch_size_phase1 $train_batch_size_phase2 $learning_rate_phase1 \
                      $learning_rate_phase2 "$precision" "$use_preln" $num_gpus $warmup_steps_phase1 \
                      $warmup_steps_phase2 $train_steps_phase1 $train_steps_phase2 $save_checkpoints_steps \
                      $num_accumulation_steps_phase1 $num_accumulation_steps_phase2 "$config_path"

set -x
# RUN PHASE 1
bash scripts/run_pretraining_phase1.sh $SCRIPT_ARGS

# RUN PHASE 2
bash scripts/run_pretraining_phase2.sh $SCRIPT_ARGS
set +x

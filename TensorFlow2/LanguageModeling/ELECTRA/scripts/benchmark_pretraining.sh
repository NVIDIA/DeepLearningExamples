#!/usr/bin/env bash

# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
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

train_batch_size_p1=${1:-"176"}
learning_rate_p1="6e-7"
precision=${2:-"amp"}
xla=${3:-"xla"}
num_gpus=${4:-8}
warmup_steps_p1="10"
train_steps_p1=10
save_checkpoint_steps=500
resume_training="false"
optimizer="lamb"
accumulate_gradients=${5:-"true"}
gradient_accumulation_steps_p1=${6:-48}
seed=42
job_name="electra_lamb_pretraining_benchmark"
train_batch_size_p2=${7:-24}
learning_rate_p2="4e-7"
warmup_steps_p2="10"
train_steps_p2=10
gradient_accumulation_steps_p2=${8:-144}
electra_model=${9:-"base"}

restore_checkpoint=false bash scripts/run_pretraining.sh $train_batch_size_p1 $learning_rate_p1 $precision $num_gpus $xla \
         $warmup_steps_p1 $train_steps_p1 $save_checkpoint_steps \
         $resume_training $optimizer $accumulate_gradients  \
         $gradient_accumulation_steps_p1 $seed $job_name \
         $train_batch_size_p2 $learning_rate_p2 \
         $warmup_steps_p2 $train_steps_p2 $gradient_accumulation_steps_p2 \
         $electra_model

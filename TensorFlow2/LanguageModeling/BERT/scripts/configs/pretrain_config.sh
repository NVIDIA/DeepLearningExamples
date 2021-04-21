#!/usr/bin/env bash
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

# Full LAMB pretraining configs for NVIDIA DGX A100 (8x NVIDIA A100 40GB GPU)

dgxa100_8gpu_fp16 ()
{
  train_batch_size_phase1=312
  train_batch_size_phase2=40
  eval_batch_size=8
  learning_rate_phase1="8.12e-4"
  learning_rate_phase2="5e-4"
  precision="fp16"
  use_xla="true"
  num_gpus=8
  warmup_steps_phase1=2000
  warmup_steps_phase2=200
  train_steps=6416
  save_checkpoints_steps=100
  num_accumulation_steps_phase1=32
  num_accumulation_steps_phase2=96
  echo $train_batch_size_phase1 $train_batch_size_phase2 $eval_batch_size $learning_rate_phase1 $learning_rate_phase2 $precision $use_xla $num_gpus $warmup_steps_phase1 $warmup_steps_phase2 $train_steps $save_checkpoint_steps $num_accumulation_steps_phase2
}

dgxa100_8gpu_tf32 ()
{
  train_batch_size_phase1=176
  train_batch_size_phase2=22
  eval_batch_size=8
  learning_rate_phase1="7.5e-4"
  learning_rate_phase2="5e-4"
  precision="tf32"
  use_xla="true"
  num_gpus=8
  warmup_steps_phase1=2000
  warmup_steps_phase2=200
  train_steps=5687
  save_checkpoints_steps=100
  num_accumulation_steps_phase1=64
  num_accumulation_steps_phase2=192
  echo $train_batch_size_phase1 $train_batch_size_phase2 $eval_batch_size $learning_rate_phase1 $learning_rate_phase2 $precision $use_xla $num_gpus $warmup_steps_phase1 $warmup_steps_phase2 $train_steps $save_checkpoint_steps $num_accumulation_steps_phase2
}

# Full LAMB pretraining configs for NVIDIA DGX-2H (16x NVIDIA V100 32GB GPU)

dgx2_16gpu_fp16 ()
{
  train_batch_size_phase1=60
  train_batch_size_phase2=10
  eval_batch_size=8
  learning_rate_phase1="3.75e-4"
  learning_rate_phase2="2.5e-4"
  precision="fp16"
  use_xla="true"
  num_gpus=16
  warmup_steps_phase1=2133
  warmup_steps_phase2=213
  train_steps=8341
  save_checkpoints_steps=100
  num_accumulation_steps_phase1=64
  num_accumulation_steps_phase2=192
  echo $train_batch_size_phase1 $train_batch_size_phase2 $eval_batch_size $learning_rate_phase1 $learning_rate_phase2 $precision $use_xla $num_gpus $warmup_steps_phase1 $warmup_steps_phase2 $train_steps $save_checkpoint_steps $num_accumulation_steps_phase2
}

dgx2_16gpu_fp32 ()
{
  train_batch_size_phase1=32
  train_batch_size_phase2=6
  eval_batch_size=8
  learning_rate_phase1="3.75e-4"
  learning_rate_phase2="2.5e-4"
  precision="fp32"
  use_xla="true"
  num_gpus=16
  warmup_steps_phase1=2000
  warmup_steps_phase2=200
  train_steps=7820
  save_checkpoints_steps=100
  num_accumulation_steps_phase1=128
  num_accumulation_steps_phase2=320
  echo $train_batch_size_phase1 $train_batch_size_phase2 $eval_batch_size $learning_rate_phase1 $learning_rate_phase2 $precision $use_xla $num_gpus $warmup_steps_phase1 $warmup_steps_phase2 $train_steps $save_checkpoint_steps $num_accumulation_steps_phase2
}

# Full LAMB pretraining configs for NVIDIA DGX-1 (8x NVIDIA V100 32GB GPU)

dgx1_8gpu_fp16 ()
{
  train_batch_size_phase1=60
  train_batch_size_phase2=10
  eval_batch_size=8
  learning_rate_phase1="7.5e-4"
  learning_rate_phase2="5e-4"
  precision="fp16"
  use_xla="true"
  num_gpus=8
  warmup_steps_phase1=2133
  warmup_steps_phase2=213
  train_steps=8341
  save_checkpoints_steps=100
  num_accumulation_steps_phase1=128
  num_accumulation_steps_phase2=384
  echo $train_batch_size_phase1 $train_batch_size_phase2 $eval_batch_size $learning_rate_phase1 $learning_rate_phase2 $precision $use_xla $num_gpus $warmup_steps_phase1 $warmup_steps_phase2 $train_steps $save_checkpoint_steps $num_accumulation_steps_phase2
}

dgx1_8gpu_fp32 ()
{
  train_batch_size_phase1=32
  train_batch_size_phase2=6
  eval_batch_size=8
  learning_rate_phase1="7.5e-4"
  learning_rate_phase2="5e-4"
  precision="fp32"
  use_xla="true"
  num_gpus=8
  warmup_steps_phase1=2000
  warmup_steps_phase2=200
  train_steps=7820
  save_checkpoints_steps=100
  num_accumulation_steps_phase1=256
  num_accumulation_steps_phase2=640
  echo $train_batch_size_phase1 $train_batch_size_phase2 $eval_batch_size $learning_rate_phase1 $learning_rate_phase2 $precision $use_xla $num_gpus $warmup_steps_phase1 $warmup_steps_phase2 $train_steps $save_checkpoint_steps $num_accumulation_steps_phase2
}

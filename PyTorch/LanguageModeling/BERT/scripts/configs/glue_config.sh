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

set -e

batch_size_and_gradient_accumulation_steps() {
  batch_size=$((global_batch_size / num_gpu))
  gradient_accumulation_steps=1

  while [ $((batch_size / gradient_accumulation_steps)) -gt $batch_size_capacity ]
  do
    gradient_accumulation_steps=$((gradient_accumulation_steps * 2))
  done
}

commons () {
  init_checkpoint=/workspace/bert/checkpoints/bert_uncased.pt
  vocab_file=${BERT_PREP_WORKING_DIR}/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt
  config_file=/workspace/bert/bert_configs/large.json
  max_steps=-1.0
}

mrpc_commons () {
  data_dir=${BERT_PREP_WORKING_DIR}/download/glue/MRPC/
  out_dir=/workspace/bert/results/MRPC
  task_name=mrpc
  global_batch_size=128
  learning_rate=1.8e-5
  warmup_proportion=0.3
  epochs=3
}

sst-2_commons () {
  data_dir=${BERT_PREP_WORKING_DIR}/download/glue/SST-2/
  out_dir=/workspace/bert/results/SST-2
  task_name=sst-2
  global_batch_size=1024
  learning_rate=1e-5
  warmup_proportion=0.1
  epochs=3
}

dgxa100-80g_fp16_commons () {
  batch_size_capacity=256
  precision=fp16
}

dgxa100-80g_tf32_commons () {
  batch_size_capacity=128
  precision=tf32
}

dgx1-32g_fp16_commons () {
  batch_size_capacity=128
  precision=fp16
}

dgx1-32g_fp32_commons () {
  batch_size_capacity=64
  precision=fp32
}

print_arguments_in_order () {
  echo \
    $init_checkpoint \
    $data_dir \
    $vocab_file \
    $config_file \
    $out_dir \
    $task_name \
    $num_gpu \
    $batch_size \
    $gradient_accumulation_steps \
    $learning_rate \
    $warmup_proportion \
    $epochs \
    $max_steps \
    $precision
}

##########################################
#             DGXA100 80G                #
##########################################

##########################
#          MRPC          #
##########################

# AMP

mrpc_dgxa100-80g_1gpu_fp16 () {
  commons
  mrpc_commons
  dgxa100-80g_fp16_commons
  num_gpu=1
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

mrpc_dgxa100-80g_2gpu_fp16 () {
  commons
  mrpc_commons
  dgxa100-80g_fp16_commons
  num_gpu=2
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

mrpc_dgxa100-80g_4gpu_fp16 () {
  commons
  mrpc_commons
  dgxa100-80g_fp16_commons
  num_gpu=4
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

mrpc_dgxa100-80g_8gpu_fp16 () {
  commons
  mrpc_commons
  dgxa100-80g_fp16_commons
  num_gpu=8
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

# TF32

mrpc_dgxa100-80g_1gpu_tf32 () {
  commons
  mrpc_commons
  dgxa100-80g_tf32_commons
  num_gpu=1
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

mrpc_dgxa100-80g_2gpu_tf32 () {
  commons
  mrpc_commons
  dgxa100-80g_tf32_commons
  num_gpu=2
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

mrpc_dgxa100-80g_4gpu_tf32 () {
  commons
  mrpc_commons
  dgxa100-80g_tf32_commons
  num_gpu=4
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

mrpc_dgxa100-80g_8gpu_tf32 () {
  commons
  mrpc_commons
  dgxa100-80g_tf32_commons
  num_gpu=8
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

##########################
#          SST-2         #
##########################

# AMP

sst-2_dgxa100-80g_1gpu_fp16 () {
  commons
  sst-2_commons
  dgxa100-80g_fp16_commons
  num_gpu=1
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

sst-2_dgxa100-80g_2gpu_fp16 () {
  commons
  sst-2_commons
  dgxa100-80g_fp16_commons
  num_gpu=2
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

sst-2_dgxa100-80g_4gpu_fp16 () {
  commons
  sst-2_commons
  dgxa100-80g_fp16_commons
  num_gpu=4
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

sst-2_dgxa100-80g_8gpu_fp16 () {
  commons
  sst-2_commons
  dgxa100-80g_fp16_commons
  num_gpu=8
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

# TF32

sst-2_dgxa100-80g_1gpu_tf32 () {
  commons
  sst-2_commons
  dgxa100-80g_tf32_commons
  num_gpu=1
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

sst-2_dgxa100-80g_2gpu_tf32 () {
  commons
  sst-2_commons
  dgxa100-80g_tf32_commons
  num_gpu=2
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

sst-2_dgxa100-80g_4gpu_tf32 () {
  commons
  sst-2_commons
  dgxa100-80g_tf32_commons
  num_gpu=4
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

sst-2_dgxa100-80g_8gpu_tf32 () {
  commons
  sst-2_commons
  dgxa100-80g_tf32_commons
  num_gpu=8
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

##########################################
#                DGX1 32G                #
##########################################

##########################
#          MRPC          #
##########################

# AMP

mrpc_dgx1-32g_1gpu_fp16 () {
  commons
  mrpc_commons
  dgx1-32g_fp16_commons
  num_gpu=1
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

mrpc_dgx1-32g_2gpu_fp16 () {
  commons
  mrpc_commons
  dgx1-32g_fp16_commons
  num_gpu=2
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

mrpc_dgx1-32g_4gpu_fp16 () {
  commons
  mrpc_commons
  dgx1-32g_fp16_commons
  num_gpu=4
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

mrpc_dgx1-32g_8gpu_fp16 () {
  commons
  mrpc_commons
  dgx1-32g_fp16_commons
  num_gpu=8
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

# FP32

mrpc_dgx1-32g_1gpu_fp32 () {
  commons
  mrpc_commons
  dgx1-32g_fp32_commons
  num_gpu=1
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

mrpc_dgx1-32g_2gpu_fp32 () {
  commons
  mrpc_commons
  dgx1-32g_fp32_commons
  num_gpu=2
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order

}

mrpc_dgx1-32g_4gpu_fp32 () {
  commons
  mrpc_commons
  dgx1-32g_fp32_commons
  num_gpu=4
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

mrpc_dgx1-32g_8gpu_fp32 () {
  commons
  mrpc_commons
  dgx1-32g_fp32_commons
  num_gpu=8
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

##########################
#          SST-2         #
##########################

# AMP

sst-2_dgx1-32g_1gpu_fp16 () {
  commons
  sst-2_commons
  dgx1-32g_fp16_commons
  num_gpu=1
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

sst-2_dgx1-32g_2gpu_fp16 () {
  commons
  sst-2_commons
  dgx1-32g_fp16_commons
  num_gpu=2
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

sst-2_dgx1-32g_4gpu_fp16 () {
  commons
  sst-2_commons
  dgx1-32g_fp16_commons
  num_gpu=4
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

sst-2_dgx1-32g_8gpu_fp16 () {
  commons
  sst-2_commons
  dgx1-32g_fp16_commons
  num_gpu=8
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

# FP32

sst-2_dgx1-32g_fp32_commons () {
  global_batch_size=512
}

sst-2_dgx1-32g_1gpu_fp32 () {
  commons
  sst-2_commons
  dgx1-32g_fp32_commons
  sst-2_dgx1-32g_fp32_commons
  num_gpu=1
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

sst-2_dgx1-32g_2gpu_fp32 () {
  commons
  sst-2_commons
  dgx1-32g_fp32_commons
  sst-2_dgx1-32g_fp32_commons
  num_gpu=2
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

sst-2_dgx1-32g_4gpu_fp32 () {
  commons
  sst-2_commons
  dgx1-32g_fp32_commons
  sst-2_dgx1-32g_fp32_commons
  num_gpu=4
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

sst-2_dgx1-32g_8gpu_fp32 () {
  commons
  sst-2_commons
  dgx1-32g_fp32_commons
  sst-2_dgx1-32g_fp32_commons
  num_gpu=8
  batch_size_and_gradient_accumulation_steps
  print_arguments_in_order
}

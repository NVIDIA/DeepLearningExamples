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

# XSUM dataset summarization configurations for NVIDIA DGX A100 (8x NVIDIA A100 40GB GPU)

dgxa100_8gpu_bf16 ()
{
  DATA_DIR=data/xsum
  CKPT_PATH=data/nvidia_pretrained/bart_large/
  CONFIG_PATH=configs/config_xsum.json
  NUM_GPU=8
  LR=1.25e-4
  BS=40
  ACCUM=1
  PRECISION="bf16"
  TRAIN_STEPS=2000
  WARMUP_STEPS=50
  MAX_SOURCE_LEN=1024
  MAX_TARGET_LEN=60
  EVAL_BEAMS=6
  EVAL_BS=128
  PRED_BS=128
  PRELN=true

  echo $DATA_DIR $CKPT_PATH $CONFIG_PATH $NUM_GPU $LR $BS $ACCUM $PRECISION $TRAIN_STEPS $WARMUP_STEPS $MAX_SOURCE_LEN $MAX_TARGET_LEN $EVAL_BEAMS $EVAL_BS $PRED_BS $PRELN
}

dgxa100_8gpu_bf16_eval ()
{
  DATA_DIR=data/xsum
  CONFIG_PATH=configs/config_xsum.json
  NUM_GPU=8
  PRECISION="bf16"
  MAX_SOURCE_LEN=1024
  MAX_TARGET_LEN=60
  EVAL_BEAMS=6
  PRED_BS=128

  echo $PRED_BS $NUM_GPU $PRECISION $EVAL_BEAMS $MAX_SOURCE_LEN $MAX_TARGET_LEN $DATA_DIR $CONFIG_PATH
}

dgxa100_8gpu_tf32 ()
{
  DATA_DIR=data/xsum
  CKPT_PATH=data/nvidia_pretrained/bart_large/
  CONFIG_PATH=configs/config_xsum.json
  NUM_GPU=8
  LR=1.25e-4
  BS=24
  ACCUM=1
  PRECISION="tf32"
  TRAIN_STEPS=3333
  WARMUP_STEPS=50
  MAX_SOURCE_LEN=1024
  MAX_TARGET_LEN=60
  EVAL_BEAMS=6
  EVAL_BS=128
  PRED_BS=64
  PRELN=true

  echo $DATA_DIR $CKPT_PATH $CONFIG_PATH $NUM_GPU $LR $BS $ACCUM $PRECISION $TRAIN_STEPS $WARMUP_STEPS $MAX_SOURCE_LEN $MAX_TARGET_LEN $EVAL_BEAMS $EVAL_BS $PRED_BS $PRELN
}

dgxa100_8gpu_tf32_eval ()
{
  DATA_DIR=data/xsum
  CONFIG_PATH=configs/config_xsum.json
  NUM_GPU=8
  PRECISION="tf32"
  MAX_SOURCE_LEN=1024
  MAX_TARGET_LEN=60
  EVAL_BEAMS=6
  PRED_BS=64

  echo $PRED_BS $NUM_GPU $PRECISION $EVAL_BEAMS $MAX_SOURCE_LEN $MAX_TARGET_LEN $DATA_DIR $CONFIG_PATH
}

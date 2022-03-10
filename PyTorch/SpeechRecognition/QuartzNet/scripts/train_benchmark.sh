#!/bin/bash

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

set -a

: ${DATA_DIR:=${1:-"/datasets/LibriSpeech"}}
: ${OUTPUT_DIR:=${3:-"/results"}}
: ${TRAIN_MANIFESTS:="$DATA_DIR/librispeech-train-clean-100-wav.json"}

: ${BENCHMARK_EPOCHS:=20}
: ${EPOCHS:=100000}
: ${RESUME:=false}
: ${SAVE_FREQUENCY:=100000}
: ${EVAL_FREQUENCY:=100000}
: ${LEARNING_RATE:=0.0001}

: ${AMP:=false}
: ${EMA:=0}
: ${DALI_DEVICE:="gpu"}
: ${NUM_GPUS_SEQ:="8 4 1"}
: ${ACC_BATCH_SIZE:="144"}
: ${GRAD_ACC_SEQ:="4 2"}

# A range of batch lengths for LibriSpeech
# with continuous speed perturbation (0.85, 1.15) and max duration 16.7s
: ${PRE_ALLOCATE:="1408 1920"}

for NUM_GPUS in $NUM_GPUS_SEQ; do
  for GRAD_ACCUMULATION in $GRAD_ACC_SEQ; do

    # Scale the number of epochs to the number of GPUs
    BMARK=$((BENCHMARK_EPOCHS * NUM_GPUS / 8))
    BMARK=$((BMARK < 2 ? 2 : BMARK))
    BMARK=$((BMARK > BENCHMARK_EPOCHS ? BENCHMARK_EPOCHS : BMARK))
    EPOCHS_THIS_JOB=$((BMARK + 1))

    GPU_BATCH_SIZE=$((ACC_BATCH_SIZE / $GRAD_ACCUMULATION * 8 / $NUM_GPUS))

    LOG_FILE="$OUTPUT_DIR/perf-train_dali-${DALI_DEVICE}_amp-${AMP}_"
    LOG_FILE+="1x${NUM_GPUS}x${GPU_BATCH_SIZE}x${GRAD_ACCUMULATION}.json"
    BENCHMARK_EPOCHS=$BMARK bash ./scripts/train.sh "$@"

  done
done

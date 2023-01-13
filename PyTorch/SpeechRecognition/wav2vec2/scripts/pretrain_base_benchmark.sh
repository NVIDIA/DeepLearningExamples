#!/usr/bin/env bash

# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

: ${NUM_WARMUP_EPOCHS:=2}  # Number of warmup epochs
: ${NUM_EPOCHS:=5}         # Number of epochs for collecting perf measurements
: ${TRAIN_SUBSET:="train-full-960"}

: ${FP16:=false}
: ${BF16:=false}
: ${NUM_GPUS:=8}
: ${MAX_TOKENS:=1400000}
: ${NUM_CONCAT_BATCHES:=8}
: ${UPDATE_FREQ:=1}

if [ "$FP16" = true ]; then PREC=fp16; elif [ "$BF16" = true ]; then PREC=bf16; else PREC=fp32; fi
: ${OUTPUT_DIR:="results/pretrain_base_benchmark_${NUM_GPUS}x${MAX_TOKENS}x${NUM_CONCAT_BATCHES}x${UPDATE_FREQ}_${PREC}"}

NO_SAVE=true
EPOCHS_THIS_JOB=$(($NUM_EPOCHS + $NUM_WARMUP_EPOCHS))
ARGS+=" --benchmark_epochs_num $NUM_EPOCHS"

bash scripts/pretrain_base.sh

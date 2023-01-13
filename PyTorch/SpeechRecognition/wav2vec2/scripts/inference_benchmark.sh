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

: ${DATASET_DIR:="/datasets/LibriSpeech"}
: ${VALID_SUBSET:="test-other"}

: ${BF16:=false}
: ${FP16:=false}
: ${NUM_GPUS:=1}
: ${BATCH_SIZE:=1}
: ${NUM_REPEATS:=10}
: ${NUM_WARMUP_REPEATS:=2}

if [ "$FP16" = true ]; then PREC=fp16; elif [ "$BF16" = true ]; then PREC=bf16; else PREC=fp32; fi
: ${OUTPUT_DIR:="results/base_inference_benchmark_bs${BATCH_SIZE}_${PREC}"}

NUM_SAMPLES=$(cat $DATASET_DIR/$VALID_SUBSET.ltr | wc -l)
NUM_BATCHES=$(((NUM_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE))

NUM_STEPS=$(($NUM_BATCHES * $NUM_REPEATS))
NUM_WARMUP_STEPS=$(($NUM_BATCHES * $NUM_WARMUP_REPEATS))

bash scripts/inference.sh

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

# measure on speed perturbed data, but so slightly that fbank length remains the same
# with pad_to_max_duration, this reduces cuDNN benchmak's burn-in period to a single step
: ${DATA_DIR:=${1:-"/datasets/LibriSpeech"}}
: ${OUTPUT_DIR:=${3:-"/results"}}
: ${TRAIN_MANIFESTS:="$DATA_DIR/librispeech-train-clean-100-wav.json"}

# run for a number of epochs, but don't finalize the training
: ${EPOCHS_THIS_JOB:=2}
: ${EPOCHS:=100000}
: ${RESUME:=false}
: ${SAVE_FREQUENCY:=100000}
: ${EVAL_FREQUENCY:=100000}
: ${GRAD_ACCUMULATION_STEPS:=1}

: ${AMP:=false}
: ${EMA:=0}
: ${DALI_DEVICE:="gpu"}
: ${NUM_GPUS_SEQ:="1 4 8"}
: ${BATCH_SIZE_SEQ:="32"}
# A probable range of batch lengths for LibriSpeech
# with BS=64 and continuous speed perturbation (0.85, 1.15)
: ${PRE_ALLOCATE:="1408 1920"}

for NUM_GPUS in $NUM_GPUS_SEQ; do
  for BATCH_SIZE in $BATCH_SIZE_SEQ; do

    LOG_FILE="$OUTPUT_DIR/perf-train_dali-${DALI_DEVICE}_amp-${AMP}_ngpus${NUM_GPUS}_bs${BATCH_SIZE}.json"
    bash ./scripts/train.sh "$@"

  done
done

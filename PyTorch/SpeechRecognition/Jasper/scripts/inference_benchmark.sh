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

: ${OUTPUT_DIR:=${3:-"/results"}}
: ${CUDNN_BENCHMARK:=true}
: ${PAD_TO_MAX_DURATION:=true}
: ${PAD_LEADING:=0}
: ${NUM_WARMUP_STEPS:=10}
: ${NUM_STEPS:=500}

: ${AMP:=false}
: ${DALI_DEVICE:="cpu"}
: ${BATCH_SIZE_SEQ:="1 2 4 8 16"}
: ${MAX_DURATION_SEQ:="2 7 16.7"}

for MAX_DURATION in $MAX_DURATION_SEQ; do
  for BATCH_SIZE in $BATCH_SIZE_SEQ; do

    LOG_FILE="$OUTPUT_DIR/perf-infer_dali-${DALI_DEVICE}_amp-${AMP}_dur${MAX_DURATION}_bs${BATCH_SIZE}.json"
    bash ./scripts/inference.sh "$@"

  done
done

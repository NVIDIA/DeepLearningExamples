#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Performs inference and measures latency and accuracy of TRT and PyTorch implementations of JASPER.

echo "Container nvidia build = " $NVIDIA_BUILD_ID

# Mandatory Arguments
CHECKPOINT=$CHECKPOINT

# Arguments with Defaults
DATA_DIR=${DATA_DIR:-"/datasets/LibriSpeech"}
DATASET=${DATASET:-"dev-clean"}
RESULT_DIR=${RESULT_DIR:-"/results"}
CREATE_LOGFILE=${CREATE_LOGFILE:-"true"}
TRT_PRECISION=${TRT_PRECISION:-"fp32"}
PYTORCH_PRECISION=${PYTORCH_PRECISION:-"fp32"}
NUM_STEPS=${NUM_STEPS:-"-1"}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_FRAMES=${NUM_FRAMES:-3600}
MAX_SEQUENCE_LENGTH_FOR_ENGINE=${MAX_SEQUENCE_LENGTH_FOR_ENGINE:-$NUM_FRAMES}
FORCE_ENGINE_REBUILD=${FORCE_ENGINE_REBUILD:-"true"}
CSV_PATH=${CSV_PATH:-"/results/res.csv"}
TRT_PREDICTION_PATH=${TRT_PREDICTION_PATH:-"/results/trt_predictions.txt"}
PYT_PREDICTION_PATH=${PYT_PREDICTION_PATH:-"/results/pyt_predictions.txt"}
VERBOSE=${VERBOSE:-"false"}



export CHECKPOINT="$CHECKPOINT"
export DATA_DIR="$DATA_DIR"
export DATASET="$DATASET"
export RESULT_DIR="$RESULT_DIR"
export CREATE_LOGFILE="$CREATE_LOGFILE"
export TRT_PRECISION="$TRT_PRECISION"
export PYTORCH_PRECISION="$PYTORCH_PRECISION"
export NUM_STEPS="$NUM_STEPS"
export BATCH_SIZE="$BATCH_SIZE"
export NUM_FRAMES="$NUM_FRAMES"
export MAX_SEQUENCE_LENGTH_FOR_ENGINE="$MAX_SEQUENCE_LENGTH_FOR_ENGINE"
export FORCE_ENGINE_REBUILD="$FORCE_ENGINE_REBUILD"
export CSV_PATH="$CSV_PATH"
export TRT_PREDICTION_PATH="$TRT_PREDICTION_PATH"
export PYT_PREDICTION_PATH="$PYT_PREDICTION_PATH"
export VERBOSE="$VERBOSE"

bash ./trt/scripts/trt_inference_benchmark.sh $1 $2 $3 $4 $5 $6 $7

trt_word_error_rate=`cat "$CSV_PATH" | awk '{print $3}'`
pyt_word_error_rate=`cat "$CSV_PATH" | awk '{print $4}'`

echo "word error rate for native PyTorch inference: "
echo "${pyt_word_error_rate}"
echo "word error rate for native TRT inference: "
echo "${trt_word_error_rate}"

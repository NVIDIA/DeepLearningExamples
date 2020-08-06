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


echo "NVIDIA container build: ${NVIDIA_BUILD_ID}"

DATA_DIR=${1:-${DATA_DIR:-"/datasets/LibriSpeech"}}
DATASET=${2:-${DATASET:-"dev-clean"}}
MODEL_CONFIG=${3:-${MODEL_CONFIG:-"configs/jasper10x5dr_sp_offline_specaugment.toml"}}
RESULT_DIR=${4:-${RESULT_DIR:-"/results"}}
CHECKPOINT=${5:-${CHECKPOINT:-"/checkpoints/jasper_fp16.pt"}}
CREATE_LOGFILE=${6:-${CREATE_LOGFILE:-"true"}}
CUDNN_BENCHMARK=${7:-${CUDNN_BENCHMARK:-"false"}}
AMP=${8:-${AMP:-"false"}}
NUM_STEPS=${9:-${NUM_STEPS:-"-1"}}
SEED=${10:-${SEED:-0}}
BATCH_SIZE=${11:-${BATCH_SIZE:-64}}
LOGITS_FILE=${12:-${LOGITS_FILE:-""}}
PREDICTION_FILE=${13:-${PREDICTION_FILE:-"${RESULT_DIR}/${DATASET}.predictions"}}
CPU=${14:-${CPU:-"false"}}
EMA=${14:-${EMA:-"false"}}

mkdir -p "$RESULT_DIR"

CMD="python inference.py "
CMD+=" --batch_size $BATCH_SIZE "
CMD+=" --dataset_dir $DATA_DIR "
CMD+=" --val_manifest $DATA_DIR/librispeech-${DATASET}-wav.json "
CMD+=" --model_toml $MODEL_CONFIG  "
CMD+=" --seed $SEED "
[ "$NUM_STEPS" -gt 0 ] && \
CMD+=" --steps $NUM_STEPS"
[ "$CUDNN_BENCHMARK" = "true" ] && \
CMD+=" --cudnn"
[ "$AMP" == "true" ] && \
CMD+=" --amp"
[ "$CPU" == "true" ] && \
CMD+=" --cpu"
[ "$EMA" == "true" ] && \
CMD+=" --ema"
[ -n "$CHECKPOINT" ] && \
CMD+=" --ckpt=${CHECKPOINT}"
[ -n "$PREDICTION_FILE" ] && \
CMD+=" --save_prediction $PREDICTION_FILE"
[ -n "$LOGITS_FILE" ] && \
CMD+=" --logits_save_to $LOGITS_FILE"

if [ "$CREATE_LOGFILE" = "true" ] ; then
   export GBS=$(expr $BATCH_SIZE)
   printf -v TAG "jasper_train_benchmark_amp-%s_gbs%d" "$AMP" $GBS
   DATESTAMP=`date +'%y%m%d%H%M%S'`
   LOGFILE="${RESULT_DIR}/${TAG}.${DATESTAMP}.log"
   printf "Logs written to %s\n" "$LOGFILE"
fi

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee "$LOGFILE"
fi
set +x
[ -n "$PREDICTION_FILE" ] && echo "PREDICTION_FILE: ${PREDICTION_FILE}"
[ -n "$LOGITS_FILE" ] && echo "LOGITS_FILE: ${LOGITS_FILE}"

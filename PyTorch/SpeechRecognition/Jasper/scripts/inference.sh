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

: ${DATA_DIR:=${1:-"/datasets/LibriSpeech"}}
: ${MODEL_CONFIG:=${2:-"configs/jasper10x5dr_speedp-online_speca.yaml"}}
: ${OUTPUT_DIR:=${3:-"/results"}}
: ${CHECKPOINT:=${4:-"/checkpoints/jasper_fp16.pt"}}
: ${DATASET:="test-other"}
: ${LOG_FILE:=""}
: ${CUDNN_BENCHMARK:=false}
: ${MAX_DURATION:=""}
: ${PAD_TO_MAX_DURATION:=false}
: ${PAD_LEADING:=16}
: ${NUM_GPUS:=1}
: ${NUM_STEPS:=0}
: ${NUM_WARMUP_STEPS:=0}
: ${AMP:=false}
: ${BATCH_SIZE:=64}
: ${EMA:=true}
: ${SEED:=0}
: ${DALI_DEVICE:="gpu"}
: ${CPU:=false}
: ${LOGITS_FILE:=}
: ${PREDICTION_FILE:="${OUTPUT_DIR}/${DATASET}.predictions"}

mkdir -p "$OUTPUT_DIR"

ARGS="--dataset_dir=$DATA_DIR"
ARGS+=" --val_manifest=$DATA_DIR/librispeech-${DATASET}-wav.json"
ARGS+=" --model_config=$MODEL_CONFIG"
ARGS+=" --output_dir=$OUTPUT_DIR"
ARGS+=" --batch_size=$BATCH_SIZE"
ARGS+=" --seed=$SEED"
ARGS+=" --dali_device=$DALI_DEVICE"
ARGS+=" --steps $NUM_STEPS"
ARGS+=" --warmup_steps $NUM_WARMUP_STEPS"
ARGS+=" --pad_leading $PAD_LEADING"

[ "$AMP" = true ] &&                 ARGS+=" --amp"
[ "$EMA" = true ] &&                 ARGS+=" --ema"
[ "$CUDNN_BENCHMARK" = true ] &&     ARGS+=" --cudnn_benchmark"
[ -n "$CHECKPOINT" ] &&              ARGS+=" --ckpt=${CHECKPOINT}"
[ -n "$LOG_FILE" ] &&                ARGS+=" --log_file $LOG_FILE"
[ -n "$PREDICTION_FILE" ] &&         ARGS+=" --save_prediction $PREDICTION_FILE"
[ -n "$LOGITS_FILE" ] &&             ARGS+=" --logits_save_to $LOGITS_FILE"
[ "$CPU" == "true" ] &&              ARGS+=" --cpu"
[ -n "$MAX_DURATION" ] &&            ARGS+=" --override_config input_val.audio_dataset.max_duration=$MAX_DURATION" \
                                     ARGS+=" --override_config input_val.filterbank_features.max_duration=$MAX_DURATION"
[ "$PAD_TO_MAX_DURATION" = true ] && ARGS+=" --override_config input_val.audio_dataset.pad_to_max_duration=True" \
                                     ARGS+=" --override_config input_val.filterbank_features.pad_to_max_duration=True"

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS inference.py $ARGS

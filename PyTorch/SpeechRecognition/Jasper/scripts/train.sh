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

export OMP_NUM_THREADS=1

: ${DATA_DIR:=${1:-"/datasets/LibriSpeech"}}
: ${MODEL_CONFIG:=${2:-"configs/jasper10x5dr_speedp-online_speca.yaml"}}
: ${OUTPUT_DIR:=${3:-"/results"}}
: ${CHECKPOINT:=${4:-}}
: ${RESUME:=true}
: ${CUDNN_BENCHMARK:=true}
: ${NUM_GPUS:=8}
: ${AMP:=false}
: ${BATCH_SIZE:=64}
: ${GRAD_ACCUMULATION_STEPS:=2}
: ${LEARNING_RATE:=0.01}
: ${MIN_LEARNING_RATE:=0.00001}
: ${LR_POLICY:=exponential}
: ${LR_EXP_GAMMA:=0.981}
: ${EMA:=0.999}
: ${SEED:=0}
: ${EPOCHS:=440}
: ${WARMUP_EPOCHS:=2}
: ${HOLD_EPOCHS:=140}
: ${SAVE_FREQUENCY:=10}
: ${EPOCHS_THIS_JOB:=0}
: ${DALI_DEVICE:="gpu"}
: ${PAD_TO_MAX_DURATION:=false}
: ${EVAL_FREQUENCY:=544}
: ${PREDICTION_FREQUENCY:=544}
: ${TRAIN_MANIFESTS:="$DATA_DIR/librispeech-train-clean-100-wav.json \
                      $DATA_DIR/librispeech-train-clean-360-wav.json \
                      $DATA_DIR/librispeech-train-other-500-wav.json"}
: ${VAL_MANIFESTS:="$DATA_DIR/librispeech-dev-clean-wav.json"}

mkdir -p "$OUTPUT_DIR"

ARGS="--dataset_dir=$DATA_DIR"
ARGS+=" --val_manifests $VAL_MANIFESTS"
ARGS+=" --train_manifests $TRAIN_MANIFESTS"
ARGS+=" --model_config=$MODEL_CONFIG"
ARGS+=" --output_dir=$OUTPUT_DIR"
ARGS+=" --lr=$LEARNING_RATE"
ARGS+=" --batch_size=$BATCH_SIZE"
ARGS+=" --min_lr=$MIN_LEARNING_RATE"
ARGS+=" --lr_policy=$LR_POLICY"
ARGS+=" --lr_exp_gamma=$LR_EXP_GAMMA"
ARGS+=" --epochs=$EPOCHS"
ARGS+=" --warmup_epochs=$WARMUP_EPOCHS"
ARGS+=" --hold_epochs=$HOLD_EPOCHS"
ARGS+=" --epochs_this_job=$EPOCHS_THIS_JOB"
ARGS+=" --ema=$EMA"
ARGS+=" --seed=$SEED"
ARGS+=" --optimizer=novograd"
ARGS+=" --weight_decay=1e-3"
ARGS+=" --save_frequency=$SAVE_FREQUENCY"
ARGS+=" --keep_milestones 100 200 300 400"
ARGS+=" --save_best_from=380"
ARGS+=" --log_frequency=1"
ARGS+=" --eval_frequency=$EVAL_FREQUENCY"
ARGS+=" --prediction_frequency=$PREDICTION_FREQUENCY"
ARGS+=" --grad_accumulation_steps=$GRAD_ACCUMULATION_STEPS "
ARGS+=" --dali_device=$DALI_DEVICE"

[ "$AMP" = true ] &&                 ARGS+=" --amp"
[ "$RESUME" = true ] &&              ARGS+=" --resume"
[ "$CUDNN_BENCHMARK" = true ] &&     ARGS+=" --cudnn_benchmark"
[ -n "$MAX_DURATION" ] &&            ARGS+=" --override_config input_train.audio_dataset.max_duration=$MAX_DURATION" \
                                     ARGS+=" --override_config input_train.filterbank_features.max_duration=$MAX_DURATION"
[ "$PAD_TO_MAX_DURATION" = true ] && ARGS+=" --override_config input_train.audio_dataset.pad_to_max_duration=True" \
                                     ARGS+=" --override_config input_train.filterbank_features.pad_to_max_duration=True"
[ -n "$CHECKPOINT" ] &&              ARGS+=" --ckpt=$CHECKPOINT"
[ -n "$LOG_FILE" ] &&                ARGS+=" --log_file $LOG_FILE"
[ -n "$PRE_ALLOCATE" ] &&            ARGS+=" --pre_allocate_range $PRE_ALLOCATE"

DISTRIBUTED="-m torch.distributed.launch --nproc_per_node=$NUM_GPUS"
python $DISTRIBUTED train.py $ARGS

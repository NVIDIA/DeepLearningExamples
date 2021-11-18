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
: ${MODEL_CONFIG:=${2:-"configs/quartznet15x5_speedp-online-1.15_speca.yaml"}}
: ${OUTPUT_DIR:=${3:-"/results"}}
: ${CHECKPOINT:=${4:-}}
: ${CUDNN_BENCHMARK:=true}
: ${NUM_GPUS:=8}
: ${AMP:=false}
: ${GPU_BATCH_SIZE:=72}
: ${GRAD_ACCUMULATION:=2}
: ${OPTIMIZER:=fused_novograd}
: ${LEARNING_RATE:=0.01}
: ${LR_POLICY:=exponential}
: ${LR_EXP_GAMMA:=0.981}
: ${EMA:=0.999}
: ${MULTI_TENSOR_EMA:=true}
: ${SEED:=0}
: ${EPOCHS:=260}
: ${WARMUP_EPOCHS:=2}
: ${HOLD_EPOCHS:=140}
: ${SAVE_FREQUENCY:=10}
: ${EPOCHS_THIS_JOB:=0}
: ${DALI_DEVICE:="gpu"}
: ${PAD_TO_MAX_DURATION:=false}
: ${EVAL_FREQUENCY:=241}
: ${PREDICTION_FREQUENCY:=241}
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
ARGS+=" --gpu_batch_size=$GPU_BATCH_SIZE"
ARGS+=" --min_lr=1e-5"
ARGS+=" --lr_policy=$LR_POLICY"
ARGS+=" --lr_exp_gamma=$LR_EXP_GAMMA"
ARGS+=" --epochs=$EPOCHS"
ARGS+=" --warmup_epochs=$WARMUP_EPOCHS"
ARGS+=" --hold_epochs=$HOLD_EPOCHS"
ARGS+=" --epochs_this_job=$EPOCHS_THIS_JOB"
ARGS+=" --ema=$EMA"
ARGS+=" --seed=$SEED"
ARGS+=" --optimizer=$OPTIMIZER"
ARGS+=" --weight_decay=1e-3"
ARGS+=" --resume"
ARGS+=" --save_frequency=$SAVE_FREQUENCY"
ARGS+=" --keep_milestones 100 200"
ARGS+=" --save_best_from=200"
ARGS+=" --log_frequency=1"
ARGS+=" --eval_frequency=$EVAL_FREQUENCY"
ARGS+=" --prediction_frequency=$PREDICTION_FREQUENCY"
ARGS+=" --grad_accumulation=$GRAD_ACCUMULATION "
ARGS+=" --dali_device=$DALI_DEVICE"

[ "$AMP" = true ] &&                 ARGS+=" --amp"
[ "$CUDNN_BENCHMARK" = true ] &&     ARGS+=" --cudnn_benchmark"
[ -n "$MAX_DURATION" ] &&            ARGS+=" --override_config input_train.audio_dataset.max_duration=$MAX_DURATION" \
                                     ARGS+=" --override_config input_train.filterbank_features.max_duration=$MAX_DURATION"
[ "$PAD_TO_MAX_DURATION" = true ] && ARGS+=" --override_config input_train.audio_dataset.pad_to_max_duration=True" \
                                     ARGS+=" --override_config input_train.filterbank_features.pad_to_max_duration=True"
[ -n "$CHECKPOINT" ] &&              ARGS+=" --ckpt=${CHECKPOINT}"
[ -n "$LOG_FILE" ] &&                ARGS+=" --log_file $LOG_FILE"
[ -n "$PRE_ALLOCATE" ] &&            ARGS+=" --pre_allocate_range $PRE_ALLOCATE"
[ "$MULTI_TENSOR_EMA" = true ] &&    ARGS+=" --multi_tensor_ema"
[ -n "$BENCHMARK_EPOCHS" ] &&        ARGS+=" --benchmark_epochs_num=$BENCHMARK_EPOCHS"

GBS=$(($NUM_GPUS * $GPU_BATCH_SIZE * $GRAD_ACCUMULATION))
if [ $GBS -ne $((8 * 144)) ]; then
    echo -e "\nWARNING: Global batch size changed from $((8 * 144)) to ${GBS}."
    sleep 3
fi
echo -e "\nAMP=$AMP,""${NUM_GPUS}x${GPU_BATCH_SIZE}x${GRAD_ACCUMULATION}" \
        "(global batch size ${GBS})\n"

: ${DISTRIBUTED:="-m torch.distributed.launch --nproc_per_node=$NUM_GPUS"}
python $DISTRIBUTED train.py $ARGS

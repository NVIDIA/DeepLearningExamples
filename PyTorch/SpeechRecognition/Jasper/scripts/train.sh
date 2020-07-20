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
MODEL_CONFIG=${2:-${MODEL_CONFIG:-"configs/jasper10x5dr_sp_offline_specaugment.toml"}}
RESULT_DIR=${3:-${RESULT_DIR:-"/results"}}
CHECKPOINT=${4:-${CHECKPOINT:-""}}
CREATE_LOGFILE=${5:-${CREATE_LOGFILE:-"true"}}
CUDNN_BENCHMARK=${6:-${CUDNN_BENCHMARK:-"true"}}
NUM_GPUS=${7:-${NUM_GPUS:-8}}
AMP=${8:-${AMP:-"false"}}
EPOCHS=${9:-${EPOCHS:-400}}
SEED=${10:-${SEED:-6}}
BATCH_SIZE=${11:-${BATCH_SIZE:-64}}
LEARNING_RATE=${12:-${LEARNING_RATE:-"0.015"}}
GRADIENT_ACCUMULATION_STEPS=${13:-${GRADIENT_ACCUMULATION_STEPS:-2}}
EMA=${EMA:-0.999}
SAVE_FREQUENCY=${SAVE_FREQUENCY:-10}

mkdir -p "$RESULT_DIR"

CMD="python3 -m torch.distributed.launch --nproc_per_node=$NUM_GPUS"
CMD+=" train.py"
CMD+=" --batch_size=$BATCH_SIZE"
CMD+=" --num_epochs=$EPOCHS"
CMD+=" --output_dir=$RESULT_DIR"
CMD+=" --model_toml=$MODEL_CONFIG"
CMD+=" --lr=$LEARNING_RATE"
CMD+=" --ema=$EMA"
CMD+=" --seed=$SEED"
CMD+=" --optimizer=novograd"
CMD+=" --dataset_dir=$DATA_DIR"
CMD+=" --val_manifest=$DATA_DIR/librispeech-dev-clean-wav.json"
CMD+=" --train_manifest=$DATA_DIR/librispeech-train-clean-100-wav.json,"
CMD+="$DATA_DIR/librispeech-train-clean-360-wav.json,"
CMD+="$DATA_DIR/librispeech-train-other-500-wav.json"
CMD+=" --weight_decay=1e-3"
CMD+=" --save_freq=$SAVE_FREQUENCY"
CMD+=" --eval_freq=100"
CMD+=" --train_freq=1"
CMD+=" --lr_decay"
CMD+=" --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS "

[ "$AMP" == "true" ] && \
CMD+=" --amp"
[ "$CUDNN_BENCHMARK" = "true" ] && \
CMD+=" --cudnn"
[ -n "$CHECKPOINT" ] && \
CMD+=" --ckpt=${CHECKPOINT}"

if [ "$CREATE_LOGFILE" = "true" ] ; then
   export GBS=$(expr $BATCH_SIZE \* $NUM_GPUS)
   printf -v TAG "jasper_train_benchmark_amp-%s_gbs%d" "$AMP" $GBS
   DATESTAMP=`date +'%y%m%d%H%M%S'`
   LOGFILE=$RESULT_DIR/$TAG.$DATESTAMP.log
   printf "Logs written to %s\n" "$LOGFILE"
fi

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi
set +x

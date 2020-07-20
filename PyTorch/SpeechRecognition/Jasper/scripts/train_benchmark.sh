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

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_DIR=${SCRIPT_DIR}/..

DATA_DIR=${1:-${DATA_DIR:-"/datasets/LibriSpeech"}}
MODEL_CONFIG=${2:-${MODEL_CONFIG:-"configs/jasper10x5dr_sp_offline_specaugment.toml"}}
RESULT_DIR=${3:-${RESULT_DIR:-"/results"}}
CREATE_LOGFILE=${4:-${CREATE_LOGFILE:-"true"}}
CUDNN_BENCHMARK=${5:-${CUDNN_BENCHMARK:-"true"}}
NUM_GPUS=${6:-${NUM_GPUS:-8}}
AMP=${7:-${AMP:-"false"}}
NUM_STEPS=${8:-${NUM_STEPS:-"-1"}}
MAX_DURATION=${9:-${MAX_DURATION:-16.7}}
SEED=${10:-${SEED:-0}}
BATCH_SIZE=${11:-${BATCH_SIZE:-32}}
LEARNING_RATE=${12:-${LEARNING_RATE:-"0.015"}}
GRADIENT_ACCUMULATION_STEPS=${13:-${GRADIENT_ACCUMULATION_STEPS:-1}}
PRINT_FREQUENCY=${14:-${PRINT_FREQUENCY:-1}}
USE_PROFILER=${USE_PROFILER:-"false"}

mkdir -p "$RESULT_DIR"

[ "${USE_PROFILER}" = "true" ] && PYTHON_ARGS="-m cProfile -s cumtime"

CMD="${PYTHON_ARGS} ${PROJECT_DIR}/train.py"
CMD+=" --batch_size=$BATCH_SIZE"
CMD+=" --num_epochs=400"
CMD+=" --output_dir=$RESULT_DIR"
CMD+=" --model_toml=$MODEL_CONFIG"
CMD+=" --lr=$LEARNING_RATE"
CMD+=" --seed=$SEED"
CMD+=" --optimizer=novograd"
CMD+=" --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
CMD+=" --dataset_dir=$DATA_DIR"
CMD+=" --val_manifest=$DATA_DIR/librispeech-dev-clean-wav.json"
CMD+=" --train_manifest=$DATA_DIR/librispeech-train-clean-100-wav.json,"
CMD+="$DATA_DIR/librispeech-train-clean-360-wav.json,"
CMD+="$DATA_DIR/librispeech-train-other-500-wav.json"
CMD+=" --weight_decay=1e-3"
CMD+=" --save_freq=100000"
CMD+=" --eval_freq=100000"
CMD+=" --max_duration=$MAX_DURATION"
CMD+=" --pad_to_max"
CMD+=" --train_freq=$PRINT_FREQUENCY"
CMD+=" --lr_decay "
[ "$AMP" == "true" ] && \
CMD+=" --amp"
[ "$CUDNN_BENCHMARK" = "true" ] && \
CMD+=" --cudnn"
[ "$NUM_STEPS" -gt 1 ] && \
CMD+=" --num_steps=$NUM_STEPS"

if [ "$NUM_GPUS" -gt 1  ] ; then
   CMD="python3 -m torch.distributed.launch --nproc_per_node=$NUM_GPUS $CMD"
else
   CMD="python3 $CMD"
fi

if [ "$CREATE_LOGFILE" = "true" ] ; then
   export GBS=$(expr $BATCH_SIZE \* $NUM_GPUS)
   printf -v TAG "jasper_train_benchmark_amp-%s_gbs%d" "$AMP" $GBS
   DATESTAMP=`date +'%y%m%d%H%M%S'`
   LOGFILE="${RESULT_DIR}/${TAG}.${DATESTAMP}.log"
   printf "Logs written to %s\n" "$LOGFILE"
fi

if [ -z "$LOGFILE" ] ; then

   set -x
   $CMD
   set +x
else

   set -x
   (
     $CMD
   ) |& tee "$LOGFILE"

   set +x

   mean_latency=`cat "$LOGFILE" | grep 'Step time' | awk '{print $3}'  | tail -n +2 | egrep -o '[0-9.]+'| awk 'BEGIN {total=0} {total+=$1} END {printf("%.2f\n",total/NR)}'`
   mean_throughput=`python -c "print($BATCH_SIZE*$NUM_GPUS/${mean_latency})"`
   training_wer_per_pgu=`cat "$LOGFILE" | grep 'training_batch_WER'| awk '{print $2}'  | tail -n 1 | egrep -o '[0-9.]+'`
   training_loss_per_pgu=`cat "$LOGFILE" | grep 'Loss@Step'| awk '{print $4}'  | tail -n 1 | egrep -o '[0-9.]+'`
   final_eval_wer=`cat "$LOGFILE" | grep 'Evaluation WER'| tail -n 1 | egrep -o '[0-9.]+'`
   final_eval_loss=`cat "$LOGFILE" | grep 'Evaluation Loss'| tail -n 1 | egrep -o '[0-9.]+'`

   echo "max duration: $MAX_DURATION s" | tee -a "$LOGFILE"
   echo "mean_latency: $mean_latency s" | tee -a "$LOGFILE"
   echo "mean_throughput: $mean_throughput sequences/s" | tee -a "$LOGFILE"
   echo "training_wer_per_pgu: $training_wer_per_pgu" | tee -a "$LOGFILE"
   echo "training_loss_per_pgu: $training_loss_per_pgu" | tee -a "$LOGFILE"
   echo "final_eval_loss: $final_eval_loss" | tee -a "$LOGFILE"
   echo "final_eval_wer: $final_eval_wer" | tee -a "$LOGFILE"
fi

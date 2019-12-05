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


#!/bin/bash

echo "Container nvidia build = " $NVIDIA_BUILD_ID


DATA_DIR=${1:-"/datasets/LibriSpeech"}
DATASET=${2:-"dev-clean"}
MODEL_CONFIG=${3:-"configs/jasper10x5dr_sp_offline_specaugment.toml"}
RESULT_DIR=${4:-"/results"}
CHECKPOINT=${5:-"/checkpoints/jasper_fp16.pt"}
CREATE_LOGFILE=${6:-"true"}
CUDNN_BENCHMARK=${7:-"true"}
PRECISION=${8:-"fp32"}
NUM_STEPS=${9:-"-1"}
MAX_DURATION=${10:-"36"}
SEED=${11:-0}
BATCH_SIZE=${12:-64}

PREC=""
if [ "$PRECISION" = "fp16" ] ; then
   PREC="--fp16"
elif [ "$PRECISION" = "fp32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi
STEPS=""
if [ "$NUM_STEPS" -gt 0 ] ; then
   STEPS=" --steps $NUM_STEPS"
fi
if [ "$CUDNN_BENCHMARK" = "true" ] ; then
    CUDNN_BENCHMARK=" --cudnn_benchmark"
else
    CUDNN_BENCHMARK=""
fi

CMD=" python inference_benchmark.py"
CMD+=" --batch_size=$BATCH_SIZE"
CMD+=" --model_toml=$MODEL_CONFIG"
CMD+=" --seed=$SEED"
CMD+=" --dataset_dir=$DATA_DIR"
CMD+=" --val_manifest $DATA_DIR/librispeech-${DATASET}-wav.json "
CMD+=" --ckpt=$CHECKPOINT"
CMD+=" --max_duration=$MAX_DURATION"
CMD+=" --pad_to=-1"
CMD+=" $CUDNN_BENCHMARK"
CMD+=" $PREC"
CMD+=" $STEPS"


if [ "$CREATE_LOGFILE" = "true" ] ; then
  export GBS=$(expr $BATCH_SIZE )
  printf -v TAG "jasper_inference_benchmark_%s_gbs%d" "$PRECISION" $GBS
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
   grep 'latency' "$LOGFILE"
fi
set +x

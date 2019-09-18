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
MODEL_CONFIG=${2:-"configs/jasper10x5dr_sp_offline_specaugment.toml"}
RESULT_DIR=${3:-"/results"}
CHECKPOINT=${4:-"none"}
CREATE_LOGFILE=${5:-"true"}
CUDNN_BENCHMARK=${6:-"true"}
NUM_GPUS=${7:-8}
PRECISION=${8:-"fp16"}
EPOCHS=${9:-400}
SEED=${10:-6}
BATCH_SIZE=${11:-64}
LEARNING_RATE=${12:-"0.015"}
GRADIENT_ACCUMULATION_STEPS=${13:-1}
LAUNCH_OPT=${LAUNCH_OPT:-"none"}


PREC=""
if [ "$PRECISION" = "fp16" ] ; then
   PREC="--fp16"
elif [ "$PRECISION" = "fp32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

CUDNN=""
if [ "$CUDNN_BENCHMARK" = "true" ] && [ "$PRECISION" = "fp16" ]; then
   CUDNN=" --cudnn"
else
   CUDNN=""
fi



if [ "$CHECKPOINT" = "none" ] ; then
   CHECKPOINT=""
else
   CHECKPOINT=" --ckpt=${CHECKPOINT}"
fi


CMD=" train.py"
CMD+=" --batch_size=$BATCH_SIZE"
CMD+=" --num_epochs=$EPOCHS"
CMD+=" --output_dir=$RESULT_DIR"
CMD+=" --model_toml=$MODEL_CONFIG"
CMD+=" --lr=$LEARNING_RATE"
CMD+=" --seed=$SEED"
CMD+=" --optimizer=novograd"
CMD+=" --dataset_dir=$DATA_DIR"
CMD+=" --val_manifest=$DATA_DIR/librispeech-dev-clean-wav.json"
CMD+=" --train_manifest=$DATA_DIR/librispeech-train-clean-100-wav.json,$DATA_DIR/librispeech-train-clean-360-wav.json,$DATA_DIR/librispeech-train-other-500-wav.json"
CMD+=" --weight_decay=1e-3"
CMD+=" --save_freq=10"
CMD+=" --eval_freq=100"
CMD+=" --train_freq=25"
CMD+=" --lr_decay"
CMD+=" --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS "
CMD+=" $CHECKPOINT"
CMD+=" $PREC"
CMD+=" $CUDNN"


if [ "${LAUNCH_OPT}" != "none" ]; then
   CMD="python -m $LAUNCH_OPT $CMD"
elif [ "$NUM_GPUS" -gt 1  ] ; then
   CMD="python3 -m torch.distributed.launch --nproc_per_node=$NUM_GPUS $CMD"
else
   CMD="python3  $CMD"
fi



if [ "$CREATE_LOGFILE" = "true" ] ; then
  export GBS=$(expr $BATCH_SIZE \* $NUM_GPUS)
  printf -v TAG "jasper_train_%s_gbs%d" "$PRECISION" $GBS
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

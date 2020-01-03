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


DATA_DIR=${1-"/datasets/LibriSpeech"}
DATASET=${2:-"dev-clean"}
MODEL_CONFIG=${3:-"configs/jasper10x5dr_sp_offline_specaugment.toml"}
RESULT_DIR=${4:-"/results"}
CHECKPOINT=${5:-"/checkpoints/jasper_fp16.pt"}
CREATE_LOGFILE=${6:-"true"}
CUDNN_BENCHMARK=${7:-"false"}
PRECISION=${8:-"fp32"}
NUM_STEPS=${9:-"-1"}
SEED=${10:-0}
BATCH_SIZE=${11:-64}
MODELOUTPUT_FILE=${12:-"none"}
PREDICTION_FILE=${13:-"$RESULT_DIR/${DATASET}.predictions"}

if [ "$CREATE_LOGFILE" = "true" ] ; then
    export GBS=$(expr $BATCH_SIZE)
    printf -v TAG "jasper_inference_${DATASET}_%s_gbs%d" "$PRECISION" $GBS
    DATESTAMP=`date +'%y%m%d%H%M%S'`
    LOGFILE="${RESULT_DIR}/${TAG}.${DATESTAMP}.log"
    printf "Logs written to %s\n" "$LOGFILE"
fi



PREC=""
if [ "$PRECISION" = "fp16" ] ; then
    PREC="--fp16"
elif [ "$PRECISION" = "fp32" ] ; then
    PREC=""
else
    echo "Unknown <precision> argument"
    exit -2
fi

PRED=""
if [ "$PREDICTION_FILE" = "none" ] ; then
    PRED=""
else
    PRED=" --save_prediction $PREDICTION_FILE"
fi

OUTPUT=""
if [ "$MODELOUTPUT_FILE" = "none" ] ; then
    OUTPUT=" "
else
    OUTPUT=" --logits_save_to $MODELOUTPUT_FILE"
fi


if [ "$CUDNN_BENCHMARK" = "true" ]; then
    CUDNN_BENCHMARK=" --cudnn_benchmark"
else
    CUDNN_BENCHMARK=""
fi

STEPS=""
if [ "$NUM_STEPS" -gt 0 ] ; then
    STEPS=" --steps $NUM_STEPS"
fi

CMD=" python inference.py "
CMD+=" --batch_size $BATCH_SIZE "
CMD+=" --dataset_dir $DATA_DIR "
CMD+=" --val_manifest $DATA_DIR/librispeech-${DATASET}-wav.json "
CMD+=" --model_toml $MODEL_CONFIG  "
CMD+=" --seed $SEED "
CMD+=" --ckpt $CHECKPOINT "
CMD+=" $CUDNN_BENCHMARK"
CMD+=" $PRED "
CMD+=" $OUTPUT "
CMD+=" $PREC "
CMD+=" $STEPS "


set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee "$LOGFILE"
fi
set +x
echo "MODELOUTPUT_FILE: ${MODELOUTPUT_FILE}"
echo "PREDICTION_FILE: ${PREDICTION_FILE}"

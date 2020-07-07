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
# Measures latency and accuracy of TRT and PyTorch implementations of JASPER.

echo "Container nvidia build = " $NVIDIA_BUILD_ID

trap "exit" INT


# Mandatory Arguments
CHECKPOINT=${CHECKPOINT:-"/checkpoints/jasper_fp16.pt"}

# Arguments with Defaults
DATA_DIR=${DATA_DIR:-"/datasets/LibriSpeech"}
DATASET=${DATASET:-"dev-clean"}
RESULT_DIR=${RESULT_DIR:-"/results"}
LOG_DIR=${RESULT_DIR}/logs
CREATE_LOGFILE=${CREATE_LOGFILE:-"true"}
TRT_PRECISION=${TRT_PRECISION:-"fp16"}
PYTORCH_PRECISION=${PYTORCH_PRECISION:-"fp16"}
NUM_STEPS=${NUM_STEPS:-"100"}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_FRAMES=${NUM_FRAMES:-512}
FORCE_ENGINE_REBUILD=${FORCE_ENGINE_REBUILD:-"false"}
CSV_PATH=${CSV_PATH:-"/results/res.csv"}
TRT_PREDICTION_PATH=${TRT_PREDICTION_PATH:-"none"}
PYT_PREDICTION_PATH=${PYT_PREDICTION_PATH:-"none"}
VERBOSE=${VERBOSE:-"false"}
USE_DYNAMIC_SHAPE=${USE_DYNAMIC_SHAPE:-"yes"}


# Set up flag-based arguments
TRT_PREC=""
if [ "$TRT_PRECISION" = "fp16" ] ; then
    TRT_PREC="--trt_fp16"
elif [ "$TRT_PRECISION" = "fp32" ] ; then
    TRT_PREC=""
else
   echo "Unknown <trt_precision> argument"
   exit -2
fi

PYTORCH_PREC=""
if [ "$PYTORCH_PRECISION" = "fp16" ] ; then
    PYTORCH_PREC="--pyt_fp16"
elif [ "$PYTORCH_PRECISION" = "fp32" ] ; then
    PYTORCH_PREC=""
else
   echo "Unknown <pytorch_precision> argument"
   exit -2
fi

SHOULD_VERBOSE=""
if [ "$VERBOSE" = "true" ] ; then
    SHOULD_VERBOSE="--verbose"
fi

STEPS=""
if [ "$NUM_STEPS" -gt 0 ] ; then
   STEPS=" --num_steps $NUM_STEPS"
fi

# Making engine and onnx directories in RESULT_DIR if they don't already exist
ONNX_DIR=$RESULT_DIR/onnxs
ENGINE_DIR=$RESULT_DIR/engines
mkdir -p $ONNX_DIR
mkdir -p $ENGINE_DIR
mkdir -p $LOG_DIR



if [ "$USE_DYNAMIC_SHAPE" = "no" ] ; then
    PREFIX=BS${BATCH_SIZE}_NF${NUM_FRAMES}
    DYNAMIC_PREFIX=" --static_shape "
else
    PREFIX=DYNAMIC
fi

# Currently, TRT parser for ONNX can't parse mixed-precision weights, so ONNX
# export will always be FP32. This is also enforced in perf.py
ONNX_FILE=fp32_${PREFIX}.onnx
ENGINE_FILE=${TRT_PRECISION}_${PREFIX}.engine


# If an ONNX with the same precision and number of frames exists, don't recreate it because
# TRT engine construction can be done on an onnx of any batch size
# "%P" only prints filenames (rather than absolute/relative path names)
EXISTING_ONNX=$(find $ONNX_DIR -name ${ONNX_FILE} -printf "%P\n" | head -n 1)
SHOULD_MAKE_ONNX=""
if [ -z "$EXISTING_ONNX" ] ; then
    SHOULD_MAKE_ONNX="--make_onnx"
else
    ONNX_FILE=${EXISTING_ONNX}
fi

# Follow FORCE_ENGINE_REBUILD about reusing existing engines.
# If false, the existing engine must match precision, batch size, and number of frames
SHOULD_MAKE_ENGINE=""
if [ "$FORCE_ENGINE_REBUILD" != "true" ] ; then
    EXISTING_ENGINE=$(find $ENGINE_DIR -name "${ENGINE_FILE}")
    if [ -n "$EXISTING_ENGINE" ] ; then
        SHOULD_MAKE_ENGINE="--use_existing_engine"
    fi
fi



if [ "$TRT_PREDICTION_PATH" = "none" ] ; then
   TRT_PREDICTION_PATH=""
else
   TRT_PREDICTION_PATH=" --trt_prediction_path=${TRT_PREDICTION_PATH}"
fi


if [ "$PYT_PREDICTION_PATH" = "none" ] ; then
   PYT_PREDICTION_PATH=""
else
   PYT_PREDICTION_PATH=" --pyt_prediction_path=${PYT_PREDICTION_PATH}"
fi

CMD="python trt/perf.py"
CMD+=" --batch_size $BATCH_SIZE"
CMD+=" --engine_batch_size $BATCH_SIZE"
CMD+=" --model_toml configs/jasper10x5dr_nomask.toml"
CMD+=" --dataset_dir $DATA_DIR"
CMD+=" --val_manifest $DATA_DIR/librispeech-${DATASET}-wav.json "
CMD+=" --ckpt_path $CHECKPOINT"
CMD+=" $SHOULD_VERBOSE"
CMD+=" $TRT_PREC"
CMD+=" $PYTORCH_PREC"
CMD+=" $STEPS"
CMD+=" --engine_path ${RESULT_DIR}/engines/${ENGINE_FILE}"
CMD+=" --onnx_path ${RESULT_DIR}/onnxs/${ONNX_FILE}"
CMD+=" --seq_len $NUM_FRAMES"
CMD+=" $SHOULD_MAKE_ONNX"
CMD+=" $SHOULD_MAKE_ENGINE"
CMD+=" $DYNAMIC_PREFIX"
CMD+=" --csv_path $CSV_PATH"
CMD+=" $1 $2 $3 $4 $5 $6 $7 $8 $9"
CMD+=" $TRT_PREDICTION_PATH"
CMD+=" $PYT_PREDICTION_PATH"


if [ "$CREATE_LOGFILE" == "true" ] ; then
  export GBS=$(expr $BATCH_SIZE )
  printf -v TAG "jasper_trt_inference_benchmark_%s_gbs%d" "$PYTORCH_PRECISION" $GBS
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$LOG_DIR/$TAG.$DATESTAMP.log
  printf "Logs written to %s\n" "$LOGFILE"
fi

mkdir -p ${RESULT_DIR}/logs

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   $CMD |& tee $LOGFILE
   grep 'latency' $LOGFILE
fi
set +x

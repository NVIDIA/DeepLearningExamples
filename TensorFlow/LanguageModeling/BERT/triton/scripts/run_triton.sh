#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

init_checkpoint=${1:-"/results/models/bert_large_fp16_384_v1/model.ckpt-5474"}
batch_size=${2:-"8"}
precision=${3:-"fp16"}
use_xla=${4:-"true"}
seq_length=${5:-"384"}
doc_stride=${6:-"128"}
bert_model=${7:-"large"}
squad_version=${8:-"1.1"}
triton_version_name=${9:-1}
triton_model_name=${10:-"bert"}
triton_export_model=${11:-"true"}
triton_dyn_batching_delay=${12:-0}
triton_engine_count=${13:-1}
triton_model_overwrite=${14:-"False"}
squad_version=${15:-"1.1"}

if [ "$bert_model" = "large" ] ; then
    export BERT_DIR=data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16
else
    export BERT_DIR=data/download/google_pretrained_weights/uncased_L-12_H-768_A-12
fi

if [ ! -d "$BERT_DIR" ] ; then
   echo "Error! $BERT_DIR directory missing. Please mount pretrained BERT dataset."
   exit -1
fi

export SQUAD_DIR=data/download/squad/v${squad_version}
if [ "$squad_version" = "1.1" ] ; then
    version_2_with_negative="False"
else
    version_2_with_negative="True"
fi

echo "Squad directory set as " $SQUAD_DIR
if [ ! -d "$SQUAD_DIR" ] ; then
   echo "Error! $SQUAD_DIR directory missing. Please mount SQuAD dataset."
   exit -1
fi

# Need to ignore case on some variables
triton_export_model=$(echo "$triton_export_model" | tr '[:upper:]' '[:lower:]')

# Explicitly save this variable to pass down to new containers
NV_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-"all"}

echo " BERT directory set as " $BERT_DIR
echo
echo "Argument: "
echo "   init_checkpoint = $init_checkpoint"
echo "   batch_size      = $batch_size"
echo "   precision       = $precision"
echo "   use_xla         = $use_xla"
echo "   seq_length      = $seq_length"
echo "   doc_stride      = $doc_stride"
echo "   bert_model      = $bert_model"
echo "   squad_version   = $squad_version"
echo "   version_name    = $triton_version_name"
echo "   model_name      = $triton_model_name"
echo "   export_model    = $triton_export_model"
echo
echo "Env: "
echo "   NVIDIA_VISIBLE_DEVICES = $NV_VISIBLE_DEVICES"
echo

# Export Model in SavedModel format if enabled
if [ "$triton_export_model" = "true" ] ; then
   echo "Exporting model as: Name - $triton_model_name Version - $triton_version_name"

      bash triton/scripts/export_model.sh $init_checkpoint $batch_size $precision $use_xla $seq_length \
         $doc_stride $BERT_DIR $triton_version_name $triton_model_name \
         $triton_dyn_batching_delay $triton_engine_count $triton_model_overwrite
fi

# Start TRTIS server in detached state
bash triton/scripts/launch_server.sh

# Wait until server is up. curl on the health of the server and sleep until its ready
bash triton/scripts/wait_for_triton_server.sh localhost

# Start TRTIS client for inference on SQuAD Dataset
bash triton/scripts/run_client.sh $batch_size $seq_length $doc_stride $triton_version_name $triton_model_name \
    $BERT_DIR --predict_file=$SQUAD_DIR/dev-v${squad_version}.json --version_2_with_negative=${version_2_with_negative}

# Evaluate SQuAD results
bash scripts/docker/launch.sh "python $SQUAD_DIR/evaluate-v${squad_version}.py \
    $SQUAD_DIR/dev-v${squad_version}.json /results/predictions.json"

#Kill the TRTIS Server
docker kill triton_server_cont

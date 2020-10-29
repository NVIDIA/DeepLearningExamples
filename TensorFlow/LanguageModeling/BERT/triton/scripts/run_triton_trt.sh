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

batch_size=${1:-"8"}
seq_length=${2:-"384"}
doc_stride=${3:-"128"}
bert_model=${4:-"large"}
squad_version=${5:-"1.1"}
triton_version_name=${6:-1}
triton_model_name=${7:-"bert"}

if [ "$bert_model" = "large" ] ; then
    export BERT_DIR=data/download/nvidia_pretrained/bert_tf_pretraining_large_lamb
else
    export BERT_DIR=data/download/nvidia_pretrained/bert_tf_squad11_base_128
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

# Explicitly save this variable to pass down to new containers
NV_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-"all"}

echo " BERT directory set as " $BERT_DIR
echo
echo "Argument: "
echo "   init_checkpoint = $init_checkpoint"
echo "   batch_size      = $batch_size"
echo "   seq_length      = $seq_length"
echo "   doc_stride      = $doc_stride"
echo "   squad_version   = $squad_version"
echo "   version_name    = $triton_version_name"
echo "   model_name      = $triton_model_name"
echo
echo "Env: "
echo "   NVIDIA_VISIBLE_DEVICES = $NV_VISIBLE_DEVICES"
echo


# Start TRTIS server in detached state
bash triton/scripts/launch_server.sh

# Wait until server is up. curl on the health of the server and sleep until its ready
bash triton/scripts/wait_for_triton_server.sh localhost

# Start TRTIS client for inference on SQuAD Dataset
bash triton/scripts/run_client.sh $batch_size $seq_length $doc_stride $triton_version_name $triton_model_name \
    $BERT_DIR --version_2_with_negative=${version_2_with_negative} --trt_engine --predict_file=$SQUAD_DIR/dev-v${squad_version}.json 

# Evaluate SQuAD results
bash scripts/docker/launch.sh "python $SQUAD_DIR/evaluate-v${squad_version}.py \
    $SQUAD_DIR/dev-v${squad_version}.json /results/predictions.json"

#Kill the TRTIS Server
docker kill triton_server_cont
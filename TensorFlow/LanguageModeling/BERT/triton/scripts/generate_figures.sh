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

# Set the number of devices to use
export NVIDIA_VISIBLE_DEVICES=0

# Always need to be overwriting models to keep memory use low
export TRITON_MODEL_OVERWRITE=True

bert_model=${1:-small}
seq_length=${2:-128}
precision=${3:-fp16}
init_checkpoint=${4:-"/results/models/bert_${bert_model}_${precision}_${seq_length}_v1/model.ckpt-5474"}

MODEL_NAME="bert_${bert_model}_${seq_length}_${precision}"

if [ "$bert_model" = "large" ] ; then
    export BERT_DIR=data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16
else
    export BERT_DIR=data/download/google_pretrained_weights/uncased_L-12_H-768_A-12
fi

doc_stride=128
use_xla=true
EXPORT_MODEL_ARGS="${precision} ${use_xla} ${seq_length} ${doc_stride} ${BERT_DIR} 1 ${MODEL_NAME}"
PERF_CLIENT_ARGS="1000 10 20 localhost"

############## Dynamic Batching Comparison ##############
SERVER_BATCH_SIZE=8
CLIENT_BATCH_SIZE=1
TRITON_ENGINE_COUNT=1

# Dynamic batching 10 ms
TRITON_DYN_BATCHING_DELAY=10
bash triton/scripts/export_model.sh ${init_checkpoint} ${SERVER_BATCH_SIZE} ${EXPORT_MODEL_ARGS} ${TRITON_DYN_BATCHING_DELAY} ${TRITON_ENGINE_COUNT} ${TRITON_MODEL_OVERWRITE}
bash triton/scripts/run_perf_client.sh ${MODEL_NAME} 1 ${precision} ${CLIENT_BATCH_SIZE} ${PERF_CLIENT_ARGS}

# Dynamic batching 5 ms
TRITON_DYN_BATCHING_DELAY=5
bash triton/scripts/export_model.sh ${init_checkpoint} ${SERVER_BATCH_SIZE} ${EXPORT_MODEL_ARGS} ${TRITON_DYN_BATCHING_DELAY} ${TRITON_ENGINE_COUNT} ${TRITON_MODEL_OVERWRITE}
bash triton/scripts/run_perf_client.sh ${MODEL_NAME} 1 ${precision} ${CLIENT_BATCH_SIZE} ${PERF_CLIENT_ARGS}

# Dynamic batching 2 ms
TRITON_DYN_BATCHING_DELAY=2
bash triton/scripts/export_model.sh ${init_checkpoint} ${SERVER_BATCH_SIZE} ${EXPORT_MODEL_ARGS} ${TRITON_DYN_BATCHING_DELAY} ${TRITON_ENGINE_COUNT} ${TRITON_MODEL_OVERWRITE}
bash triton/scripts/run_perf_client.sh ${MODEL_NAME} 1 ${precision} ${CLIENT_BATCH_SIZE} ${PERF_CLIENT_ARGS}


# Static Batching (i.e. Dynamic batching 0 ms)
TRITON_DYN_BATCHING_DELAY=0
bash triton/scripts/export_model.sh ${init_checkpoint} ${SERVER_BATCH_SIZE} ${EXPORT_MODEL_ARGS} ${TRITON_DYN_BATCHING_DELAY} ${TRITON_ENGINE_COUNT} ${TRITON_MODEL_OVERWRITE}
bash triton/scripts/run_perf_client.sh ${MODEL_NAME} 1 ${precision} ${CLIENT_BATCH_SIZE} ${PERF_CLIENT_ARGS}


# ############## Engine Count Comparison ##############
SERVER_BATCH_SIZE=1
CLIENT_BATCH_SIZE=1
TRITON_DYN_BATCHING_DELAY=0

# Engine Count = 4
TRITON_ENGINE_COUNT=4
bash triton/scripts/export_model.sh ${init_checkpoint} ${SERVER_BATCH_SIZE} ${EXPORT_MODEL_ARGS} ${TRITON_DYN_BATCHING_DELAY} ${TRITON_ENGINE_COUNT} ${TRITON_MODEL_OVERWRITE}
bash triton/scripts/run_perf_client.sh ${MODEL_NAME} 1 ${precision} ${CLIENT_BATCH_SIZE} ${PERF_CLIENT_ARGS}

# Engine Count = 2
TRITON_ENGINE_COUNT=2
bash triton/scripts/export_model.sh ${init_checkpoint} ${SERVER_BATCH_SIZE} ${EXPORT_MODEL_ARGS} ${TRITON_DYN_BATCHING_DELAY} ${TRITON_ENGINE_COUNT} ${TRITON_MODEL_OVERWRITE}
bash triton/scripts/run_perf_client.sh ${MODEL_NAME} 1 ${precision} ${CLIENT_BATCH_SIZE} ${PERF_CLIENT_ARGS}

# Engine Count = 1
TRITON_ENGINE_COUNT=1
bash triton/scripts/export_model.sh ${init_checkpoint} ${SERVER_BATCH_SIZE} ${EXPORT_MODEL_ARGS} ${TRITON_DYN_BATCHING_DELAY} ${TRITON_ENGINE_COUNT} ${TRITON_MODEL_OVERWRITE}
bash triton/scripts/run_perf_client.sh ${MODEL_NAME} 1 ${precision} ${CLIENT_BATCH_SIZE} ${PERF_CLIENT_ARGS}


############## Batch Size Comparison ##############
# BATCH=1 Generate model and perf
SERVER_BATCH_SIZE=1
CLIENT_BATCH_SIZE=1
TRITON_ENGINE_COUNT=1
TRITON_DYN_BATCHING_DELAY=0

bash triton/scripts/export_model.sh ${init_checkpoint} ${SERVER_BATCH_SIZE} ${EXPORT_MODEL_ARGS} ${TRITON_DYN_BATCHING_DELAY} ${TRITON_ENGINE_COUNT} ${TRITON_MODEL_OVERWRITE}
bash triton/scripts/run_perf_client.sh ${MODEL_NAME} 1 ${precision} ${CLIENT_BATCH_SIZE} 1000 10 64 localhost

# BATCH=2 Generate model and perf
SERVER_BATCH_SIZE=2
CLIENT_BATCH_SIZE=2
bash triton/scripts/export_model.sh ${init_checkpoint} ${SERVER_BATCH_SIZE} ${EXPORT_MODEL_ARGS} ${TRITON_DYN_BATCHING_DELAY} ${TRITON_ENGINE_COUNT} ${TRITON_MODEL_OVERWRITE}
bash triton/scripts/run_perf_client.sh ${MODEL_NAME} 1 ${precision} ${CLIENT_BATCH_SIZE} 1000 10 32 localhost

# BATCH=4 Generate model and perf
SERVER_BATCH_SIZE=4
CLIENT_BATCH_SIZE=4
bash triton/scripts/export_model.sh ${init_checkpoint} ${SERVER_BATCH_SIZE} ${EXPORT_MODEL_ARGS} ${TRITON_DYN_BATCHING_DELAY} ${TRITON_ENGINE_COUNT} ${TRITON_MODEL_OVERWRITE}
bash triton/scripts/run_perf_client.sh ${MODEL_NAME} 1 ${precision} ${CLIENT_BATCH_SIZE} 1000 10 16 localhost

# BATCH=8 Generate model and perf
SERVER_BATCH_SIZE=8
CLIENT_BATCH_SIZE=8
bash triton/scripts/export_model.sh ${init_checkpoint} ${SERVER_BATCH_SIZE} ${EXPORT_MODEL_ARGS} ${TRITON_DYN_BATCHING_DELAY} ${TRITON_ENGINE_COUNT} ${TRITON_MODEL_OVERWRITE}
bash triton/scripts/run_perf_client.sh ${MODEL_NAME} 1 ${precision} ${CLIENT_BATCH_SIZE} 1000 10 8 localhost


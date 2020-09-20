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
BERT_DIR=${7:-"data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16"}
triton_model_version=${8:-1}
triton_model_name=${9:-"bert"}
triton_dyn_batching_delay=${10:-0}
triton_engine_count=${11:-1}
triton_model_overwrite=${12:-"False"}

additional_args="--triton_model_version=$triton_model_version --triton_model_name=$triton_model_name --triton_max_batch_size=$batch_size \
                 --triton_model_overwrite=$triton_model_overwrite --triton_dyn_batching_delay=$triton_dyn_batching_delay \
                 --triton_engine_count=$triton_engine_count"

if [ "$precision" = "fp16" ] ; then
   echo "fp16 activated!"
   additional_args="$additional_args --amp"
else
   echo "fp32/tf32 activated!"
   additional_args="$additional_args --noamp"
fi

if [ "$use_xla" = "true" ] ; then
    echo "XLA activated"
    additional_args="$additional_args --use_xla"
else
    additional_args="$additional_args --nouse_xla"
fi

echo "Additional args: $additional_args"

bash scripts/docker/launch.sh \
    python run_squad.py \
       --vocab_file=${BERT_DIR}/vocab.txt \
       --bert_config_file=${BERT_DIR}/bert_config.json \
       --init_checkpoint=${init_checkpoint} \
       --max_seq_length=${seq_length} \
       --doc_stride=${doc_stride} \
       --predict_batch_size=${batch_size} \
       --output_dir=/results \
       --export_triton=True \
       ${additional_args}




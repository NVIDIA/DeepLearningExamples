#!/usr/bin/env bash

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

rm -rf /result_tmp/

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export CUDA_VISIBLE_DEVICES=0

python ${BASEDIR}/../mask_rcnn_main.py \
    --mode="eval" \
    --eval_batch_size=8 \
    --eval_samples=5000 \
    --learning_rate_steps="480000,640000" \
    --model_dir="/result_tmp/" \
    --validation_file_pattern="/data/val*.tfrecord" \
    --val_json_file="/data/annotations/instances_val2017.json" \
    --use_batched_nms \
    --use_amp \
    --nouse_xla \
    --nouse_custom_box_proposals_op
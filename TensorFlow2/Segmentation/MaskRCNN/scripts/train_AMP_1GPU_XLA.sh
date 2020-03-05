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

rm -rf /results

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export CUDA_VISIBLE_DEVICES=0

python ${BASEDIR}/../mask_rcnn_main.py \
    --mode="train_and_eval" \
    --checkpoint="/model/resnet/resnet-nhwc-2018-10-14/model.ckpt-112602" \
    --eval_samples=5000 \
    --init_learning_rate=0.005 \
    --learning_rate_steps="240000,320000" \
    --model_dir="/results/" \
    --num_steps_per_eval=29568 \
    --total_steps=360000 \
    --train_batch_size=4 \
    --eval_batch_size=8 \
    --training_file_pattern="/data/train*.tfrecord" \
    --validation_file_pattern="/data/val*.tfrecord" \
    --val_json_file="/data/annotations/instances_val2017.json" \
    --use_amp \
    --use_batched_nms \
    --use_xla \
    --nouse_custom_box_proposals_op
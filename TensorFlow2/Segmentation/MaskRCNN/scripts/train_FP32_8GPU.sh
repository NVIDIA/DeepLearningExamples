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

mpirun \
    -np 8 \
    -H localhost:8 \
    -bind-to none \
    -map-by slot \
    -x NCCL_DEBUG=VERSION \
    -x LD_LIBRARY_PATH \
    -x PATH \
    -mca pml ob1 -mca btl ^openib \
    --allow-run-as-root \
    python ${BASEDIR}/../mask_rcnn_main.py \
        --mode="train_and_eval" \
        --checkpoint="/model/resnet/resnet-nhwc-2018-10-14/model.ckpt-112602" \
        --eval_samples=5000 \
        --init_learning_rate=0.04 \
        --learning_rate_steps="30000,40000" \
        --model_dir="/results/" \
        --num_steps_per_eval=3696 \
        --total_steps=45000 \
        --train_batch_size=4 \
        --eval_batch_size=8 \
        --training_file_pattern="/data/train*.tfrecord" \
        --validation_file_pattern="/data/val*.tfrecord" \
        --val_json_file="/data/annotations/instances_val2017.json" \
        --nouse_amp \
        --use_batched_nms \
        --nouse_xla \
        --nouse_custom_box_proposals_op

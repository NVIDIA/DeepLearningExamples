#!/usr/bin/env bash

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

# This script launches UNet training in FP32 on 4 GPUs using 16 batch size (4 per GPU)
# Usage ./UNet_FP32_4GPU_XLA.sh <path to result repository> <path to dataset> <dagm classID (1-10)>

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export TF_CPP_MIN_LOG_LEVEL=3

mpirun \
    -np 4 \
    -H localhost:4 \
    -bind-to none \
    -map-by slot \
    -x NCCL_DEBUG=VERSION \
    -x LD_LIBRARY_PATH \
    -x PATH \
    -mca pml ob1 -mca btl ^openib \
    --allow-run-as-root \
    python "${BASEDIR}/../main.py" \
        --unet_variant='tinyUNet' \
        --activation_fn='relu' \
        --exec_mode='train_and_evaluate' \
        --iter_unit='batch' \
        --num_iter=2500 \
        --batch_size=4 \
        --warmup_step=10 \
        --results_dir="${1}" \
        --data_dir="${2}" \
        --dataset_name='DAGM2007' \
        --dataset_classID="${3}" \
        --data_format='NCHW' \
        --use_auto_loss_scaling \
        --noamp \
        --xla \
        --learning_rate=1e-4 \
        --learning_rate_decay_factor=0.8 \
        --learning_rate_decay_steps=500 \
        --rmsprop_decay=0.9 \
        --rmsprop_momentum=0.8 \
        --loss_fn_name='adaptive_loss' \
        --weight_decay=1e-5 \
        --weight_init_method='he_uniform' \
        --augment_data \
        --display_every=250 \
        --debug_verbosity=0

#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#       http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

WORKSPACE=${1:-"/workspace/rn50v15_tf"}
DATA_DIR=${2:-"/data"}

OTHER=${@:3}

if [[ ! -z "${BIND_TO_SOCKET}" ]]; then
    BIND_TO_SOCKET="--bind-to socket"
fi

mpiexec --allow-run-as-root ${BIND_TO_SOCKET} -np 8 python3 main.py --arch=resnext101-32x4d \
    --mode=train_and_evaluate --iter_unit=epoch --num_iter=90 \
    --batch_size=128 --warmup_steps=100 --cosine_lr --label_smoothing 0.1 \
    --lr_init=0.256 --lr_warmup_epochs=8 --momentum=0.875 --weight_decay=6.103515625e-05 \
    --amp --static_loss_scale 128 \
    --data_dir=${DATA_DIR}/tfrecords --data_idx_dir=${DATA_DIR}/dali_idx \
    --results_dir=${WORKSPACE}/results --weight_init=fan_in ${OTHER}


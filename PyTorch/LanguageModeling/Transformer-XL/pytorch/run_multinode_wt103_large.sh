#!/bin/bash

# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

export OMP_NUM_THREADS=1

if [[ $1 == 'train' ]] || [[ $1 == 'all' ]]; then
    echo 'Run training...'
    python train.py \
        --config_file wt103_large.yaml \
        --config 8dgx2_16gpu_fp16 \
        ${@:2}
fi

if [[ $1 == 'eval' ]] || [[ $1 == 'all' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --config_file wt103_large.yaml \
        --config 8dgx2_16gpu_fp16 \
        ${@:2}
fi

if [[ $1 != 'train' ]] && [[ $1 != 'eval' ]] && [[ $1 != 'all' ]]; then
    echo 'unknown argment 1'
fi

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

MAX_FP32_BS=${1:-64}
MAX_AMP_BS=${2:-96}

GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | uniq)
GPU_COUNT=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)


function run_benchmark() {
    BATCH_SIZE=$1
    MODE_SIZE=$2
    
    if [[ $4 -eq "1" ]]; then
        XLA="--xla"
    else
        XLA=""
    fi

    case $2 in
        "amp") MODE_FLAGS="--amp --static_loss_scale=128";;
        "fp32"|"tf32") MODE_FLAGS="";;
        *) echo "Unsupported configuration, use amp, tf32 or fp32";;
    esac

    CMD_LINE="--arch=se-resnext101-32x4d --mode=training_benchmark --warmup_steps 200 --num_iter 500 --iter_unit batch --batch_size $BATCH_SIZE \
        --data_dir=/data/tfrecords/ --results_dir=/tmp/result $MODE_FLAGS $XLA"

    mkdir -p /tmp/result/
    if [[ $G3 -eq "1" ]]; then
        python ./main.py ${CMD_LINE} > /tmp/result/logs.txt
    else
        mpiexec --allow-run-as-root --bind-to socket -np $3 python3 main.py ${CMD_LINE} > /tmp/result/logs.txt
    fi

    tail -n1 /tmp/result/logs.txt | sed \
            's/^DLL \([0-9]*-\)*[0-9]* \([0-9]*:\)*[0-9]*.[0-9]* - ()/BS='$BATCH_SIZE','$2',XLA='$4'/' >> ./training_benchmark.txt
    rm -rf /tmp/result
}

run_benchmark $MAX_AMP_BS amp 1 0
run_benchmark $MAX_AMP_BS amp 1 1
run_benchmark $MAX_FP32_BS fp32 1 0
run_benchmark $MAX_FP32_BS fp32 1 1

if [[ $GPU_COUNT -ne "1" ]]; then
    run_benchmark $MAX_AMP_BS amp $GPU_COUNT 0
    run_benchmark $MAX_AMP_BS amp $GPU_COUNT 1
    run_benchmark $MAX_FP32_BS fp32 $GPU_COUNT 0
    run_benchmark $MAX_FP32_BS fp32 $GPU_COUNT 1
fi

cat ./training_benchmark.txt
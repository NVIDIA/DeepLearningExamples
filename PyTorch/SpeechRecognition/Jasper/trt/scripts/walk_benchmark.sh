#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# A usage example of trt_inference_benchmark.sh.


export NUM_STEPS=100
export FORCE_ENGINE_REBUILD="false"
export CHECKPOINT=${CHECKPOINT:-"/checkpoints/jasper_fp16.pt"}
export CREATE_LOGFILE="true"
prec=fp16
export TRT_PRECISION=$prec
export PYTORCH_PRECISION=$prec

trap "exit" INT

for use_dynamic in yes no;
do
    export USE_DYNAMIC_SHAPE=${use_dynamic}
    export CSV_PATH="/results/${prec}.csv"
    for nf in 208 304 512 704 1008 1680;
    do
        export NUM_FRAMES=$nf
        for bs in 1 2 4 8 16 32 64;
        do
            export BATCH_SIZE=$bs

            echo "Doing batch size ${bs}, sequence length ${nf}, precision ${prec}"
            bash trt/scripts/trt_inference_benchmark.sh $1 $2 $3 $4 $5 $6
        done
    done
done

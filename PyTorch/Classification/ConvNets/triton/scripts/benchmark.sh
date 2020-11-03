#!/bin/bash

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

MODEL_REPO=${1:-"/repo"}
OUTPUT=${2:-"/logs"}
MODEL_ARCH=${3:-"resnet50"}
MODEL_CHECKPOINT=${4:-"/checkpoint.pth"}

for backend in ts onnx trt; do
    if [[ "$backend" = "ts" ]]; then
        EXPORT_NAME="ts-script"
    else
        EXPORT_NAME="${backend}"
    fi

    for precision in 16 32; do
        if [[ $precision -eq 16 ]]; then
            CUSTOM_FLAGS="--fp16"
            CUSTON_TRTFLAGS="--trt-fp16 --max_workspace_size 2147483648"
        else
            CUSTOM_FLAGS=""
            CUSTON_TRTFLAGS=""
        fi

        echo "Exporting model as ${EXPORT_NAME} with precision ${precision}"

        python -m triton.deployer --${EXPORT_NAME} --triton-model-name model_${backend} --triton-max-batch-size 64 \
            --triton-engine-count 2 --save-dir ${MODEL_REPO} ${CUSTON_TRTFLAGS} -- --config ${MODEL_ARCH} ${CUSTOM_FLAGS}
        sleep 30

        /workspace/bin/perf_client --max-threads 10 -m model_${backend} -x 1 -p 10000 -v -i gRPC -u localhost:8001 -b 1 \
            -l 5000 --concurrency-range 1:2 -f ${OUTPUT}/${backend}_dynamic_${precision}.csv
        for CONCURENCY_LEVEL in 4 8 16 32 64 128 256; do
            /workspace/bin/perf_client --max-threads 10 -m model_${backend} -x 1 -p 10000 -v -i gRPC -u localhost:8001 -b 1 \
                -l 5000 --concurrency-range $CONCURENCY_LEVEL:$CONCURENCY_LEVEL -f >(tail -n +2 >> ${OUTPUT}/${backend}_dynamic_${precision}.csv)
        done
        rm -rf ${MODEL_REPO}/model_${backend}
    done
    cat ${OUTPUT}/*_dynamic_*.csv
done

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

set -e

REPO_DIR=${REPO_DIR:-"/workspace/transformer-xl/pytorch/"}
REFERENCE_FILE=$REPO_DIR/scripts/tests/reference_training_throughput

MATH=$1
if [[ ${MATH} != "fp16" && ${MATH} != "fp32" ]]; then
   echo "Unsupported option for MATH, use either 'fp16' or 'fp32'"
   exit 1
fi

if [[ ${MATH} == 'fp16' ]]; then
   MATH_OPT='--fp16'
elif [[ ${MATH} == 'fp32' ]]; then
   MATH_OPT=''
fi

PERF_TOLERANCE=0.9
GLOBAL_BATCH_SIZE=256

GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader |uniq)
echo 'GPU_NAME:' "${GPU_NAME}"
GPU_COUNT=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader |wc -l)
echo 'GPU_COUNT:' "${GPU_COUNT}"
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader |head -n 1 |cut -f 1 -d " ")
echo 'GPU_MEM:' "${GPU_MEM}"

if (( GPU_MEM > 16500 )); then
   LOCAL_BATCH_SIZE=32
else
   if [[ ${MATH} == 'fp16' ]]; then
      LOCAL_BATCH_SIZE=32
   elif [[ ${MATH} == 'fp32' ]]; then
      LOCAL_BATCH_SIZE=16
   fi
fi

BATCH_CHUNK=$((GLOBAL_BATCH_SIZE / (GPU_COUNT * LOCAL_BATCH_SIZE)))
BATCH_CHUNK=$((BATCH_CHUNK < 1 ? 1 : BATCH_CHUNK))

REFERENCE_PERF=$(grep "${MATH},${GPU_COUNT},${GPU_NAME}" \
   ${REFERENCE_FILE} | \cut -f 4 -d ',')

if [ -z "${REFERENCE_PERF}" ]; then
   echo "WARNING: COULD NOT FIND REFERENCE PERFORMANCE FOR EXECUTED CONFIG"
   TARGET_PERF=''
else
   PERF_THRESHOLD=$(awk 'BEGIN {print ('"${REFERENCE_PERF}"' * '"${PERF_TOLERANCE}"')}')
   TARGET_PERF='--target_throughput '${PERF_THRESHOLD}
fi

cd $REPO_DIR

bash run_wt103_base.sh train "${GPU_COUNT}" \
   --debug \
   --max_step 5000 \
   --max_step_scheduler 40000 \
   --target_perplexity 43.5 \
   --batch_chunk "${BATCH_CHUNK}" \
   --log_interval 1 \
   --adaptive \
   --vocab word \
   "${MATH_OPT}" \
   "${TARGET_PERF}"

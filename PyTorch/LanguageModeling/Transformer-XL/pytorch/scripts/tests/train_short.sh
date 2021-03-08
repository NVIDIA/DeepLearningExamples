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

set -e

REPO_DIR=${REPO_DIR:-"/workspace/transformer-xl/pytorch/"}
REFERENCE_FILE=$REPO_DIR/scripts/tests/reference_training_throughput

MATH=$1
if [[ ${MATH} != "fp16" && ${MATH} != "fp32" ]]; then
   echo "Unsupported option for MATH, use either 'fp16' or 'fp32'"
   exit 1
fi

PERF_TOLERANCE=0.9

GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader |uniq)
echo 'GPU_NAME:' "${GPU_NAME}"
GPU_COUNT=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader |wc -l)
echo 'GPU_COUNT:' "${GPU_COUNT}"

if (( GPU_COUNT == 16 )); then
   SYSTEM=dgx2
else
   SYSTEM=dgx1
fi

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
   --config ${SYSTEM}_${GPU_COUNT}gpu_${MATH} \
   --debug \
   --max_step 5000 \
   --max_step_scheduler 40000 \
   --target_perplexity 43.5 \
   --log_interval 1 \
   ${TARGET_PERF}

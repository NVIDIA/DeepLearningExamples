#!/bin/bash

# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


set -e

DATASET_DIR='data/wmt16_de_en'
REPO_DIR='/workspace/gnmt'
REFERENCE_FILE=$REPO_DIR/scripts/tests/reference_inference_performance

MATH=$1
if [[ ${MATH} != "fp16" && ${MATH} != "fp32" ]]; then
   echo "Unsupported option for MATH, use either 'fp16' or 'fp32'"
   exit 1
fi

BATCH_SIZE=128
BEAM_SIZE=5
PERF_TOLERANCE=0.95

GPU_NAME=`nvidia-smi --query-gpu=gpu_name --format=csv,noheader |uniq`
echo 'GPU_NAME:' ${GPU_NAME}

REFERENCE_PERF=`grep "${MATH},${BATCH_SIZE},${BEAM_SIZE},${GPU_NAME}" \
   ${REFERENCE_FILE} | \cut -f 5 -d ','`

if [ -z "${REFERENCE_PERF}" ]; then
   echo "WARNING: COULD NOT FIND REFERENCE PERFORMANCE FOR EXECUTED CONFIG"
   TARGET_PERF=''
else
   PERF_THRESHOLD=$(awk 'BEGIN {print ('${REFERENCE_PERF}' * '${PERF_TOLERANCE}')}')
   TARGET_PERF='--target-perf '${PERF_THRESHOLD}
fi

cd $REPO_DIR

python3 translate.py \
   --input ${DATASET_DIR}/newstest2014.en \
   --reference ${DATASET_DIR}/newstest2014.de \
   --output /tmp/output \
   --model results/gnmt/model_best.pth \
   --batch-size ${BATCH_SIZE} \
   --beam-size ${BEAM_SIZE} \
   --math ${MATH} \
   --warmup 1 \
   --target-bleu 24.3 \
   ${TARGET_PERF}

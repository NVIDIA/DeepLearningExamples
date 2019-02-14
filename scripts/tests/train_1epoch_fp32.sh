#!/bin/bash

set -e

DATASET_DIR='data/wmt16_de_en'
REPO_DIR='/workspace/gnmt'
REFERENCE_FILE=$REPO_DIR/scripts/tests/reference_performance

MATH='fp32'
PERF_TOLERANCE=0.9

GPU_NAME=`nvidia-smi --query-gpu=gpu_name --format=csv,noheader |uniq`
echo 'GPU_NAME:' ${GPU_NAME}
GPU_COUNT=`nvidia-smi --query-gpu=gpu_name --format=csv,noheader |wc -l`
echo 'GPU_COUNT:' ${GPU_COUNT}

REFERENCE_PERF=`grep "${MATH},${GPU_COUNT},${GPU_NAME}" \
   ${REFERENCE_FILE} | \cut -f 4 -d ','`

if [ -z "${REFERENCE_PERF}" ]; then
   echo "WARNING: COULD NOT FIND REFERENCE PERFORMANCE FOR EXECUTED CONFIG"
   TARGET_PERF=''
else
   PERF_THRESHOLD=$(awk 'BEGIN {print ('${REFERENCE_PERF}' * '${PERF_TOLERANCE}')}')
   TARGET_PERF='--target-perf '${PERF_THRESHOLD}
fi

cd $REPO_DIR

python3 -m launch train.py \
   --dataset-dir $DATASET_DIR \
   --seed 1 \
   --epochs 1 \
   --remain-steps 1.0 \
   --target-bleu 17.20 \
   --math ${MATH} \
   ${TARGET_PERF}

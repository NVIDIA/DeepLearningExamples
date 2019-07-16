#!/bin/bash

set -e

DATASET_DIR='data/wmt16_de_en'
REPO_DIR='/workspace/gnmt'
REFERENCE_FILE=$REPO_DIR/scripts/tests/reference_training_performance

MATH=$1
if [[ ${MATH} != "fp16" && ${MATH} != "fp32" ]]; then
   echo "Unsupported option for MATH, use either 'fp16' or 'fp32'"
   exit 1
fi

PERF_TOLERANCE=0.9

GPU_NAME=`nvidia-smi --query-gpu=gpu_name --format=csv,noheader |uniq`
echo 'GPU_NAME:' ${GPU_NAME}
GPU_COUNT=`nvidia-smi --query-gpu=gpu_name --format=csv,noheader |wc -l`
echo 'GPU_COUNT:' ${GPU_COUNT}

if [[ ${GPU_COUNT} -eq 1 || ${GPU_COUNT} -eq 2 || ${GPU_COUNT} -eq 4 || ${GPU_COUNT} -eq 8 ]]; then
   GLOBAL_BATCH_SIZE=1024
elif [ ${GPU_COUNT} -eq 16 ]; then
   GLOBAL_BATCH_SIZE=2048
else
   echo "Unsupported number of GPUs"
   exit 1
fi

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
   --no-eval \
   --train-max-size $((128 * ${GPU_COUNT} * 300)) \
   --math ${MATH} \
   --train-global-batch-size ${GLOBAL_BATCH_SIZE} \
   ${TARGET_PERF}

#!/bin/bash

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
   --input ${DATASET_DIR}/newstest2014.tok.bpe.32000.en \
   --reference ${DATASET_DIR}/newstest2014.de \
   --output /tmp/output \
   --model results/gnmt/model_best.pth \
   --batch-size ${BATCH_SIZE} \
   --beam-size ${BEAM_SIZE} \
   --math ${MATH} \
   --target-bleu 24.3 \
   ${TARGET_PERF}

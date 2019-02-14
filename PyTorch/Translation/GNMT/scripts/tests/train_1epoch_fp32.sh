#!/bin/bash

set -e

DATASET_DIR='data/wmt16_de_en'
RESULTS_DIR='gnmt_wmt16_test'
REFERENCE_FILE=scripts/tests/reference_performance
LOGFILE=results/${RESULTS_DIR}/log_gpu_0.log

REFERENCE_ACCURACY=17.2
MATH='fp32'
PERFORMANCE_TOLERANCE=0.9

python3 -m multiproc train.py \
  --save ${RESULTS_DIR} \
  --dataset-dir ${DATASET_DIR} \
  --seed 1 \
  --epochs 1 \
  --math ${MATH} \
  --print-freq 10 \
  --batch-size 128 \
  --model-config "{'num_layers': 4, 'hidden_size': 1024, 'dropout':0.2, 'share_embedding': True}" \
  --optimization-config "{'optimizer': 'Adam', 'lr': 5e-4}"

GPU_NAME=`nvidia-smi --query-gpu=gpu_name --format=csv,noheader |uniq`
echo 'GPU_NAME:' ${GPU_NAME}
GPU_COUNT=`nvidia-smi --query-gpu=gpu_name --format=csv,noheader |wc -l`
echo 'GPU_COUNT:' ${GPU_COUNT}

# Accuracy test
ACHIEVED_ACCURACY=`cat ${LOGFILE} \
   |grep Summary \
   |tail -n 1 \
   |cut -f 4 \
   |egrep -o [0-9.]+`

echo 'REFERENCE_ACCURACY:' ${REFERENCE_ACCURACY}
echo 'ACHIEVED_ACCURACY:' ${ACHIEVED_ACCURACY}

ACCURACY_TEST_RESULT=$(awk 'BEGIN {print ('${ACHIEVED_ACCURACY}' >= '${REFERENCE_ACCURACY}')}')

if (( ${ACCURACY_TEST_RESULT} )); then
    echo "&&&& ACCURACY TEST PASSED"
else
    echo "&&&& ACCURACY TEST FAILED"
fi

# Performance test
ACHIEVED_PERFORMANCE=`cat ${LOGFILE} \
   |grep Performance \
   |tail -n 1 \
   |cut -f 2 \
   |egrep -o [0-9.]+`

REFERENCE_PERFORMANCE=`grep "${MATH},${GPU_COUNT},${GPU_NAME}" ${REFERENCE_FILE} \
   | \cut -f 4 -d ','`

echo 'REFERENCE_PERFORMANCE:' ${REFERENCE_PERFORMANCE}
echo 'ACHIEVED_PERFORMANCE:' ${ACHIEVED_PERFORMANCE}

PERFORMANCE_TEST_RESULT=1

if [ -z "${REFERENCE_PERFORMANCE}" ]; then
   echo "WARNING: COULD NOT FIND REFERENCE PERFORMANCE FOR EXECUTED CONFIG"
   echo "&&&& PERFORMANCE TEST WAIVED"
else
   PERFORMANCE_TEST_RESULT=$(awk 'BEGIN {print ('${ACHIEVED_PERFORMANCE}' >= \
      ('${REFERENCE_PERFORMANCE}' * '${PERFORMANCE_TOLERANCE}'))}')

   if (( ${PERFORMANCE_TEST_RESULT} )); then
      echo "&&&& PERFORMANCE TEST PASSED"
   else
      echo "&&&& PERFORMANCE TEST FAILED"
   fi
fi

if (( ${ACCURACY_TEST_RESULT} )) && (( ${PERFORMANCE_TEST_RESULT} )); then
   echo "&&&& PASSED"
   exit 0
else
   echo "&&&& FAILED"
   exit 1
fi

#!/usr/bin/env bash

# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
# ==============================================================================

PRED_BS=${1:-128}
EVAL_BEAMS=${2:-6}
MAX_SOURCE_LEN=${3:-1024}
MAX_TARGET_LEN=${4:-60}

INIT_CKPT=${5:-"facebook/bart-large"}
DATA_DIR=${6:-data/xsum}
CONFIG_PATH=${7:-"configs/config_xsum.json"}

printf -v TAG "bart_pyt_inference_benchmark"
DATESTAMP=`date +'%y%m%d%H%M%S'`
RESULTS_DIR=${RESULTS_DIR:-results/${TAG}_${DATESTAMP}}

mkdir -p $RESULTS_DIR

echo "Inference for Batch size $PRED_BS Eval Beams $EVAL_BEAMS Source Length $MAX_SOURCE_LEN Target Length $MAX_TARGET_LEN
	  Model at $INIT_CKPT Data at $DATA_DIR and Config at $CONFIG_PATH $DATA_DIR $CONFIG_PATH" |& tee ${RESULTS_DIR}/inference_benchmark.log

echo "NUM_GPU Precision Throughput" |& tee ${RESULTS_DIR}/inference_benchmark.log

for NUM_GPU in 1 4 8; do

	for precision in fp16 fp32; do

		if [ "$precision" = "fp16" ] ; then
		    echo "fp16 activated!"
		    USE_FP16="--fp16"

		else
		    echo "fp32/tf32 activated!"
		    USE_FP16=""
		fi

		python -m torch.distributed.launch --nproc_per_node=$NUM_GPU run_eval.py \
		    --task summarization \
		    --bs ${PRED_BS} --max_source_length=${MAX_SOURCE_LEN} --max_target_length=${MAX_TARGET_LEN} \
		    --eval_max_gen_length=${MAX_TARGET_LEN} --eval_beams=${EVAL_BEAMS} ${USE_FP16} \
		    ${INIT_CKPT} ${CONFIG_PATH} ${DATA_DIR} ${RESULTS_DIR} |& tee -a ${RESULTS_DIR}/log_${NUM_GPU}_${precision}.log


        perf=`cat ${RESULTS_DIR}/log_${NUM_GPU}_${precision}.log | grep -F 'INFO:tensorflow:Throughput Average (sentences/sec) =' | tail -1 | awk -F'= ' '{print $2}'`

        echo "$NUM_GPU $precision $perf"  |& tee ${RESULTS_DIR}/inference_benchmark.log


	done
done

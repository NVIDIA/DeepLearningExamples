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

BS=${1:-24}
MAX_SOURCE_LEN=${2:-1024}
MAX_TARGET_LEN=${3:-60}
DATA_DIR=${4:-data/xsum}

printf -v TAG "bart_pyt_training_benchmark"
DATESTAMP=`date +'%y%m%d%H%M%S'`
RESULTS_DIR=${RESULTS_DIR:-results/${TAG}_${DATESTAMP}}

mkdir -p $RESULTS_DIR

echo "Training for Batch size $BS Source Length $MAX_SOURCE_LEN Target Length $MAX_TARGET_LEN Data at $DATA_DIR and Config at $CONFIG_PATH" |& tee ${RESULTS_DIR}/training_benchmark.log
echo "NUM_GPU Precision Throughput" |& tee ${RESULTS_DIR}/training_benchmark.log


for NUM_GPU in 1 4 8; do
	for precision in fp16 fp32; do

		if [ "$precision" = "fp16" ] ; then
		    echo "fp16 activated!"
		    USE_FP16="--fp16"

		else
		    echo "fp32/tf32 activated!"
		    USE_FP16=""
		fi

		python finetune.py \
		    --data_dir=${DATA_DIR} \
		    --config_path=configs/config_xsum.json \
		    --output_dir=${RESULTS_DIR} \
		    --gpus ${NUM_GPU} \
		    --learning_rate=1e-4 \
		    ${USE_FP16} \
		    --do_train \
		    --n_val -1 \
		    --train_batch_size=${BS} --gradient_accumulation_steps=1 \
		    --max_epochs 1 --warmup_steps 0 \
		    --min_epochs=0 --val_check_interval 1.0 \
		    --max_source_length=${MAX_SOURCE_LEN} --max_target_length=${MAX_TARGET_LEN} \
		    --val_max_target_length=${MAX_TARGET_LEN} --eval_max_gen_length=${MAX_TARGET_LEN} \
		    --sortish_sampler \
		    --lr_scheduler polynomial \
		    --label_smoothing 0.1 \
		    --weight_decay 0.1 \
		    --dropout 0.1 --attention_dropout 0.1 --gradient_clip_val=0.1 \
		    --early_stopping_patience=2 \
		    --num_sanity_val_steps=0 --eval_beams 0 --freeze_embeds \
		    --amp_level=O1 --seed ${SEED:-42} |& tee -a ${RESULTS_DIR}/log_${NUM_GPU}_${precision}.log

        perf=`cat ${RESULTS_DIR}/log_${NUM_GPU}_${precision}.log | grep -F 'INFO:tensorflow:Throughput Average (sentences/sec) =' | tail -1 | awk -F'= ' '{print $2}'`

        echo "$NUM_GPU $precision $perf"  |& tee ${RESULTS_DIR}/training_benchmark.log

	done
done

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

DATA_DIR=${1:-data/cnn_dm/}
CKPT_PATH=${2:-data/nvidia_pretrained/bart_large}
CONFIG_PATH=${3:-"configs/config.json"}
NUM_GPU=${4:-8}
LR=${5:-1.25e-4}
BS=${6:-40}
ACCUM=${7:-1}
PREC=${8:-"bf16"}
TRAIN_STEPS=${9:-2000}
WARMUP_STEPS=${10:-50}
MAX_SOURCE_LEN=${11:-1024}
MAX_TARGET_LEN=${12:-142}
EVAL_BEAMS=${13:-4}
EVAL_BS=${14:-128}
PRED_BS=${15:-64}
PRELN=${16:-true}

if [ "$PREC" = "fp16" ] ; then
    echo "fp16 activated!"
    USE_FP16="--fp16"
elif [ "$PREC" = "bf16" ] ; then
    echo "bf16 activated!"
    USE_FP16="--bf16"
else
    echo "fp32/tf32 activated!"
    USE_FP16=""
fi

if [ "$PRELN" = "true" ] ; then
    echo "Trained with PreLN"
    USE_FP16="--pre_ln $USE_FP16"
else
    echo "Trained with PostLN"
fi

printf -v TAG "bart_pyt"
DATESTAMP=`date +'%y%m%d%H%M%S'`
RESULTS_DIR=${RESULTS_DIR:-results/${TAG}_${DATESTAMP}}
mkdir -p ${RESULTS_DIR}

export TOKENIZERS_PARALLELISM="true"
python -m torch.distributed.launch --nproc_per_node=${NUM_GPU:-8} finetune.py \
    --data_dir=${DATA_DIR} \
    --config_path=${CONFIG_PATH} \
    --output_dir=${RESULTS_DIR} \
    --gpus ${NUM_GPU} \
    --learning_rate=${LR:-1e-4} \
    ${USE_FP16} \
    --do_train \
    --n_val -1 \
    --train_batch_size=${BS} --gradient_accumulation_steps=${ACCUM} \
    --eval_batch_size=${EVAL_BS} \
    --max_steps ${TRAIN_STEPS} --warmup_steps ${WARMUP_STEPS} \
    --max_source_length=${MAX_SOURCE_LEN} --max_target_length=${MAX_TARGET_LEN} \
    --val_max_target_length=${MAX_TARGET_LEN} --eval_max_gen_length=${MAX_TARGET_LEN} \
    --sortish_sampler \
    --lr_scheduler polynomial \
    --label_smoothing 0.1 \
    --weight_decay 0.1 \
    --dropout 0.1 --attention_dropout 0.1 --gradient_clip_val=0.1 \
    --eval_beams 0 --freeze_embeds \
    --seed ${SEED:-42} \
    --resume_from_checkpoint=${CKPT_PATH} --load_model_weights_only \
    ${@:17} |& tee ${RESULTS_DIR}/joblog.log


echo "completed training! Begin test" |& tee -a ${RESULTS_DIR}/joblog.log

INIT_CKPT=$(ls ${RESULTS_DIR}/final_step.ckpt | sort -n | tail -1)


python -m torch.distributed.launch --nproc_per_node=${NUM_GPU:-8} run_eval.py \
    --task summarization \
    --bs ${PRED_BS} --max_source_length=${MAX_SOURCE_LEN} --max_target_length=${MAX_TARGET_LEN} \
    --eval_max_gen_length=${MAX_TARGET_LEN} --eval_beams=${EVAL_BEAMS} ${USE_FP16} \
    ${INIT_CKPT} ${CONFIG_PATH} ${DATA_DIR} ${RESULTS_DIR} |& tee -a ${RESULTS_DIR}/joblog.log

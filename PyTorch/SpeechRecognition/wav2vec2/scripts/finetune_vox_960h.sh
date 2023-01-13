#!/usr/bin/env bash

# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

set -e

export OMP_NUM_THREADS=1
export CUDNN_V8_API_ENABLED=1  # For older containers (~22.01)
export TORCH_CUDNN_V8_API_ENABLED=1

# IO
: ${DATASET_DIR:="/datasets/LibriSpeech"}
: ${TRAIN_SUBSET:="train-full-960"}
: ${VALID_SUBSET:="dev-other"}
: ${OUTPUT_DIR:="results/finetune_large_960h"}
# Batching
# To best utilize hw, increase batch size by increasing NUM_CONCAT_BATCHES, and lowering UPDATE_FREQ.
# Keep NUM_NODES x $NUM_GPUS x $NUM_CONCAT_BATCHES x $UPDATE_FREQ = 24.
# Note that this script does not control NUM_NODES.
: ${NUM_GPUS:=8}
: ${MAX_TOKENS:=1280000}
: ${NUM_CONCAT_BATCHES:=3}
: ${UPDATE_FREQ:=1}
# Training
: ${MAX_UPDATE:=320000}
: ${WARMUP_UPDATES:=$(($MAX_UPDATE / 10 * 1))}
: ${HOLD_UPDATES:=$(($MAX_UPDATE / 10 * 4))}
: ${FREEZE_FINETUNE_UPDATES:=10000}
: ${BATCH_SIZE:=}
: ${LEARNING_RATE:=0.00003}
: ${FP16:=false}
: ${BF16:=false}
: ${EMA:=0.0}  # XXX
: ${SEED:=1}  # XXX
: ${CUDNN_BENCHMARK:=false}
# Model
: ${PRETRAINED_MODEL:=pretrained_models/libri960_big.pt}
: ${MASK_PROB:=0.5}
: ${MASK_CHANNEL_PROB:=0.25}
: ${ENCODER_LAYERDROP:=0.1}
# Misc
: ${NO_SAVE:=false}
: ${SAVE_FREQUENCY:=10}
: ${DISTRIBUTED="-m torch.distributed.launch --nproc_per_node=$NUM_GPUS"}

mkdir -p "$OUTPUT_DIR"

# ARGS+=" --no_epoch_checkpoints"
ARGS+=" --resume"
ARGS+=" --save_frequency $SAVE_FREQUENCY"

ARGS+=" --labels ltr"
ARGS+=" --w2v_path $PRETRAINED_MODEL"

ARGS+=" --data $DATASET_DIR"
ARGS+=" --train_subset $TRAIN_SUBSET"
ARGS+=" --valid_subset $VALID_SUBSET"
ARGS+=" --output_dir $OUTPUT_DIR"
ARGS+=" --ema $EMA"
ARGS+=" --adam_eps 1e-8"
ARGS+=" --lr $LEARNING_RATE"
ARGS+=" --lr_policy exp"
ARGS+=" --initial_lr_scale 0.01"
ARGS+=" --final_lr_scale 0.05"
ARGS+=" --warmup_updates $WARMUP_UPDATES"
ARGS+=" --hold_updates $HOLD_UPDATES"
ARGS+=" --max_update $MAX_UPDATE"
ARGS+=" --num_concat_batches $NUM_CONCAT_BATCHES"
ARGS+=" --update_freq $UPDATE_FREQ "
ARGS+=" --max_tokens $MAX_TOKENS"
ARGS+=" --max_tokens_valid $MAX_TOKENS"
ARGS+=" --freeze_finetune_updates $FREEZE_FINETUNE_UPDATES"
# Overrides
ARGS+=" --apply_mask"
ARGS+=" --mask_prob $MASK_PROB"
ARGS+=" --mask_channel_prob $MASK_CHANNEL_PROB"
ARGS+=" --mask_channel_length 64"
ARGS+=" --encoder_layerdrop $ENCODER_LAYERDROP"  # NOTE This is called `layerdrop` in fairseq finetuning yamls
ARGS+=" --activation_dropout 0.1"
ARGS+=" --feature_grad_mult 0.0"
ARGS+=" --dropout_input 0.0"
ARGS+=" --dropout 0.0"
ARGS+=" --weight_decay 0.0"
ARGS+=" --mha pyt"

# float16
[ "$FP16" = true ]                       && ARGS+=" --fp16"
[ "$FP16" = true ]                       && ARGS+=" --fp32_cosine_sim"
[ "$FP16" = true ]                       && ARGS+=" --fp32_conv_norms"
[ "$FP16" = true ]                       && ARGS+=" --fp32_pos_conv"
# bfloat16
[ "$BF16" = true ]                       && ARGS+=" --bf16"
[ "$BF16" = true ]                       && ARGS+=" --fp32_pos_conv"
# Misc
[ -n "$SEED" ]                           && ARGS+=" --seed $SEED"
[ -n "$EPOCHS_THIS_JOB" ]                && ARGS+=" --epochs_this_job $EPOCHS_THIS_JOB"
[ -n "$BATCH_SIZE" ]                     && ARGS+=" --batch_size $BATCH_SIZE"
[ "$CUDNN_BENCHMARK" = true ]            && ARGS+=" --cudnn_benchmark"
[ "$FP32_TRANSFORMER_LAYERNORM" = true ] && ARGS+=" --fp32_transformer_layernorm"
[ "$FP32_MHA_SOFTMAX" = true ]           && ARGS+=" --fp32_mha_softmax"
[ "$FP32_COSINE_SIM" = true ]            && ARGS+=" --fp32_cosine_sim"
[ "$FP32_POS_CONV" = true ]              && ARGS+=" --fp32_pos_conv"
[ "$FP32_CONV_NORMS" = true ]            && ARGS+=" --fp32_conv_norms"
[ "$NO_SAVE" = true ]                    && ARGS+=" --no_save"

echo -e "\nFP16=$FP16, BP16=$BF16, ${NUM_GPUS}x(${MAX_TOKENS}x${NUM_CONCAT_BATCHES})x${UPDATE_FREQ}\n"

set -x
python3 $DISTRIBUTED train.py finetune $ARGS "$@"


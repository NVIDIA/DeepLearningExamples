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

# Pre-trains a LARGE model on LibriSpeech

set -e

export OMP_NUM_THREADS=1
export CUDNN_V8_API_ENABLED=1  # For older containers (~22.01)
export TORCH_CUDNN_V8_API_ENABLED=1

# IO
: ${DATASET_DIR:="/datasets/LibriSpeech"}
: ${TRAIN_SUBSET:="train-full-960"}
: ${VALID_SUBSET:="dev-other"}
: ${OUTPUT_DIR:="results/pretrain_large"}
# Batching
# To best utilize hw, increase batch size by increasing NUM_CONCAT_BATCHES, and lowering UPDATE_FREQ.
# Keep NUM_NODES x $NUM_GPUS x $NUM_CONCAT_BATCHES x $UPDATE_FREQ = 128.
# Note that this script does not control NUM_NODES.
: ${NUM_GPUS:=8}
: ${MAX_TOKENS:=1200000}
: ${NUM_CONCAT_BATCHES:=1}
: ${UPDATE_FREQ:=16}
: ${MIN_SAMPLE_SIZE:=32000}
: ${MAX_SAMPLE_SIZE:=320000}
# Training
# Fairseq 'Wav2Vec 2.0 Large (LV-60 + CV + SWBD + FSH)' model has been trained
# for 800k steps with 25.6k warmup (wav2vec2_large_librivox.yaml sets 1M/32k)
: ${MAX_UPDATE:=800000}
: ${WARMUP_UPDATES:=32000}
: ${OPTIMIZER:=adam}
: ${LEARNING_RATE:=0.005}
: ${LOSS_WEIGHTS:="0.1 0.0"}
: ${FP16:=false}
: ${BF16:=false}
: ${EMA:=0.0}
: ${SEED=""}  # Disable seed - TODO check if it is working
: ${CUDNN_BENCHMARK:=false}
# Model
: ${NORMALIZE:=true}
: ${MASK_PROB:=0.65}
: ${EXTRACTOR_MODE:="layer_norm"}
: ${LAYER_NORM_FIRST:=true}  # enabled in the `large` model
: ${FINAL_DIM:=768}
: ${LATENT_TEMP:="2.0 0.1 0.999995"}
: ${ENCODER_LAYERDROP:=0.0}
: ${DROPOUT_INPUT:=0.0}
: ${DROPOUT_FEATURES:=0.0}
: ${DROPOUT:=0.0}
: ${ATTENTION_DROPOUT:=0.0}
: ${CONV_BIAS:=true}
: ${ENCODER_LAYERS:=24}
: ${ENCODER_EMBED_DIM:=1024}
: ${ENCODER_FFN_EMBED_DIM:=4096}
: ${ENCODER_ATTENTION_HEADS:=16}
: ${FEATURE_GRAD_MULT:=1.0}
# Misc
: ${NO_SAVE:=false}
: ${SAVE_FREQUENCY=1}
: ${DISTRIBUTED="-m torch.distributed.launch --nproc_per_node=$NUM_GPUS"}

mkdir -p "$OUTPUT_DIR"

ARGS+=" --resume"
ARGS+=" --save_frequency $SAVE_FREQUENCY"
ARGS+=" --data $DATASET_DIR"
ARGS+=" --train_subset $TRAIN_SUBSET"
ARGS+=" --valid_subset $VALID_SUBSET"
ARGS+=" --output_dir $OUTPUT_DIR"
ARGS+=" --ema $EMA"
ARGS+=" --optimizer $OPTIMIZER"
ARGS+=" --lr $LEARNING_RATE"
ARGS+=" --clip_norm 25"
ARGS+=" --weight_decay 0.01"
ARGS+=" --lr_policy poly"
ARGS+=" --lr_poly_power 1.0"
ARGS+=" --loss_weights $LOSS_WEIGHTS"
ARGS+=" --warmup_updates $WARMUP_UPDATES"
ARGS+=" --max_update $MAX_UPDATE"
ARGS+=" --num_concat_batches $NUM_CONCAT_BATCHES"
ARGS+=" --update_freq $UPDATE_FREQ "
ARGS+=" --max_tokens $MAX_TOKENS"
ARGS+=" --max_tokens_valid $MAX_TOKENS"
ARGS+=" --skip_invalid_size_inputs_valid_test"  # XXX ??? ??? ???
ARGS+=" --infonce"
ARGS+=" --min_sample_size $MIN_SAMPLE_SIZE"
ARGS+=" --max_sample_size $MAX_SAMPLE_SIZE"
ARGS+=" --mask_prob $MASK_PROB"
ARGS+=" --quantize_targets"
ARGS+=" --extractor_mode $EXTRACTOR_MODE"
ARGS+=" --final_dim $FINAL_DIM"
ARGS+=" --latent_temp $LATENT_TEMP"
ARGS+=" --encoder_layerdrop $ENCODER_LAYERDROP"
ARGS+=" --dropout_input $DROPOUT_INPUT"
ARGS+=" --dropout_features $DROPOUT_FEATURES"
ARGS+=" --dropout $DROPOUT"
ARGS+=" --attention_dropout $ATTENTION_DROPOUT"
ARGS+=" --encoder_layers $ENCODER_LAYERS"
ARGS+=" --encoder_embed_dim $ENCODER_EMBED_DIM"
ARGS+=" --encoder_ffn_embed_dim $ENCODER_FFN_EMBED_DIM"
ARGS+=" --encoder_attention_heads $ENCODER_ATTENTION_HEADS"
ARGS+=" --feature_grad_mult $FEATURE_GRAD_MULT"
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
[ "$NORMALIZE" = true ]                  && ARGS+=" --normalize"
[ "$CONV_BIAS" = true ]                  && ARGS+=" --conv_bias"
[ "$LAYER_NORM_FIRST" = true ]           && ARGS+=" --layer_norm_first"
[ -n "$SEED" ]                           && ARGS+=" --seed $SEED"
[ -n "$EPOCHS_THIS_JOB" ]                && ARGS+=" --epochs_this_job $EPOCHS_THIS_JOB"
[ "$CUDNN_BENCHMARK" = true ]            && ARGS+=" --cudnn_benchmark"
[ "$FP32_TRANSFORMER_LAYERNORM" = true ] && ARGS+=" --fp32_transformer_layernorm"
[ "$FP32_MHA_SOFTMAX" = true ]           && ARGS+=" --fp32_mha_softmax"
[ "$FP32_COSINE_SIM" = true ]            && ARGS+=" --fp32_cosine_sim"
[ "$FP32_POS_CONV" = true ]              && ARGS+=" --fp32_pos_conv"
[ "$FP32_CONV_NORMS" = true ]            && ARGS+=" --fp32_conv_norms"
[ -n "$HOURGLASS_CONFIG" ]               && ARGS+=" --hourglass_transformer $HOURGLASS_CONFIG"
[ "$NO_SAVE" = true ]                    && ARGS+=" --no_save"

echo -e "\nFP16=$FP16, BP16=$BF16, ${NUM_GPUS}x(${MAX_TOKENS}x${NUM_CONCAT_BATCHES})x${UPDATE_FREQ}\n"

set -x
python3 $DISTRIBUTED train.py pretrain $ARGS "$@"

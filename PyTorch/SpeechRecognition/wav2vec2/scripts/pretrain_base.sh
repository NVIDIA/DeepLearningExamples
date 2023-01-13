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

# Pre-trains a BASE model on LibriSpeech

set -a

# IO
: ${OUTPUT_DIR:="results/pretrain_base"}
# Batching
# To best utilize hw, increase batch size by increasing NUM_CONCAT_BATCHES, and lowering UPDATE_FREQ.
# Keep NUM_NODES x $NUM_GPUS x $NUM_CONCAT_BATCHES x $UPDATE_FREQ = 64.
# Note that this script does not control NUM_NODES.
: ${NUM_GPUS:=8}
: ${MAX_TOKENS:=1400000}
: ${NUM_CONCAT_BATCHES:=8}
: ${UPDATE_FREQ:=1}
: ${MAX_SAMPLE_SIZE:=250000}
# Training
: ${MAX_UPDATE:=400000}
: ${LOSS_WEIGHTS:="0.1 10.0"}
: ${LEARNING_RATE:=0.0005}
# Model
: ${NORMALIZE:=false}
: ${MASK_PROB:=0.65}
: ${EXTRACTOR_MODE:="default"}
: ${LAYER_NORM_FIRST:=false}
: ${FINAL_DIM:=256}
: ${LATENT_TEMP:="2.0 0.5 0.999995"}
: ${ENCODER_LAYERDROP:=0.05}
: ${DROPOUT_INPUT:=0.1}
: ${DROPOUT_FEATURES:=0.1}
: ${DROPOUT:=0.1}
: ${ATTENTION_DROPOUT:=0.1}
: ${CONV_BIAS:=false}
: ${ENCODER_LAYERS:=12}
: ${ENCODER_EMBED_DIM:=768}
: ${ENCODER_FFN_EMBED_DIM:=3072}
: ${ENCODER_ATTENTION_HEADS:=12}
: ${FEATURE_GRAD_MULT:=0.1}
: ${HOURGLASS_CONFIG="[2,(8,4),2]"}

bash scripts/pretrain_large.sh "$@"

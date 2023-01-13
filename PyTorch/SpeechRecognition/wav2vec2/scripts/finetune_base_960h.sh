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

set -a

# A100 80GiB BF16: NUM_CONCAT_BATCHES=1
# A100 80GiB TF32: NUM_CONCAT_BATCHES=1

# IO
: ${DATASET_DIR:="/datasets/LibriSpeech"}
: ${TRAIN_SUBSET:="train-full-960"}
: ${OUTPUT_DIR:="results/finetune_base_960h"}
: ${PRETRAINED_MODEL:=results/pretrain_base/wav2vec2_update400000.pt}
# Batching
: ${NUM_GPUS:=8}
: ${MAX_TOKENS:=3200000}
: ${NUM_CONCAT_BATCHES:=1}
: ${UPDATE_FREQ:=1}
# Training
: ${LEARNING_RATE:=0.0001}
: ${FREEZE_FINETUNE_UPDATES:=0}
: ${MAX_UPDATE:=320000}
: ${MASK_CHANNEL_PROB:=0.1}
: ${MASK_PROB:=0.5}

bash scripts/finetune_vox_960h.sh "$@"

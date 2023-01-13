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

# A100 80GiB FP16: UPDATE_FREQ=1
# A100 80GiB TF32: UPDATE_FREQ=1

# IO
: ${DATASET_DIR:="/datasets/LibriSpeech"}
: ${TRAIN_SUBSET:="train-clean-100"}
: ${OUTPUT_DIR:="results/finetune_large_100h"}
# Batching
# We train with effective world_size=16; the reference sets for world_size=20
: ${NUM_GPUS:=8}
: ${MAX_TOKENS:=1280000}
: ${NUM_CONCAT_BATCHES:=2}
: ${UPDATE_FREQ:=1}
# Training
: ${MAX_UPDATE:=80000}
: ${MASK_CHANNEL_PROB:=0.5}
: ${MASK_PROB:=0.5}

bash scripts/finetune_vox_960h.sh "$@"

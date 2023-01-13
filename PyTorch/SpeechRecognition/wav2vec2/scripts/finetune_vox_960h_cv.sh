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

# The model `Wav2Vec 2.0 Large (LV-60 + CV + SWBD + FSH)` fine-tuned on LS960
# has these changes wrt `wav2vec2_large_librivox.yaml`

: ${MAX_UPDATE:=80000}
: ${FREEZE_FINETUNE_UPDATES:=0}
: ${LEARNING_RATE:=0.00002}
: ${MASK_PROB:=0.25}
: ${MASK_CHANNEL_PROB:=0.5}

# Other changes (minor)
# --clip_norm=0  # =25
# --required_seq_len_multiple=1  # =2

bash scripts/finetune_vox_960h.sh

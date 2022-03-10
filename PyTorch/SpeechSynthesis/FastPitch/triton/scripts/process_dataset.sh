#!/usr/bin/env bash
# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

DATASET_DIR="${DATASETS_DIR}/LJSpeech-1.1/LJSpeech-1.1_fastpitch"
: ${F0_METHOD:="pyin"}
: ${ARGS="--extract-mels"}


if [ ! -d "${DATASET_DIR}/mels" ]; then

    python prepare_dataset.py \
        --wav-text-filelists filelists/ljs_audio_text_val.txt \
        --n-workers 16 \
        --batch-size 1 \
        --dataset-path $DATASET_DIR \
        --extract-pitch \
	--f0-method $F0_METHOD \
	$ARGS
fi

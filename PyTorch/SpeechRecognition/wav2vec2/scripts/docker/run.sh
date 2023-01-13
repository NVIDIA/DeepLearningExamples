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


SCRIPT_DIR=$(cd $(dirname $0); pwd)

: ${DATASET_DIR:=$SCRIPT_DIR/../../datasets}

set -eux

docker run -it --rm \
    --gpus all \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --ipc=host \
    -v "$DATASET_DIR:/datasets" \
    -v "$SCRIPT_DIR/../..:/workspace/wav2vec2" \
    wav2vec2:latest bash

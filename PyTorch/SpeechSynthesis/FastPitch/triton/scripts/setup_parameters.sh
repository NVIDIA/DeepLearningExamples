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
export PRECISION="fp16"
export FORMAT="ts-trace"
export BATCH_SIZE="1,2,4,8"
export BACKEND_ACCELERATOR="none"
export MAX_BATCH_SIZE="8"
export NUMBER_OF_MODEL_INSTANCES="2"
export TRITON_MAX_QUEUE_DELAY="1"
export TRITON_PREFERRED_BATCH_SIZES="4 8"
export SEQUENCE_LENGTH="128"
export CONFIG_FORMAT="torchscript"

#!/usr/bin/env bash

# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
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


mode=${1:-"train"}
num_gpu=${2:-"8"}
batch_size=${3:-"16"}
infer_batch_size=${4:-"$batch_size"}
precision=${5:-"amp"}
SQUAD_VERSION=${6:-"1.1"}
squad_dir=${7:-"/workspace/electra/data/download/squad/v$SQUAD_VERSION"}
OUT_DIR=${8:-"results/"}
init_checkpoint=${9:-"None"}
cache_dir=${10:-"$squad_dir"}

bash scripts/run_squad.sh google/electra-base-discriminator 1 $batch_size $infer_batch_size 8e-4 $precision $num_gpu $RANDOM $SQUAD_VERSION $squad_dir $OUT_DIR $init_checkpoint $mode interactive $cache_dir 200

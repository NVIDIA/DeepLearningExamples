#!/bin/bash

# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

CHECKPOINT=${CHECKPOINT:-"LM-TFM/checkpoint_best.pt"}
MODEL=${MODEL:-"base"}
GPU=${GPU:-"v100"}

BATCH_SIZES=(1 2 4 8 16 32)
TYPES=("pytorch" "torchscript")
# "empty" MATH corresponds to fp32
MATHS=("" "--fp16")
MATHS_FULL=("fp32" "fp16")


for (( i = 0; i < ${#TYPES[@]}; i++ )); do
   for (( j = 0; j < ${#BATCH_SIZES[@]}; j++ )); do
      for (( k = 0; k < ${#MATHS[@]}; k++ )); do
         echo type: ${TYPES[i]} batch size: ${BATCH_SIZES[j]} math: ${MATHS[k]}

         DIR="LM-TFM/inference/${GPU}_${BATCH_SIZES[j]}_${MATHS_FULL[k]}_${TYPES[i]}"
         mkdir -p "${DIR}"

         bash run_wt103_"${MODEL}".sh eval 1 \
            --work_dir "${DIR}" \
            --model "${CHECKPOINT}" \
            --type "${TYPES[i]}" \
            --batch_size "${BATCH_SIZES[j]}" \
            --log_interval 1 \
            --no_env \
            "${MATHS[k]}" \
            --save_data \
            "${@:1}"
      done
   done
done

#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

BATCH_SIZES=(1 2 4 8 16 32)
TYPES=("pytorch" "torchscript")
# "empty" MATH corresponds to fp32
MATHS=("" "--fp16")


for (( i = 0; i < ${#TYPES[@]}; i++ )); do
   for (( j = 0; j < ${#BATCH_SIZES[@]}; j++ )); do
      for (( k = 0; k < ${#MATHS[@]}; k++ )); do
         echo type: ${TYPES[i]} batch size: ${BATCH_SIZES[j]} math: ${MATHS[k]}

         taskset -c 0 bash run_wt103_"${MODEL}".sh eval 1 \
            --model "${CHECKPOINT}" \
            --type "${TYPES[i]}" \
            --batch_size "${BATCH_SIZES[j]}" \
            "${MATHS[k]}" \
            --save_data \
            "${@:1}"
      done
   done
done

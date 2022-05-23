#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

bs=40
ema=0.9999

mkdir -p /tmp/evaluate-FP32-8xV100-32G
mpirun -np 8 --allow-run-as-root --bind-to none \
-map-by slot -x LD_LIBRARY_PATH -x PATH \
-mca pml ob1 -mca btl ^openib \
-x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 eval.py \
--val_file_pattern=/workspace/coco/val-* \
--val_json_file=/workspace/coco/annotations/instances_val2017.json \
--ckpt_path=${CKPT:-/checkpoints/emackpt-300} \
--batch_size=$bs \
--amp=False \
--hparams="moving_average_decay=$ema" \
2>&1 | tee /tmp/evaluate-FP32-8xV100-32G/eval.log
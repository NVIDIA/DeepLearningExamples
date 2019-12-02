# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#! /bin/bash

set -e
set -x

python prepare_dataset.py

# performance with AMP
for i in 1 2 4 8 16; do
  horovodrun -np $i -H localhost:$i python3 /code/main.py --train --use_tf_amp --results_dir /data/performance_amp_results/${i}gpu
  rm -rf /tmp/checkpoints
done

# performance without AMP
for i in 1 2 4 8 16; do
  horovodrun -np $i -H localhost:$i python3 /code/main.py --train --results_dir /data/performance_fp32_results/${i}gpu
  rm -rf /tmp/checkpoints
done

# AMP accuracy for multiple seeds
for i in $(seq 20); do
  horovodrun -np 8 -H localhost:8 python3 /code/main.py --train --use_tf_amp  --seed $i --results_dir /data/amp_accuracy_results/seed_${i}
  rm -rf /tmp/checkpoints
done

# FP32 accuracy for multiple seeds
for i in $(seq 20); do
  horovodrun -np 8 -H localhost:8 python3 /code/main.py --train --seed $i --results_dir /data/fp32_accuracy_results/seed_${i}
  rm -rf /tmp/checkpoints
done


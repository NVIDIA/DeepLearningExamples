# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

# This script launches ResNet50 training in FP32 on 1 GPUs using 128 batch size (128 per GPU)
# Usage ./RN50_FP32_1GPU.sh <path to this repository> <path to dataset> <path to results directory>

python $1/main.py --mode=train_and_evaluate --iter_unit=epoch --num_iter=50 --batch_size=128 --warmup_steps=100 --use_cosine_lr --label_smoothing 0.1 --lr_init=0.256 --lr_warmup_epochs=8 --momentum=0.875 --weight_decay=3.0517578125e-05 --data_dir=$2 --results_dir=$3
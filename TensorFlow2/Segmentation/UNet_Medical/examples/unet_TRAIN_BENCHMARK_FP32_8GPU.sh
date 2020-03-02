# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

# This script launches U-Net run in FP32 on 8 GPUs for training benchmarking. Usage:
# bash unet_TRAIN_BENCHMARK_FP32_8GPU.sh <path to dataset> <path to results directory> <batch size>

horovodrun -np 8 python main.py --data_dir $1 --model_dir $2 --batch_size $3 --exec_mode train --augment --benchmark --warmup_steps 200 --max_steps 1000 --use_xla
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

# This script launches U-Net run in FP32 on 8 GPUs and runs 5-fold cross-validation training for 40000 iterations.
# Usage:
# bash unet_TRAIN_FP32_8GPU.sh <path to dataset> <path to results directory> <batch size>

horovodrun -np 8 python main.py --data_dir $1 --model_dir $2 --log_every 100 --max_steps 40000 --batch_size $3 --exec_mode train_and_evaluate --crossvalidation_idx 0 --augment --use_xla > $2/log_FP32_8GPU_fold0.txt
horovodrun -np 8 python main.py --data_dir $1 --model_dir $2 --log_every 100 --max_steps 40000 --batch_size $3 --exec_mode train_and_evaluate --crossvalidation_idx 1 --augment --use_xla > $2/log_FP32_8GPU_fold1.txt
horovodrun -np 8 python main.py --data_dir $1 --model_dir $2 --log_every 100 --max_steps 40000 --batch_size $3 --exec_mode train_and_evaluate --crossvalidation_idx 2 --augment --use_xla > $2/log_FP32_8GPU_fold2.txt
horovodrun -np 8 python main.py --data_dir $1 --model_dir $2 --log_every 100 --max_steps 40000 --batch_size $3 --exec_mode train_and_evaluate --crossvalidation_idx 3 --augment --use_xla > $2/log_FP32_8GPU_fold3.txt
horovodrun -np 8 python main.py --data_dir $1 --model_dir $2 --log_every 100 --max_steps 40000 --batch_size $3 --exec_mode train_and_evaluate --crossvalidation_idx 4 --augment --use_xla > $2/log_FP32_8GPU_fold4.txt
python utils/parse_results.py --model_dir $2 --exec_mode convergence --env FP32_8GPU
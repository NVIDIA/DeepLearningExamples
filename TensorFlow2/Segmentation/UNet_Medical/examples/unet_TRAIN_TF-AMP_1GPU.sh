# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

# This script launches U-Net run in TF-AMP on 1 GPU and runs 5-fold cross-validation training for 6400 iterations.
# Usage:
# bash unet_TRAIN_TF-AMP_1GPU.sh <path to dataset> <path to results directory> <batch size>

horovodrun -np 1 python main.py --data_dir $1 --model_dir $2 --log_every 100 --max_steps 6400 --batch_size $3 --exec_mode train_and_evaluate --fold 0 --augment --xla --amp > $2/log_TF-AMP_1GPU_fold0.txt
horovodrun -np 1 python main.py --data_dir $1 --model_dir $2 --log_every 100 --max_steps 6400 --batch_size $3 --exec_mode train_and_evaluate --fold 1 --augment --xla --amp > $2/log_TF-AMP_1GPU_fold1.txt
horovodrun -np 1 python main.py --data_dir $1 --model_dir $2 --log_every 100 --max_steps 6400 --batch_size $3 --exec_mode train_and_evaluate --fold 2 --augment --xla --amp > $2/log_TF-AMP_1GPU_fold2.txt
horovodrun -np 1 python main.py --data_dir $1 --model_dir $2 --log_every 100 --max_steps 6400 --batch_size $3 --exec_mode train_and_evaluate --fold 3 --augment --xla --amp > $2/log_TF-AMP_1GPU_fold3.txt
horovodrun -np 1 python main.py --data_dir $1 --model_dir $2 --log_every 100 --max_steps 6400 --batch_size $3 --exec_mode train_and_evaluate --fold 4 --augment --xla --amp > $2/log_TF-AMP_1GPU_fold4.txt
python runtime/parse_results.py --model_dir $2 --exec_mode convergence --env TF-AMP_1GPU
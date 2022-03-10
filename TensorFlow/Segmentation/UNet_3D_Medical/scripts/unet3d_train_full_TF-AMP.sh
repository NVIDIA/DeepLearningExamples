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

# This script launches 3D-UNet run 5-fold cross-validation TF-AMP training for 16000 iterations each.
# Usage:
# bash examples/unet3d_train_full_TF-AMP.sh <number/of/gpus> <path/to/dataset> <path/to/results/directory> <batch/size>

horovodrun -np $1 python main.py --data_dir $2 --model_dir $3 --log_dir $3/log.json --exec_mode train_and_evaluate --max_steps 16000 --augment --batch_size $4 --fold 0 --use_xla --use_amp > $3/log_TF-AMP_$1GPU_fold0.txt
horovodrun -np $1 python main.py --data_dir $2 --model_dir $3 --log_dir $3/log.json --exec_mode train_and_evaluate --max_steps 16000 --augment --batch_size $4 --fold 1 --use_xla --use_amp > $3/log_TF-AMP_$1GPU_fold1.txt
horovodrun -np $1 python main.py --data_dir $2 --model_dir $3 --log_dir $3/log.json --exec_mode train_and_evaluate --max_steps 16000 --augment --batch_size $4 --fold 2 --use_xla --use_amp > $3/log_TF-AMP_$1GPU_fold2.txt
horovodrun -np $1 python main.py --data_dir $2 --model_dir $3 --log_dir $3/log.json --exec_mode train_and_evaluate --max_steps 16000 --augment --batch_size $4 --fold 3 --use_xla --use_amp > $3/log_TF-AMP_$1GPU_fold3.txt
horovodrun -np $1 python main.py --data_dir $2 --model_dir $3 --log_dir $3/log.json --exec_mode train_and_evaluate --max_steps 16000 --augment --batch_size $4 --fold 4 --use_xla --use_amp > $3/log_TF-AMP_$1GPU_fold4.txt

python runtime/parse_results.py --model_dir $3 --env TF-AMP_$1GPU

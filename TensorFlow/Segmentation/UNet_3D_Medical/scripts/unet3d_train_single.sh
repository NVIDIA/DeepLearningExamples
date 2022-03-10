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

# This script launches 3D-UNet run FP32 training on fold 0 for 16000 iterations each.
# Usage:
# bash examples/unet3d_train_single.sh <number/of/gpus> <path/to/dataset> <path/to/results/directory> <batch/size>

horovodrun -np $1 python main.py --data_dir $2 --model_dir $3 --exec_mode train_and_evaluate --augment --max_steps 16000 --batch_size $4 --xla --fold 0
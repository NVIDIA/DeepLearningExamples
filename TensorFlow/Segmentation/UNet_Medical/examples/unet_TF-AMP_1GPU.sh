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

# This script launches U-Net run in FP16 on 1 GPU and trains for 6400 iterations batch_size 8. Usage:
# bash unet_TF-AMP_1GPU.sh <path to dataset> <path to results directory>

horovodrun -np 1 python main.py --data_dir $1 --model_dir $2 --log_every 100 --max_steps 6400 --batch_size 8 --exec_mode train_and_evaluate --crossvalidation_idx 0 --augment --xla --amp
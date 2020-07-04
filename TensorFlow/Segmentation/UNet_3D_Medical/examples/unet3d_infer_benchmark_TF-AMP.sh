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

# This script launches 3D-UNet run TF-AMP inference benchmark.
# Usage:
# bash examples/unet3d_infer_benchmark_TF-AMP.sh <path/to/dataset> <path/to/results/directory> <batch/size>

python main.py --data_dir $1 --model_dir $2 --exec_mode predict --warmup_steps 20 --fold 0 --batch_size $3 --benchmark --amp --xla
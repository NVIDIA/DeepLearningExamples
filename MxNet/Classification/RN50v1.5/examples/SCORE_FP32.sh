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


# This script score ResNet50 checkpoint in FP32 on 1 GPUs using 64 batch size
# Usage ./SCORE_FP32.sh <model prefix> <epoch> <additionals flags>

./runner -n 1 -b 64 --dtype float32 --only-inference --model-prefix $1 --load-epoch $2 -e 1 ${@:3}

# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
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

python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 train.py \
    --epochs 90 \
    --amp \
    --scale-loss 128.0 \
    --use-dynamic-loss-scaling \
    --data-layout NHWC \
    --fuse-resunit \
    --inference-dir ./inference_amp

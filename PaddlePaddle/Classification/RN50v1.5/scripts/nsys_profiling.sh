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

# Nsys Profile Flags
export ENABLE_PROFILE=1
export PROFILE_START_STEP=100
export PROFILE_STOP_STEP=120

NSYS_CMD=" \
        nsys profile --stats=true \
        --output ./log/%p.qdrep \
        --force-overwrite true \
        -t cuda,nvtx,osrt,cudnn,cublas \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --gpu-metrics-device=0 \
        --sample=cpu \
        -d 60 \
        --kill=sigkill \
        -x true"

PADDLE_CMD=" \
        python -m paddle.distributed.launch \
        --gpus=0,1,2,3,4,5,6,7 \
        train.py \
        --epochs 1"

if [[ ${ENABLE_PROFILE} -ge 1 ]]; then
        ${NSYS_CMD} ${PADDLE_CMD}
else
        ${PADDLE_CMD}
fi
export ENABLE_PROFILE=0

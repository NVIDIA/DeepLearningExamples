#!/bin/bash 

# Copyright (c) 2019-2021 NVIDIA CORPORATION. All rights reserved.
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

NV_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-"0"}

# Start Triton server 
docker run --rm -it \
   --gpus $NV_VISIBLE_DEVICES \
   --shm-size=1g \
   --ulimit memlock=-1 \
   --ulimit stack=67108864 \
   -p8000:8000 \
   -p8001:8001 \
   -p8002:8002 \
   --name trt_server_asr \
   -v $PWD/data:/data \
   -v $PWD/model-repo:/mnt/model-repo \
   triton_kaldi_server

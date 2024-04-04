# Copyright (c) 2021-2024 NVIDIA CORPORATION. All rights reserved.
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

DOCKER_BRIDGE=${1:-"bridge"}
NAV_DIR=${2:-"/workspace/"}
NV_VISIBLE_DEVICES=${3:-"0"}

# Start TRITON server in detached state
docker run --rm -d \
   --gpus device=${NV_VISIBLE_DEVICES} \
   --shm-size=1g \
   --ulimit memlock=-1 \
   --ulimit stack=67108864 \
   --network=${DOCKER_BRIDGE} \
   -p 8000:8000 \
   -p 8001:8001 \
   -p 8002:8002 \
   --name trt_server_cont \
   -v ${NAV_DIR}/navigator_workspace/model-store/:/models \
   nvcr.io/nvidia/tritonserver:22.04-py3 tritonserver --model-store=/models --log-verbose=1 --exit-on-error=true --strict-model-config=false --cuda-memory-pool-byte-size 0:10000000000

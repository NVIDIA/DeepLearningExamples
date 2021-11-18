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

# Start Triton server container for download - need some kaldi tools
docker run --rm \
   --shm-size=1g \
   --ulimit memlock=-1 \
   --ulimit stack=67108864 \
   -v $PWD/data:/mnt/data \
   triton_kaldi_server /workspace/scripts/docker/dataset_setup.sh $(id -u) $(id -g)

#   --user $(id -u):$(id -g) \

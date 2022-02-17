# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
#!/bin/bash
# Install Docker
. /etc/os-release && \
curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add - && \
echo "deb [arch=amd64] https://download.docker.com/linux/debian buster stable" > /etc/apt/sources.list.d/docker.list && \
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey| apt-key add - && \
curl -s -L https://nvidia.github.io/nvidia-docker/$ID$VERSION_ID/nvidia-docker.list > /etc/apt/sources.list.d/nvidia-docker.list && \
apt-get update && \
apt-get install -y docker-ce docker-ce-cli containerd.io nvidia-docker2

# Install packages
pip install -r triton/runner/requirements.txt

# Evaluate Runner
python3 -m "triton.runner.__main__" \
    --config-path "triton/runner/config_NVIDIA-T4.yaml" \
    --device 0
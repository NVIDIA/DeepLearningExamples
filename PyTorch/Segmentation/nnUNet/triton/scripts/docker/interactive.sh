#Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

docker run -it --rm \
    --gpus "device=all" \
    --net=host \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e WORKDIR="$(pwd)" \
    -e PYTHONPATH=$(pwd) \
    -v $(pwd):$(pwd) \
    -v /mnt/nvdl/usr/jzarzycki/nnunet_pyt/results:/data \
    -v /mnt/nvdl/usr/jzarzycki/nnunet_pyt/results:/results \
    -w $(pwd) \
    nnunet:latest bash

#!/bin/bash

# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#****** instantiate a container on my workstation
set -euxo pipefail
arg1=$1
nvidia-docker run -it --rm --net=host --runtime=nvidia --ipc=host --cap-add=SYS_PTRACE --cap-add SYS_ADMIN --cap-add DAC_READ_SEARCH --security-opt seccomp=unconfined \
	-v $(pwd)/:/workspace/ \
	-v "/home/ntajbakhsh/data/imagenet_tfrecords/":/data/ \
	-v "/home/ntajbakhsh/data/imagenet_infer/":/infer_data/ \
	"$arg1" # nvcr.io/nvidian/efficientnet-tf2:v2-merged-tf2-py3

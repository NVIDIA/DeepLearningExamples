#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

set -e

if [[ "$(docker ps | grep trtis_kaldi_server | wc -l)" == "0" ]]; then
	printf "\nThe Triton server is currently not running. Please run scripts/docker/launch_server.sh\n\n"
	exit 1
fi

printf "\nOffline benchmarks:\n"

scripts/docker/launch_client.sh -i 5 -c 1000

printf "\nOnline benchmarks:\n"

scripts/docker/launch_client.sh -i 10 -c 700 -o
scripts/docker/launch_client.sh -i 10 -c 400 -o

#!/usr/bin/env bash

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
NV_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-"all"}

to_download=${1:-"wiki_only"} # By default, we don't download BooksCorpus dataset due to recent issues with the host server

docker run --gpus $NV_VISIBLE_DEVICES \
    --rm -it \
    --net=host \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $PWD:/workspace/bert \
    bert bash -c "bash data/create_datasets_from_start.sh ${to_download}"
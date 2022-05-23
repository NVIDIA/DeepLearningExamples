#!/bin/bash

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


# Get local process ID from OpenMPI or alternatively from SLURM
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if [ -n "${OMPI_COMM_WORLD_LOCAL_RANK:-}" ]; then
        LOCAL_RANK="${OMPI_COMM_WORLD_LOCAL_RANK}"
    elif [ -n "${SLURM_LOCALID:-}" ]; then
        LOCAL_RANK="${SLURM_LOCALID}"
    fi
    export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}
fi

exec "$@"

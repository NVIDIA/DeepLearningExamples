#!/bin/bash

# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
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


# the environment variables to run spark job
# should modify below environment variables

# below numbers should be adjusted according to the resource of your running environment
# set the total number of CPU cores, spark can use
export TOTAL_CORES=80

# set the number of executors
export NUM_EXECUTORS=16

# the cores for each executor, it'll be calculated
export NUM_EXECUTOR_CORES=$((${TOTAL_CORES}/${NUM_EXECUTORS}))

# unit: GB,  set the max memory you want to use
export TOTAL_MEMORY=800

# unit: GB, set the memory for driver
export DRIVER_MEMORY=32

# the memory per executor
export EXECUTOR_MEMORY=$(((${TOTAL_MEMORY}-${DRIVER_MEMORY})/${NUM_EXECUTORS}-16))

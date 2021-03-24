#!/bin/bash

# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
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

#########################################################################
# File Name: run_spark.sh


echo "Input mode option: $1"
if [ "$1" = "CPU" ]
then
    echo "Run with CPU.";
    shift
    ./run_spark_cpu.sh ${@}
elif [ "$1" = "GPU" ]
then
    echo "Run with GPU.";
    shift
    if [ "$DGX_VERSION" = "DGX-2" ]
    then
        ./run_spark_gpu_DGX-2.sh ${@}
    else
        ./run_spark_gpu_DGX-A100.sh ${@}
    fi
else
   echo "Please choose mode (CPU/GPU).";
fi

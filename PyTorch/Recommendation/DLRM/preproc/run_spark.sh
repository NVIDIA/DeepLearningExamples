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

#########################################################################
# File Name: run_spark.sh


echo "Input mode option: $1"
if [ "$1" = "CPU" ]
then
   echo "Run with CPU.";
   shift
   ./run_spark_cpu.sh ${@}
elif [ "$1" = "DGX2" ]
then
   echo "Run with GPU.";
   shift
   ./run_spark_gpu.sh ${@} DGX2
else
   echo "Please choose mode (CPU/DGX2).";
fi

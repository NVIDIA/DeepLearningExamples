#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

if [ $# -lt 2 ] ; then
    echo "usage: $0 raw_dataset prepared_dataset"
    exit 1
fi

cd "$2" &&
python /opt/mxnet/tools/im2rec.py --list --recursive train "$1/train" &&
python /opt/mxnet/tools/im2rec.py --list --recursive val "$1/val" &&
python /opt/mxnet/tools/im2rec.py --pass-through --num-thread 40 train "$1/train" &&
python /opt/mxnet/tools/im2rec.py --pass-through --num-thread 40 val "$1/val" &&
echo "Dataset was prepared succesfully!"

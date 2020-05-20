#!/bin/bash

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

if [ $# -ge 1 ]
then
  PREBATCH_SIZE=$1
else
  PREBATCH_SIZE=4096
fi

echo "Starting preprocessing 1/4..."
time python -m preproc.preproc1
echo "Preprocessing 1/4 done.\n"
echo "Starting preprocessing 2/4..."
time python -m preproc.preproc2
echo "Preprocessing 2/4 done.\n"
echo "Starting preprocessing 3/4..."
time python -m preproc.preproc3
echo "Preprocessing 3/4 done.\n"
echo "Starting preprocessing 4/4..."
time python -m preproc.preproc4 --prebatch_size ${PREBATCH_SIZE}
echo "Preprocessing 4/4 done.\n"

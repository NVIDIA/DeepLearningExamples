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

SRC_DIR=${1}
DST_DIR=${2}

echo "Creating training file indexes"
mkdir -p ${DST_DIR}

for file in ${SRC_DIR}/train-*; do
    BASENAME=$(basename $file)
    DST_NAME=$DST_DIR/$BASENAME

    echo "Creating index $DST_NAME for $file"
    tfrecord2idx $file $DST_NAME
done

echo "Creating validation file indexes"
for file in ${SRC_DIR}/validation-*; do
    BASENAME=$(basename $file)
    DST_NAME=$DST_DIR/$BASENAME

    echo "Creating index $DST_NAME for $file"
    tfrecord2idx $file $DST_NAME
done

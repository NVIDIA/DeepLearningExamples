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

set -e

function usage() {
  echo "Usage: bash scripts/preproc.sh nvtabular/spark [tfrecords]"
}

if [ ! -d "scripts" ] || [ ! "$(ls -A 'scripts')" ]; then
  echo "You are probably calling this script from wrong directory"
  usage
  exit 1
fi

if [ $# -ne 1 ] && [ $# -ne 2 ]; then
  usage
  exit 1
fi

tfrecords=${2:-40}

if ! [ "$tfrecords" -ge 0 ] 2>/dev/null; then
  echo "Expected tfrecords (${tfrecords}) to be positive integer"
  usage
  exit 1
fi

case "$1" in
  nvtabular)
    time python -m data.outbrain.nvtabular.preproc --workers "${tfrecords}"
    ;;

  spark)
    echo "Starting preprocessing 1/3..."
    time python data/outbrain/spark/preproc1.py
    echo "Starting preprocessing 2/3..."
    time python data/outbrain/spark/preproc2.py
    echo "Starting preprocessing 3/3..."
    time python data/outbrain/spark/preproc3.py --num_train_partitions "${tfrecords}" --num_valid_partitions "${tfrecords}"
    ;;

  *)
    usage
    exit 1
    ;;
esac

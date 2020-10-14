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

MODEL_NAME=${1}
OUTPUT_FILE=${2:-result.csv}

for i in 1 2 4 8 16 32 64 128; do
    echo "Model $MODEL_NAME evaluation with BS $i"
    /workspace/bin/perf_client --max-threads 10 -m $MODEL_NAME -x 1 -p 10000 -v -i gRPC -u localhost:8001 -b $i -l 5000 \
        --concurrency-range 1 -f >(tail -n +2 | sed -e 's/^/BS='${i}',/' >> $OUTPUT_FILE)
done
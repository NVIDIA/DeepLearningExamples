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
set -x

for test_name in more_pos less_pos less_user less_item more_user more_item other_names;
do
    CACHED_DATADIR='/data/cache/ml-20m'
    NEW_DIR=${CACHED_DATADIR}/${test_name}
    echo "Trying to run on modified dataset: $test_name"
    python -m torch.distributed.launch --nproc_per_node=1 --use_env ncf.py --data ${NEW_DIR} --epochs 1
    echo "Model runs on modified dataset: $test_name"
done

for test_sample in '0' '10' '200';
do
    CACHED_DATADIR='/data/cache/ml-20m'
    NEW_DIR=${CACHED_DATADIR}/sample_${test_name}
    echo "Trying to run on dataset with test sampling: $test_sample"
    python -m torch.distributed.launch --nproc_per_node=1 --use_env ncf.py --data ${NEW_DIR} --epochs 1
    echo "Model runs on dataset with test sampling: $test_sample"
done

for online_sample in '0' '1' '10';
do
    CACHED_DATADIR='/data/cache/ml-20m'
    echo "Trying to run with train sampling: $online_sample"
    python -m torch.distributed.launch --nproc_per_node=1 --use_env ncf.py --data ${CACHED_DATADIR} --epochs 1 -n ${online_sample}
    echo "Model runs with train sampling: $online_sample"
done
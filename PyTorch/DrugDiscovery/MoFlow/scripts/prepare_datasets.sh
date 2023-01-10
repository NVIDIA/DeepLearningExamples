#!/bin/bash

# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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


REPO_URL='https://raw.githubusercontent.com/calvin-zcx/moflow'
GIT_HASH='3026b2e9bb8de027f3887deb96ccdd876ba51664'
DATA_DIR="/data"

wget -O "${DATA_DIR}/zinc250k.csv" "${REPO_URL}/${GIT_HASH}/data/zinc250k.csv"
wget -O "${DATA_DIR}/valid_idx_zinc250k.json" "${REPO_URL}/${GIT_HASH}/data/valid_idx_zinc.json"

python ${PWD}/scripts/data_preprocess.py --data_name "zinc250k" --data_dir ${DATA_DIR}

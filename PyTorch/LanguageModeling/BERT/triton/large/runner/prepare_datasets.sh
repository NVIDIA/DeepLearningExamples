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
#!/usr/bin/env bash

mkdir -p datasets/data/squad/v1.1

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O datasets/data/squad/v1.1/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O datasets/data/squad/v1.1/dev-v1.1.json
wget https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/ -O datasets/data/squad/v1.1/evaluate-v1.1.py

if [[ ! -d "/workspace/bert/data/download/google_pretrained_weights" ]]
then
  python3 data/bertPrep.py --action download --dataset google_pretrained_weights
fi

ln -s /workspace/bert/data/download/google_pretrained_weights datasets/data/google_pretrained_weights


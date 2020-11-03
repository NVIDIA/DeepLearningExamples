#! /bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

MAIN_PATH=$PWD

mkdir -p $MAIN_PATH/pytorch/translation/models/

cd $MAIN_PATH/pytorch/translation/models/
if [ ! -f "sentencepiece.model" ] || [ ! -f "averaged-10-epoch.pt" ]; then
    wget -c https://s3.amazonaws.com/opennmt-models/transformer-ende-wmt-pyOnmt.tar.gz
    tar -xzvf transformer-ende-wmt-pyOnmt.tar.gz
    rm transformer-ende-wmt-pyOnmt.tar.gz
fi

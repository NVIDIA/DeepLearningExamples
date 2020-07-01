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

# Install the OpenNMT-tf v1
pip install opennmt-tf==1.25.1

# Download the vocabulary and test data
# wget https://s3.amazonaws.com/opennmt-trainingdata/wmt_ende_sp.tar.gz

# Download the pretrained model
wget https://s3.amazonaws.com/opennmt-models/averaged-ende-ckpt500k.tar.gz

mkdir translation
mkdir translation/ckpt
tar xf averaged-ende-ckpt500k.tar.gz -C translation/ckpt
rm averaged-ende-ckpt500k.tar.gz

# convert the pretrained model to fit our model structure 
python tensorflow/utils/dump_model.py translation/ckpt/model.ckpt-500000

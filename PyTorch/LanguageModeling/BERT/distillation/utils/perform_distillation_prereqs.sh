# coding=utf-8
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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


#bert-base-uncased
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/bert_pyt_ckpt_base_pretraining_amp_lamb/versions/19.09.0/zip -O bert-base-uncased.zip
unzip bert-base-uncased.zip

#bert-base-uncased-qa
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/bert_pyt_ckpt_base_qa_squad11_amp/versions/19.09.0/zip -O bert-base-uncased-qa.zip
unzip bert-base-uncased-qa.zip

#bert-base-uncased-sst-2
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/bert_pyt_ckpt_base_ft_sst2_amp_128/versions/20.12.0/zip -O bert-base-uncased-sst-2.zip
unzip bert-base-uncased-sst-2.zip

#call convert ckpt on bert-base-uncased
python3 utils/convert_ckpts.py bert_base.pt
python3 utils/convert_ckpts.py bert_base_qa.pt

mkdir -p checkpoints/bert-base-uncased
mv bert_base.pt checkpoints/bert-base-uncased/pytorch_model.bin
cp ../vocab/vocab checkpoints/bert-base-uncased/vocab.txt
cp ../bert_base_config.json checkpoints/bert-base-uncased/config.json

mkdir -p checkpoints/bert-base-uncased-qa
mv bert_base_qa.pt checkpoints/bert-base-uncased-qa/pytorch_model.bin
cp ../vocab/vocab checkpoints/bert-base-uncased-qa/vocab.txt
cp ../bert_base_config.json checkpoints/bert-base-uncased-qa/config.json

mkdir -p checkpoints/bert-base-uncased-sst-2
mv pytorch_model.bin checkpoints/bert-base-uncased-sst-2/pytorch_model.bin
cp ../vocab/vocab checkpoints/bert-base-uncased-sst-2/vocab.txt
cp ../bert_base_config.json checkpoints/bert-base-uncased-sst-2/config.json

wget --content-disposition http://nlp.stanford.edu/data/glove.6B.zip -O glove.zip
unzip glove.zip -d /workspace/bert/data/download/glove

rm bert-base-uncased.zip
rm bert-base-uncased-qa.zip
rm bert-base-uncased-sst-2.zip

rm glove.zip

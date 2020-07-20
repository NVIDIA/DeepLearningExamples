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

if [ "$1" != "ori" ] && [ "$1" != "ths" ] && [ "$1" != "ext" ] && [ "$1" != "thsext" ]; then
    echo "wrong model type"
    echo "[Usage]: bash PATH_TO_THIS_SCRIPT model_type[ori, ths, ext, thsext] data_type[fp32, fp16]"
    exit 1
fi
if [ "$2" != "fp32" ] && [ "$2" != "fp16" ]; then
    echo "wrong data type"
    echo "[Usage]: bash PATH_TO_THIS_SCRIPT model_type[ori, ext] data_type[fp32, fp16]"
    exit 1
fi

batch_size=8
seq_len=384

MAIN_PATH=$PWD

mkdir -p $MAIN_PATH/pytorch/bert_squad/models/bert-large-uncased-whole-word-masking-finetuned-squad
mkdir -p $MAIN_PATH/pytorch/bert_squad/squad_data
mkdir -p $MAIN_PATH/pytorch/bert_squad/output

cd $MAIN_PATH/pytorch/bert_squad/squad_data
# wget -c https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget -c https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

cd $MAIN_PATH/pytorch/bert_squad/models/bert-large-uncased-whole-word-masking-finetuned-squad
if [ ! -f "config.json" ]; then
    wget -c https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json
    mv bert-large-uncased-whole-word-masking-finetuned-squad-config.json config.json
fi
if [ ! -f "pytorch_model.bin" ]; then
    wget -c https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin
    mv bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin pytorch_model.bin
fi
if [ ! -f "vocab.txt" ]; then
    wget -c https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt
    mv bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt vocab.txt
fi
cd $MAIN_PATH

if [ "$1" == "ext" ] || [ "$1" == "thsext" ]; then
    if [ "$2" == "fp32" ]; then
        $MAIN_PATH/bin/encoder_gemm ${batch_size} ${seq_len} 16 64 0
    else
        $MAIN_PATH/bin/encoder_gemm ${batch_size} ${seq_len} 16 64 1
    fi
fi

python $MAIN_PATH/pytorch/run_squad.py \
    --model_name_or_path $MAIN_PATH/pytorch/bert_squad/models/bert-large-uncased-whole-word-masking-finetuned-squad \
    --do_eval \
    --do_lower_case \
    --predict_file $MAIN_PATH/pytorch/bert_squad/squad_data/dev-v1.1.json \
    --output_dir $MAIN_PATH/pytorch/bert_squad/output/ \
    --cache_dir $MAIN_PATH/pytorch/bert_squad/models/ \
    --max_seq_length ${seq_len} \
    --doc_stride 128 \
    --max_query_length 64 \
    --per_gpu_eval_batch_size ${batch_size} \
    --model_type $1 \
    --data_type $2 \

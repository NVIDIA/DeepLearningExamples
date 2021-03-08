#!/usr/bin/env bash

# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
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

dgxa100_8gpu_fp16 ()
{
    init_checkpoint="/workspace/bert/checkpoints/bert_uncased.pt"
    epochs="2.0"
    batch_size="32"
    learning_rate="3e-5"
    precision="fp16"
    num_gpu="8"
    seed="1"
    squad_dir="$BERT_PREP_WORKING_DIR/download/squad/v1.1"
    vocab_file="$BERT_PREP_WORKING_DIR/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt"
    OUT_DIR="/workspace/bert/results/SQuAD"
    echo $init_checkpoint $epochs $batch_size $learning_rate \
     $precision $num_gpu $seed $squad_dir $vocab_file \
     $OUT_DIR
}

dgxa100_8gpu_tf32 ()
{
    init_checkpoint="/workspace/bert/checkpoints/bert_uncased.pt"
    epochs="2.0"
    batch_size="16"
    learning_rate="3e-5"
    precision="tf32"
    num_gpu="8"
    seed="1"
    squad_dir="$BERT_PREP_WORKING_DIR/download/squad/v1.1"
    vocab_file="$BERT_PREP_WORKING_DIR/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt"
    OUT_DIR="/workspace/bert/results/SQuAD"
    echo $init_checkpoint $epochs $batch_size $learning_rate \
     $precision $num_gpu $seed $squad_dir $vocab_file \
     $OUT_DIR
}

# Full SQuAD training configs for NVIDIA DGX-2H (16x NVIDIA V100 32GB GPU)

dgx2_16gpu_fp16 ()
{
    init_checkpoint="/workspace/bert/checkpoints/bert_uncased.pt"
    epochs="2.0"
    batch_size="16"
    learning_rate="3e-5"
    precision="fp16"
    num_gpu="16"
    seed="1"
    squad_dir="$BERT_PREP_WORKING_DIR/download/squad/v1.1"
    vocab_file="$BERT_PREP_WORKING_DIR/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt"
    OUT_DIR="/workspace/bert/results/SQuAD"
    echo $init_checkpoint $epochs $batch_size $learning_rate \
     $precision $num_gpu $seed $squad_dir $vocab_file \
     $OUT_DIR
}

dgx2_16gpu_fp32 ()
{
    init_checkpoint="/workspace/bert/checkpoints/bert_uncased.pt"
    epochs="2.0"
    batch_size="8"
    learning_rate="3e-5"
    precision="fp16"
    num_gpu="16"
    seed="1"
    squad_dir="$BERT_PREP_WORKING_DIR/download/squad/v1.1"
    vocab_file="$BERT_PREP_WORKING_DIR/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt"
    OUT_DIR="/workspace/bert/results/SQuAD"
    echo $init_checkpoint $epochs $batch_size $learning_rate \
     $precision $num_gpu $seed $squad_dir $vocab_file \
     $OUT_DIR
}

# Full SQuAD training configs for NVIDIA DGX-1 (8x NVIDIA V100 16GB GPU)

dgx1_8gpu_fp16 ()
{
    init_checkpoint="/workspace/bert/checkpoints/bert_uncased.pt"
    epochs="2.0"
    batch_size="10"
    learning_rate="3e-5"
    precision="fp16"
    num_gpu="8"
    seed="1"
    squad_dir="$BERT_PREP_WORKING_DIR/download/squad/v1.1"
    vocab_file="$BERT_PREP_WORKING_DIR/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt"
    OUT_DIR="/workspace/bert/results/SQuAD"
    echo $init_checkpoint $epochs $batch_size $learning_rate \
     $precision $num_gpu $seed $squad_dir $vocab_file \
     $OUT_DIR
}

dgx1_8gpu_fp32 ()
{
    init_checkpoint="/workspace/bert/checkpoints/bert_uncased.pt"
    epochs="2.0"
    batch_size="4"
    learning_rate="3e-5"
    precision="fp32"
    num_gpu="8"
    seed="1"
    squad_dir="$BERT_PREP_WORKING_DIR/download/squad/v1.1"
    vocab_file="$BERT_PREP_WORKING_DIR/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt"
    OUT_DIR="/workspace/bert/results/SQuAD"
    echo $init_checkpoint $epochs $batch_size $learning_rate \
     $precision $num_gpu $seed $squad_dir $vocab_file \
     $OUT_DIR
}

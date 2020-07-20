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

dgxa100_8gpu_amp ()
{
    electra_model="google/electra-base-discriminator"
    epochs="2"
    batch_size="32"
    learning_rate="8e-4"
    precision="amp"
    num_gpu="8"
    seed="1"
    SQUAD_VERSION="1.1"
    squad_dir="/workspace/electra/data/download/squad/v$SQUAD_VERSION"
    OUT_DIR="results/"
    init_checkpoint="checkpoints/electra_base_qa_v2_False_epoch_2_ckpt"
    echo $electra_model $epochs $batch_size $batch_size $learning_rate \
     $precision $num_gpu $seed $SQUAD_VERSION $squad_dir \
     $OUT_DIR $init_checkpoint
}

dgxa100_8gpu_tf32 ()
{
    electra_model="google/electra-base-discriminator"
    epochs="2"
    batch_size="32"
    learning_rate="8e-4"
    precision="tf32"
    num_gpu="8"
    seed="1"
    SQUAD_VERSION="1.1"
    squad_dir="/workspace/electra/data/download/squad/v$SQUAD_VERSION"
    OUT_DIR="results/"
    init_checkpoint="checkpoints/electra_base_qa_v2_False_epoch_2_ckpt"
    echo $electra_model $epochs $batch_size $batch_size $learning_rate \
     $precision $num_gpu $seed $SQUAD_VERSION $squad_dir \
     $OUT_DIR $init_checkpoint
}

# Full SQuAD training configs for NVIDIA DGX-2H (16x NVIDIA V100 32GB GPU)

dgx2_16gpu_amp ()
{
    electra_model="google/electra-base-discriminator"
    epochs="2"
    batch_size="32"
    learning_rate="1e-3"
    precision="amp"
    num_gpu="16"
    seed="1"
    SQUAD_VERSION="1.1"
    squad_dir="/workspace/electra/data/download/squad/v$SQUAD_VERSION"
    OUT_DIR="results/"
    init_checkpoint="checkpoints/electra_base_qa_v2_False_epoch_2_ckpt"
    echo $electra_model $epochs $batch_size $batch_size $learning_rate \
     $precision $num_gpu $seed $SQUAD_VERSION $squad_dir \
     $OUT_DIR $init_checkpoint
}

dgx2_16gpu_fp32 ()
{
    electra_model="google/electra-base-discriminator"
    epochs="2"
    batch_size="32"
    learning_rate="1e-3"
    precision="fp32"
    num_gpu="16"
    seed="1"
    SQUAD_VERSION="1.1"
    squad_dir="/workspace/electra/data/download/squad/v$SQUAD_VERSION"
    OUT_DIR="results/"
    init_checkpoint="checkpoints/electra_base_qa_v2_False_epoch_2_ckpt"
    echo $electra_model $epochs $batch_size $batch_size $learning_rate \
     $precision $num_gpu $seed $SQUAD_VERSION $squad_dir \
     $OUT_DIR $init_checkpoint
}

# Full SQuAD training configs for NVIDIA DGX-1 (8x NVIDIA V100 16GB GPU)

dgx1_8gpu_amp ()
{
    electra_model="google/electra-base-discriminator"
    epochs="2"
    batch_size="16"
    learning_rate="4e-4"
    precision="amp"
    num_gpu="8"
    seed="1"
    SQUAD_VERSION="1.1"
    squad_dir="/workspace/electra/data/download/squad/v$SQUAD_VERSION"
    OUT_DIR="results/"
    init_checkpoint="checkpoints/electra_base_qa_v2_False_epoch_2_ckpt"
    echo $electra_model $epochs $batch_size $batch_size $learning_rate \
     $precision $num_gpu $seed $SQUAD_VERSION $squad_dir \
     $OUT_DIR $init_checkpoint
}

dgx1_8gpu_fp32 ()
{
    electra_model="google/electra-base-discriminator"
    epochs="2"
    batch_size="8"
    learning_rate="3e-4"
    precision="fp32"
    num_gpu="8"
    seed="1"
    SQUAD_VERSION="1.1"
    squad_dir="/workspace/electra/data/download/squad/v$SQUAD_VERSION"
    OUT_DIR="results/"
    init_checkpoint="checkpoints/electra_base_qa_v2_False_epoch_2_ckpt"
    echo $electra_model $epochs $batch_size $batch_size $learning_rate \
     $precision $num_gpu $seed $SQUAD_VERSION $squad_dir \
     $OUT_DIR $init_checkpoint
}

# 1GPU configs

dgxa100_1gpu_amp ()
{
    electra_model="google/electra-base-discriminator"
    epochs="2"
    batch_size="32"
    learning_rate="2e-4"
    precision="amp"
    num_gpu="1"
    seed="1"
    SQUAD_VERSION="1.1"
    squad_dir="/workspace/electra/data/download/squad/v$SQUAD_VERSION"
    OUT_DIR="results/"
    init_checkpoint="checkpoints/electra_base_qa_v2_False_epoch_2_ckpt"
    echo $electra_model $epochs $batch_size $batch_size $learning_rate \
     $precision $num_gpu $seed $SQUAD_VERSION $squad_dir \
     $OUT_DIR $init_checkpoint
}

dgxa100_1gpu_tf32 ()
{
    electra_model="google/electra-base-discriminator"
    epochs="2"
    batch_size="32"
    learning_rate="2e-4"
    precision="tf32"
    num_gpu="1"
    seed="1"
    SQUAD_VERSION="1.1"
    squad_dir="/workspace/electra/data/download/squad/v$SQUAD_VERSION"
    OUT_DIR="results/"
    init_checkpoint="checkpoints/electra_base_qa_v2_False_epoch_2_ckpt"
    echo $electra_model $epochs $batch_size $batch_size $learning_rate \
     $precision $num_gpu $seed $SQUAD_VERSION $squad_dir \
     $OUT_DIR $init_checkpoint
}

# Full SQuAD training configs for NVIDIA DGX-2H (16x NVIDIA V100 32GB GPU)

dgx2_1gpu_amp ()
{
    electra_model="google/electra-base-discriminator"
    epochs="2"
    batch_size="32"
    learning_rate="2e-4"
    precision="amp"
    num_gpu="1"
    seed="1"
    SQUAD_VERSION="1.1"
    squad_dir="/workspace/electra/data/download/squad/v$SQUAD_VERSION"
    OUT_DIR="results/"
    init_checkpoint="checkpoints/electra_base_qa_v2_False_epoch_2_ckpt"
    echo $electra_model $epochs $batch_size $batch_size $learning_rate \
     $precision $num_gpu $seed $SQUAD_VERSION $squad_dir \
     $OUT_DIR $init_checkpoint
}

dgx2_1gpu_fp32 ()
{
    electra_model="google/electra-base-discriminator"
    epochs="2"
    batch_size="32"
    learning_rate="2e-4"
    precision="fp32"
    num_gpu="1"
    seed="1"
    SQUAD_VERSION="1.1"
    squad_dir="/workspace/electra/data/download/squad/v$SQUAD_VERSION"
    OUT_DIR="results/"
    init_checkpoint="checkpoints/electra_base_qa_v2_False_epoch_2_ckpt"
    echo $electra_model $epochs $batch_size $batch_size $learning_rate \
     $precision $num_gpu $seed $SQUAD_VERSION $squad_dir \
     $OUT_DIR $init_checkpoint
}

# Full SQuAD training configs for NVIDIA DGX-1 (8x NVIDIA V100 16GB GPU)

dgx1_1gpu_amp ()
{
    electra_model="google/electra-base-discriminator"
    epochs="2"
    batch_size="16"
    learning_rate="1e-4"
    precision="amp"
    num_gpu="1"
    seed="1"
    SQUAD_VERSION="1.1"
    squad_dir="/workspace/electra/data/download/squad/v$SQUAD_VERSION"
    OUT_DIR="results/"
    init_checkpoint="checkpoints/electra_base_qa_v2_False_epoch_2_ckpt"
    echo $electra_model $epochs $batch_size $batch_size $learning_rate \
     $precision $num_gpu $seed $SQUAD_VERSION $squad_dir \
     $OUT_DIR $init_checkpoint
}

dgx1_1gpu_fp32 ()
{
    electra_model="google/electra-base-discriminator"
    epochs="2"
    batch_size="8"
    learning_rate="1e-4"
    precision="fp32"
    num_gpu="1"
    seed="1"
    SQUAD_VERSION="1.1"
    squad_dir="/workspace/electra/data/download/squad/v$SQUAD_VERSION"
    OUT_DIR="results/"
    init_checkpoint="checkpoints/electra_base_qa_v2_False_epoch_2_ckpt"
    echo $electra_model $epochs $batch_size $batch_size $learning_rate \
     $precision $num_gpu $seed $SQUAD_VERSION $squad_dir \
     $OUT_DIR $init_checkpoint
}

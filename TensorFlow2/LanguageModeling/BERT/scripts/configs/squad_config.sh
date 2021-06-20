#!/usr/bin/env bash
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
# ==============================================================================

# Full SQuAD training configs for NVIDIA DGX A100 (8x NVIDIA A100 40GB GPU)

dgxa100_8gpu_fp16 ()
{
  batch_size=76
  learning_rate=3e-5
  precision=fp16
  use_xla=true
  num_gpu=8
  bert_model="large"
  echo $num_gpu $batch_size $learning_rate $precision $use_xla $bert_model
}

dgxa100_8gpu_tf32 ()
{
  batch_size=38
  learning_rate=7.5e-6
  precision=tf32
  use_xla=true
  num_gpu=8
  bert_model="large"
  echo $num_gpu $batch_size $learning_rate $precision $use_xla $bert_model
}

# Full SQuAD training configs for NVIDIA DGX-2H (16x NVIDIA V100 32GB GPU)

dgx2_16gpu_fp16 ()
{
  batch_size=12
  learning_rate=3.75e-6
  precision=fp16
  use_xla=true
  num_gpu=16
  bert_model="large"
  echo $num_gpu $batch_size $learning_rate $precision $use_xla $bert_model
}

dgx2_16gpu_fp32 ()
{
  batch_size=8
  learning_rate=2.5e-6
  precision=fp32
  use_xla=true
  num_gpu=16
  bert_model="large"
  echo $num_gpu $batch_size $learning_rate $precision $use_xla $bert_model
}

# Full SQuAD training configs for NVIDIA DGX-1 (8x NVIDIA V100 16GB GPU)

dgx1_8gpu_fp16 ()
{
  batch_size=6
  learning_rate=5e-6
  precision=fp16
  use_xla=true
  num_gpu=8
  bert_model="large"
  echo $num_gpu $batch_size $learning_rate $precision $use_xla $bert_model
}

dgx1_8gpu_fp32 ()
{
  batch_size=3
  learning_rate=5e-6
  precision=fp32
  use_xla=true
  num_gpu=8
  bert_model="large"
  echo $num_gpu $batch_size $learning_rate $precision $use_xla $bert_model
}

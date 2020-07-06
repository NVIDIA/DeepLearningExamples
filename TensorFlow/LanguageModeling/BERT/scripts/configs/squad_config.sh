#!/usr/bin/env bash

# Full SQuAD training configs for NVIDIA DGX A100 (8x NVIDIA A100 40GB GPU)

dgxa100_8gpu_fp16 ()
{
  batch_size=32
  learning_rate=5e-6
  precision=fp16
  use_xla=true
  num_gpu=8
  seq_length=384
  doc_stride=128
  bert_model="large"
  echo $batch_size $learning_rate $precision $use_xla $num_gpu $seq_length $doc_stride $bert_model
}

dgxa100_8gpu_tf32 ()
{
  batch_size=16
  learning_rate=5e-6
  precision=tf32
  use_xla=true
  num_gpu=8
  seq_length=384
  doc_stride=128
  bert_model="large"
  echo $batch_size $learning_rate $precision $use_xla $num_gpu $seq_length $doc_stride $bert_model
}

# Full SQuAD training configs for NVIDIA DGX-2H (16x NVIDIA V100 32GB GPU)

dgx2_16gpu_fp16 ()
{
  batch_size=24
  learning_rate=2.5e-6
  precision=fp16
  use_xla=true
  num_gpu=16
  seq_length=384
  doc_stride=128
  bert_model="large"
  echo $batch_size $learning_rate $precision $use_xla $num_gpu $seq_length $doc_stride $bert_model
}

dgx2_16gpu_fp32 ()
{
  batch_size=8
  learning_rate=2.5e-6
  precision=fp32
  use_xla=true
  num_gpu=16
  seq_length=384
  doc_stride=128
  bert_model="large"
  echo $batch_size $learning_rate $precision $use_xla $num_gpu $seq_length $doc_stride $bert_model
}

# Full SQuAD training configs for NVIDIA DGX-1 (8x NVIDIA V100 16GB GPU)

dgx1_8gpu_fp16 ()
{
  batch_size=4
  learning_rate=5e-6
  precision=fp16
  use_xla=true
  num_gpu=8
  seq_length=384
  doc_stride=128
  bert_model="large"
  echo $batch_size $learning_rate $precision $use_xla $num_gpu $seq_length $doc_stride $bert_model
}

dgx1_8gpu_fp32 ()
{
  batch_size=2
  learning_rate=5e-6
  precision=fp32
  use_xla=true
  num_gpu=8
  seq_length=384
  doc_stride=128
  bert_model="large"
  echo $batch_size $learning_rate $precision $use_xla $num_gpu $seq_length $doc_stride $bert_model
}

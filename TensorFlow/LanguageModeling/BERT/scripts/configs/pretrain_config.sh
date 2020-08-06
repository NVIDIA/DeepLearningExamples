#!/usr/bin/env bash

# Full LAMB pretraining configs for NVIDIA DGX A100 (8x NVIDIA A100 40GB GPU)

dgxa100_8gpu_fp16 ()
{
  train_batch_size_phase1=64
  train_batch_size_phase2=16
  eval_batch_size=8
  learning_rate_phase1="7.5e-4"
  learning_rate_phase2="5e-4"
  precision="fp16"
  use_xla="true"
  num_gpus=8
  echo $train_batch_size_phase1 $train_batch_size_phase2 $eval_batch_size $learning_rate_phase1 $learning_rate_phase2 $precision $use_xla $num_gpu
}

dgxa100_8gpu_tf32 ()
{
  train_batch_size_phase1=64
  train_batch_size_phase2=8
  eval_batch_size=8
  learning_rate_phase1="7.5e-4"
  learning_rate_phase2="5e-4"
  precision="tf32"
  use_xla="true"
  num_gpus=8
  echo $train_batch_size_phase1 $train_batch_size_phase2 $eval_batch_size $learning_rate_phase1 $learning_rate_phase2 $precision $use_xla $num_gpu
}

# Full LAMB pretraining configs for NVIDIA DGX-2H (16x NVIDIA V100 32GB GPU)

dgx2_16gpu_fp16 ()
{
  train_batch_size_phase1=64
  train_batch_size_phase2=8
  eval_batch_size=8
  learning_rate_phase1="3.75e-4"
  learning_rate_phase2="2.5e-4"
  precision="fp16"
  use_xla="true"
  num_gpus=16
  echo $train_batch_size_phase1 $train_batch_size_phase2 $eval_batch_size $learning_rate_phase1 $learning_rate_phase2 $precision $use_xla $num_gpu
}

dgx2_16gpu_fp32 ()
{
  train_batch_size_phase1=32
  train_batch_size_phase2=8
  eval_batch_size=8
  learning_rate_phase1="3.75e-4"
  learning_rate_phase2="2.5e-4"
  precision="fp32"
  use_xla="true"
  num_gpus=16
  echo $train_batch_size_phase1 $train_batch_size_phase2 $eval_batch_size $learning_rate_phase1 $learning_rate_phase2 $precision $use_xla $num_gpu
}

# Full LAMB pretraining configs for NVIDIA DGX-1 (8x NVIDIA V100 16GB GPU)

dgx1_8gpu_fp16 ()
{
  train_batch_size_phase1=16
  train_batch_size_phase2=4
  eval_batch_size=8
  learning_rate_phase1="7.5e-4"
  learning_rate_phase2="5e-4"
  precision="fp16"
  use_xla="true"
  num_gpus=8
  echo $train_batch_size_phase1 $train_batch_size_phase2 $eval_batch_size $learning_rate_phase1 $learning_rate_phase2 $precision $use_xla $num_gpu
}

dgx1_8gpu_fp32 ()
{
  train_batch_size_phase1=8
  train_batch_size_phase2=2
  eval_batch_size=8
  learning_rate_phase1="7.5e-4"
  learning_rate_phase2="5e-4"
  precision="fp32"
  use_xla="true"
  num_gpus=8
  echo $train_batch_size_phase1 $train_batch_size_phase2 $eval_batch_size $learning_rate_phase1 $learning_rate_phase2 $precision $use_xla $num_gpu
}

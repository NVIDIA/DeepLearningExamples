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

# Full  pretraining configs for NVIDIA DGX-A100 (8x NVIDIA A100 40GB GPU)

dgxa100_8gpu_amp ()
{
    train_batch_size_p1="176"
    learning_rate_p1="6e-3"
    precision="amp"
    xla="xla"
    num_gpus=8
    warmup_steps_p1="2000"
    train_steps_p1=10000
    save_checkpoint_steps=500
    resume_training="false"
    optimizer="lamb"
    accumulate_gradients="true"
    gradient_accumulation_steps_p1=48
    seed=42
    job_name="electra_lamb_pretraining"
    train_batch_size_p2=24
    learning_rate_p2="4e-3"
    warmup_steps_p2="200"
    train_steps_p2=933
    gradient_accumulation_steps_p2=144
    electra_model="base"
    echo $train_batch_size_p1 $learning_rate_p1 $precision $num_gpus $xla \
         $warmup_steps_p1 $train_steps_p1 $save_checkpoint_steps \
         $resume_training $optimizer $accumulate_gradients  \
         $gradient_accumulation_steps_p1 $seed $job_name \
         $train_batch_size_p2 $learning_rate_p2 \
         $warmup_steps_p2 $train_steps_p2 $gradient_accumulation_steps_p2 \
         $electra_model

}

dgxa100_8gpu_tf32 ()
{
    train_batch_size_p1="88"
    learning_rate_p1="6e-3"
    precision="tf32"
    xla="xla"
    num_gpus=8
    warmup_steps_p1="2000"
    train_steps_p1=10000
    save_checkpoint_steps=500
    resume_training="false"
    optimizer="lamb"
    accumulate_gradients="true"
    gradient_accumulation_steps_p1=96
    seed=42
    job_name="electra_lamb_pretraining"
    train_batch_size_p2=12
    learning_rate_p2="4e-3"
    warmup_steps_p2="200"
    train_steps_p2=933
    gradient_accumulation_steps_p2=288
    electra_model="base"
    echo $train_batch_size_p1 $learning_rate_p1 $precision $num_gpus $xla \
         $warmup_steps_p1 $train_steps_p1 $save_checkpoint_steps \
         $resume_training $optimizer $accumulate_gradients  \
         $gradient_accumulation_steps_p1 $seed $job_name \
         $train_batch_size_p2 $learning_rate_p2 \
         $warmup_steps_p2 $train_steps_p2 $gradient_accumulation_steps_p2 \
         $electra_model 

}


# Full  pretraining configs for NVIDIA DGX-2H (16x NVIDIA V100 32GB GPU)

dgx2_16gpu_amp ()
{
    train_batch_size_p1="176"
    learning_rate_p1="6e-3"
    precision="amp"
    xla="xla"
    num_gpus=16
    warmup_steps_p1="2000"
    train_steps_p1=10000
    save_checkpoint_steps=500
    resume_training="false"
    optimizer="lamb"
    accumulate_gradients="true"
    gradient_accumulation_steps_p1=24
    seed=42
    job_name="electra_lamb_pretraining"
    train_batch_size_p2=24
    learning_rate_p2="4e-3"
    warmup_steps_p2="200"
    train_steps_p2=933
    gradient_accumulation_steps_p2=72
    electra_model="base"
    echo $train_batch_size_p1 $learning_rate_p1 $precision $num_gpus $xla \
         $warmup_steps_p1 $train_steps_p1 $save_checkpoint_steps \
         $resume_training $optimizer $accumulate_gradients  \
         $gradient_accumulation_steps_p1 $seed $job_name \
         $train_batch_size_p2 $learning_rate_p2 \
         $warmup_steps_p2 $train_steps_p2 $gradient_accumulation_steps_p2 \
         $electra_model 

}

dgx2_16gpu_fp32 ()
{
    train_batch_size_p1="88"
    learning_rate_p1="6e-3"
    precision="fp32"
    xla="xla"
    num_gpus=16
    warmup_steps_p1="2000"
    train_steps_p1=10000
    save_checkpoint_steps=500
    resume_training="false"
    optimizer="lamb"
    accumulate_gradients="true"
    gradient_accumulation_steps_p1=48
    seed=42
    job_name="electra_lamb_pretraining"
    train_batch_size_p2=12
    learning_rate_p2="4e-3"
    warmup_steps_p2="200"
    train_steps_p2=933
    gradient_accumulation_steps_p2=144
    electra_model="base"
    echo $train_batch_size_p1 $learning_rate_p1 $precision $num_gpus $xla \
         $warmup_steps_p1 $train_steps_p1 $save_checkpoint_steps \
         $resume_training $optimizer $accumulate_gradients  \
         $gradient_accumulation_steps_p1 $seed $job_name \
         $train_batch_size_p2 $learning_rate_p2 \
         $warmup_steps_p2 $train_steps_p2 $gradient_accumulation_steps_p2 \
         $electra_model 

}

# Full pretraining configs for NVIDIA DGX-1 (8x NVIDIA V100 16GB GPU)

dgx1_8gpu_amp ()
{
    train_batch_size_p1="88"
    learning_rate_p1="6e-3"
    precision="amp"
    xla="xla"
    num_gpus=8
    warmup_steps_p1="2000"
    train_steps_p1=10000
    save_checkpoint_steps=500
    resume_training="false"
    optimizer="lamb"
    accumulate_gradients="true"
    gradient_accumulation_steps_p1=96
    seed=42
    job_name="electra_lamb_pretraining"
    train_batch_size_p2=12
    learning_rate_p2="4e-3"
    warmup_steps_p2="200"
    train_steps_p2=933
    gradient_accumulation_steps_p2=288
    electra_model="base"
    echo $train_batch_size_p1 $learning_rate_p1 $precision $num_gpus $xla \
         $warmup_steps_p1 $train_steps_p1 $save_checkpoint_steps \
         $resume_training $optimizer $accumulate_gradients  \
         $gradient_accumulation_steps_p1 $seed $job_name \
         $train_batch_size_p2 $learning_rate_p2 \
         $warmup_steps_p2 $train_steps_p2 $gradient_accumulation_steps_p2 \
         $electra_model 

}

dgx1_8gpu_fp32 ()
{
    train_batch_size_p1="40"
    learning_rate_p1="6e-3"
    precision="fp32"
    xla="xla"
    num_gpus=8
    warmup_steps_p1="2000"
    train_steps_p1=10000
    save_checkpoint_steps=500
    resume_training="false"
    optimizer="lamb"
    accumulate_gradients="true"
    gradient_accumulation_steps_p1=211
    seed=42
    job_name="electra_lamb_pretraining"
    train_batch_size_p2=6
    learning_rate_p2="4e-3"
    warmup_steps_p2="200"
    train_steps_p2=933
    gradient_accumulation_steps_p2=576
    electra_model="base"
    echo $train_batch_size_p1 $learning_rate_1 $precision $num_gpus $xla \
         $warmup_steps_p1 $train_steps_p1 $save_checkpoint_steps \
         $resume_training $optimizer $accumulate_gradients  \
         $gradient_accumulation_steps_p1 $seed $job_name \
         $train_batch_size_p2 $learning_rate_p2 \
         $warmup_steps_p2 $train_steps_p2 $gradient_accumulation_steps_p2 \
         $electra_model 

}

# Full  pretraining configs for NVIDIA DGX-A100 (1x NVIDIA A100 40GB GPU)

dgxa100_1gpu_amp ()
{
    train_batch_size_p1="176"
    learning_rate_p1="6e-3"
    precision="amp"
    xla="xla"
    num_gpus=1
    warmup_steps_p1="2000"
    train_steps_p1=10000
    save_checkpoint_steps=500
    resume_training="false"
    optimizer="lamb"
    accumulate_gradients="true"
    gradient_accumulation_steps_p1=384
    seed=42
    job_name="electra_lamb_pretraining"
    train_batch_size_p2=24
    learning_rate_p2="4e-3"
    warmup_steps_p2="200"
    train_steps_p2=933
    gradient_accumulation_steps_p2=1152
    electra_model="base"
    echo $train_batch_size_p1 $learning_rate_p1 $precision $num_gpus $xla \
         $warmup_steps_p1 $train_steps_p1 $save_checkpoint_steps \
         $resume_training $optimizer $accumulate_gradients  \
         $gradient_accumulation_steps_p1 $seed $job_name \
         $train_batch_size_p2 $learning_rate_p2 \
         $warmup_steps_p2 $train_steps_p2 $gradient_accumulation_steps_p2 \
         $electra_model 

}

dgxa100_1gpu_tf32 ()
{
    train_batch_size_p1="88"
    learning_rate_p1="6e-3"
    precision="tf32"
    xla="xla"
    num_gpus=1
    warmup_steps_p1="2000"
    train_steps_p1=10000
    save_checkpoint_steps=500
    resume_training="false"
    optimizer="lamb"
    accumulate_gradients="true"
    gradient_accumulation_steps_p1=768
    seed=42
    job_name="electra_lamb_pretraining"
    train_batch_size_p2=12
    learning_rate_p2="4e-3"
    warmup_steps_p2="200"
    train_steps_p2=933
    gradient_accumulation_steps_p2=2304
    electra_model="base"
    echo $train_batch_size_p1 $learning_rate_p1 $precision $num_gpus $xla \
         $warmup_steps_p1 $train_steps_p1 $save_checkpoint_steps \
         $resume_training $optimizer $accumulate_gradients  \
         $gradient_accumulation_steps_p1 $seed $job_name \
         $train_batch_size_p2 $learning_rate_p2 \
         $warmup_steps_p2 $train_steps_p2 $gradient_accumulation_steps_p2 \
         $electra_model 

}

# Full  pretraining configs for NVIDIA DGX-2H (1x NVIDIA V100 32GB GPU)

dgx2_1gpu_amp ()
{
    train_batch_size_p1="176"
    learning_rate_p1="6e-3"
    precision="amp"
    xla="xla"
    num_gpus=1
    warmup_steps_p1="2000"
    train_steps_p1=10000
    save_checkpoint_steps=500
    resume_training="false"
    optimizer="lamb"
    accumulate_gradients="true"
    gradient_accumulation_steps_p1=384
    seed=42
    job_name="electra_lamb_pretraining"
    train_batch_size_p2=24
    learning_rate_p2="4e-3"
    warmup_steps_p2="200"
    train_steps_p2=933
    gradient_accumulation_steps_p2=1152
    electra_model="base"
    echo $train_batch_size_p1 $learning_rate_p1 $precision $num_gpus $xla \
         $warmup_steps_p1 $train_steps_p1 $save_checkpoint_steps \
         $resume_training $optimizer $accumulate_gradients  \
         $gradient_accumulation_steps_p1 $seed $job_name \
         $train_batch_size_p2 $learning_rate_p2 \
         $warmup_steps_p2 $train_steps_p2 $gradient_accumulation_steps_p2 \
         $electra_model 

}

dgx2_1gpu_fp32 ()
{
    train_batch_size_p1="88"
    learning_rate_p1="6e-3"
    precision="fp32"
    xla="xla"
    num_gpus=1
    warmup_steps_p1="2000"
    train_steps_p1=10000
    save_checkpoint_steps=500
    resume_training="false"
    optimizer="lamb"
    accumulate_gradients="true"
    gradient_accumulation_steps_p1=768
    seed=42
    job_name="electra_lamb_pretraining"
    train_batch_size_p2=12
    learning_rate_p2="4e-3"
    warmup_steps_p2="200"
    train_steps_p2=933
    gradient_accumulation_steps_p2=2304
    electra_model="base"
    echo $train_batch_size_p1 $learning_rate_p1 $precision $num_gpus $xla \
         $warmup_steps_p1 $train_steps_p1 $save_checkpoint_steps \
         $resume_training $optimizer $accumulate_gradients  \
         $gradient_accumulation_steps_p1 $seed $job_name \
         $train_batch_size_p2 $learning_rate_p2 \
         $warmup_steps_p2 $train_steps_p2 $gradient_accumulation_steps_p2 \
         $electra_model 

}

# Full pretraining configs for NVIDIA DGX-1 (1x NVIDIA V100 16GB GPU)

dgx1_1gpu_amp ()
{
    train_batch_size_p1="88"
    learning_rate_p1="6e-3"
    precision="amp"
    xla="xla"
    num_gpus=1
    warmup_steps_p1="2000"
    train_steps_p1=10000
    save_checkpoint_steps=500
    resume_training="false"
    optimizer="lamb"
    accumulate_gradients="true"
    gradient_accumulation_steps_p1=768
    seed=42
    job_name="electra_lamb_pretraining"
    train_batch_size_p2=12
    learning_rate_p2="4e-3"
    warmup_steps_p2="200"
    train_steps_p2=933
    gradient_accumulation_steps_p2=2304
    electra_model="base"
    echo $train_batch_size_p1 $learning_rate_p1 $precision $num_gpus $xla \
         $warmup_steps_p1 $train_steps_p1 $save_checkpoint_steps \
         $resume_training $optimizer $accumulate_gradients  \
         $gradient_accumulation_steps_p1 $seed $job_name \
         $train_batch_size_p2 $learning_rate_p2 \
         $warmup_steps_p2 $train_steps_p2 $gradient_accumulation_steps_p2 \
         $electra_model 

}

dgx1_1gpu_fp32 ()
{
    train_batch_size_p1="40"
    learning_rate_p1="6e-3"
    precision="fp32"
    xla="xla"
    num_gpus=1
    warmup_steps_p1="2000"
    train_steps_p1=10000
    save_checkpoint_steps=500
    resume_training="false"
    optimizer="lamb"
    accumulate_gradients="true"
    gradient_accumulation_steps_p1=1689
    seed=42
    job_name="electra_lamb_pretraining"
    train_batch_size_p2=6
    learning_rate_p2="4e-3"
    warmup_steps_p2="200"
    train_steps_p2=933
    gradient_accumulation_steps_p2=4608
    electra_model="base"
    echo $train_batch_size_p1 $learning_rate_1 $precision $num_gpus $xla \
         $warmup_steps_p1 $train_steps_p1 $save_checkpoint_steps \
         $resume_training $optimizer $accumulate_gradients  \
         $gradient_accumulation_steps_p1 $seed $job_name \
         $train_batch_size_p2 $learning_rate_p2 \
         $warmup_steps_p2 $train_steps_p2 $gradient_accumulation_steps_p2 \
         $electra_model 

}

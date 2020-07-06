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
    train_batch_size="8192"
    learning_rate="6e-3"
    precision="fp16"
    num_gpus=8
    warmup_proportion="0.2843"
    train_steps=7038
    save_checkpoint_steps=200
    resume_training="false"
    create_logfile="true"
    accumulate_gradients="true"
    gradient_accumulation_steps=128
    seed=42
    job_name="bert_lamb_pretraining"
    allreduce_post_accumulation="true"
    allreduce_post_accumulation_fp16="true"
    train_batch_size_phase2=4096
    learning_rate_phase2="4e-3"
    warmup_proportion_phase2="0.128"
    train_steps_phase2=1563
    gradient_accumulation_steps_phase2=256
    DATASET=hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/training # change this for other datasets
    DATA_DIR_PHASE1="$BERT_PREP_WORKING_DIR/${DATASET}/"
    BERT_CONFIG=bert_config.json
    CODEDIR="/workspace/bert"
    init_checkpoint="None"
    DATASET2=hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/training # change this for other datasets
    DATA_DIR_PHASE2="$BERT_PREP_WORKING_DIR/${DATASET2}/"
    echo $train_batch_size $learning_rate $precision $num_gpus \
         $warmup_proportion $train_steps $save_checkpoint_steps \
         $resume_training $create_logfile $accumulate_gradients  \
         $gradient_accumulation_steps $seed $job_name $allreduce_post_accumulation \
         $allreduce_post_accumulation_fp16 $train_batch_size_phase2 $learning_rate_phase2 \
         $warmup_proportion_phase2 $train_steps_phase2 $gradient_accumulation_steps_phase2 \
         $DATA_DIR_PHASE1 $DATA_DIR_PHASE2 $CODEDIR

}

dgxa100_8gpu_tf32 ()
{
    train_batch_size="8192"
    learning_rate="6e-3"
    precision="tf32"
    num_gpus=8
    warmup_proportion="0.2843"
    train_steps=7038
    save_checkpoint_steps=200
    resume_training="false"
    create_logfile="true"
    accumulate_gradients="true"
    gradient_accumulation_steps=128
    seed=42
    job_name="bert_lamb_pretraining"
    allreduce_post_accumulation="true"
    allreduce_post_accumulation_fp16="false"
    train_batch_size_phase2=4096
    learning_rate_phase2="4e-3"
    warmup_proportion_phase2="0.128"
    train_steps_phase2=1563
    gradient_accumulation_steps_phase2=512
    DATASET=hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/training # change this for other datasets
    DATA_DIR_PHASE1="$BERT_PREP_WORKING_DIR/${DATASET}/"
    BERT_CONFIG=bert_config.json
    CODEDIR="/workspace/bert"
    init_checkpoint="None"
    DATASET2=hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/training # change this for other datasets
    DATA_DIR_PHASE2="$BERT_PREP_WORKING_DIR/${DATASET2}/"
    echo $train_batch_size $learning_rate $precision $num_gpus \
         $warmup_proportion $train_steps $save_checkpoint_steps \
         $resume_training $create_logfile $accumulate_gradients  \
         $gradient_accumulation_steps $seed $job_name $allreduce_post_accumulation \
         $allreduce_post_accumulation_fp16 $train_batch_size_phase2 $learning_rate_phase2 \
         $warmup_proportion_phase2 $train_steps_phase2 $gradient_accumulation_steps_phase2 \
         $DATA_DIR_PHASE1 $DATA_DIR_PHASE2 $CODEDIR

}

# Full  pretraining configs for NVIDIA DGX-2H (16x NVIDIA V100 32GB GPU)

dgx2_16gpu_fp16 ()
{
    train_batch_size="4096"
    learning_rate="6e-3"
    precision="fp16"
    num_gpus=16
    warmup_proportion="0.2843"
    train_steps=7038
    save_checkpoint_steps=200
    resume_training="false"
    create_logfile="true"
    accumulate_gradients="true"
    gradient_accumulation_steps=64
    seed=42
    job_name="bert_lamb_pretraining"
    allreduce_post_accumulation="true"
    allreduce_post_accumulation_fp16="true"
    train_batch_size_phase2=2048
    learning_rate_phase2="4e-3"
    warmup_proportion_phase2="0.128"
    train_steps_phase2=1563
    gradient_accumulation_steps_phase2=128
    DATASET=hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/training # change this for other datasets
    DATA_DIR_PHASE1="$BERT_PREP_WORKING_DIR/${DATASET}/"
    BERT_CONFIG=bert_config.json
    CODEDIR="/workspace/bert"
    init_checkpoint="None"
    DATASET2=hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/training # change this for other datasets
    DATA_DIR_PHASE2="$BERT_PREP_WORKING_DIR/${DATASET2}/"
    echo $train_batch_size $learning_rate $precision $num_gpus \
         $warmup_proportion $train_steps $save_checkpoint_steps \
         $resume_training $create_logfile $accumulate_gradients  \
         $gradient_accumulation_steps $seed $job_name $allreduce_post_accumulation \
         $allreduce_post_accumulation_fp16 $train_batch_size_phase2 $learning_rate_phase2 \
         $warmup_proportion_phase2 $train_steps_phase2 $gradient_accumulation_steps_phase2 \
         $DATA_DIR_PHASE1 $DATA_DIR_PHASE2 $CODEDIR

}

dgx2_16gpu_fp32 ()
{
    train_batch_size="4096"
    learning_rate="6e-3"
    precision="fp32"
    num_gpus=16
    warmup_proportion="0.2843"
    train_steps=7038
    save_checkpoint_steps=200
    resume_training="false"
    create_logfile="true"
    accumulate_gradients="true"
    gradient_accumulation_steps=128
    seed=42
    job_name="bert_lamb_pretraining"
    allreduce_post_accumulation="true"
    allreduce_post_accumulation_fp16="false"
    train_batch_size_phase2=2048
    learning_rate_phase2="4e-3"
    warmup_proportion_phase2="0.128"
    train_steps_phase2=1563
    gradient_accumulation_steps_phase2=256
    DATASET=hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/training # change this for other datasets
    DATA_DIR_PHASE1="$BERT_PREP_WORKING_DIR/${DATASET}/"
    BERT_CONFIG=bert_config.json
    CODEDIR="/workspace/bert"
    init_checkpoint="None"
    DATASET2=hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/training # change this for other datasets
    DATA_DIR_PHASE2="$BERT_PREP_WORKING_DIR/${DATASET2}/"
    echo $train_batch_size $learning_rate $precision $num_gpus \
         $warmup_proportion $train_steps $save_checkpoint_steps \
         $resume_training $create_logfile $accumulate_gradients  \
         $gradient_accumulation_steps $seed $job_name $allreduce_post_accumulation \
         $allreduce_post_accumulation_fp16 $train_batch_size_phase2 $learning_rate_phase2 \
         $warmup_proportion_phase2 $train_steps_phase2 $gradient_accumulation_steps_phase2 \
         $DATA_DIR_PHASE1 $DATA_DIR_PHASE2 $CODEDIR

}

# Full pretraining configs for NVIDIA DGX-1 (8x NVIDIA V100 16GB GPU)

dgx1_8gpu_fp16 ()
{
    train_batch_size="8192"
    learning_rate="6e-3"
    precision="fp16"
    num_gpus=8
    warmup_proportion="0.2843"
    train_steps=7038
    save_checkpoint_steps=200
    resume_training="false"
    create_logfile="true"
    accumulate_gradients="true"
    gradient_accumulation_steps=512
    seed=42
    job_name="bert_lamb_pretraining"
    allreduce_post_accumulation="true"
    allreduce_post_accumulation_fp16="true"
    train_batch_size_phase2=4096
    learning_rate_phase2="4e-3"
    warmup_proportion_phase2="0.128"
    train_steps_phase2=1563
    gradient_accumulation_steps_phase2=512
    DATASET=hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/training # change this for other datasets
    DATA_DIR_PHASE1="$BERT_PREP_WORKING_DIR/${DATASET}/"
    BERT_CONFIG=bert_config.json
    CODEDIR="/workspace/bert"
    init_checkpoint="None"
    DATASET2=hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/training # change this for other datasets
    DATA_DIR_PHASE2="$BERT_PREP_WORKING_DIR/${DATASET2}/"
    echo $train_batch_size $learning_rate $precision $num_gpus \
         $warmup_proportion $train_steps $save_checkpoint_steps \
         $resume_training $create_logfile $accumulate_gradients  \
         $gradient_accumulation_steps $seed $job_name $allreduce_post_accumulation \
         $allreduce_post_accumulation_fp16 $train_batch_size_phase2 $learning_rate_phase2 \
         $warmup_proportion_phase2 $train_steps_phase2 $gradient_accumulation_steps_phase2 \
         $DATA_DIR_PHASE1 $DATA_DIR_PHASE2 $CODEDIR

}

dgx1_8gpu_fp32 ()
{
    train_batch_size="8192"
    learning_rate="6e-3"
    precision="fp32"
    num_gpus=8
    warmup_proportion="0.2843"
    train_steps=7038
    save_checkpoint_steps=200
    resume_training="false"
    create_logfile="true"
    accumulate_gradients="true"
    gradient_accumulation_steps=1024
    seed=42
    job_name="bert_lamb_pretraining"
    allreduce_post_accumulation="true"
    allreduce_post_accumulation_fp16="false"
    train_batch_size_phase2=4096
    learning_rate_phase2="4e-3"
    warmup_proportion_phase2="0.128"
    train_steps_phase2=1563
    gradient_accumulation_steps_phase2=1024
    DATASET=hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/training # change this for other datasets
    DATA_DIR_PHASE1="$BERT_PREP_WORKING_DIR/${DATASET}/"
    BERT_CONFIG=bert_config.json
    CODEDIR="/workspace/bert"
    init_checkpoint="None"
    DATASET2=hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/training # change this for other datasets
    DATA_DIR_PHASE2="$BERT_PREP_WORKING_DIR/${DATASET2}/"
    echo $train_batch_size $learning_rate $precision $num_gpus \
         $warmup_proportion $train_steps $save_checkpoint_steps \
         $resume_training $create_logfile $accumulate_gradients  \
         $gradient_accumulation_steps $seed $job_name $allreduce_post_accumulation \
         $allreduce_post_accumulation_fp16 $train_batch_size_phase2 $learning_rate_phase2 \
         $warmup_proportion_phase2 $train_steps_phase2 $gradient_accumulation_steps_phase2 \
         $DATA_DIR_PHASE1 $DATA_DIR_PHASE2 $CODEDIR

}

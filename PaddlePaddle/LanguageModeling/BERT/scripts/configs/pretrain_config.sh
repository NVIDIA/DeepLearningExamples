# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
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

dgxa100-80g_8gpu_amp ()
{
    train_batch_size="256"
    learning_rate="6e-3"
    precision="amp"
    num_gpus=8
    warmup_proportion="0.2843"
    train_steps=7038
    save_checkpoint_steps=200
    create_logfile="false"
    gradient_accumulation_steps=32
    seed=42
    job_name="bert_lamb_pretraining"
    train_batch_size_phase2=32
    learning_rate_phase2="4e-3"
    warmup_proportion_phase2="0.128"
    train_steps_phase2=1563
    gradient_accumulation_steps_phase2=128
    DATASET=hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en # change this for other datasets
    DATA_DIR_PHASE1="$BERT_PREP_WORKING_DIR/${DATASET}/"
    DATASET2=hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en # change this for other datasets
    DATA_DIR_PHASE2="$BERT_PREP_WORKING_DIR/${DATASET2}/"
    CODEDIR=/workspace/bert
    init_checkpoint="None"
    RESULTS_DIR=$CODEDIR/results
    CHECKPOINTS_DIR=$RESULTS_DIR
    BERT_CONFIG=bert_configs/bert-large-uncased.json
    enable_benchmark="false"
    benchmark_steps=10  # It takes effect only after the enable_benchmark is set to true
    benchmark_warmup_steps=10  # It takes effect only after the enable_benchmark is set to true
    echo $train_batch_size $learning_rate $precision $num_gpus \
         $warmup_proportion $train_steps $save_checkpoint_steps \
         $create_logfile $gradient_accumulation_steps $seed $job_name \
	 $train_batch_size_phase2 $learning_rate_phase2 \
         $warmup_proportion_phase2 $train_steps_phase2 $gradient_accumulation_steps_phase2 \
         $DATA_DIR_PHASE1 $DATA_DIR_PHASE2 $CODEDIR $init_checkpoint \
         $BERT_CONFIG $enable_benchmark $benchmark_steps $benchmark_warmup_steps
}

dgxa100-80g_8gpu_tf32 ()
{
    train_batch_size="128"
    learning_rate="6e-3"
    precision="tf32"
    num_gpus=8
    warmup_proportion="0.2843"
    train_steps=7038
    save_checkpoint_steps=200
    create_logfile="false"
    gradient_accumulation_steps=64
    seed=42
    job_name="bert_lamb_pretraining"
    train_batch_size_phase2=16
    learning_rate_phase2="4e-3"
    warmup_proportion_phase2="0.128"
    train_steps_phase2=1563
    gradient_accumulation_steps_phase2=256
    DATASET=hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en # change this for other datasets
    DATA_DIR_PHASE1="$BERT_PREP_WORKING_DIR/${DATASET}/"
    DATASET2=hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en # change this for other datasets
    DATA_DIR_PHASE2="$BERT_PREP_WORKING_DIR/${DATASET2}/"
    CODEDIR=/workspace/bert
    init_checkpoint="None"
    RESULTS_DIR=$CODEDIR/results
    CHECKPOINTS_DIR=$RESULTS_DIR
    BERT_CONFIG=bert_configs/bert-large-uncased.json
    enable_benchmark="false"
    benchmark_steps=10  # It takes effect only after the enable_benchmark is set to true
    benchmark_warmup_steps=10  # It takes effect only after the enable_benchmark is set to true
    echo $train_batch_size $learning_rate $precision $num_gpus \
         $warmup_proportion $train_steps $save_checkpoint_steps \
         $create_logfile $gradient_accumulation_steps $seed $job_name \
	 $train_batch_size_phase2 $learning_rate_phase2 \
         $warmup_proportion_phase2 $train_steps_phase2 $gradient_accumulation_steps_phase2 \
         $DATA_DIR_PHASE1 $DATA_DIR_PHASE2 $CODEDIR $init_checkpoint \
         $BERT_CONFIG $enable_benchmark $benchmark_steps $benchmark_warmup_steps
}

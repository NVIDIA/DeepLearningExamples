#!/bin/bash
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --overcommit

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

set -eux

readonly docker_image="nvcr.io/nvidia/tensorflow:21.02-tf2-py3"
readonly datadir="/raid/data/bert"
readonly checkpointdir="$PWD/checkpoints"

readonly mounts=".:/workspace/bert_tf2,${datadir}:/workspace/bert_tf2/data,${checkpointdir}:/results"


srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 mkdir -p "${checkpointdir}/phase_1"
srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 mkdir -p "${checkpointdir}/phase_2"

PHASE1="\
     --train_batch_size=${BATCHSIZE:-16} \
     --learning_rate=${LEARNING_RATE:-1.875e-4} \
     --num_accumulation_steps=${NUM_ACCUMULATION_STEPS:-128} \
     --input_files=/workspace/bert_tf2/data/tfrecord/lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/training/* \
     --max_seq_length=128 \
     --max_predictions_per_seq=20 \
     --num_steps_per_epoch=7038 --num_train_epochs=1 \
     --warmup_steps=2000 \
     --model_dir=/results/phase_1 \
     "

PHASE2="\
     --train_batch_size=${BATCHSIZE:-2} \
     --learning_rate=${LEARNING_RATE:-1.25e-4} \
     --num_accumulation_steps=${NUM_ACCUMULATION_STEPS:-512} \
     --input_files=/workspace/bert_tf2/data/tfrecord/lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/training/* \
     --max_seq_length=512 \
     --max_predictions_per_seq=80 \
     --num_steps_per_epoch=1564 --num_train_epochs=1 \
     --warmup_steps=200 \
     --model_dir=/results/phase_2 \
     --init_checkpoint=/results/phase_1/pretrained/bert_model.ckpt-1 \
    "

PHASES=( "$PHASE1" "$PHASE2" )

PHASE=${PHASE:-1}

PIP_CMD="pip3 install \
  requests \
  tqdm \
  horovod \
  sentencepiece \
  tensorflow_hub \
  pynvml \
  wget \
  progressbar \
  git+https://github.com/NVIDIA/dllogger"

BERT_CMD="\
    python /workspace/bert_tf2/run_pretraining.py \
     ${PHASES[$((PHASE-1))]} \
     --save_checkpoint_steps=100 \
     --steps_per_loop=100 \
     --optimizer_type=LAMB \
     --bert_config_file=data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/bert_config.json \
     --use_horovod --use_fp16 --enable_xla"

srun --mpi=pmi2 -l --container-image="${docker_image}" --container-mounts="${mounts}" bash -c "${PIP_CMD}; ${BERT_CMD}"

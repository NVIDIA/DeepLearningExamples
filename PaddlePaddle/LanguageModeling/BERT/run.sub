#!/bin/bash
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --parsable

# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
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

set -eux

#
# Job Configurations
#
# Tag to the built image.
IMAGE_VERSION=${IMAGE_VERSION:-"22.12-py3"}
# Number of processes per node used for the LDDL preprocessor.
DASK_TASKS_PER_NODE=${DASK_TASKS_PER_NODE:-128}
# 1 or 2 .
PHASE=${PHASE:-1}
# An integer that specifies the pretraining seed. 
SEED=${SEED:-42}
# The percentage of the articles from the Wikipedia dataset to sample and used
# for pretraining. 0 < ${SAMPLE_RATIO} < 1.0
SAMPLE_RATIO=${SAMPLE_RATIO:-0.9}
# Number of GPUs per node. 0 < ${GPUS} <= 8.
GPUS=${GPUS:-"8"}
# The bin size for binned LDDL data loading. 'none' or an integer that divides 
# 128 (for Phase1) or 512 (for Phase2).
BIN_SIZE=${BIN_SIZE:-"none"}
# Number of parquet shards per each LDDL data loader worker process. 'none' or 
# an integer.
NUM_SHARDS_PER_WORKER=${NUM_SHARDS_PER_WORKER:-"none"}
# Number of LDDL data loader worker processes per rank.
NUM_WORKERS=${NUM_WORKERS:-4}
# Should we rerun the LDDL preprocessor every time? 'true' or 'false' .
RERUN_DASK=${RERUN_DASK:-"true"}
# 'static' or 'dynamic' .
MASKING=${MASKING:-"static"}
# Should we use jemalloc for the LDDL preprocessor? 'true' or 'false' .
USE_JEMALLOC=${USE_JEMALLOC:-"true"}
# 'fp16' or 'tf32' .
PRECISION=${PRECISION:-"fp16"}
# The path to the initial checkpoint (from Phase1) used to start Phase2. 'none'
# or an absolute path.
INIT_CHECKPOINT=${INIT_CHECKPOINT:-"none"}
# The per-rank batch size before being divided by the gradient accumulation
# steps.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-"256"}
# The gradient accumulation steps.
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-"32"}

#
# Static Configurations
#
# Container URL.
# Replace this with the URL of the docker image that you build 
# with scripts/docker/build.sh .
readonly docker_image="bert:${IMAGE_VERSION}" 
# Where the datasets are stored on the system.
readonly host_datadir="/home/${USER}/datasets"
readonly container_datadir="/datasets"
# Replace these with the path to the 'source' subdirectory of the LDDL Wikipedia
# dataset.
readonly host_wikipedia_source="${host_datadir}/wikipedia/source"
readonly container_wikipedia_source="${container_datadir}/wikipedia/source"
readonly wikipedia_mount="${host_wikipedia_source}:${container_wikipedia_source}"
# Replace these with where you want to store the Parquet shards in case 
# ${RERUN_DASK} is 'false'.
readonly host_pretrain="${host_datadir}/pretrain"
readonly container_pretrain="${container_datadir}/pretrain"
readonly pretrain_mount="${host_pretrain}:${container_pretrain}"
# Replace these with where you want to store the pretrained checkpoints on 
# the system.
readonly host_output="$PWD/results/${SLURM_JOB_ID}"
mkdir -p "${host_output}"
readonly container_output="/results"
readonly output_mount="${host_output}:${container_output}"
# If INIT_CHECKPOINT is 'none', infer INIT_CHECKPOINT based on job dependency.
if [ "${INIT_CHECKPOINT}" == "none" ] && [ "${PHASE}" == "2" ] ; then
  INIT_CHECKPOINT="$PWD/results/${SLURM_JOB_DEPENDENCY}/bert-large-uncased/phase1/7038"
fi
# Define mounts.
mounts="${PWD}:/workspace/bert,${wikipedia_mount},${pretrain_mount},${output_mount}"
# Add the mount path of the initial checkpoint for Phase2.
if [ "${PHASE}" == "1" ]; then
  echo "No init. mounted for Phase1!"
  readonly container_init_checkpoint=""
elif [ "${PHASE}" == "2" ]; then
  if [ ! -f "${INIT_CHECKPOINT}" ]; then
    echo "No init. checkpoint found for Phase2!"
    exit 1
  else
    mounts="${mounts},$(dirname "${INIT_CHECKPOINT}"):/checkpoints"
    readonly container_init_checkpoint="/checkpoints"
  fi
else
  echo "\${PHASE} = ${PHASE} unknown!"
  exit 1
fi
# Determine where the parquet shards should be stored.
if [ "${RERUN_DASK}" == "true" ]; then
  # Always rerun the dask pipeline. Therefore, use the output directory to store
  # the parquets.
  readonly host_pretrain_parquet="${host_output}/parquet"
  readonly container_pretrain_parquet="${container_output}/parquet"
elif [ "${RERUN_DASK}" == "false" ]; then
  echo "Use existing parquets if they exists."
  if [ "${BIN_SIZE}" == "none" ]; then
      readonly host_pretrain_parquet="${host_pretrain}/phase${PHASE}/unbinned/parquet"
      readonly container_pretrain_parquet="${container_pretrain}/phase${PHASE}/unbinned/parquet"
  else
      readonly host_pretrain_parquet="${host_pretrain}/phase${PHASE}/bin_size_${BIN_SIZE}/parquet"
      readonly container_pretrain_parquet="${container_pretrain}/phase${PHASE}/bin_size_${BIN_SIZE}/parquet"
  fi
else
  echo "\${RERUN_DASK} = ${RERUN_DASK} unknown!"
  exit 1
fi

readonly PHASE1="\
    --learning-rate=6e-3 \
    --warmup-proportion=0.2843 \
    --phase1 \
    --max-seq-length=128 \
    --max-predictions-per-seq=20 \
    --max-steps=7038 \
    --num-steps-per-checkpoint=2500 \
    "

readonly PHASE2="\
    --learning-rate=4e-3 \
    --warmup-proportion=0.128 \
    --phase2 \
    --max-seq-length=512 \
    --max-predictions-per-seq=80 \
    --max-steps=1563 \
    --num-steps-per-checkpoint=1000 \
    --from-pretrained-params=${container_init_checkpoint} \
    "

# Arguments for fp16.
if [ "${PRECISION}" == "fp16" ]; then
  readonly fp16_flags="--amp --use-dynamic-loss-scaling --scale-loss=1048576"
elif [ "${PRECISION}" == "tf32" ]; then
  readonly fp16_flags=""
else
  echo "\${PRECISION} = ${PRECISION} unknown!"
  exit 1
fi

# Get the ip address of all nodes.
IP_CMD="hostname -i"
IP_STR=$(srun -pmix --ntasks-per-node=1 bash -c "${IP_CMD}")
IP_STR=$(echo $IP_STR | sed 's/ /,/g')
echo "\${IP_STR} = ${IP_STR}"

# Get the actual pretraining command.
readonly PHASES=( "$PHASE1" "$PHASE2" ) 
readonly BERT_CMD="\
    python -m paddle.distributed.launch \
    --gpus=0,1,2,3,4,5,6,7 \
    --ips="${IP_STR}" \
    /workspace/bert/run_pretraining.py \
    ${PHASES[$((PHASE - 1))]} \
    --batch-size=${TRAIN_BATCH_SIZE} \
    --input-dir=${container_pretrain_parquet} \
    --output-dir=${container_output} \
    --vocab-file=/workspace/bert/vocab/bert-large-uncased-vocab.txt \
    --bert-model=bert-large-uncased \
    --config-file=/workspace/bert/bert_configs/bert-large-uncased.json \
    --gradient-merge-steps=${GRADIENT_ACCUMULATION_STEPS} \
    --log-freq=1 \
    --seed=12345 \
    --optimizer=Lamb \
    ${fp16_flags} "

echo "nodes: ${SLURM_JOB_NUM_NODES}, TRAIN_BATCH_SIZE: ${TRAIN_BATCH_SIZE}, GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}"

#
# Running the LDDL preprocessor and load balancer.
# 
# Determine the number of parquet shards in total.
if [ "${NUM_SHARDS_PER_WORKER}" == "none" ]; then
  readonly num_blocks=4096
else
  readonly num_blocks=$((NUM_SHARDS_PER_WORKER * $(( NUM_WORKERS > 0 ? NUM_WORKERS : 1 )) * SLURM_JOB_NUM_NODES * GPUS))
fi
echo "num_blocks: ${num_blocks}"
# Run the LDDL preprocessor and load balancer only when there is no file in 
# where the parquets are supposed to be stored.
if [ ! -d "${host_pretrain_parquet}" ] || [ -z "$(ls -A "${host_pretrain_parquet}")" ]; then
  # The sequence length is 128 for Phase1, but 512 for Phase2.
  if [ "${PHASE}" == "1" ]; then
    readonly target_seq_len_flag=""
  elif [ "${PHASE}" == "2" ]; then
    readonly target_seq_len_flag="--target-seq-length 512"
  else
    echo "\${PHASE} = ${PHASE} unknown!"
    exit 1
  fi
  # Should we use sequence binning?
  if [ "${BIN_SIZE}" == "none" ]; then
    readonly bin_size_flag=""
  else
    readonly bin_size_flag="--bin-size ${BIN_SIZE}"
  fi
  # Static masking or dynamic masking?
  if [ "${MASKING}" == "dynamic" ]; then
    readonly masking_flag=""
  elif [ "${MASKING}" == "static" ]; then
    readonly masking_flag="--masking"
  else
    echo "\${MASKING} = ${MASKING} unknown!"
    exit 1
  fi
  # Should we use jemalloc for the LDDL preprocessor?
  if [ "${USE_JEMALLOC}" == "true" ]; then
    readonly use_jemalloc_flag="--export=ALL,LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so"
  elif [ "${USE_JEMALLOC}" == "false" ]; then
    readonly use_jemalloc_flag=""
  else
    echo "\${USE_JEMALLOC} = ${USE_JEMALLOC} unknown!"
    exit 1
  fi
  # Run the LDDL preprocessor.
  srun -l \
    --mpi=pmix \
    --container-image="${docker_image}" \
    --container-mounts="${mounts}"  \
    --ntasks-per-node="${DASK_TASKS_PER_NODE}" \
    ${use_jemalloc_flag} \
    preprocess_bert_pretrain \
      --schedule mpi \
      ${target_seq_len_flag} \
      --wikipedia ${container_wikipedia_source} \
      --sink "${container_pretrain_parquet}" \
      --vocab-file /workspace/bert/vocab/bert-large-uncased-vocab.txt \
      --num-blocks "${num_blocks}" \
      --sample-ratio "${SAMPLE_RATIO}" \
      ${bin_size_flag} \
      ${masking_flag} \
      --seed "${SEED}"
  # Run the LDDL load balancer.
  srun -l \
    --mpi=pmix \
    --container-image="${docker_image}" \
    --container-mounts="${mounts}"  \
    --ntasks-per-node="${DASK_TASKS_PER_NODE}" \
    balance_dask_output \
      --indir "${container_pretrain_parquet}" \
      --num-shards "${num_blocks}"
fi

# 
# Run pretraining.
#
srun -l -pmix --container-image="${docker_image}" --container-mounts="${mounts}" --ntasks-per-node=1 bash -c "${BERT_CMD}"
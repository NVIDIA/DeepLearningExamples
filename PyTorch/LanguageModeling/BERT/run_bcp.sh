#!/bin/bash

# Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
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


# DEFAULTS adjusted to match table:
# https://github.com/NVIDIA/DeepLearningExamples/tree/544a2d6fdc685ca7ea1ca94f518508f765d6f856/PyTorch/LanguageModeling/BERT#pre-training-nvidia-dgx-a100-8x-a100-80gb-multi-node-scaling
# Refer to README_BCP.md for examples of running this script.

# ---------------------------------------------------------------------------- #
# Job Configurations
# ---------------------------------------------------------------------------- #
# DEFAULTS adjusted to match table:
# https://github.com/NVIDIA/DeepLearningExamples/tree/544a2d6fdc685ca7ea1ca94f518508f765d6f856/PyTorch/LanguageModeling/BERT#pre-training-nvidia-dgx-a100-8x-a100-80gb-multi-node-scaling


# Number of Nodes
NUM_NODES=${NGC_ARRAY_SIZE:-"1"}
# Number of processes per node used for the LDDL preprocessor.
DASK_TASKS_PER_NODE=${DASK_TASKS_PER_NODE:-128}
# 1 or 2 .
PHASE=${PHASE:-1}
# An integer that specifies the pretraining seed. 
SEED=${SEED:-42}
# The percentage of the articles from the Wikipedia dataset to sample and used
# for pretraining. 0 < ${SAMPLE_RATIO} < 1.0
SAMPLE_RATIO=${SAMPLE_RATIO:-0.9}
# How many global steps to run before ending the pretraining job. This argument
# does not impact the learning rate schedule, but only if the pretraining job 
# should exit early. 'none' or an integer.
STEPS_THIS_RUN=${STEPS_THIS_RUN:-"none"}
# Number of GPUs per node. 0 < ${GPUS} <= 8.
NGPUS=$(nvidia-smi -L | wc -l)
NUM_GPUS=${NUM_GPUS:-"$NGPUS"}

# The bin size for binned LDDL data loading. 'none' or an integer that divides 
# 128 (for Phase1) or 512 (for Phase2).
# The batch size presets.
if [ "${PHASE}" == "1" ]; then
  BIN_SIZE_DEFAULT="none"
  BIN_SIZE=${BIN_SIZE:-"$BIN_SIZE_DEFAULT"}
else
  # assume PHASE 2
  BIN_SIZE_DEFAULT=64
  BIN_SIZE=${BIN_SIZE:-$BIN_SIZE_DEFAULT}
  if [ ! "${BIN_SIZE}" == "none" ] && [[ ! "${BIN_SIZE}" =~ ^(32|64|128|256|512)$ ]]; then
    echo "Error! PHASE2 BIN_SIZE=${BIN_SIZE} not supported! Set to one of: none|32|64|128|256|512"
    exit 1
  fi
fi

# Number of parquet shards per each LDDL data loader worker process. 'none' or 
# an integer.
# NUM_SHARDS_PER_WORKER=${NUM_SHARDS_PER_WORKER:-"none"}
NUM_SHARDS_PER_WORKER=${NUM_SHARDS_PER_WORKER:-128}
# Number of LDDL data loader worker processes per rank.
NUM_WORKERS=${NUM_WORKERS:-4}
# Should we rerun the LDDL preprocessor every time? 'true' or 'false' .
# RERUN_DASK=${RERUN_DASK:-"true"}
RERUN_DASK=${RERUN_DASK:-"false"}
# 'static' or 'dynamic' .
MASKING=${MASKING:-"static"}
# Should we use jemalloc for the LDDL preprocessor? 'true' or 'false' .
USE_JEMALLOC=${USE_JEMALLOC:-"true"}
# 'fp16' or 'tf32' .
PRECISION=${PRECISION:-"fp16"}
# 'base' or 'large' .
CONFIG=${CONFIG:-"large"}

# The batch size presets.
if [ "${PHASE}" == "1" ]; then
  TRAIN_BATCH_SIZE_DEFAULT=8192
else
  # assume PHASE 2
  TRAIN_BATCH_SIZE_DEFAULT=4096
fi
TRAIN_BATCH_SIZE_DEFAULT=$(( TRAIN_BATCH_SIZE_DEFAULT / NUM_NODES ))
# The per-rank batch size before being divided by the gradient accumulation
# steps.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-"$TRAIN_BATCH_SIZE_DEFAULT"}

# The gradient accumulation steps presets.
if [ "${PRECISION}" == "fp16" ]; then
  if [ "${PHASE}" == "1" ]; then
    GRADIENT_ACCUMULATION_STEPS_DEFAULT=32
  else
    GRADIENT_ACCUMULATION_STEPS_DEFAULT=128
  fi
else
  # assume TF32
  if [ "${PHASE}" == "1" ]; then
    GRADIENT_ACCUMULATION_STEPS_DEFAULT=64
  else
    GRADIENT_ACCUMULATION_STEPS_DEFAULT=256
  fi
fi
GRADIENT_ACCUMULATION_STEPS_DEFAULT=$(( GRADIENT_ACCUMULATION_STEPS_DEFAULT / NUM_NODES ))
# The gradient accumulation steps.
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-"$GRADIENT_ACCUMULATION_STEPS_DEFAULT"}

# ---------------------------------------------------------------------------- #
# Static Configurations
# ---------------------------------------------------------------------------- #
# Replace these with where you want to store the pretrained checkpoints on 
# the system.
CODEDIR=${CODEDIR:-"/workspace/bert"}

# Where the datasets are stored on the system.
# readonly DATADIR="/datasets"
DATADIR=${DATADIR:-"/datasets"}

OUTPUTDIR=${OUTPUTDIR:-"/results"}

# Replace these with where you want to store the Parquet shards.
PRETRAINDIR=${PRETRAINDIR:-"${DATADIR}/pretrain"}

# Replace these with the path to the 'source' subdirectory of the LDDL Wikipedia
# dataset.
readonly wikipedia_source="${DATADIR}/wikipedia/source"

# Replace these with where you want to store the pretrained checkpoints on 
# the system.
CHECKPOINTSDIR=${CHECKPOINTSDIR:-"${OUTPUTDIR}/checkpoints"}
mkdir -p ${CHECKPOINTSDIR}

# The path to the initial checkpoint (from Phase1) used to start Phase2. 'none'
# or an absolute path.
INIT_CHECKPOINT=${INIT_CHECKPOINT:-"none"}

# If INIT_CHECKPOINT is 'none', infer INIT_CHECKPOINT based on job dependency.
if [ "${INIT_CHECKPOINT}" == "none" ] && [ "${PHASE}" == "2" ] ; then
  INIT_CHECKPOINT="${CHECKPOINTSDIR}/ckpt_7038.pt"
fi

# Add the mount path of the initial checkpoint for Phase2.
if [ "${PHASE}" == "1" ]; then
  echo "No init. mounted for Phase1!"
  readonly init_checkpoint_dir=""
elif [ "${PHASE}" == "2" ]; then
  if [ ! -f "${INIT_CHECKPOINT}" ]; then
    echo "No init. checkpoint found for Phase2!"
    exit 1
  fi
  readonly init_checkpoint_dir="${INIT_CHECKPOINT}"
else
  echo "\${PHASE} = ${PHASE} unknown!"
  exit 1
fi


# ---------------------------------------------------------------------------- #
# Determine which launcher to use.
# ---------------------------------------------------------------------------- #
# If bcprun exists, that implies multinode.
isbcprun=false
command -v bcprun >/dev/null
if [ $? -eq 0 ]; then
    isbcprun=true
fi

if [ "$isbcprun" = false ]; then
  envopt="-x"
else
  envopt="--env"
fi

# ---------------------------------------------------------------------------- #
# Running the LDDL preprocessor and load balancer.
# ---------------------------------------------------------------------------- #

# Determine the number of parquet shards in total.
if [ "${NUM_SHARDS_PER_WORKER}" == "none" ]; then
  readonly num_blocks=4096
else
  readonly num_blocks=$((NUM_SHARDS_PER_WORKER * $(( NUM_WORKERS > 0 ? NUM_WORKERS : 1 )) * NUM_NODES * NUM_GPUS ))
fi
echo "num_blocks: ${num_blocks}"

# Determine where the parquet shards should be stored.
if [ "${BIN_SIZE}" == "none" ]; then
  readonly pretrain_parquet="${PRETRAINDIR}/phase${PHASE}_numblocks${num_blocks}/unbinned/parquet"
else
  readonly pretrain_parquet="${PRETRAINDIR}/phase${PHASE}_numblocks${num_blocks}/bin_size_${BIN_SIZE}/parquet"
fi

if [ "${RERUN_DASK}" == "true" ]; then
  mkdir -p ${pretrain_parquet}
  rm -rf ${pretrain_parquet}
fi

nshards=$(find ${pretrain_parquet} -maxdepth 1 -name "shard*.parquet*" 2>/dev/null | wc -l)
num_shards=$num_blocks
if [ ! "${BIN_SIZE}" == "none" ]; then
  if [ "${PHASE}" == "2" ]; then
    num_shards=$(( num_blocks * 512 / BIN_SIZE ))
  else
    num_shards=$(( num_blocks * 128 / BIN_SIZE ))
  fi
fi
# Run the LDDL preprocessor and load balancer only when there is no file in 
# where the parquets are supposed to be stored.
if [ ! -d "${pretrain_parquet}" ] || [ $nshards -ne $num_shards ]; then
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
    readonly use_jemalloc_flag="${envopt} LD_PRELOAD=/opt/conda/lib/libjemalloc.so"
  elif [ "${USE_JEMALLOC}" == "false" ]; then
    readonly use_jemalloc_flag=""
  else
    echo "\${USE_JEMALLOC} = ${USE_JEMALLOC} unknown!"
    exit 1
  fi

  # Run the LDDL preprocessor.
  PRETRAIN_CMD="preprocess_bert_pretrain \
    --schedule mpi \
    ${target_seq_len_flag} \
    --wikipedia ${wikipedia_source} \
    --sink "${pretrain_parquet}" \
    --vocab-file ${CODEDIR}/vocab/vocab \
    --num-blocks "${num_blocks}" \
    --sample-ratio "${SAMPLE_RATIO}" \
    ${bin_size_flag} \
    ${masking_flag} \
    --seed ${SEED}"

  if [ "$isbcprun" = false ]; then
    mpirun \
      --allow-run-as-root \
      --oversubscribe \
      -np ${DASK_TASKS_PER_NODE} \
      ${use_jemalloc_flag} \
      ${PRETRAIN_CMD}
  else
    NGC_ARRAY_TYPE=MPIJob bcprun \
      --launcher 'mpirun --allow-run-as-root --oversubscribe' \
      --nnodes $NUM_NODES \
      --npernode ${DASK_TASKS_PER_NODE} \
      ${use_jemalloc_flag} \
      --cmd "${PRETRAIN_CMD}"
  fi

  # Run the LDDL load balancer.
  BALANCE_CMD="balance_dask_output \
    --indir \"${pretrain_parquet}\" \
    --num-shards ${num_blocks}"
  if [ "$isbcprun" = false ]; then
    mpirun \
      --oversubscribe \
      --allow-run-as-root \
      -np ${DASK_TASKS_PER_NODE} \
      ${BALANCE_CMD}
  else
      NGC_ARRAY_TYPE=MPIJob bcprun \
      --launcher 'mpirun --allow-run-as-root --oversubscribe' \
      --nnodes $NUM_NODES \
      --npernode ${DASK_TASKS_PER_NODE} \
      --cmd "${BALANCE_CMD}"
  fi
fi

# ---------------------------------------------------------------------------- #
# Determine the pretraining arguments.
# ---------------------------------------------------------------------------- #

# Should we exit pretraining early?
if [ "${STEPS_THIS_RUN}" == "none" ]; then
  readonly steps_this_run_flag=""
else
  readonly steps_this_run_flag="--steps_this_run ${STEPS_THIS_RUN}"
fi

# Arguments that are specific to Phase1 and Phase2.
readonly PHASE1="\
  --learning_rate=6e-3 \
  --warmup_proportion=0.2843 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --max_steps=7038 \
  --num_steps_per_checkpoint=2500 \
  "
readonly PHASE2="\
  --learning_rate=4e-3 \
  --warmup_proportion=0.128 \
  --phase2 \
  --max_seq_length=512 \
  --max_predictions_per_seq=80 \
  --max_steps=1563 \
  --num_steps_per_checkpoint=1000 \
  --resume_from_checkpoint --phase1_end_step=7038 \
  --init_checkpoint=${init_checkpoint_dir} \
  "

# Arguments for fp16.
if [ "${PRECISION}" == "fp16" ]; then
  readonly fp16_flags="--fp16 --allreduce_post_accumulation_fp16"
elif [ "${PRECISION}" == "tf32" ]; then
  readonly fp16_flags=""
else
  echo "\${PRECISION} = ${PRECISION} unknown!"
  exit 1
fi

# ---------------------------------------------------------------------------- #
# Determine the pretraining command.
# ---------------------------------------------------------------------------- #

# CPU-GPU binding.
readonly BIND_CMD="${CODEDIR}/bind.sh --cpu=exclusive --cluster=selene --"

# Get the actual pretraining command.
readonly PHASES=( "$PHASE1" "$PHASE2" ) 
# readonly BERT_CMD="\
#     ${BIND_CMD} python -u ${CODEDIR}/run_pretraining.py \
readonly BERT_CMD="\
    python -u ${CODEDIR}/run_pretraining.py \
    --seed=${SEED} \
    --train_batch_size=${TRAIN_BATCH_SIZE} \
    ${PHASES[$((PHASE - 1))]} \
    --do_train \
    --config_file=${CODEDIR}/bert_configs/${CONFIG}.json \
    --input_dir=${pretrain_parquet} \
    --vocab_file=${CODEDIR}/vocab/vocab \
    --output_dir=${CHECKPOINTSDIR} \
    ${fp16_flags} \
    --allreduce_post_accumulation \
    --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    --log_freq=1 \
    --json-summary=${OUTPUTDIR}/summary.json \
    --disable_progress_bar \
    --num_workers=${NUM_WORKERS} \
    ${steps_this_run_flag} "

echo "nodes: ${NUM_NODES}, TRAIN_BATCH_SIZE: ${TRAIN_BATCH_SIZE}, \
GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}"

# ---------------------------------------------------------------------------- #
# Run pretraining.
# ---------------------------------------------------------------------------- #

if [ "$isbcprun" = false ]; then
  mpirun --allow-run-as-root -np $NUM_GPUS --bind-to none bash -c "\
$BIND_CMD \
bash -c \"\
WORLD_SIZE=\$OMPI_COMM_WORLD_SIZE \
RANK=\$OMPI_COMM_WORLD_RANK \
LOCAL_RANK=\$OMPI_COMM_WORLD_LOCAL_RANK \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29500 \
$BERT_CMD\"\
"
else
  # This works well, but the binding is not as sophisticated as "bind.sh".
  # The performance on BCP seems to be the same as with the explicit "bind.sh"
  # appraoch commented out below.
  NGC_ARRAY_TYPE="PyTorchJob" bcprun \
    --nnodes=${NUM_NODES} \
    --npernode=${NUM_GPUS} \
    --binding exclusive \
    --cmd "$BERT_CMD"

  # Using the bind.sh approach. This involved using bcprun with the launcher
  # "bcpPyTorchLaunch.py", which like bcprun is injected into multinode jobs.
  # But unlike bcprun it is not a documented API, therefore "bcpPyTorchLaunch.py"
  # should be treated as a protected/private API of BCP.
  # ---------- Beginning of approach with bcpPyTorchLaunch.py ---------------- #
#   LAUNCHSCRIPT=${CODEDIR}/bert_job_${NGC_JOB_ID}.sh

# cat <<EOF > ${LAUNCHSCRIPT}
# #!/bin/bash

# $BIND_CMD $BERT_CMD

# EOF

#   chmod +x ${LAUNCHSCRIPT}

#   NGC_ARRAY_TYPE="" bcprun \
#     --nnodes=${NUM_NODES} \
#     --npernode=1 \
#     --cmd "python /etc/mpi/bcpPyTorchLaunch.py \
#       --nnodes=${NUM_NODES} --nproc_per_node=${NUM_GPUS} \
#       --node_rank=\${NGC_ARRAY_INDEX} --master_addr=\${NGC_MASTER_ADDR} \
#       --no_python --use_env ${LAUNCHSCRIPT}"
  # --- End bcpPyTorchLaunch.py approach ------------------------------------- #

fi


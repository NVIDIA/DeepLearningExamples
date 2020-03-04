#!/bin/bash
#SBATCH -N 8                       # number of nodes
#SBATCH -t 4:00:00                 # wall time
#SBATCH -J "transformer-xl_pyt"    # job name
#SBATCH --exclusive                # exclusive node access
#SBATCH --mem=0                    # all mem avail
#SBATCH --mail-type=FAIL           # only send email on failure
#SBATCH --overcommit               # Needed for pytorch

# Configure the multinode script
SCRIPT="run_multinode_wt103_large.sh"

# Configure workspace in the file system
WORK_DIR="/gpfs/fs1/${USER}/transformer-xl_pyt"
CONT_WORK_DIR="/workspace/transformer-xl"

# Configure container and mounting paths
CONT="<YOUR DOCKER REGISTRY>/transformer-xl:latest"
MOUNTS="${WORK_DIR}/data:${CONT_WORK_DIR}/data,${WORK_DIR}/results:${CONT_WORK_DIR}/results"

# Create directory for the results
srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 mkdir -p "${WORK_DIR}/results"

# Configure DGX-2H parameters
export DGXNSOCKET=2
export DGXSOCKETCORES=24
export DGXNGPU=16

# Overwrite default parameters in the script
EXTRA_TRAIN_PARAMS=""
EXTRA_EVAL_PARAMS=""

if [ -n "$MAX_STEP" ]; then
   EXTRA_TRAIN_PARAMS+="--max_step ${MAX_STEP} "
fi
if [ -n "$SEED" ]; then
   EXTRA_TRAIN_PARAMS+="--seed ${SEED} "
fi
if [ -n "$BATCH_SIZE" ]; then
   EXTRA_TRAIN_PARAMS+="--batch_size ${BATCH_SIZE} "
fi
if [ -n "$LOCAL_BATCH_SIZE" ]; then
   EXTRA_TRAIN_PARAMS+="--local_batch_size ${LOCAL_BATCH_SIZE} "
fi
if [ -n "$BATCH_CHUNK" ]; then
   EXTRA_TRAIN_PARAMS+="--batch_chunk ${BATCH_CHUNK} "
fi
if [ -n "$EVAL_INTERVAL" ]; then
   EXTRA_TRAIN_PARAMS+="--eval_interval ${EVAL_INTERVAL} "
fi
if [ -n "$EVAL_BATCH_SIZE" ]; then
   EXTRA_TRAIN_PARAMS+="--eval_batch_size ${EVAL_BATCH_SIZE} "
   EXTRA_EVAL_PARAMS+="--batch_size ${EVAL_BATCH_SIZE} "
fi
if [ -n "$LR" ]; then
   EXTRA_TRAIN_PARAMS+="--lr ${LR} "
fi
if [ -n "$FP16" ]; then
   EXTRA_TRAIN_PARAMS+="--fp16 "
   EXTRA_EVAL_PARAMS+="--fp16 "
fi
if [ -n "$CONFIG" ]; then
   EXTRA_TRAIN_PARAMS+="--config ${CONFIG} "
   EXTRA_EVAL_PARAMS+="--config ${CONFIG} "
fi

if [[ $1 == 'train' ]] || [[ $1 == 'all' ]]; then
   # Run training
   srun --mpi=none \
      --container-image="${CONT}" \
      --container-mounts="${MOUNTS}" \
      bash "${SCRIPT}" train \
      --work_dir ${CONT_WORK_DIR}/results/${SLURM_JOB_ID} \
      "${EXTRA_TRAIN_PARAMS}"
fi

if [[ $1 == 'eval' ]] || [[ $1 == 'all' ]]; then
   # Run final evaluation
   srun --mpi=none \
      --container-image="${CONT}" \
      --container-mounts="${MOUNTS}" \
      bash "${SCRIPT}" eval \
      --work_dir ${CONT_WORK_DIR}/results/${SLURM_JOB_ID} \
      "${EXTRA_EVAL_PARAMS}"
fi

if [[ $1 != 'train' ]] && [[ $1 != 'eval' ]] && [[ $1 != 'all' ]]; then
    echo 'unknown argment 1'
fi

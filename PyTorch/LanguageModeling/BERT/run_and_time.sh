#!/bin/bash

#echo "Multi-node $MULTI_NODE"
#echo "Dataset $DATASET"
## DL vars -- Change your parameters below
# To change the number of GPUs per node, change the sbatch param --ntasks-per-node in the launching script

## Need to avoid virtualenv and do python directly
#        train.py --data=/dev/shm/$DATASET \
#        train.py --data=/raid/datasets/$DATASET \

DGXSYSTEM=${DGXSYSTEM:-"DGX1"}
if [[ -f config_${DGXSYSTEM}.sh ]]; then
  source config_${DGXSYSTEM}.sh
else
  source config_DGX1.sh
  echo "Unknown system, assuming DGX1"
fi
SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-$DGXNGPU}
SLURM_JOB_ID=${SLURM_JOB_ID:-$RANDOM}
MULTI_NODE=${MULTI_NODE:-''}
echo "Run vars: id $SLURM_JOB_ID gpus $SLURM_NTASKS_PER_NODE mparams $MULTI_NODE"

# run training
BIND_LAUNCH=1  ## should be the default

if [[ $BIND_LAUNCH -eq 1 ]]; then
  LAUNCH_OPT="bind_pyt  --nsockets_per_node 2  --ncores_per_socket ${DGXSOCKETCORES} --nproc_per_node ${SLURM_NTASKS_PER_NODE} ${MULTI_NODE}"
else
  LAUNCH_OPT="torch.distributed.launch --nproc_per_node ${SLURM_NTASKS_PER_NODE} ${MULTI_NODE}"
fi

# Options
python -m $LAUNCH_OPT \
run_pretraining.py --seed=${SEED} \
--train_batch_size=${BATCHSIZE} \
--learning_rate=${LEARNING_RATE} \
--warmup_proportion=${WARMUP_UPDATES} \
$EXTRA_PARAMS

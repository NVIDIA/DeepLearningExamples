#!/bin/bash
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --time=8:00:00
#SBATCH --ntasks-per-node=8     # n tasks per machine (one task per gpu)

set -eux

curr_dt=`date +"%Y-%m-%d-%H-%M-%S"`

readonly ro_mounts="${checkpointdir}:/workspace/checkpoints,${datadir}:/workspace/coco"

CREATE_FOLDER_CMD="mkdir -p /tmp/convergence-AMP-32xV100-32G; chmod -R 02775 /tmp/convergence-AMP-32xV100-32G"
srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 sh -c "${CREATE_FOLDER_CMD}"

bs=64
ep=350
lr=1.5
wu=90
ema=0.999
momentum=0.95

EFFDET_CMD="\
        python3 train.py \
        --training_mode=${training_mode:=traineval} \
        --training_file_pattern=/workspace/coco/train-* \
        --val_file_pattern=/workspace/coco/val-* \
        --val_json_file=/workspace/coco/annotations/instances_val2017.json \
        --model_name=efficientdet-d0 \
        --model_dir=/tmp/convergence-AMP-32xV100-32G  \
        --backbone_init=/workspace/checkpoints/efficientnet-b0-joc \
        --batch_size=$bs \
        --eval_batch_size=$bs \
        --num_epochs=$ep \
        --use_xla=True \
        --amp=True \
        --lr=$lr \
        --warmup_epochs=$wu \
        --hparams="moving_average_decay=$ema,momentum=$momentum" \
        2>&1 | tee /tmp/convergence-AMP-32xV100-32G/train-$curr_dt.log"

srun --mpi=pmix -l --container-image=nvcr.io/nvidia/effdet:21.09-tf2 --no-container-entrypoint --container-mounts="${ro_mounts}" bash -c "${EFFDET_CMD}"
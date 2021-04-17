#!/bin/bash
###SBATCH -t 8:00:00                 # wall time
#SBATCH --ntasks-per-node=8        # tasks per node
#SBATCH --exclusive                # exclusive node access
#SBATCH --mem=0                    # all mem avail
#SBATCH --mail-type=FAIL           # only send email on failure
#SBATCH --overcommit               # Needed for pytorch

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

# Data dir
readonly datadir="/datasets/imagenet/train-val-tfrecord"
# Path to where trained checkpoints will be saved on the system
readonly checkpointdir="$PWD/B4_mulitnode_AMP/"

CREATE_FOLDER_CMD="if [ ! -d ${checkpointdir} ]; then mkdir -p ${checkpointdir} ; fi && nvidia-smi"

srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 sh -c "${CREATE_FOLDER_CMD}"

OUTFILE="${checkpointdir}/slurm-%j.out"
ERRFILE="${checkpointdir}/error-%j.out"

readonly mounts="${datadir}:/data,${checkpointdir}:/model"

srun -p ${PARTITION} -l -o $OUTFILE -e $ERRFILE --container-image nvcr.io/nvidia/efficientnet-tf2:21.02-tf2-py3 --container-mounts ${mounts} --mpi=pmix bash ./scripts/bind.sh --cpu=exclusive --ib=single -- python3 main.py --mode train_and_eval --arch efficientnet-b4 --model_dir /model --data_dir /data --use_amp --use_xla --lr_decay cosine --weight_init fan_out --max_epochs 500 --log_steps 100 --save_checkpoint_freq 3 --train_batch_size 128 --eval_batch_size 128 --lr_init 0.005 --batch_norm syncbn --resume_checkpoint --augmenter_name autoaugment --mixup_alpha 0.2 --weight_decay 5e-6 --epsilon 0.001
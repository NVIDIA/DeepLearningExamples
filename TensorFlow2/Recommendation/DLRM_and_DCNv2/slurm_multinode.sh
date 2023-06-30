#!/bin/bash
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
#
# author: Tomasz Grel (tgrel@nvidia.com)


# This is a generic SLURM batch script. It runs $cmd
# command in $cont docker image while mounting $mounts directories.
# You can use the $srun_flags variable to pass additional
# arguments to srun.
#
# It is designed to work with enroot/pyxis, but could be modified
# to run on bare-metal machines as well.
#
# Example usage to train a 1.68TB DLRM variant using 32xA100-80GB GPUs on 4 nodes:
#
#  cmd='numactl --interleave=all -- python -u main.py --dataset_path /data/dlrm/full_criteo_data --amp \
#  --embedding_dim 512 --bottom_mlp_dims 512,256,512' \
#  srun_flags='--mpi=pmix' \
#  cont=dlrm_tf_adam \
#  mounts=/data/dlrm:/data/dlrm \
#  sbatch -n 32 -N 4 -t 00:20:00 slurm_multinode.sh
#

srun --mpi=none ${srun_flags} --ntasks-per-node=1 \
     --container-image="${cont}"  --container-mounts=${mounts} /bin/bash -c "$cmd"
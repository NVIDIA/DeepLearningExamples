# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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


#!/bin/bash
set -e
set -x
NAMES=${1:-'*.yaml'}
TARGET=feature_specs/${NAMES}
OPTIONS=${2-""}
for file in ${TARGET};
do
  echo "${file}";
done
for fspec_file in ${TARGET};
do
  SYNTH_DATA_DIR=/tmp/generated_data/${fspec_file}
  # generate data based on fspec
  python /dlrm/prepare_synthetic_dataset.py --feature_spec ${fspec_file} --synthetic_dataset_dir ${SYNTH_DATA_DIR}

  # single-GPU A100-80GB
  #horovodrun -np 1 -H localhost:1 --mpi-args=--oversubscribe numactl --interleave=all -- python -u /dlrm/main.py --dataset_path ${SYNTH_DATA_DIR} ${OPTIONS}

  # single-GPU V100-32GB
  #horovodrun -np 1 -H localhost:1 --mpi-args=--oversubscribe numactl --interleave=all -- python -u /dlrm/main.py --dataset_path ${SYNTH_DATA_DIR} ${OPTIONS}

  # delete the data
  rm -r ${SYNTH_DATA_DIR}
done
#
#        usage:
#        docker build . -t nvidia_dlrm_tf
#        docker run --security-opt seccomp=unconfined --runtime=nvidia -it --rm --ipc=host  -v ${PWD}/data:/data nvidia_dlrm_tf bash
#        cd tests
#        bash test_with_opts.sh

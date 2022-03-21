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
  #horovodrun -np 1 -H localhost:1 --mpi-args=--oversubscribe numactl --interleave=all -- python -u /dlrm/main.py --dataset_path ${SYNTH_DATA_DIR} ${OPTIONS} --tf_gpu_memory_limit_gb 73

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
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
  python -m dlrm.scripts.prepare_synthetic_dataset --feature_spec ${fspec_file} --synthetic_dataset_dir ${SYNTH_DATA_DIR}
  # train on the data
  python -m dlrm.scripts.main --mode train --dataset ${SYNTH_DATA_DIR} ${OPTIONS}
  # delete the data
  rm -r ${SYNTH_DATA_DIR}
done
#
#        usage:
#        docker build . -t nvidia_dlrm_pyt
#        docker run --security-opt seccomp=unconfined --runtime=nvidia -it --rm --ipc=host  -v ${PWD}/data:/data nvidia_dlrm_pyt bash
#        cd tests
#        bash test_with_opts.sh
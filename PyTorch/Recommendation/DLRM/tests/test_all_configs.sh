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

  for mlp in True False;
  do
    for graphs in True;
    do
      for dot in cuda_dot dot;
      do
        for amp in True False;
        do
          python -m dlrm.scripts.main --mode train --dataset ${SYNTH_DATA_DIR} --optimized_mlp=${mlp} --cuda_graphs=${graphs} --interaction_op=${dot} --embedding_type=joint_sparse --amp=${amp}
          #DGX-2
          python -m torch.distributed.launch --no_python --use_env --nproc_per_node 8 bash -c "/workspace/dlrm/bind.sh --cpu=exclusive -- python -m dlrm.scripts.main --dataset ${SYNTH_DATA_DIR} --optimized_mlp=${mlp} --cuda_graphs=${graphs} --interaction_op=${dot} --embedding_type=joint_sparse --amp=${amp}"
          #DGX A100
          #python -m torch.distributed.launch --no_python --use_env --nproc_per_node 8 bash -c "/workspace/dlrm/bind.sh --cpu=/workspace/dlrm/dgxa100_ccx.sh --mem=/workspace/dlrm/dgxa100_ccx.sh python -m dlrm.scripts.main --dataset ${SYNTH_DATA_DIR} --optimized_mlp=${mlp} --cuda_graphs=${graphs} --interaction_op=${dot} --embedding_type=joint_sparse  --amp=${amp}"
        done;
      done
    done
  done
  # delete the data
  rm -r ${SYNTH_DATA_DIR}
done
#
#        usage:
#        docker build . -t nvidia_dlrm_pyt
#        docker run --security-opt seccomp=unconfined --runtime=nvidia -it --rm --ipc=host  -v ${PWD}/data:/data nvidia_dlrm_pyt bash
#        cd tests
#        bash test_custom_dot.sh
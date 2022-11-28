#!/bin/bash

# Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

usage() {
  cat <<EOF
Usage: bash scripts/run_model.sh
--data_path             Data path. Default: /data.
--gpus                  Number of gpus.
--amp                   Use amp (0 or 1).
--xla                   Use xla (0 or 1).
--benchmark             Use benchmark mode (0 or 1).
--benchmark_steps       Number of bench steps.
--mode                  One of: train, inference.
--epochs                Number of epochs (only valid with mode=train).
--batch_size            Batch size.
--results_dir           Path to output directory. Default: /tmp/sim.
--log_filename          Name of output log file within results_dir. Default: log.json.
--save_checkpoint_path  Path to output checkpoint after training.
--load_checkpoint_path  Path from which to restore checkpoint for inference or suspend/resume training.
--prebatch_train_size
--prebatch_test_size
EOF
}

if [ ! -d "scripts" ] || [ ! "$(ls -A 'scripts')" ]; then
  echo "[ERROR] You are probably calling this script from wrong directory"
  usage
  exit 1
fi

gpus=${gpus:-1}
data_path=${data_path:-/data}

xla=${xla:-0}
amp=${amp:-0}
benchmark=${benchmark:-0}

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi

  shift
done

xla_map_arg=("" "--xla")
amp_map_arg=("" "--amp")
benchmark_map_arg=("" "--benchmark")

xla_arg=${xla_map_arg[$xla]}
amp_arg=${amp_map_arg[$amp]}
benchmark_arg=${benchmark_map_arg[$benchmark]}

function get_option_or_use_default() {
  if [ -z $2 ]
  then
    echo ""
  else
    echo $1 $2
  fi
}

data_path_option=$(get_option_or_use_default --dataset_dir $data_path)
mode_option=$(get_option_or_use_default --mode $mode)
benchmark_steps_option=$(get_option_or_use_default --benchmark_steps $benchmark_steps)
batch_size_option=$(get_option_or_use_default --global_batch_size $batch_size)
epochs_option=$(get_option_or_use_default --epochs $epochs)
results_dir_option=$(get_option_or_use_default --results_dir $results_dir)
log_filename_option=$(get_option_or_use_default --log_filename $log_filename)
save_checkpoint_path_option=$(get_option_or_use_default --save_checkpoint_path $save_checkpoint_path)
load_checkpoint_path_option=$(get_option_or_use_default --load_checkpoint_path $load_checkpoint_path)
prebatch_train_size_option=$(get_option_or_use_default --prebatch_train_size $prebatch_train_size)
prebatch_test_size_option=$(get_option_or_use_default --prebatch_test_size $prebatch_test_size)

command="mpiexec --allow-run-as-root --bind-to socket -np ${gpus} python main.py --dataset_dir ${data_path} --drop_remainder ${epochs_option} 
${xla_arg} ${amp_arg} ${benchmark_arg} ${mode_option} ${benchmark_steps_option} ${batch_size_option} ${results_dir_option} ${log_filename_option}
${save_checkpoint_path_option} ${load_checkpoint_path_option} ${prebatch_train_size_option} ${prebatch_test_size_option}"

printf "[INFO] Running:\n%s\n" "${command}"
# run
$command
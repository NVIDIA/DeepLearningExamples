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

set -e

usage() {
  cat <<EOF
Usage: bash scripts/evaluating_benchmark.sh -g gpu
-g    | --gpu      (Required)            Number of gpus
-b    | --bs       (Optional)            Global batch size, default 131072
-a    | --amp      (Optional)            Use amp
-x    | --xla      (Optional)            Use xla
EOF
}

if [ ! -d "scripts" ] || [ ! "$(ls -A 'scripts')" ]; then
  echo "You are probably calling this script from wrong directory"
  usage
  exit 1
fi

amp=
xla=
gpu=
bs=131072

while [ "$1" != "" ]; do
  case $1 in
    -g | --gpu)
      shift
      gpu="$1"
      ;;
    -b | --bs)
      shift
      bs="$1"
      ;;
    -a | --amp)
      amp="--amp"
      ;;
    -x | --xla)
      xla="--xla"
      ;;
    *)
      usage
      exit 1
      ;;
  esac
  shift

done

if [ -z "$gpu" ]; then
  echo "Missing number of gpus param"
  usage
  exit 1
fi

if ! [ "$bs" -ge 0 ] 2>/dev/null; then
  echo "Expected global batch size (${bs}) to be positive integer"
  usage
  exit 1
fi

if ! [ "$gpu" -ge 0 ] || [[ ! "$gpu" =~ ^(1|4|8)$ ]] 2>/dev/null; then
  echo "Expected number of gpus (${gpu}) to be equal 1, 4 or 8"
  usage
  exit 1
fi

cmd="mpiexec --allow-run-as-root --bind-to socket -np ${gpu} \
	python main.py \
	--evaluate \
	--benchmark \
	--benchmark_warmup_steps 500 \
	--benchmark_steps 1000 \
	-eval_batch_size ${bs} \
	${amp} \
	${xla}"

set -x

$cmd

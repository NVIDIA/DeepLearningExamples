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

OUTBRAIN_DIR='/outbrain'
SPARK_DIR='/tmp/spark'

usage() {
  cat <<EOF
Usage: bash scripts/preproc_benchmark.sh -m nvtabular/spark
-m    | --mode           (Required)            Preprocessing to be executed from [nvtabular, spark].
-t    | --tfrecords      (Optional)            Number of tfrecords to be created, default 40.
-i    | --iteration      (Optional)            Number of benchmark iterations, default 10.
EOF
}

if [ ! -d "scripts" ] || [ ! "$(ls -A 'scripts')" ]; then
  echo "You are probably calling this script from wrong directory"
  usage
  exit 1
fi

mode=
iter=10
tfrecords=40

while [ "$1" != "" ]; do
  case $1 in
    -m | --mode)
      shift
      mode="$1"
      ;;
    -t | --tfrecords)
      shift
      tfrecords="$1"
      ;;
    -i | --iteration)
      shift
      iter="$1"
      ;;
    *)
      usage
      exit 1
      ;;
  esac
  shift

done

if [ -z "$mode" ]; then
  echo "Missing preprocessing mode"
  usage
  exit 1
fi

if [[ ! "$mode" =~ ^(spark|nvtabular)$ ]]; then
  echo "Expected mode (${mode}) to be equal spark or nvtabular"
  usage
  exit 1
fi

if ! [ "$tfrecords" -ge 0 ] 2>/dev/null; then
  echo "Expected tfrecords (${tfrecords}) to be positive integer"
  usage
  exit 1
fi

if ! [ "$iter" -ge 0 ] 2>/dev/null; then
  echo "Expected iteration (${iter}) to be positive integer"
  usage
  exit 1
fi

function clean() {
  case "$1" in
    nvtabular)
      rm -rf "$OUTBRAIN_DIR/data"
      rm -rf "$OUTBRAIN_DIR/tfrecords"
      ;;

    spark)
      rm -rf "$SPARK_DIR"
      rm -rf "$OUTBRAIN_DIR/tfrecords"
      ;;
  esac
}

SECONDS=0

for i in $(seq 1 "$iter"); do
	echo "[BENCHMARK] Cleaning directories"
	clean "${mode}"
	echo "[BENCHMARK] Running iteration ${i}"	
	bash scripts/memscript.sh & bash scripts/preproc.sh "${mode}" "${tfrecords}" 
  echo "[BENCHMARK] Memory consumption during iteration ${i} (kB): $(cat mem_consumption.txt)"
done
echo -e "\n[BENCHMARK] Benchmark finished:\n"
echo "[BENCHMARK] Memory consumption (kB): $(cat mem_consumption.txt)"
rm mem_consumption.txt
echo "[BENCHMARK] Mode=${mode}"
echo "[BENCHMARK] Iteration=${iter}"
echo "[BENCHMARK] Tfrecords=${tfrecords}"
AVG_SECONDS=$((("$SECONDS" + "$iter" / 2) / "$iter"))
printf '[BENCHMARK] Total time elapsed: %dh:%dm:%ds\n' $(("$SECONDS" / 3600)) $(("$SECONDS" % 3600 / 60)) $(("$SECONDS" % 60))
printf '[BENCHMARK] Average iteration time: %dh:%dm:%ds\n\n' $(("$AVG_SECONDS" / 3600)) $(("$AVG_SECONDS" % 3600 / 60)) $(("$AVG_SECONDS" % 60))

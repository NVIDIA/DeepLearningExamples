
#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

trap "exit" INT

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_DIR=${SCRIPT_DIR}/../..

TRITON_CLIENT_CONTAINER_TAG=${TRITON_CLIENT_CONTAINER_TAG:-jasper:triton}

SERVER_HOSTNAME=${SERVER_HOSTNAME:-localhost}
MODEL_NAME=${MODEL_NAME:-jasper-tensorrt-ensemble}
MODEL_VERSION=${MODEL_VERSION:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
AUDIO_LENGTH=${AUDIO_LENGTH:-32000}
RESULT_DIR=${RESULT_DIR:-${PROJECT_DIR}/results}
MAX_LATENCY=${MAX_LATENCY:-500}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-64}
MEASUREMENT_WINDOW=${MEASUREMENT_WINDOW:-3000}

TIMESTAMP=$(date "+%y%m%d_%H%M")

# RESULT_DIR_H is the path on the host, outside the container. Inside the container RESULT_DIR_H is always mounted to /results
RESULT_DIR_H="${RESULT_DIR}/perf_client/${MODEL_NAME}"

# Set the output folder using the first argument or pick a default
if [ -z ${1+x} ]; then
   RESULT_DIR_H=${RESULT_DIR_H}/batch_${BATCH_SIZE}_len_${AUDIO_LENGTH}
else
   RESULT_DIR_H=${RESULT_DIR_H}/"$1"
   shift
fi

# Make the directory if it doesnt exist
if [ ! -d "${RESULT_DIR_H}" ]; then
   mkdir -p "${RESULT_DIR_H}"
fi

echo "Saving output to ${RESULT_DIR_H}"

LOGNAME="${RESULT_DIR_H}/log_${TIMESTAMP}.log"
OUTPUT_FILE_CSV="results_${TIMESTAMP}.csv"

ARGS="\
   -m ${MODEL_NAME} \
   -x ${MODEL_VERSION} \
   -p ${MEASUREMENT_WINDOW} \
   -v \
   -i gRPC \
   -u ${SERVER_HOSTNAME}:8001 \
   -b ${BATCH_SIZE} \
   -l ${MAX_LATENCY} \
   --max-threads ${MAX_CONCURRENCY} "

curl -s "http://${SERVER_HOSTNAME}:8000/api/status/${MODEL_NAME}" | grep ready_state | grep SERVER_READY || (echo "Model ${MODEL_NAME} is not ready, perf_client skipped..." && exit 1)

echo "=== STARTING: perf client ${ARGS} --concurrency-range 1:4:1 ==="
set -x
docker run  -e DISPLAY=${DISPLAY}  --runtime nvidia --rm \
	      --privileged --net=host \
	      -v ${RESULT_DIR_H}:/results --name jasper-perf-client \
	      ${TRITON_CLIENT_CONTAINER_TAG}  perf_client $ARGS -f /results/${OUTPUT_FILE_CSV}_p1 --shape AUDIO_SIGNAL:${AUDIO_LENGTH} --concurrency-range 1:4:1 2>&1 | tee -a $LOGNAME
set +x

echo "=== STARTING: perf client ${ARGS} --concurrency-range 8:${MAX_CONCURRENCY}:8 ==="
set -x
docker run  -e DISPLAY=${DISPLAY}  --runtime nvidia --rm \
	      --privileged --net=host \
	      -v ${RESULT_DIR_H}:/results --name jasper-perf-client \
	      ${TRITON_CLIENT_CONTAINER_TAG}  perf_client $ARGS -f /results/${OUTPUT_FILE_CSV}_p2 --shape AUDIO_SIGNAL:${AUDIO_LENGTH} --concurrency-range 8:${MAX_CONCURRENCY}:8 2>&1 | tee -a $LOGNAME
set +x

cat ${RESULT_DIR_H}/${OUTPUT_FILE_CSV}_p1 ${RESULT_DIR_H}/${OUTPUT_FILE_CSV}_p2 > ${RESULT_DIR_H}/${OUTPUT_FILE_CSV}

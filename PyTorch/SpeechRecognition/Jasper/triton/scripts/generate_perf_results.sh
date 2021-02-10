#!/bin/bash

#### input arguments
RESULT_DIR=${RESULT_DIR} # used by perf_client to store results
NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-"0"}
SERVER_HOSTNAME=${SERVER_HOSTNAME:-localhost}
AUDIO_LENGTH=${AUDIO_LENGTH:-80000}
BATCH_SIZE=${BATCH_SIZE:-16}
####

set -e
SCRIPT_DIR=$(cd $(dirname $0); pwd)
TRITON_DIR=${SCRIPT_DIR}/..
PROJECT_DIR=${TRITON_DIR}/..

GPU_TESTS=${GPU_TESTS:-"tensorrt ts-trace onnx"}
ENGINE_COUNT_TESTS=${ENGINE_COUNT_TESTS:-"1"}

# Export the set variables in case they used the default
export NVIDIA_VISIBLE_DEVICES
export SERVER_HOSTNAME
export AUDIO_LENGTH
export MAX_LATENCY=2000 # Set max latency high to prevent errors
TRITON=${TRITON:-jasper-triton-server}
MAX_QUEUE_DELAYS=${MAX_QUEUE_DELAYS:-"10 5 2"} #ms

# Ensure that the server is closed when the script exits
function cleanup_server {
    current_time=$(date "+%Y.%m.%d-%H.%M.%S")
    logfile="/tmp/${TRITON}-${current_time}.log"
    echo "Shutting down ${TRITON} container, log is in ${logfile}"
    docker logs ${TRITON} > ${logfile} 2>&1
    docker stop ${TRITON} > /dev/null 2>&1
}

trap cleanup_server EXIT

trap "exit" INT

function wait_for_triton {
    TIMEOUT=${1:-60}
    timeout ${TIMEOUT} ${SCRIPT_DIR}/wait_for_triton_server.sh || (echo '\nServer timeout!!!\n' && exit 1)
}

function modify_ensemble {

    PLAT=$1

    REPO=${TRITON_DIR}/deploy/model_repo
    INPLACE="--in_place"

    CONF=${REPO}/jasper-${PLAT}/config.pbtxt
    CONF_E=${REPO}/jasper-${PLAT}-ensemble/config.pbtxt


    echo "Modifying ${CONF} : batch size ${BATCH_SIZE} engines=${NUM_ENGINES} ..."
    cleanup_server || true
    sed -i -e "s/1#NUM_ENGINES/${NUM_ENGINES}/g" -e "s/8#MAX_BATCH/${BATCH_SIZE}/g" ${CONF}
    if [ "$MAX_QUEUE" != "" ] ; then
        sed -i -e "s/#db#//g" -e "s/#MAX_QUEUE/${MAX_QUEUE}/g" ${CONF}
    fi


    echo "Modifying ${CONF_E} for size $2, batch size ${BATCH_SIZE} ${TRITON_DYN_BATCH_ARGS}.."
    sed -i -e "s/-1#AUDIO_LENGTH/${AUDIO_LENGTH}/g" -e "s/8#MAX_BATCH/${BATCH_SIZE}/g" ${CONF_E}

    ${SCRIPT_DIR}/run_server.sh

    wait_for_triton

    echo "done."
}

echo "GPU tests: ${GPU_TESTS}"
echo "PRECISION: ${PRECISION}"

for plat in ${GPU_TESTS}; do
    if [ "$plat" == "none" ]; then
	    continue
    else
        export MAX_LATENCY=2000
        export MEASUREMENT_WINDOW=3000
    fi

    export BASE_SAVE_NAME="${plat}_${PRECISION}_${AUDIO_LENGTH}_BS${BATCH_SIZE}"
    export MODEL_NAME=jasper-${plat}-ensemble

    MODELS="jasper-${plat} jasper-${plat}-ensemble" PRECISION=${PRECISION} ${SCRIPT_DIR}/prepare_model_repository.sh

    ############## Engine Count Comparison (static batcing) ##############
    for num_engines in ${ENGINE_COUNT_TESTS}; do
	SAVE_RESULTS_DIR="${BASE_SAVE_NAME}/static/${num_engines}_engines"
	NUM_ENGINES=${num_engines} BATCH_SIZE=${BATCH_SIZE} modify_ensemble ${plat} ${AUDIO_LENGTH}
	echo "Running engines comparison, ${num_engines} engines..."
	MAX_CONCURRENCY=8 BATCH_SIZE=${BATCH_SIZE} ${SCRIPT_DIR}/run_perf_client.sh  ${SAVE_RESULTS_DIR} || echo '\nPerf Client Failure!!!\n'
    done

    ############## Dynamic Batching Comparison ##############
    for delay in ${MAX_QUEUE_DELAYS}; do
	echo "Running dynamic batching comparison, models=${MODELS}, delay ${delay}..."
	TRITON_DYN_BATCHING_DELAY=$((delay * 1000))
	SAVE_RESULTS_DIR="${BASE_SAVE_NAME}/batching/${TRITON_DYN_BATCHING_DELAY}"
	NUM_ENGINES=1 MAX_QUEUE=${TRITON_DYN_BATCHING_DELAY} BATCH_SIZE=${BATCH_SIZE} modify_ensemble ${plat} ${AUDIO_LENGTH}
	BATCH_SIZE=1 MAX_CONCURRENCY=$((BATCH_SIZE*2)) ${SCRIPT_DIR}/run_perf_client.sh ${SAVE_RESULTS_DIR} || echo '\nPerf Client Failure!!!\n'
    done
done

echo "Complete!"

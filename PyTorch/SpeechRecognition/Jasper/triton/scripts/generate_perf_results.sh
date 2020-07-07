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
TRTIS_DIR=${SCRIPT_DIR}/..
PROJECT_DIR=${TRTIS_DIR}/..

GPU_TESTS=${GPU_TESTS:-"trt pyt onnx"}
CPU_TESTS=${CPU_TESTS:-""}

ENGINE_COUNT_TESTS=${ENGINE_COUNT_TESTS:-"1"}

# Export the set variables in case they used the default
export NVIDIA_VISIBLE_DEVICES
export SERVER_HOSTNAME
export AUDIO_LENGTH
export MAX_LATENCY=2000 # Set max latency high to prevent errors
TRTIS=${TRTIS:-jasper-trtis}
max_queue_delays="10 5 2" #ms


# Ensure that the server is closed when the script exits
function cleanup_server {
    current_time=$(date "+%Y.%m.%d-%H.%M.%S")
    logfile="/tmp/${TRTIS}-${current_time}.log"
    echo "Shutting down ${TRTIS} container, log is in ${logfile}"
    docker logs ${TRTIS} > ${logfile} 2>&1
    docker stop ${TRTIS} > /dev/null 2>&1
}

trap cleanup_server EXIT

trap "exit" INT

function wait_for_trtis {
    TIMEOUT=${1:-30}
    timeout ${TIMEOUT} ${SCRIPT_DIR}/wait_for_trtis_server.sh || (echo '\nServer timeout!!!\n' && exit 1) 
}

function modify_ensemble {
    
    PLAT=$1
    
    REPO=${TRTIS_DIR}/deploy/model_repo
    INPLACE="--in_place"
            
    CONF=${REPO}/jasper-${PLAT}/config.pbtxt
    CONF_E=${REPO}/jasper-${PLAT}-ensemble/config.pbtxt
    

    echo "Modifying ${CONF} : batch size ${BATCH_SIZE} engines=${NUM_ENGINES} ..."
    cleanup_server || true
    sed -i -e "s/1#NUM_ENGINES/${NUM_ENGINES}/g" -e "s/64#MAX_BATCH/${BATCH_SIZE}/g" ${CONF}
    if [ "$MAX_QUEUE" != "" ] ; then
        sed -i -e "s/#db#//g" -e "s/#MAX_QUEUE/${MAX_QUEUE}/g" ${CONF}
    fi


    echo "Modifying ${CONF_E} for size $2, batch size ${BATCH_SIZE} ${TRTIS_DYN_BATCH_ARGS}.."
    sed -i -e "s/-1#AUDIO_LENGTH/${AUDIO_LENGTH}/g" -e "s/64#MAX_BATCH/${BATCH_SIZE}/g" ${CONF_E} 

    ${SCRIPT_DIR}/run_server.sh ${NORESTART}
    
    wait_for_trtis
    
    echo "done."
}

echo "GPU tests: ${GPU_TESTS}" 
echo "CPU tests: ${CPU_TESTS}"
echo "PRECISION: ${PRECISION}"

for plat in ${GPU_TESTS} ${CPU_TESTS}; do
    if [ "$plat" == "onnx-cpu" ]; then
        export MAX_LATENCY=10000
        export MEASUREMENT_WINDOW=15000
    elif [ "$plat" == "none" ]; then
	    continue
    else
        export MAX_LATENCY=2000
        export MEASUREMENT_WINDOW=3000    
    fi
    

    export BASE_SAVE_NAME="${plat}_${PRECISION}_${AUDIO_LENGTH}_BS${BATCH_SIZE}"
    export MODEL_NAME=jasper-${plat}-ensemble

    MODELS="jasper-${plat} jasper-${plat}-ensemble" ${SCRIPT_DIR}/prepare_model_repository.sh

    if [ "$plat" == "onnx-cpu" ]; then
        export MAX_LATENCY=10000
        export MEASUREMENT_WINDOW=15000
    fi    

    # ############## Engine Count Comparison (static batcing) ##############
    for num_engines in ${ENGINE_COUNT_TESTS}; do
	SAVE_RESULTS_DIR="${BASE_SAVE_NAME}/static/${num_engines}_engines"
        NUM_ENGINES=${num_engines} BATCH_SIZE=${BATCH_SIZE} modify_ensemble ${plat} ${AUDIO_LENGTH}
        echo "Running engines comparison, ${num_engines} engines..." 
        MAX_CONCURRENCY=8 BATCH_SIZE=${BATCH_SIZE} ${SCRIPT_DIR}/run_perf_client.sh  ${SAVE_RESULTS_DIR} || echo '\nPerf Client Failure!!!\n'
    done
    
    ############## Dynamic Batching Comparison ##############
    for delay in ${max_queue_delays}; do
	    echo "Running dynamic batching comparison, models=${MODELS}, delay ${delay}..."
        TRTIS_DYN_BATCHING_DELAY=$((delay * 1000))
	    SAVE_RESULTS_DIR="${BASE_SAVE_NAME}/batching/${TRTIS_DYN_BATCHING_DELAY}"
        NUM_ENGINES=1 MAX_QUEUE=${TRTIS_DYN_BATCHING_DELAY} BATCH_SIZE=${BATCH_SIZE} modify_ensemble ${plat} ${AUDIO_LENGTH}
        BATCH_SIZE=1 MAX_CONCURRENCY=$((BATCH_SIZE*2)) ${SCRIPT_DIR}/run_perf_client.sh ${SAVE_RESULTS_DIR} || echo '\nPerf Client Failure!!!\n'
    done

    
    
done

echo "Complete!"


#!/bin/bash

MODEL_NAME=${1:-"bert"}
MODEL_VERSION=${2:-1}
precision=${3:-"fp16"}
BATCH_SIZE=${4:-1}
MAX_LATENCY=${5:-500}
MAX_CLIENT_THREADS=${6:-10}
MAX_CONCURRENCY=${7:-50}
SERVER_HOSTNAME=${8:-"localhost"}

if [[ $SERVER_HOSTNAME == *":"* ]]; then
  echo "ERROR! Do not include the port when passing the Server Hostname. These scripts require that the TRTIS HTTP endpoint is on Port 8000 and the gRPC endpoint is on Port 8001. Exiting..."
  exit 1
fi

if [ "$SERVER_HOSTNAME" = "localhost" ]
then
    if [ ! "$(docker inspect -f "{{.State.Running}}" trt_server_cont)" = "true" ] ; then

        echo "Launching TRTIS server"
        bash scripts/docker/launch_server.sh $precision
        SERVER_LAUNCHED=true

        function cleanup_server {
            echo "Killing TRTIS server"
            docker kill trt_server_cont
        }

        # Ensure we cleanup the server on exit
        # trap "exit" INT TERM
        trap cleanup_server EXIT
    fi
fi

# Wait until server is up. curl on the health of the server and sleep until its ready
bash scripts/trtis/wait_for_trtis_server.sh $SERVER_HOSTNAME

TIMESTAMP=$(date "+%y%m%d_%H%M")

bash scripts/docker/launch.sh mkdir -p /results/perf_client/${MODEL_NAME}
OUTPUT_FILE_CSV="/results/perf_client/${MODEL_NAME}/results_${TIMESTAMP}.csv"

ARGS="\
   --max-threads ${MAX_CLIENT_THREADS} \
   -m ${MODEL_NAME} \
   -x ${MODEL_VERSION} \
   -p 3000 \
   -d \
   -v \
   -i gRPC \
   -u ${SERVER_HOSTNAME}:8001 \
   -b ${BATCH_SIZE} \
   -l ${MAX_LATENCY} \
   -c ${MAX_CONCURRENCY} \
   -f ${OUTPUT_FILE_CSV}"

echo "Using args:  $(echo "$ARGS" | sed -e 's/   -/\n-/g')"

bash scripts/docker/launch.sh /workspace/build/perf_client $ARGS

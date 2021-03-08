#!/bin/bash

if [[ -v SLURM_LOCALID ]]; then
    echo "Bind using slurm localid"
    LOCAL_ID=$SLURM_LOCALID
elif [[ -v OMPI_COMM_WORLD_LOCAL_RANK ]]; then
    echo "Bind using OpenMPI env"
    LOCAL_ID=$OMPI_COMM_WORLD_LOCAL_RANK
else
    echo "Bind to first node"
    LOCAL_ID=0
fi

case $LOCAL_ID in
    0|1) exec numactl --cpunodebind=3 --membind=3 $@;;
    2|3) exec numactl --cpunodebind=1 --membind=1 $@;;
    4|5) exec numactl --cpunodebind=7 --membind=7 $@;;
    6|7) exec numactl --cpunodebind=5 --membind=5 $@;;
    *) echo "unknown binding"; exec $@;;
esac

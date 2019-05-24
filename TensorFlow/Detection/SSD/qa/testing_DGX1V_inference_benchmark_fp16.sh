#!/bin/bash
BASELINES=(93.6 136.3 172.1 190.8 188.2 189.4 192.2)
TOLERANCE=0.07
PRECISION=FP16

for i in `seq 0 6`
do
    BS=$((2 ** $i))
    
    MSG="Testing mixed precision inference speed on batch size = $BS"
    CMD="bash ../../examples/SSD320_${PRECISION}_inference.sh ../../configs --batch_size $BS"

    if CMD=$CMD BASELINE=${BASELINES[$i]} TOLERANCE=$TOLERANCE MSG=$MSG bash ../../qa/testing_DGX1V_performance.sh
    then
        exit $?
    fi

done

return $result

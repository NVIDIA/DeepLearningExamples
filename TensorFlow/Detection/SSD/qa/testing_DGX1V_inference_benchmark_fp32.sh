#!/bin/bash
BASELINES=(93.2 136.2 171.2 189.4 188.0 188.7 192.5)
PRECISION=FP32
TOLERANCE=0.07

for i in `seq 0 6`
do
    BS=$((2 ** $i))
    
    MSG="Testing single precision inference speed on batch size = $BS"
    CMD="bash ../../examples/SSD320_${PRECISION}_inference.sh ../../configs --batch_size $BS"

    if CMD=$CMD BASELINE=${BASELINES[$i]} TOLERANCE=$TOLERANCE MSG=$MSG bash ../../qa/testing_DGX1V_performance.sh
    then
        exit $?
    fi

done

return $result

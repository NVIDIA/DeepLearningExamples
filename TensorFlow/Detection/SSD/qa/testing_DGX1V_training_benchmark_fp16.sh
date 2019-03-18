#!/bin/bash
BASELINES=(125 430 750)

i=0
for GPUS in 1 4 8
do
        echo "Testing mixed precision training speed on $GPUS GPUs"
        bash examples/SSD320_FP16_${GPUS}_BENCHMARK.sh > tmp 2> /dev/null
        echo -n "img/s: "; tail -n 1 tmp | awk '{print $7}'; echo "expected img/s: ${BASELINES[$i]}"; echo -n "relative error: "; err=`tail -n 1 tmp | awk -v BASELINE=${BASELINES[$i]} '{print sqrt(($7 - BASELINE)^2)/$7}'`; echo $err
        rm tmp
        if [[ $err > 0.1 ]]; then echo "FAILED" && exit 1; else echo "PASSED"; fi
        i=$(($i + 1))
done

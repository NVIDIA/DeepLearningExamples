#!/bin/bash
BASELINES=(193.6 135.2 171.5 188.3 187 187.6 191.4)

for i in `seq 0 6`
do
        echo "Testing mixed precision inference speed on batch size = $((2 ** $i))"
        bash examples/SSD320_FP16_inference.sh --batch_size $((2 ** $i)) > tmp 2> /dev/null
        echo -n "img/s: "; tail -n 1 tmp | awk '{print $3}'; echo "expected img/s: ${BASELINES[$i]}"; echo -n "relative error: "; err=`tail -n 1 tmp | awk -v BASELINE=${BASELINES[$i]} '{print sqrt(($3 - BASELINE)^2)/$3}'`; echo $err
        rm tmp
        if [[ $err > 0.1 ]]; then echo "FAILED" && exit 1; else echo "PASSED"; fi
done

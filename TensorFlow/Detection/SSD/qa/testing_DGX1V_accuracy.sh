TARGET_mAP=${TARGET_mAP:-0.137}
TARGET_loss=${TARGET_loss:-2.3}
TOLERANCE=${TOLERANCE:-0.1}

PRECISION=${PRECISION:-FP16}

bash ../../examples/SSD320_${PRECISION}_8GPU_BENCHMARK.sh /results/SSD320_${PRECISION}_8GPU ../../configs

mAP=$(cat /results/SSD320_${PRECISION}_8GPU/train_log | sed -n 's|.*DetectionBoxes_Precision/mAP = \([^,]*\),.*|\1|p' | tail -n1)
loss=$(cat /results/SSD320_${PRECISION}_8GPU/train_log | sed -n 's|.*Loss for final step: \(.*\)\.|\1|p' | tail -n1)

mAP_error=$( python -c "print(abs($TARGET_mAP  - $mAP)/$mAP)")
loss_error=$(python -c "print(abs($TARGET_loss - $loss)/$loss)")


cat /results/SSD320_${PRECISION}_8GPU/train_log
echo expected: mAP=$TARGET_mAP loss=$TARGET_loss
echo got:      mAP=$mAP        loss=$loss

if [[ -n $mAP_error && $mAP_error < $TOLERANCE && -n $loss_error && $loss_error < $TOLERANCE ]]
then
    echo PASS
else
    echo FAIL
    exit 1
fi

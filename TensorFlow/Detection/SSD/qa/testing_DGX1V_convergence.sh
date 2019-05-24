TARGET_mAP=${TARGET_mAP:-0.281}
TOLERANCE=${TOLERANCE:-0.04}

PRECISION=${PRECISION:-FP16}

bash ../../examples/SSD320_${PRECISION}_8GPU.sh /results/SSD320_${PRECISION}_8GPU ../../configs

mAP=$(cat /results/SSD320_${PRECISION}_8GPU/train_log | sed -n 's|.*DetectionBoxes_Precision/mAP = \([^,]*\),.*|\1|p' | tail -n1)

mAP_error=$( python -c "print(abs($TARGET_mAP  - $mAP)/$TARGET_mAP)")

echo expected: mAP=$TARGET_mAP
echo got:      mAP=$mAP
if [[ $mAP_error < $TOLERANCE ]]
then
   echo PASS
else
    echo FAIL
    exit 1
fi

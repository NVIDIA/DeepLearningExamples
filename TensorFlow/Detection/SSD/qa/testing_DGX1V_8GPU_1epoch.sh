TARGET_mAP=${TARGET_mAP:-0.0020408058}
TARGET_loss=${TARGET_loss:-2.2808013}
TOLERANCE=${TOLERANCE:-0.1}

PRECISION=${PRECISION:-FP16}

TRAIN_LOG=$(bash examples/SSD320_${PRECISION}_8GPU.sh /results/SSD320_${PRECISION}_8GPU --num_train_steps $((12500/27)) 2>&1 | tee /dev/tty)

mAP=$( echo $TRAIN_LOG | sed -n 's|.*DetectionBoxes_Precision/mAP = \([^,]*\),.*|\1|p' | tail -n1)
loss=$(echo $TRAIN_LOG | sed -n 's|.*Loss for final step: \(.*\)\.|\1|p' | tail -n1)

mAP_error=$( python -c "print(abs($TARGET_mAP  - $mAP)/$mAP)")
loss_error=$(python -c "print(abs($TARGET_loss - $loss)/$loss)")

if [[ $mAP_error < $TOLERANCE && $loss_error < $TOLERANCE ]]
then
    echo PASS
else
    echo expected: mAP=$TARGET_mAP loss=$TARGET_loss
    echo got:      mAP=$mAP        loss=$loss
    echo FAIL
    exit 1
fi

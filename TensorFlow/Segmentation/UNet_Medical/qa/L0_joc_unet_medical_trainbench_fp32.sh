echo ------------------------------------------------------
echo TRAINING BECHMARK FOR UNET MEDICAL - FP32 BS=1 - 1 GPU
echo ------------------------------------------------------

python main.py \
--data_dir /data/unet_medical_tf \
--model_dir /results \
--batch_size 1 \
--benchmark \
--exec_mode train \
--augment \
--warmup_steps 100 \
--log_every 100 \
--max_steps 200 > /results/qa_log.txt

if [[ $? -ne 0 ]]; then
    cat /results/qa_log.txt
    echo LOG SCRIPT NOT FOUND
    exit 1
fi

PERF=$(grep 'average_images_per_second' /results/qa_log.txt | tail -1 | cut -d '"' -f2 )

if [[ -z "$PERF" ]]; then
    cat /results/qa_log.txt
    echo COULD NOT FIND VALUE FOR PERF IN LOG
    exit 1
fi

echo REACHED $PERF IMG/SEC

BASELINE=10

if [[ $PERF < $BASELINE ]]; then
    cat /results/qa_log.txt
    echo EXPECTED BASELINE OF $BASELINE IMG/SEC --- FAILED
    exit 1
fi

echo EXPECTED BASELINE OF $BASELINE IMG/SEC --- SURPASSED

exit 0

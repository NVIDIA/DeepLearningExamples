echo ------------------------------------------------------
echo TRAINING BECHMARK FOR UNET MEDICAL - FP32 BS=1 - 8 GPU
echo ------------------------------------------------------

mpirun \
    -np 8 \
    -H localhost:8 \
    -bind-to none \
    -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH \
    -mca pml ob1 -mca btl ^openib \
    --allow-run-as-root \
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

BASELINE=85

if [[ $PERF < $BASELINE ]]; then
    cat /results/qa_log.txt
    echo EXPECTED BASELINE OF $BASELINE IMG/SEC --- FAILED
    exit 1
fi

echo EXPECTED BASELINE OF $BASELINE IMG/SEC --- SURPASSED

exit 0

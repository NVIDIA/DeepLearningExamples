#! /bin/bash
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
[ $NUM_GPUS -eq 16 ] && WORKER_NUMS=(1 8 16) || WORKER_NUMS=(1 8)
DATASETS=(electricity volatility traffic favorita)

rm -r /tmp/benchmark_results

for DATASET in ${DATASETS[@]}
do
    for NGPU in ${WORKER_NUMS[@]}
    do
        for BATCH_SIZE in 512 1024 1536 2048 2560
        do
            for USE_AMP in --use_amp ""
            do
                for AFFINITY in "--affinity disabled" "--affinity single" "--affinity socket_unique_interleaved"
                do 
                    EXP_NAME="TFT_benchmark_${DATASET}_BS_${BATCH_SIZE}_${NGPU}GPU${USE_AMP}_${AFFINITY}"
                    python -m torch.distributed.run --nproc_per_node=${NGPU} train.py \
                            --dataset ${DATASET} \
                            --data_path /data/processed/${DATASET}_bin \
                            --batch_size=${BATCH_SIZE} \
                            --lr 5e-4 \
                            --epochs 1 \
                            --sample 100000 5000 \
                            --seed 1 \
                            ${USE_AMP} \
                            ${AFFINITY} \
                            --clip_grad 0.1 \
                            --results /tmp/benchmark_results/${EXP_NAME}
                done
            done
        done
    done
done
for P in `ls /tmp/benchmark_results/`;
do
    echo ${P}
    tail -n 1 /tmp/benchmark_results/${P}/dllogger.json
done

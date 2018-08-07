#!/bin/bash

DATASET_DIR='data/wmt16_de_en'

batches=(128)
maths=(fp16 fp32)
gpus=(1 2 4 8)

for math in "${maths[@]}"
do
   for batch in "${batches[@]}"
   do
      for gpu in "${gpus[@]}"
      do
         export CUDA_VISIBLE_DEVICES=`seq -s "," 0 $((gpu - 1))`
         python3 -m multiproc train.py \
         --save benchmark_gpu_${gpu}_math_${math}_batch_${batch} \
         --dataset-dir ${DATASET_DIR} \
         --seed 1 \
         --epochs 1 \
         --math ${math} \
         --print-freq 1 \
         --batch-size ${batch} \
         --disable-eval \
         --max-size $((512 * ${batch} * ${gpu}))
      done
   done
done

#!/bin/bash

batches=(128)
maths=(fp16 fp32)
gpus=(1 2 4 8)

sentences=3498161

echo -e [parameters] "\t\t\t" [tokens / s]  [second per epoch]

for batch in "${batches[@]}"
do
   for math in "${maths[@]}"
   do
      for gpu in "${gpus[@]}"
      do
      dir=results/benchmark_gpu_${gpu}_math_${math}_batch_${batch}/
      if [ ! -d $dir ]; then
         echo Directory $dir does not exist
         continue
      fi

      total_tokens_per_s=0
      for gpu_id in `seq 0 $((gpu - 1))`
      do
         tokens_per_s=`cat ${dir}/log_gpu_${gpu_id}.log \
            |grep TRAIN \
            |cut -f 4 \
            |sed -E -n 's/.*\(([0-9]+)\).*/\1/p' \
            |tail -n 1`
         total_tokens_per_s=$((total_tokens_per_s + tokens_per_s))
      done

      batch_time=`cat ${dir}/log_gpu_0.log \
         |grep TRAIN \
         |cut -f 2 \
         |sed -E -n 's/.*\(([.0-9]+)\).*/\1/p' \
         |tail -n 1`

      n_batches=$(( $sentences / ($batch * $gpu)))
      epoch_time=`awk "BEGIN {print $n_batches * $batch_time}"`

      echo -e math: $math batch: $batch gpus: $gpu "\t\t" $total_tokens_per_s "\t" $epoch_time
      done
   done
done

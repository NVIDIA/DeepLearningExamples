#! /bin/bash

source /workspace/bert/data/bookcorpus/config.sh

SHARD_COUNT=0
rm -rf /workspace/bert/data/bookcorpus/xarg_list.txt
touch /workspace/bert/data/bookcorpus/xarg_list.txt
for file in /workspace/bert/data/bookcorpus/final_text_files_sharded/*; do
  echo ${SHARD_COUNT} >> /workspace/bert/data/bookcorpus/xarg_list.txt
  SHARD_COUNT=$((SHARD_COUNT+1))
done

xargs -n 1 --max-procs=${N_PROCS_PREPROCESS} --arg-file=/workspace/bert/data/bookcorpus/xarg_list.txt /workspace/bert/data/bookcorpus/preprocessing.sh

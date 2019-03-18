#! /bin/bash

source /workspace/bert/data/bookcorpus/config.sh

SHARD_COUNT=0
rm -rf /workspace/bert/data/bookcorpus/xarg_list.txt
touch /workspace/bert/data/bookcorpus/xarg_list.txt
for file in /workspace/bert/data/bookcorpus/test_set_text_files/*; do
  echo ${file} >> /workspace/bert/data/bookcorpus/xarg_list.txt
done

xargs -n 1 --max-procs=${N_PROCS_PREPROCESS} --arg-file=/workspace/bert/data/bookcorpus/xarg_list.txt /workspace/bert/data/bookcorpus/preprocessing_test_set.sh

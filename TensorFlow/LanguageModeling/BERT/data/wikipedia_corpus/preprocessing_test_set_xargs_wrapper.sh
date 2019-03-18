#! /bin/bash

source /workspace/bert/data/wikipedia_corpus/config.sh

SHARD_COUNT=0
rm -rf /workspace/bert/data/wikipedia_corpus/xarg_list.txt
touch /workspace/bert/data/wikipedia_corpus/xarg_list.txt
for file in /workspace/bert/data/wikipedia_corpus/test_set_text_files/*; do
  echo ${file} >> /workspace/bert/data/wikipedia_corpus/xarg_list.txt
done

xargs -n 1 --max-procs=${N_PROCS_PREPROCESS} --arg-file=/workspace/bert/data/wikipedia_corpus/xarg_list.txt /workspace/bert/data/wikipedia_corpus/preprocessing_test_set.sh

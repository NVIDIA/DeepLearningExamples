#! /bin/bash

source /workspace/bert/data/pubmed/config_gvocab_base.sh

SHARD_COUNT=0
rm -rf /workspace/bert/data/pubmed/xarg_list_gvocab_base.txt
touch /workspace/bert/data/pubmed/xarg_list_gvocab_base.txt
for file in /workspace/bert/data/pubmed/final_text_files_sharded_gvocab_base/*; do
  echo ${SHARD_COUNT} >> /workspace/bert/data/pubmed/xarg_list_gvocab_base.txt
  SHARD_COUNT=$((SHARD_COUNT+1))
done

xargs -n 1 --max-procs=${N_PROCS_PREPROCESS} --arg-file=/workspace/bert/data/pubmed/xarg_list_gvocab_base.txt /workspace/bert/data/pubmed/preprocessing_gvocab_base.sh

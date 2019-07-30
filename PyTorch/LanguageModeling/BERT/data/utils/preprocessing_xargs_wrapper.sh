#! /bin/bash

source /workspace/bert/data/utils/config.sh

SHARD_COUNT=0
rm -rf ${TARGET_DIR}/xarg_list.txt
touch ${TARGET_DIR}/xarg_list.txt
for file in ${TARGET_DIR}/final_text_files_sharded/*; do
  echo ${SHARD_COUNT} >> ${TARGET_DIR}/xarg_list.txt
  SHARD_COUNT=$((SHARD_COUNT+1))
done

xargs -n 1 --max-procs=${N_PROCS_PREPROCESS} --arg-file=${TARGET_DIR}/xarg_list.txt /workspace/bert/data/utils/preprocessing.sh

rm ${TARGET_DIR}/xarg_list.txt

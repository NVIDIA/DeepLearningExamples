#! /bin/bash

SHARD_INDEX=${1}
INPUT_FILE="${TARGET_DIR}/final_text_files_sharded/corpus.segmented.part.${SHARD_INDEX}.txt"

source /workspace/bert/data/utils/config.sh

OUTPUT_DIR=${TARGET_DIR}/hdf5_shards
mkdir -p ${OUTPUT_DIR}

OUTPUT_FILE="${OUTPUT_DIR}/${SHARD_INDEX}.hdf5"

python /workspace/bert/create_pretraining_data.py \
  --input_file=${INPUT_FILE} \
  --output_file=${OUTPUT_FILE} \
  --vocab_file=${VOCAB_FILE} \
  --do_lower_case \
  --max_seq_length=${MAX_SEQUENCE_LENGTH} \
  --max_predictions_per_seq=${MAX_PREDICTIONS_PER_SEQUENCE} \
  --masked_lm_prob=${MASKED_LM_PROB} \
  --random_seed=${SEED} \
  --dupe_factor=${DUPE_FACTOR}


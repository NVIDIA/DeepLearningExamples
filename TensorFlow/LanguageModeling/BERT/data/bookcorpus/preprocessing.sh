#! /bin/bash

SHARD_INDEX=${1}
INPUT_FILE="${WORKING_DIR}/final_text_files_sharded/bookcorpus.segmented.part.${SHARD_INDEX}.txt"

source /workspace/bert/data/bookcorpus/config.sh

OUTPUT_DIR=${WORKING_DIR}/final_tfrecords_sharded
mkdir -p ${OUTPUT_DIR}

OUTPUT_FILE="${OUTPUT_DIR}/tf_examples.tfrecord000${SHARD_INDEX}"

python /workspace/bert/create_pretraining_data.py \
  --input_file=${INPUT_FILE} \
  --output_file=${OUTPUT_FILE} \
  --vocab_file=${VOCAB_FILE} \
  --do_lower_case=${DO_LOWER_CASE} \
  --max_seq_length=${MAX_SEQUENCE_LENGTH} \
  --max_predictions_per_seq=${MAX_PREDICTIONS_PER_SEQUENCE} \
  --masked_lm_prob=${MASKED_LM_PROB} \
  --random_seed=${SEED} \
  --dupe_factor=${DUPE_FACTOR}


#! /bin/bash

INPUT_FILE=${1}

source /workspace/bert/data/wikipedia_corpus/config.sh

OUTPUT_DIR=${WORKING_DIR}/test_set_tfrecords
mkdir -p ${OUTPUT_DIR}

#SHARD_INDEX=$(( echo ${INPUT_FILE} | egrep -o [0-9]+ ))
SHARD_INDEX=$( eval echo ${INPUT_FILE} | sed -e s/[^0-9]//g )
OUTPUT_FILE="${OUTPUT_DIR}/tf_examples.tfrecord000${SHARD_INDEX}"

SEED=13254

echo "Shard index ${SHARD_INDEX}"

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


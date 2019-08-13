#! /bin/bash

set -e

USE_BERT_LARGE=true
MAX_SEQUENCE_LENGTH=512
MAX_PREDICTIONS_PER_SEQUENCE=80
MASKED_LM_PROB=0.15
SEED=12345
DUPE_FACTOR=5
DO_LOWER_CASE="True"
N_LINES_PER_SHARD_APPROX=396000   # Default=396000 creates 256 shards

N_PROCS_PREPROCESS=4    # Adjust this based on memory requirements and available number of cores
export WORKING_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

WIKI_DUMP="ftp://ftpmirror.your.org/pub/wikimedia/dumps/enwiki/20190301/enwiki-20190301-pages-articles-multistream.xml.bz2"
BERT_BASE_DIR="${WORKING_DIR}/../pretrained_models_google/uncased_L-12_H-768_A-12"
BERT_LARGE_DIR="${WORKING_DIR}/../pretrained_models_google/uncased_L-24_H-1024_A-16"

if [ "$USE_BERT_LARGE" = true ] ; then
  VOCAB_FILE="${BERT_LARGE_DIR}/vocab.txt"
else
  VOCAB_FILE="${BERT_BASE_DIR}/vocab.txt"
fi

OUTPUT_DIR="${WORKING_DIR}/final_tfrecords_sharded/bert_large_wikipedia_seq_${MAX_SEQUENCE_LENGTH}_pred_${MAX_PREDICTIONS_PER_SEQUENCE}"


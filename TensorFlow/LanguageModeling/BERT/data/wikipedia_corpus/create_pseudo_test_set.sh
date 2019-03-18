#! /bin/bash

source /workspace/bert/data/wikipedia_corpus/config.sh

# Convert test set sharded text files into tfrecords that are ready for BERT pretraining
echo "Creating test set tfrecords for each text shard"
mkdir -p ${WORKING_DIR}/test_set_text_files
mkdir -p ${WORKING_DIR}/test_set_tfrecords
python3 ${WORKING_DIR}/create_pseudo_test_set.py
. ${WORKING_DIR}/preprocessing_test_set_xargs_wrapper.sh ${N_PROCS_PREPROCESS}

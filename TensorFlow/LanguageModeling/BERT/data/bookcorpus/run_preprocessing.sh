#! /bin/bash

source /workspace/bert/data/bookcorpus/config.sh

# Download books
mkdir -p download
python3 /workspace/bookcorpus/download_files.py --list /workspace/bookcorpus/url_list.jsonl --out ${WORKING_DIR}/download --trash-bad-count

# Clean and prep (one book per line)
mkdir -p ${WORKING_DIR}/intermediate_files
python3 ${WORKING_DIR}/clean_and_merge_text.py

# Split books into one-sentence-per-line format for use with BERT scripts
echo "Applying sentence segmentation to get one sentence per line"
mkdir -p ${WORKING_DIR}/final_text_file_single
python3 ${WORKING_DIR}/sentence_segmentation_nltk.py
# Note: NLTK can be replaced with Spacy, although it is slower (2 variations provided)

# Shard finalized text so that it has a chance of fitting in memory when creating pretraining data into tfrecords (choose appropriate number of shards for distributed training)
echo "Shard text files - size is approximate to prevent splitting a book across shards"
mkdir -p ${WORKING_DIR}/final_text_files_sharded
python3 ${WORKING_DIR}/shard_text_input_file.py

# Convert sharded text files into tfrecords that are ready for BERT pretraining
echo "Creating tfrecords for each text shard"
mkdir -p ${WORKING_DIR}/final_tfrecords_sharded
. ${WORKING_DIR}/preprocessing_xargs_wrapper.sh ${N_PROCS_PREPROCESS}


#!/bin/bash

# Note: There are several directories created to make it clear what has been performed at each stage of preprocessing. The intermediate files may be useful if you want to further clean/prepare/augment the data for your own applications.
# NLTK was chosen as the default over spaCy simply due to speed of sentence segmentation on the large files.

MERGED_DIR=$1
args="${*:2}"

source utils/config.sh

mkdir -p ${MERGED_DIR}

corpus_file=${MERGED_DIR}/corpus.txt
## Shuffle the full corpus texts
if [ ! -z $3 ]
then
  echo "Merging $args"
  cat $args | sed "/^$/d" | shuf > $corpus_file
else
  corpus_file=$2
fi

# Split articles into one-sentence-per-line format for use with BERT scripts
echo "Applying sentence segmentation to get one sentence per line"
mkdir -p ${MERGED_DIR}/final_text_file_single
python3 utils/sentence_segmentation_nltk.py $corpus_file ${MERGED_DIR}/final_text_file_single/corpus.segmented.nltk.txt

## Shard finalized text so that it has a chance of fitting in memory when creating pretraining data into hdf5 (choose appropriate number of shards for distributed training)
echo "Shard text files - size is approximate to prevent splitting an article across shards"
mkdir -p ${MERGED_DIR}/final_text_files_sharded
python3 utils/shard_text_input_file.py ${MERGED_DIR}/final_text_file_single/corpus.segmented.nltk.txt ${MERGED_DIR}/final_text_files_sharded/corpus.segmented.part.

# Convert sharded text files into hdf5 that are ready for BERT pretraining
echo "Creating hdf5 for each text shard"
mkdir -p ${MERGED_DIR}/hdf5_shards
export TARGET_DIR=${MERGED_DIR}
. utils/preprocessing_xargs_wrapper.sh ${N_PROCS_PREPROCESS}


#! /bin/bash

source /workspace/bert/data/wikipedia_corpus/config.sh

# Note: There are several directories created to make it clear what has been performed at each stage of preprocessing. The intermediate files may be useful if you want to further clean/prepare/augment the data for your own applications.
# NLTK was chosen as the default over spaCy simply due to speed of sentence segmentation on the large files.

# Download Wikipedia dump file
mkdir -p ${WORKING_DIR}/download

# Not using --noclobber since it emits an error if exists (incompatible with bash 'set -e')
echo "Downloading Wikidump"
if [ ! -f ${WORKING_DIR}/download/wikidump.xml.bz2 ]; then
  cd ${WORKING_DIR}/download && wget -O wikidump.xml.bz2 ${WIKI_DUMP}
fi

# Extract dump
echo "Extracting Wikidump"
mkdir -p ${WORKING_DIR}/raw_data
#cd ${WORKING_DIR}/raw_data && pv ${WORKING_DIR}/download/wikidump.xml.bz2 | pbzip2 -kdc > ${WORKING_DIR}/raw_data/wikidump.xml
cd ${WORKING_DIR}/raw_data && pv ${WORKING_DIR}/download/wikidump.xml.bz2 | bunzip2 -kdc > ${WORKING_DIR}/raw_data/wikidump.xml
#cd ${WORKING_DIR}/raw_data && bunzip2 -kdc ${WORKING_DIR}/download/wikidump.xml.bz2 > ${WORKING_DIR}/raw_data/wikidump.xml
 
# Wikiextractor.py - Creates lots of folders/files in "doc format"
echo "Running Wikiextractor"
mkdir -p ${WORKING_DIR}/extracted_articles
/workspace/wikiextractor/WikiExtractor.py ${WORKING_DIR}/raw_data/wikidump.xml -b 1000M --processes ${N_PROCS_PREPROCESS} -o ${WORKING_DIR}/extracted_articles

# Remove XML Tags and extraneous titles (since they are not sentences)
# Also clean to remove lines between paragraphs within article and use space-separated articles
echo "Cleaning and formatting files (one article per line)"
mkdir -p ${WORKING_DIR}/intermediate_files
python3 ${WORKING_DIR}/remove_tags_and_clean.py

# Split articles into one-sentence-per-line format for use with BERT scripts
echo "Applying sentence segmentation to get one sentence per line"
mkdir -p ${WORKING_DIR}/final_text_file_single
python3 ${WORKING_DIR}/wiki_sentence_segmentation_nltk.py
# Note: NLTK can be replaced with Spacy, although it is slower (2 variations provided)

# Shard finalized text so that it has a chance of fitting in memory when creating pretraining data into tfrecords (choose appropriate number of shards for distributed training)
echo "Shard text files - size is approximate to prevent splitting an article across shards"
mkdir -p ${WORKING_DIR}/final_text_files_sharded
python3 ${WORKING_DIR}/shard_text_input_file.py

# Convert sharded text files into tfrecords that are ready for BERT pretraining
echo "Creating tfrecords for each text shard"
mkdir -p ${WORKING_DIR}/final_tfrecords_sharded
. ${WORKING_DIR}/preprocessing_xargs_wrapper.sh ${N_PROCS_PREPROCESS}

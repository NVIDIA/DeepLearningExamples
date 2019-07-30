#! /bin/bash

WIKI_DUMP="https://dumps.wikimedia.org/enwiki/20190320/enwiki-20190320-pages-articles-multistream.xml.bz2"
N_PROCS_PREPROCESS=$(nproc)    # Adjust this based on memory requirements and available number of cores

# Download Wikipedia dump file
mkdir -p ./download

# Not using --noclobber since it emits an error if exists (incompatible with bash 'set -e')
echo "Downloading Wikidump"
if [ ! -f ./download/wikidump.xml.bz2 ]; then
  wget -O ./download/wikidump.xml.bz2 ${WIKI_DUMP}
fi

# Extract dump
echo "Extracting Wikidump"
mkdir -p ./raw_data
if [ ! -f ./raw_data/wikidump.xml ]; then
  pv ./download/wikidump.xml.bz2 | bunzip2 -kdc > ./raw_data/wikidump.xml
fi
 
# Wikiextractor.py - Creates lots of folders/files in "doc format"
echo "Running Wikiextractor"
mkdir -p ./extracted_articles
/workspace/wikiextractor/WikiExtractor.py ./raw_data/wikidump.xml -b 1000M --processes ${N_PROCS_PREPROCESS} -o ./extracted_articles

# Remove XML Tags and extraneous titles (since they are not sentences)
# Also clean to remove lines between paragraphs within article and use space-separated articles
echo "Cleaning and formatting files (one article per line)"
python3 ./remove_tags_and_clean.py ./extracted_articles ./wikipedia_corpus.txt

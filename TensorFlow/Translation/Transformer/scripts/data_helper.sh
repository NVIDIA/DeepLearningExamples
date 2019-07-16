#!/bin/bash

set -e

SEED=$1

cd /research/transformer

# TODO: Add SEED to process_data.py since this uses a random generator (future PR)

mkdir -p data

sacrebleu -t wmt14/full -l en-de --echo ref > data/newstest2014.de
sacrebleu -t wmt14/full -l en-de --echo src > data/newstest2014.en

mkdir -p /research/transformer/data/raw_data
mkdir -p /research/transformer/data/processed_data/
mkdir -p /research/transformer/data/processed_data/utf8

cp /research/transformer/transformer/vocab/vocab.translate_ende_wmt32k.32768.subwords /research/transformer/data/processed_data/vocab.ende.32768

python3 transformer/data/process_data.py --raw_dir data/raw_data --data_dir data/processed_data --sentencepiece && python3 transformer/data/convert_utf8_to_tfrecord.py --data_dir /research/transformer/data/processed_data/utf8

bash scripts/verify_dataset.sh

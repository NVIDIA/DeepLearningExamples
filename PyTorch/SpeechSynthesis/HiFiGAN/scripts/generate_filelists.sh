#!/usr/bin/env bash

set -e

: ${DATASET_PATH:=data/LJSpeech-1.1}

# Generate filelists
python common/split_lj.py --metadata-path "${DATASET_PATH}/metadata.csv" --subsets train val test all
python common/split_lj.py --metadata-path "${DATASET_PATH}/metadata.csv" --add-transcript --subsets all  # used to extract ground-truth mels or pitch
python common/split_lj.py --metadata-path "${DATASET_PATH}/metadata.csv" --add-pitch --add-transcript --subsets all  # used during extracting fastpitch mels

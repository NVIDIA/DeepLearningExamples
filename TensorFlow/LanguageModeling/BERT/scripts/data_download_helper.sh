#!/usr/bin/env bash

# Download SQUAD
cd /workspace/bert/data/squad && . squad_download.sh

# Download pretrained_models
cd /workspace/bert/data/pretrained_models_google && python3 download_models.py

# WIKI Download, set config in data_generators/wikipedia_corpus/config.sh
cd /workspace/bert/data/wikipedia_corpus && . run_preprocessing.sh

cd /workspace/bert/data/bookcorpus && . run_preprocessing.sh

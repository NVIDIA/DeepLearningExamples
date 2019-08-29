#!/bin/bash

# Download
python3 /workspace/bert/data/bertPrep.py --action download --dataset bookscorpus
python3 /workspace/bert/data/bertPrep.py --action download --dataset wikicorpus_en

python3 /workspace/bert/data/bertPrep.py --action download --dataset google_pretrained_weights  # Includes vocab

python3 /workspace/bert/data/bertPrep.py --action download --dataset squad
#python3 /workspace/bert/data/bertPrep.py --action download --dataset mrpc


# Properly format the text files
python3 /workspace/bert/data/bertPrep.py --action text_formatting --dataset bookscorpus
python3 /workspace/bert/data/bertPrep.py --action text_formatting --dataset wikicorpus_en


# Shard the text files (group wiki+books then shard)
python3 /workspace/bert/data/bertPrep.py --action sharding --dataset books_wiki_en_corpus


# Create HDF5 files Phase 1
python3 /workspace/bert/data/bertPrep.py --action create_hdf5_files --dataset books_wiki_en_corpus --max_seq_length 128 --max_predictions_per_seq 20


# Create HDF5 files Phase 2
python3 /workspace/bert/data/bertPrep.py --action create_hdf5_files --dataset books_wiki_en_corpus --max_seq_length 512 --max_predictions_per_seq 80

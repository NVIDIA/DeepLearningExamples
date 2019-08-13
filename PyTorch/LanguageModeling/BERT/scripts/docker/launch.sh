#!/bin/bash

DATA_DIR=${1:-"${PWD}/data/hdf5/books_wiki_en_corpus"}
VOCAB_DIR=${2:-"${PWD}/data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16"}
CHECKPOINT_DIR=${3:-"${PWD}/checkpoints"}
RESULTS_DIR=${4:-"${PWD}/results"}

docker run -it --rm \
  --runtime=nvidia \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v ${PWD}:/workspace/bert \
  -v $DATA_DIR:/workspace/bert/data/hdf5/books_wiki_en_corpus \
  -v $CHECKPOINT_DIR:/workspace/checkpoints \
  -v $VOCAB_DIR:/workspace/bert/data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16 \
  -v $RESULTS_DIR:/results \
  bert_pyt bash

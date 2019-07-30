#!/bin/bash


MERGED_DIR=$1 # e.g wikipedia+bookcorpus
INPUTFILES=$2 # directories with hdf5 files separated by comma
NUM_SHARDS=$3

source utils/config.sh


META_DIR=$MERGED_DIR/meta
mkdir -p ${MERGED_DIR}
mkdir -p ${META_DIR}

echo "create mixed dataset ids"
echo "python utils/create_mixed_dataset_ids.py --input_files=${INPUTFILES} --num_output_shards=${NUM_SHARDS} --output_dir=${META_DIR} --random_seed=${SEED}"
python utils/create_mixed_dataset_ids.py --input_files=${INPUTFILES} --num_output_shards=${NUM_SHARDS} --output_dir=${META_DIR} --random_seed=${SEED}


echo "Creating hdf5 for each text shard"
mkdir -p ${MERGED_DIR}/hdf5_shards
echo "create mixed datasets with hdf5 files"
echo "python utils/create_mixed_dataset.py --input_files=${INPUTFILES} --output_dir=${MERGED_DIR}/hdf5_shards --lookup=${META_DIR}/lookup_table.pkl --indices_dir=${META_DIR} --index_range=0-${NUM_SHARDS} --random_seed=${SEED}"
python utils/create_mixed_dataset.py --input_files=${INPUTFILES} --output_dir=${MERGED_DIR}/hdf5_shards --lookup=${META_DIR}/lookup_table.pkl --indices_dir=${META_DIR} --index_range=0-$((NUM_SHARDS-1)) --random_seed=${SEED}


rm -rf ${META_DIR}



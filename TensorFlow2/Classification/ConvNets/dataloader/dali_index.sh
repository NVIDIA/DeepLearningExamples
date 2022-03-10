#!/bin/bash

SRC_DIR=${1}
DST_DIR=${2}

echo "Creating training file indexes"
mkdir -p ${DST_DIR}

for file in ${SRC_DIR}/train-*; do
    BASENAME=$(basename $file)
    DST_NAME=$DST_DIR/$BASENAME

    echo "Creating index $DST_NAME for $file"
    tfrecord2idx $file $DST_NAME
done

echo "Creating validation file indexes"
for file in ${SRC_DIR}/validation-*; do
    BASENAME=$(basename $file)
    DST_NAME=$DST_DIR/$BASENAME

    echo "Creating index $DST_NAME for $file"
    tfrecord2idx $file $DST_NAME
done
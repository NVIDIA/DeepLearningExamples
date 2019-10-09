#!/bin/bash

##############################################################################
# Copyright (c) Jonathan Dekhtiar - contact@jonathandekhtiar.eu
# All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

# Usage: ./download_and_preprocess_dagm2007.sh /path/to/dataset/directory/

if [[ ! "$BASH_VERSION" ]] ; then
    echo "Please do not use sh to run this script ($0), just execute it directly" 1>&2
    exit 1
fi

if [[ -z "$1" ]]
  then
    echo -e "Error: Argument is missing. No dataset directory received."
    echo -e "Usage: '$0 /path/to/dataset/directory/'"
    exit 1
fi

DATASET_DIR=$(realpath -s $1)

ZIP_FILES_DIR=${DATASET_DIR}/zip_files
RAW_IMAGES_DIR=${DATASET_DIR}/raw_images

PUBLIC_ZIP_FILES_DIR=${ZIP_FILES_DIR}/public
PUBLIC_RAW_IMAGES_DIR=${RAW_IMAGES_DIR}/public

if [[ ! -e ${PUBLIC_ZIP_FILES_DIR} ]]; then
    echo "creating ${PUBLIC_ZIP_FILES_DIR} ..."
    mkdir -p ${PUBLIC_ZIP_FILES_DIR}
fi

if [[ ! -e ${PUBLIC_RAW_IMAGES_DIR} ]]; then
    echo "creating ${PUBLIC_RAW_IMAGES_DIR} ..."
    mkdir -p ${PUBLIC_RAW_IMAGES_DIR}
fi

PRIVATE_ZIP_FILES_DIR=${ZIP_FILES_DIR}/private
PRIVATE_RAW_IMAGES_DIR=${RAW_IMAGES_DIR}/private

if [[ ! -e ${PRIVATE_ZIP_FILES_DIR} ]]; then
    echo "creating ${PRIVATE_ZIP_FILES_DIR} ..."
    mkdir -p ${PRIVATE_ZIP_FILES_DIR}
fi

if [[ ! -e ${PRIVATE_RAW_IMAGES_DIR} ]]; then
    echo "creating ${PRIVATE_RAW_IMAGES_DIR} ..."
    mkdir -p ${PRIVATE_RAW_IMAGES_DIR}
fi

echo -e "\n################################################"
echo -e "Processing Public Dataset"
echo -e "################################################\n"

sleep 2

BASE_PUBLIC_URL="https://resources.mpi-inf.mpg.de/conference/dagm/2007"

declare -a arr=(
    "Class1.zip"
    "Class1_def.zip"
    "Class2.zip"
    "Class2_def.zip"
    "Class3.zip"
    "Class3_def.zip"
    "Class4.zip"
    "Class4_def.zip"
    "Class5.zip"
    "Class5_def.zip"
    "Class6.zip"
    "Class6_def.zip"
)

for file in "${arr[@]}"
do
    if [[ ! -e ${PUBLIC_ZIP_FILES_DIR}/${file} ]]; then
        echo -e "Downloading File: $BASE_PUBLIC_URL/$file ..."
        wget -N ${BASE_PUBLIC_URL}/${file} -O ${PUBLIC_ZIP_FILES_DIR}/${file}
    fi

    # Unzip without overwriting
    unzip -n ${PUBLIC_ZIP_FILES_DIR}/${file} -d ${PUBLIC_RAW_IMAGES_DIR}

done

chmod -R 744 ${PUBLIC_ZIP_FILES_DIR}
chmod -R 744 ${PUBLIC_RAW_IMAGES_DIR}
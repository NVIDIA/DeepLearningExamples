#!/bin/bash
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Script to download and preprocess ImageNet Challenge 2012
# training and validation data set.
#
# The final output of this script are sharded TFRecord files containing
# serialized Example protocol buffers. See build_imagenet_data.py for
# details of how the Example protocol buffers contain the ImageNet data.
#
# The final output of this script appears as such:
#
#   data_dir/train-00000-of-01024
#   data_dir/train-00001-of-01024
#    ...
#   data_dir/train-01023-of-01024
#
# and
#
#   data_dir/validation-00000-of-00128
#   data_dir/validation-00001-of-00128
#   ...
#   data_dir/validation-00127-of-00128
#
# Note that this script may take several hours to run to completion. The
# conversion of the ImageNet data to TFRecords alone takes 2-3 hours depending
# on the speed of your machine. Please be patient.
#
# **IMPORTANT**
# To download the raw images, the user must create an account with image-net.org
# and generate a username and access_key. The latter two are required for
# downloading the raw images.
#
# usage:
#  ./preprocess_imagenet.sh [data-dir]
set -e

if [ -z "$1" ]; then
  echo "Usage: preprocess_imagenet.sh [data dir]"
  exit
fi

DATA_DIR="${1%/}"
SCRATCH_DIR="${DATA_DIR}/raw-data/"
mkdir -p ${SCRATCH_DIR}

# Convert the XML files for bounding box annotations into a single CSV.
echo "Extracting bounding box information from XML."
BOUNDING_BOX_SCRIPT="./dataprep/process_bounding_boxes.py"
BOUNDING_BOX_FILE="${DATA_DIR}/imagenet_2012_bounding_boxes.csv"
BOUNDING_BOX_DIR="${DATA_DIR}/bounding_boxes/"

LABELS_FILE="./dataprep/imagenet_lsvrc_2015_synsets.txt"

"${BOUNDING_BOX_SCRIPT}" "${BOUNDING_BOX_DIR}" "${LABELS_FILE}" \
 | sort > "${BOUNDING_BOX_FILE}"
echo "preprocessing the ImageNet data."

# Build the TFRecords version of the ImageNet data.
OUTPUT_DIRECTORY="${DATA_DIR}"
IMAGENET_METADATA_FILE="./dataprep/imagenet_metadata.txt"

python ./dataprep/build_imagenet_data.py \
  --train_directory="${DATA_DIR}/train" \
  --validation_directory="${DATA_DIR}/val" \
  --output_directory="${DATA_DIR}/result" \
  --imagenet_metadata_file="${IMAGENET_METADATA_FILE}" \
  --labels_file="${LABELS_FILE}" \
  --bounding_box_file="${BOUNDING_BOX_FILE}"

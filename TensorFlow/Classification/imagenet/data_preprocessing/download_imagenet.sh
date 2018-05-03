#"!/bin/bash
# Copyright 2016 Google Inc. All Rights Reserved.
# Copyright 2017 NVIDIA Corp. All Rights Reserved.  
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

# Script to download ImageNet Challenge 2012 training and validation data set.
#
# Downloads and decompresses raw images and bounding boxes.
#
# **IMPORTANT**
# To download the raw images, the user must create an account with image-net.org
# and generate a username and access_key. The latter two are required for
# downloading the raw images.
#
# usage:
#  ./download_imagenet.sh [dirname]
set -e

cat <<END

In order to download the imagenet data, you have to create an account with
image-net.org. This will get you a username and an access key. You can set the
IMAGENET_USERNAME and IMAGENET_ACCESS_KEY environment variables, or you can
enter the credentials interactively if env vars are not found.
 
END

if [ "x$IMAGENET_USERNAME" == x ]; then
  read -p "Username: " IMAGENET_USERNAME
fi

if [ "x$IMAGENET_ACCESS_KEY" == x ]; then
  read -p "Access key: " IMAGENET_ACCESS_KEY
fi

if [ "x$IMAGENET_OUT_DIR" == x ]; then
  read -p "Dataset output dir: " IMAGENET_OUT_DIR
fi

ORIG_DIR=$(pwd)

# We need to create a data subdir under the ImageNet data dir
# root to keep the tarballs at the root of the data dir.
IMAGENET_DATA_DIR="${IMAGENET_OUT_DIR}/data"
SYNSETS_FILE="${ORIG_DIR}/imagenet_lsvrc_2015_synsets.txt"

echo "Saving downloaded files to $IMAGENET_DATA_DIR"
mkdir -p "${IMAGENET_DATA_DIR}"
BBOX_DIR="${IMAGENET_OUT_DIR}/bounding_boxes"
mkdir -p "${BBOX_DIR}"
cd "${IMAGENET_DATA_DIR}"

# Download and process all of the ImageNet bounding boxes.
BASE_URL="http://www.image-net.org/challenges/LSVRC/2012/nonpub"

# See here for details: http://www.image-net.org/download-bboxes
BOUNDING_BOX_ANNOTATIONS="${BASE_URL}/ILSVRC2012_bbox_train_v2.tar.gz"
BBOX_TAR_BALL="${BBOX_DIR}/annotations.tar.gz"
if [ -e ${BBOX_TAR_BALL} ]; then
  echo "${BBOX_TAR_BALL} exists. Re-using cached file."
else
  echo "Downloading bounding box annotations."
  wget "${BOUNDING_BOX_ANNOTATIONS}" -O "${BBOX_TAR_BALL}" || BASE_URL_CHANGE=1
  if [ $BASE_URL_CHANGE ]; then
    BASE_URL="http://www.image-net.org/challenges/LSVRC/2012/nnoupb"
    BOUNDING_BOX_ANNOTATIONS="${BASE_URL}/ILSVRC2012_bbox_train_v2.tar.gz"
    BBOX_TAR_BALL="${BBOX_DIR}/annotations.tar.gz"
    wget "${BOUNDING_BOX_ANNOTATIONS}" -O "${BBOX_TAR_BALL}"
  fi
fi
# TODO: check if uncompressed annotations are cached as well
echo "Uncompressing bounding box annotations."

python ${ORIG_DIR}/check_uncompressed.py ${BBOX_TAR_BALL} ${BBOX_DIR} || STATUS=$?
if [[ $STATUS -eq 0 ]]; then
  echo "Nothing to decompress. Using cached directories for ${BBOX_DIR}"
else 
  echo "Uncompressing ${BBOX_TAR_BALL}."
  tar xzf "${BBOX_TAR_BALL}" -C "${BBOX_DIR}"
fi

LABELS_ANNOTATED="${BBOX_DIR}/*"
NUM_XML=$(ls -1 ${LABELS_ANNOTATED} | wc -l)
echo "Identified ${NUM_XML} bounding box annotations."

# Download and uncompress all images from the ImageNet 2012 validation dataset.
VALIDATION_TARBALL="ILSVRC2012_img_val.tar"
OUTPUT_PATH="${IMAGENET_DATA_DIR}/validation/"
mkdir -p "${OUTPUT_PATH}"
cd "${IMAGENET_OUT_DIR}"
if [ -e ${VALIDATION_TARBALL} ]; then
  echo "${VALIDATION_TARBALL} exists. Re-using cached file."
else
  echo "Downloading ${VALIDATION_TARBALL} to ${IMAGENET_OUT_DIR}."
  wget -nd -c "${BASE_URL}/${VALIDATION_TARBALL}"
fi
echo "Extracting ${VALIDATION_TARBALL} to ${OUTPUT_PATH}"
tar xf "${VALIDATION_TARBALL}" -C "${OUTPUT_PATH}"

# Download all images from the ImageNet 2012 train dataset.
TRAIN_TARBALL="ILSVRC2012_img_train.tar"
OUTPUT_PATH="${IMAGENET_DATA_DIR}/train/"
mkdir -p "${OUTPUT_PATH}"
cd "${IMAGENET_OUT_DIR}"
if [ -e ${TRAIN_TARBALL} ]; then
  echo "${TRAIN_TARBALL} exists. Re-using cached file."
else
  echo "Downloading ${TRAIN_TARBALL} to ${IMAGENET_OUT_DIR}."
  wget -nd -c "${BASE_URL}/${TRAIN_TARBALL}"
fi

# Un-compress the individual tar-files within the train tar-file.
echo "Uncompressing individual train tar-balls in the training data to ${OUTPUT_PATH}."

SYNSET=${SYNSETS_FILE}

while read SYNSET; do
 echo "Processing: ${SYNSET}"

# Create a directory and delete anything there.
  mkdir -p "${OUTPUT_PATH}/${SYNSET}"
  rm -rf "${OUTPUT_PATH}/${SYNSET}/*"

# Uncompress into the directory.
  tar xf "${TRAIN_TARBALL}" "${SYNSET}.tar"
  tar xf "${SYNSET}.tar" -C "${OUTPUT_PATH}/${SYNSET}/"
  rm -f "${SYNSET}.tar"

  echo "Finished processing: ${SYNSET}"
 done < "${SYNSETS_FILE}"

# Bounding box label generation 

TRAIN_DIR="${IMAGENET_DATA_DIR}/train"
VALIDATION_DIR="${IMAGENET_DATA_DIR}/validation"

# Convert the XML files for bounding box annotations into a single CSV.
#echo "Extracting bounding box information from XML."
BOUNDING_BOX_SCRIPT="./process_bounding_boxes.py"
BOUNDING_BOX_FILE="${BBOX_DIR}/imagenet_2012_bounding_boxes.csv"

LABELS_FILE="imagenet_lsvrc_2015_synsets.txt"

cd ${ORIG_DIR}
"${BOUNDING_BOX_SCRIPT}" "${BBOX_DIR}" "${LABELS_FILE}" \
 | sort >"${BOUNDING_BOX_FILE}"

echo "Organizing the validation data into sub-directories."
PREPROCESS_VAL_SCRIPT="${ORIG_DIR}/preprocess_imagenet_validation_data.py"
VAL_LABELS_FILE="${ORIG_DIR}/imagenet_2012_validation_synset_labels.txt"

"${PREPROCESS_VAL_SCRIPT}" "${VALIDATION_DIR}" "${VAL_LABELS_FILE}"


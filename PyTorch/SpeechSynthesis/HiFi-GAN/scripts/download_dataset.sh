#!/usr/bin/env bash

set -e

DATA_BASE_DIR="data"
DATA_DIR="${DATA_BASE_DIR}/LJSpeech-1.1"
LJS_ARCH_NAME="LJSpeech-1.1.tar.bz2"
LJS_ARCH="${DATA_BASE_DIR}/${LJS_ARCH_NAME}"
LJS_URL="http://data.keithito.com/data/speech/${LJS_ARCH_NAME}"

if [ ! -d ${DATA_DIR} ]; then
  echo "Downloading ${LJS_ARCH} ..."
  wget -q ${LJS_URL} -P ${DATA_BASE_DIR}
  echo "Extracting ${LJS_ARCH} ..."
  tar jxvf ${LJS_ARCH} -C ${DATA_BASE_DIR}
  rm -f ${LJS_ARCH}
fi

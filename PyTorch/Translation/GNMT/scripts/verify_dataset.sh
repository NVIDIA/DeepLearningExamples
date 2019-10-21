#!/bin/bash

# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set -e

DATASET_DIR='data/wmt16_de_en'

ACTUAL_SRC_TRAIN=`cat ${DATASET_DIR}/train.tok.clean.bpe.32000.en |md5sum`
EXPECTED_SRC_TRAIN='b7482095b787264a310d4933d197a134  -'
if [[ $ACTUAL_SRC_TRAIN = $EXPECTED_SRC_TRAIN ]]; then
  echo "OK: correct ${DATASET_DIR}/train.tok.clean.bpe.32000.en"
else
  echo "ERROR: incorrect ${DATASET_DIR}/train.tok.clean.bpe.32000.en"
  echo "ERROR: expected $EXPECTED_SRC_TRAIN"
  echo "ERROR: found $ACTUAL_SRC_TRAIN"
fi

ACTUAL_TGT_TRAIN=`cat ${DATASET_DIR}/train.tok.clean.bpe.32000.de |md5sum`
EXPECTED_TGT_TRAIN='409064aedaef5b7c458ff19a7beda462  -'
if [[ $ACTUAL_TGT_TRAIN = $EXPECTED_TGT_TRAIN ]]; then
  echo "OK: correct ${DATASET_DIR}/train.tok.clean.bpe.32000.de"
else
  echo "ERROR: incorrect ${DATASET_DIR}/train.tok.clean.bpe.32000.de"
  echo "ERROR: expected $EXPECTED_TGT_TRAIN"
  echo "ERROR: found $ACTUAL_TGT_TRAIN"
fi

ACTUAL_SRC_VAL=`cat ${DATASET_DIR}/newstest_dev.tok.clean.bpe.32000.en |md5sum`
EXPECTED_SRC_VAL='704c4ba8c8b63df1f6678d32b91438b5  -'
if [[ $ACTUAL_SRC_VAL = $EXPECTED_SRC_VAL ]]; then
  echo "OK: correct ${DATASET_DIR}/newstest_dev.tok.clean.bpe.32000.en"
else
  echo "ERROR: incorrect ${DATASET_DIR}/newstest_dev.tok.clean.bpe.32000.en"
  echo "ERROR: expected $EXPECTED_SRC_VAL"
  echo "ERROR: found $ACTUAL_SRC_VAL"
fi

ACTUAL_TGT_VAL=`cat ${DATASET_DIR}/newstest_dev.tok.clean.bpe.32000.de |md5sum`
EXPECTED_TGT_VAL='d27f5a64c839e20c5caa8b9d60075dde  -'
if [[ $ACTUAL_TGT_VAL = $EXPECTED_TGT_VAL ]]; then
  echo "OK: correct ${DATASET_DIR}/newstest_dev.tok.clean.bpe.32000.de"
else
  echo "ERROR: incorrect ${DATASET_DIR}/newstest_dev.tok.clean.bpe.32000.de"
  echo "ERROR: expected $EXPECTED_TGT_VAL"
  echo "ERROR: found $ACTUAL_TGT_VAL"
fi

ACTUAL_SRC_TEST=`cat ${DATASET_DIR}/newstest2014.tok.bpe.32000.en |md5sum`
EXPECTED_SRC_TEST='cb014e2509f86cd81d5a87c240c07464  -'
if [[ $ACTUAL_SRC_TEST = $EXPECTED_SRC_TEST ]]; then
  echo "OK: correct ${DATASET_DIR}/newstest2014.tok.bpe.32000.en"
else
  echo "ERROR: incorrect ${DATASET_DIR}/newstest2014.tok.bpe.32000.en"
  echo "ERROR: expected $EXPECTED_SRC_TEST"
  echo "ERROR: found $ACTUAL_SRC_TEST"
fi

ACTUAL_TGT_TEST=`cat ${DATASET_DIR}/newstest2014.tok.bpe.32000.de |md5sum`
EXPECTED_TGT_TEST='d616740f6026dc493e66efdf9ac1cb04  -'
if [[ $ACTUAL_TGT_TEST = $EXPECTED_TGT_TEST ]]; then
  echo "OK: correct ${DATASET_DIR}/newstest2014.tok.bpe.32000.de"
else
  echo "ERROR: incorrect ${DATASET_DIR}/newstest2014.tok.bpe.32000.de"
  echo "ERROR: expected $EXPECTED_TGT_TEST"
  echo "ERROR: found $ACTUAL_TGT_TEST"
fi

ACTUAL_TGT_TEST_TARGET=`cat ${DATASET_DIR}/newstest2014.de |md5sum`
EXPECTED_TGT_TEST_TARGET='f6c3818b477e4a25cad68b61cc883c17  -'
if [[ $ACTUAL_TGT_TEST_TARGET = $EXPECTED_TGT_TEST_TARGET ]]; then
  echo "OK: correct ${DATASET_DIR}/newstest2014.de"
else
  echo "ERROR: incorrect ${DATASET_DIR}/newstest2014.de"
  echo "ERROR: expected $EXPECTED_TGT_TEST_TARGET"
  echo "ERROR: found $ACTUAL_TGT_TEST_TARGET"
fi

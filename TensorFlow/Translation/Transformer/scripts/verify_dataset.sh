#!/bin/bash

set -e
ACTUAL_TEST=`cat data/newstest2014.en | md5sum`
EXPECTED_TEST='dabf51a9c02b2235632f3cee75c72d49  -'
if [[ $ACTUAL_TEST = $EXPECTED_TEST ]]; then
  echo "OK: correct data/newstest2014.en"
else
  echo "ERROR: incorrect data/newstest2014.en"
  echo "ERROR: expected $EXPECTED_TEST"
  echo "ERROR: found $ACTUAL_TEST"
fi
ACTUAL_TEST=`cat data/newstest2014.de | md5sum`
EXPECTED_TEST='f6c3818b477e4a25cad68b61cc883c17  -'
if [[ $ACTUAL_TEST = $EXPECTED_TEST ]]; then
  echo "OK: correct data/newstest2014.de"
else
  echo "ERROR: incorrect data/newstest2014.de"
  echo "ERROR: expected $EXPECTED_TEST"
  echo "ERROR: found $ACTUAL_TEST"
fi
ACTUAL_RAW=`find data/raw_data/ -type f -exec md5sum {} \; | sort -k 2 | md5sum`
EXPECTED_RAW='8fd41a5c658948dfbc1ec83751d9c7fe  -'
if [[ $ACTUAL_RAW = $EXPECTED_RAW ]]; then
  echo "OK: correct raw_data/"
else
  echo "ERROR: incorrect data/raw_data/"
  echo "ERROR: expected $EXPECTED_RAW"
  echo "ERROR: found $ACTUAL_RAW"
fi

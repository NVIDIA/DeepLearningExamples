#!/usr/bin/env bash

echo "Downloading and verifying dataset for squad..."

# Download SQuAD

v1="v1.1"
echo "Downloading dataset $v1"
mkdir -p $v1
wget -q https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O $v1/train-v1.1.json
wget -q https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O $v1/dev-v1.1.json
wget -q https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/ -O $v1/evaluate-v1.1.py

echo "Download done; calculating md5sums for dataset $v1"
EXP_TRAIN_v1='981b29407e0affa3b1b156f72073b945  -'
EXP_DEV_v1='3e85deb501d4e538b6bc56f786231552  -'
EXP_EVAL_v1='afb04912d18ff20696f7f88eed49bea9  -'
CALC_TRAIN_v1=`md5sum ${v1}/train-v1.1.json`
CALC_DEV_v1=`md5sum ${v1}/dev-v1.1.json`
CALC_EVAL_v1=`md5sum ${v1}/evaluate-v1.1.py`

v2="v2.0"
echo "Downloading dataset $v2"
mkdir -p $v2
wget -q https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O $v2/train-v2.0.json
wget -q https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O $v2/dev-v2.0.json
wget -q https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/ -O $v2/evaluate-v2.0.py

echo "Download done; calculating md5sums for dataset $v2"
EXP_TRAIN_v2='62108c273c268d70893182d5cf8df740  -'
EXP_DEV_v2='246adae8b7002f8679c027697b0b7cf8  -'
EXP_EVAL_v2='ff23213bed5516ea4a6d9edb6cd7d627  -'
CALC_TRAIN_v2=`md5sum ${v2}/train-v2.0.json`
CALC_DEV_v2=`md5sum ${v2}/dev-v2.0.json`
CALC_EVAL_v2=`md5sum ${v2}/evaluate-v2.0.py`

echo "Squad data download done!"

echo "Verifying dataset...."

if [ "$EXP_TRAIN_v1" != "$CALC_TRAIN_v1" ]; then
    echo "WARN: train-v1.1.json is corrupted! md5sum doesn't match"
fi

if [ "$EXP_DEV_v1" != "$CALC_DEV_v1" ]; then
    echo "WARN: dev-v1.1.json is corrupted! md5sum doesn't match"
fi
if [ "$EXP_EVAL_v1" != "$CALC_EVAL_v1" ]; then
    echo "WARN: evaluate-v1.1.py is corrupted! md5sum doesn't match"
fi


if [ "$EXP_TRAIN_v2" != "$CALC_TRAIN_v2" ]; then
    echo "WARN: train-v2.0.json is corrupted! md5sum doesn't match"
fi
if [ "$EXP_DEV_v2" != "$CALC_DEV_v2" ]; then
    echo "WARN: dev-v2.0.json is corrupted! md5sum doesn't match"
fi
if [ "$EXP_EVAL_v2" != "$CALC_EVAL_v2" ]; then
    echo "WARN: evaluate-v2.0.py is corrupted! md5sum doesn't match"
fi

echo "Complete!"

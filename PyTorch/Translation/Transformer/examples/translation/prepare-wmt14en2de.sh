#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh
#
#-------------------------------------------------------------------------
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt
BPE_TOKENS=40000

URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz"
    "http://data.statmt.org/wmt17/translation-task/dev.tgz"
    "http://statmt.org/wmt14/test-full.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-nc-v12.tgz"
    "dev.tgz"
    "test-full.tgz"
)
CORPORA=(
    "training/europarl-v7.de-en"
    "commoncrawl.de-en"
    "training/news-commentary-v12.de-en"
)

# This will make the dataset compatible to the one used in "Convolutional Sequence to Sequence Learning"
# https://arxiv.org/abs/1705.03122
if [ "$1" == "--icml17" ]; then
    URLS[2]="http://statmt.org/wmt14/training-parallel-nc-v9.tgz"
    FILES[2]="training-parallel-nc-v9.tgz"
    CORPORA[2]="training/news-commentary-v9.de-en"
fi

# This will make the dataset comparable to the one used in "Scaling Neural Machine Translation"
# https://arxiv.org/abs/1806.00187
if [ "$1" == "--scaling18" ]; then
    BPE_TOKENS=33708
fi

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=en
tgt=de
lang=en-de
prep=wmt14_en_de
tmp=$prep/tmp
orig=orig
dev=dev/newstest2013

mkdir -p $orig $tmp $prep

cd $orig

for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        fi
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        fi
    fi
done
cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    rm -f $tmp/train.tags.$lang.tok.$l
    for f in "${CORPORA[@]}"; do
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l
    done
done

echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/test-full/newstest2014-deen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.$l
    echo ""
done

if [ "$1" == "--scaling18" ]; then
    # apply length filtering before BPE for --scaling18
    perl $CLEAN $tmp/train.tags.$lang.tok $src $tgt $tmp/train 1 256

    # use newstest2013 for valid
    echo "pre-processing valid data..."
    for l in $src $tgt; do
        rm -f $tmp/valid.$l
        cat $orig/$dev.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | tee -a $tmp/valid.raw.$l | \
            perl $TOKENIZER -threads 8 -a -l $l >> $tmp/valid.$l
    done
else
    echo "splitting train and valid..."
    for l in $src $tgt; do
        awk '{if (NR%100 == 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/valid.$l
        awk '{if (NR%100 != 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/train.$l
    done
fi

TRAIN=$tmp/train.de-en
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
    done
done

if [ "$1" == "--scaling18" ]; then
    for L in $src $tgt; do
        cp $tmp/bpe.train.$L $prep/train.$L
        cp $tmp/bpe.valid.$L $prep/valid.$L
    done
else
    perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
    perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250
fi

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $prep/test.$L
done

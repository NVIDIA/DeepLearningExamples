# BSD 3-Clause License
# 
# Copyright (c) 2017, 
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

echo "=== Acquiring datasets ==="
echo "---"

mkdir -p data
cd data

if [[ ! -d 'wikitext-2' ]]; then
    echo "- Downloading WikiText-2 (WT2)"
    wget --quiet --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
    unzip -q wikitext-2-v1.zip
    cd wikitext-2
    mv wiki.train.tokens train.txt
    mv wiki.valid.tokens valid.txt
    mv wiki.test.tokens test.txt
    cd ..
fi

echo "- Downloading WikiText-103 (WT2)"
if [[ ! -d 'wikitext-103' ]]; then
    wget --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
    unzip -q wikitext-103-v1.zip
    cd wikitext-103
    mv wiki.train.tokens train.txt
    mv wiki.valid.tokens valid.txt
    mv wiki.test.tokens test.txt
    cd ..
fi

echo "- Downloading enwik8 (Character)"
if [[ ! -d 'enwik8' ]]; then
    mkdir -p enwik8
    cd enwik8
    wget --continue http://mattmahoney.net/dc/enwik8.zip
    wget https://raw.githubusercontent.com/salesforce/awd-lstm-lm/master/data/enwik8/prep_enwik8.py
    python3 prep_enwik8.py
    cd ..
fi

echo "- Downloading text8 (Character)"
if [[ ! -d 'text8' ]]; then
    mkdir -p text8
    cd text8
    wget --continue http://mattmahoney.net/dc/text8.zip
    python ../../prep_text8.py
    cd ..
fi

echo "- Downloading Penn Treebank (PTB)"
if [[ ! -d 'penn' ]]; then
    wget --quiet --continue http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
    tar -xzf simple-examples.tgz

    mkdir -p penn
    cd penn
    mv ../simple-examples/data/ptb.train.txt train.txt
    mv ../simple-examples/data/ptb.test.txt test.txt
    mv ../simple-examples/data/ptb.valid.txt valid.txt
    cd ..

    echo "- Downloading Penn Treebank (Character)"
    mkdir -p pennchar
    cd pennchar
    mv ../simple-examples/data/ptb.char.train.txt train.txt
    mv ../simple-examples/data/ptb.char.test.txt test.txt
    mv ../simple-examples/data/ptb.char.valid.txt valid.txt
    cd ..

    rm -rf simple-examples/
fi

echo "- Downloading 1B words"

if [[ ! -d 'one-billion-words' ]]; then
    mkdir -p one-billion-words
    cd one-billion-words

    wget --no-proxy http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
    tar xzvf 1-billion-word-language-modeling-benchmark-r13output.tar.gz

    path="1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/"
    cat ${path}/news.en.heldout-00000-of-00050 > valid.txt
    cat ${path}/news.en.heldout-00000-of-00050 > test.txt

    wget https://github.com/rafaljozefowicz/lm/raw/master/1b_word_vocab.txt

    cd ..
fi

echo "---"
echo "Happy language modeling :)"

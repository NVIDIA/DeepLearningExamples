#!/bin/bash

model_path="/data/models/LibriSpeech"
librispeech_path="/data/datasets/LibriSpeech/test_clean"
result_path="/data/results"

# Correctness

cat $model_path/words.txt | tr '[:upper:]' '[:lower:]' > $result_path/words.txt
cat $librispeech_path/$test_set/text | tr '[:upper:]' '[:lower:]' > $result_path/text
oovtok=$(cat $result_path/words.txt | grep "<unk>" | awk '{print $2}')
/opt/kaldi/egs/wsj/s5/utils/sym2int.pl --map-oov $oovtok -f 2- $result_path/words.txt $result_path/text > $result_path/text_ints 2> /dev/null


# convert lattice to transcript
/opt/kaldi/src/latbin/lattice-best-path \
	"ark:gunzip -c $result_path/lat.cuda-asr.gz |"\
	"ark,t:$result_path/trans.cuda-asr" 2> /dev/null

# calculate wer
/opt/kaldi/src/bin/compute-wer --mode=present \
	"ark:$result_path/text_ints" \
	"ark:$result_path/trans.cuda-asr" 2> /dev/null


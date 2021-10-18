#!/bin/bash
set -e 
results_dir=/data/results

if [ -d "$results_dir" ]
then
	rm -rf $results_dir
fi	
mkdir $results_dir
kaldi-asr-parallel-client $@
echo "Computing WER..."
/workspace/scripts/compute_wer.sh
rm -rf $results_dir

#!/usr/bin/env bash

mkdir -p $HOME/DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/SLRDATA
mkdir -p $HOME/DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/SLRDATA/mel
gcsfuse research-datasets/bn_slr/SLR37/bn_bd $HOME/DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/SLRDATA
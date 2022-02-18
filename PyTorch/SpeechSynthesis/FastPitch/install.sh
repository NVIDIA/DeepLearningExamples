#!/usr/bin/env bash

## install miniconda:
while getopts "cu:" opt; do
      case $opt in
        c ) CONDA="true";;
        u ) USER="$OPTARG";;
        \?) echo "Invalid option: -"$OPTARG"" >&2
            exit 1;;
      esac
    done
: ${CONDA-"false"}  # default value
: ${USER-`woami`}  # default value

if [ "$CONDA" = "true" ]
then
  cd /disk/scratch1/${USER}/
  wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh
  echo "!! Change install location to /disk/scratch1/${USER}/miniconda3 !!"
  bash ./Miniconda3-py39_4.9.2-Linux-x86_64.sh
  rm Miniconda3-py39_4.9.2-Linux-x86_64.sh*
  source ~/.bashrc
fi

source /disk/scratch1/${USER}/miniconda3/bin/activate

SERVERNAME=`hostname -s`
conda create -n fastpitch_${SERVERNAME} python=3.8
source activate fastpitch_${SERVERNAME}

## Get a version of gcc > 5.0. The current anaconda default (June 2021) is 9.3 which seems to work (so far!)
conda install gcc_linux-64 gxx_linux-64
## Make these the default C and C++ compilers (you could also set CC and CXX environment variabes)
## But aliasing seems to work well enough
alias gcc=x86_64-conda_cos6-linux-gnu-cc
alias g++=x86_64-conda_cos6-linux-gnu-c++

export CUDA_HOME=/opt/cuda-10.2.89_440_33
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

conda uninstall pytorch

## Then we reinstall and this for some reason downgrades the gcc to 7 and then installing apex works/
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

## Apex
cd /disk/scratch1/${USER}/FastPitches/PyTorch/SpeechSynthesis/FastPitch/
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../

## Python requirements
pip install -r requirements.txt
pip install tqdm tensorboard
pip install librosa
pip install wandb
pip install llvmlite==0.35.0
## Ignore warning around here
pip install numba==0.49.1

## for logging
## if needed, create a free account here: https://app.wandb.ai/login?signup=true
wandb login

export CUDA_VISIBLE_DEVICES=1


## Test installation
./scripts/download_fastpitch.sh
./scripts/download_waveglow.sh
mkdir output
python inference.py --cuda   --fastpitch pretrained_models/fastpitch/nvidia_fastpitch_210824.pt   --waveglow pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt   --wn-channels 256   -i phrases/devset10.tsv   -o output/wavs_devset10


## Get set up with LJ
./scripts/download_dataset.sh
./scripts/prepare_dataset.sh

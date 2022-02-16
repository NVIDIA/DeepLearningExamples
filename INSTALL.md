## Installation

Head to server

```bash
ssh dudley
```

Go to your own directory or make a new one in scratch

```bash
cd /disk/scratch/s2132904
```

Install miniconda

Activate conda

```bash
source miniconda_22/bin/activate
```

5. Clone repo

git clone https://github.com/NVIDIA/DeepLearningExamples.git

6. Create conda environement

conda create -n fastpitch_dudley python=3.8 
source activate fastpitch_dudley

```bash
## Get a version of gcc > 5.0. The current anaconda default (June 2021) is 9.3 which seems to work (so far!)
conda install gcc_linux-64 gxx_linux-64
## make these the default C and C++ compilers (you could also set CC and CXX environment variabes)
## but aliasing seems to work well enough
alias gcc=x86_64-conda_cos6-linux-gnu-cc
alias g++=x86_64-conda_cos6-linux-gnu-c++
```

```bash
export CUDA_HOME=/opt/cuda-10.2.89_440_33
```
```bash
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch 
```
So something is wrong with having gcc9, so we uninstall pytorch

```bash
conda uninstall pytorch

Then we reinstall and this for some reason downgrades the gcc to 7 and then installing apex works/
```bash
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch 
```


```bash
git clone https://github.com/NVIDIA/apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```


```bash
pip install -r requirements.txt
pip install tqdm tensorboard 
pip install librosa 
pip install llvmlite==0.35.0 
pip install numba==0.49.1
```

```bash
export CUDA_VISIBLE_DEVICES=1
python inference.py --cuda   --fastpitch pretrained_models/fastpitch/nvidia_fastpitch_210824.pt   --waveglow pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt   --wn-channels 256   -i phrases/devset10.tsv   -o output/wavs_devset10
```


```bash
bash scripts/download_dataset.sh
bash scripts/prepare_dataset.sh
```

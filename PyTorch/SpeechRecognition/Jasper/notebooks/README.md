## Overview

This notebook provides scripts for you to run Jasper with TRT for inference step by step. You can run inference using either LibriSpeech dataset or your own audio input in .wav format, to generate the corresponding text file for the audio file. 

## Requirements

This repository contains a Dockerfile which extends the PyTorch 19.09-py3 NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:

* [NVIDIA Turing](https://www.nvidia.com/en-us/geforce/turing/) or [Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) based GPU    
* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch 19.09-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
* [Pretrained Jasper Model Checkpoint](https://ngc.nvidia.com/catalog/models/nvidia:jasperpyt_fp16)

## Quick Start Guide

Running the following scripts will build and launch the container containing all required dependencies for both TensorRT as well as native PyTorch. This is necessary for using inference with TensorRT and can also be used for data download, processing and training of the model.

1. Clone the repository.

```bash
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/PyTorch/SpeechRecognition/Jasper
```
2. Build the Jasper PyTorch with TRT 6 container:

```bash
bash trt/scripts/docker/trt_build.sh
```
3. Prepare to start a detached session in the NGC container
Create three directories on your local machine for dataset, checkpoint, and result, respectively, naming "data" "checkpoint" "result":

```bash
mkdir data checkpoint result
```
Download the checkpoint file `jasperpyt_fp16` to the directory `checkpoint` from NGC Model Repository: https://ngc.nvidia.com/catalog/models/nvidia:jasperpyt_fp16

Assume you will download the dataset to /dev/sdb and mount the data on /dev/sdb to "data", please replace "/dev/sdb" with your own directories if you use other directories:

```bash
sudo mount /dev/sdb data
```

The Jasper PyTorch container will be launched in the Jupyter notebook. Within the container, the contents of the root repository will be copied to the /workspace/jasper directory. The /datasets, /checkpoints, /results directories are mounted as volumes and mapped to the corresponding directories "data" "checkpoint" "result" on the host.

Copy the JasperTRT.ipynb notebook to the root directory of Jasper:

```bash
cp notebooks/JasperTRT.ipynb .
```

For running the notebook on your local machine, run:

```bash
jupyter notebook JasperTRT.ipynb
```
For running the notebook on another machine remotely, run: 

```bash
jupyter notebook --ip=0.0.0.0 --allow-root
```
And navigate a web browser to the IP address or hostname of the host machine at port 8888: http://[host machine]:8888

Use the token listed in the output from running the jupyter command to log in, for example: http://[host machine]:8888/?token=aae96ae9387cd28151868fee318c3b3581a2d794f3b25c6b




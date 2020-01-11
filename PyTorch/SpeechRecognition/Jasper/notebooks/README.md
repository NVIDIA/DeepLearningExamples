# Jasper notebooks

This folder provides different notebooks to run Jasper inference step by step. 

## Table Of Contents

- [Jasper Jupyter Notebook for TensorRT](#jasper-jupyter-notebook-for-tensorrt)
   * [Requirements](#requirements)
    * [Quick Start Guide](#quick-start-guide)
- [Jasper Colab Notebook for TensorRT](#jasper-colab-notebook-for-tensorrt)
   * [Requirements](#requirements)
    * [Quick Start Guide](#quick-start-guide)
- [Jasper Jupyter Notebook for TensorRT Inference Server](#jasper-colab-notebook-for-tensorrt-inference-server)
   * [Requirements](#requirements)
    * [Quick Start Guide](#quick-start-guide)

## Jasper Jupyter Notebook for TensorRT
### Requirements

`./trt/` contains a Dockerfile which extends the PyTorch 19.09-py3 NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:

* [NVIDIA Turing](https://www.nvidia.com/en-us/geforce/turing/) or [Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) based GPU    
* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch 19.09-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
* [NVIDIA machine learning repository](https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb) and [NVIDIA cuda repository](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb) for NVIDIA TensorRT 6
* [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) based GPU
* [Pretrained Jasper Model Checkpoint](https://ngc.nvidia.com/catalog/models/nvidia:jasperpyt_fp16)

### Quick Start Guide

Running the following scripts will build and launch the container containing all required dependencies for both TensorRT as well as native PyTorch. This is necessary for using inference with TensorRT and can also be used for data download, processing and training of the model.

#### 1. Clone the repository.

```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/PyTorch/SpeechRecognition/Jasper
```

#### 2. Build the Jasper PyTorch with TRT 6 container:

```
bash trt/scripts/docker/build.sh
```

#### 3. Create directories
Prepare to start a detached session in the NGC container.
Create three directories on your local machine for dataset, checkpoint, and result, respectively, naming "data" "checkpoint" "result":

```
mkdir data checkpoint result
```

#### 4. Download the checkpoint
Download the checkpoint file jasperpyt_fp16 from NGC Model Repository:  
- https://ngc.nvidia.com/catalog/models/nvidia:jasperpyt_fp16

to the directory: _checkpoint_

The Jasper PyTorch container will be launched in the Jupyter notebook. Within the container, the contents of the root repository will be copied to the /workspace/jasper directory.

The /datasets, /checkpoints, /results directories are mounted as volumes and mapped to the corresponding directories "data" "checkpoint" "result" on the host.

#### 5. Run the notebook

For running the notebook on your local machine, run:

```
jupyter notebook -- notebooks/JasperTRT.ipynb
```

For running the notebook on another machine remotely, run:

```
jupyter notebook --ip=0.0.0.0 --allow-root
```

And navigate a web browser to the IP address or hostname of the host machine at port 8888: `http://[host machine]:8888`

Use the token listed in the output from running the jupyter command to log in, for example: `http://[host machine]:8888/?token=aae96ae9387cd28151868fee318c3b3581a2d794f3b25c6b`



## Jasper Colab Notebook for TensorRT
### Requirements

`./trt/` contains a Dockerfile which extends the PyTorch 19.09-py3 NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:

* [NVIDIA Turing](https://www.nvidia.com/en-us/geforce/turing/) or [Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) based GPU    
* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch 19.09-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
* [NVIDIA machine learning repository](https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb) and [NVIDIA cuda repository](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb) for NVIDIA TensorRT 6
* [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) based GPU
* [Pretrained Jasper Model Checkpoint](https://ngc.nvidia.com/catalog/models/nvidia:jasperpyt_fp16)

### Quick Start Guide

Running the following scripts will build and launch the container containing all required dependencies for both TensorRT as well as native PyTorch. This is necessary for using inference with TensorRT and can also be used for data download, processing and training of the model.

#### 1. Clone the repository.

```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/PyTorch/SpeechRecognition/Jasper
```

#### 2. Build the Jasper PyTorch with TRT 6 container:

```
bash trt/scripts/docker/build.sh
```

#### 3. Create directories
Prepare to start a detached session in the NGC container.
Create three directories on your local machine for dataset, checkpoint, and result, respectively, naming "data" "checkpoint" "result":

```
mkdir data checkpoint result
```

#### 4. Download the checkpoint
Download the checkpoint file jasperpyt_fp16 from NGC Model Repository:  
- https://ngc.nvidia.com/catalog/models/nvidia:jasperpyt_fp16

to the directory: _checkpoint_

The Jasper PyTorch container will be launched in the Jupyter notebook. Within the container, the contents of the root repository will be copied to the /workspace/jasper directory.

The /datasets, /checkpoints, /results directories are mounted as volumes and mapped to the corresponding directories "data" "checkpoint" "result" on the host.

#### 5. Run the notebook

>>>>>>> 2deaddbc2ea58d5318b06203ae30ace2dd576ecb
For running the notebook on your local machine, run:

```
jupyter notebook -- notebooks/Colab_Jasper_TRT_inference_demo.ipynb
```

For running the notebook on another machine remotely, run:

```
jupyter notebook --ip=0.0.0.0 --allow-root
```

And navigate a web browser to the IP address or hostname of the host machine at port 8888: `http://[host machine]:8888`

Use the token listed in the output from running the jupyter command to log in, for example: `http://[host machine]:8888/?token=aae96ae9387cd28151868fee318c3b3581a2d794f3b25c6b`



## Jasper Jupyter Notebook for TensorRT Inference Server
### Requirements

`./trtis/` contains a Dockerfile which extends the PyTorch 19.09-py3 NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:

* [NVIDIA Turing](https://www.nvidia.com/en-us/geforce/turing/) or [Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) based GPU    
* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch 19.09-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
* [TensorRT Inference Server 19.09 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrtserver)
* [NVIDIA machine learning repository](https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb) and [NVIDIA cuda repository](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb) for NVIDIA TensorRT 6
* [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) based GPU
* [Pretrained Jasper Model Checkpoint](https://ngc.nvidia.com/catalog/models/nvidia:jasperpyt_fp16)

### Quick Start Guide


#### 1. Clone the repository.

```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/PyTorch/SpeechRecognition/Jasper
```

#### 2. Build a container that extends NGC PyTorch 19.09, TensorRT, TensorRT Inference Server, and TensorRT Inference Client.

```
bash trtis/scripts/docker/build.sh
```

#### 3. Download the checkpoint
Download the checkpoint file jasper_fp16.pt from NGC Model Repository:  
- https://ngc.nvidia.com/catalog/models/nvidia:jasperpyt_fp16

to an user specified directory _CHECKPOINT_DIR_

#### 4. Run the notebook

For running the notebook on your local machine, run:

```
jupyter notebook -- notebooks/JasperTRTIS.ipynb
```

For running the notebook on another machine remotely, run:

```
jupyter notebook --ip=0.0.0.0 --allow-root
```

And navigate a web browser to the IP address or hostname of the host machine at port 8888: `http://[host machine]:8888`

Use the token listed in the output from running the jupyter command to log in, for example: `http://[host machine]:8888/?token=aae96ae9387cd28151868fee318c3b3581a2d794f3b25c6b`

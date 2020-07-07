# Jasper For PyTorch

This repository provides scripts to train the Jasper model to achieve near state of the art accuracy and perform high-performance inference using NVIDIA TensorRT. This repository is tested and maintained by NVIDIA.

## Table Of Contents
- [Model overview](#model-overview)
   * [Model architecture](#model-architecture)
   * [Default configuration](#default-configuration)
   * [Feature support matrix](#feature-support-matrix)
       * [Features](#features)
   * [Mixed precision training](#mixed-precision-training)
       * [Enabling mixed precision](#enabling-mixed-precision)
       * [Enabling TF32](#enabling-tf32)
   * [Glossary](#glossary)
- [Setup](#setup)
   * [Requirements](#requirements)
- [Quick Start Guide](#quick-start-guide)
- [Advanced](#advanced)
   * [Scripts and sample code](#scripts-and-sample-code)
   * [Parameters](#parameters)
   * [Command-line options](#command-line-options)
   * [Getting the data](#getting-the-data)
       * [Dataset guidelines](#dataset-guidelines)
   * [Training process](#training-process)
   * [Inference process](#inference-process)
   * [Evaluation process](#evaluation-process)
   * [Deploying Jasper using TensorRT](#deploying-jasper-using-tensorrt)
   * [Deploying Jasper using Triton Inference Server](#deploying-jasper-using-triton-inference)
- [Performance](#performance)
   * [Benchmarking](#benchmarking)
       * [Training performance benchmark](#training-performance-benchmark)
       * [Inference performance benchmark](#inference-performance-benchmark)
   * [Results](#results)
       * [Training accuracy results](#training-accuracy-results)
           * [Training accuracy: NVIDIA DGX A100 (8x A100 40GB)](#training-accuracy-nvidia-dgx-a100-8x-a100-40gb)
           * [Training accuracy: NVIDIA DGX-1 (8x V100 32GB)](#training-accuracy-nvidia-dgx-1-8x-v100-32gb)
           * [Training stability test](#training-stability-test)
       * [Training performance results](#training-performance-results)
         * [Training performance: NVIDIA DGX A100 (8x A100 40GB)](#training-performance-nvidia-dgx-a100-8x-a100-40gb)
         * [Training performance: NVIDIA DGX-1 (8x V100 16GB)](#training-performance-nvidia-dgx-1-8x-v100-16gb)
         * [Training performance: NVIDIA DGX-1 (8x V100 32GB)](#training-performance-nvidia-dgx-1-8x-v100-32gb)
         * [Training performance: NVIDIA DGX-2 (16x V100 32GB)](#training-performance-nvidia-dgx-2-16x-v100-32gb)
       * [Inference performance results](#inference-performance-results)
           * [Inference performance: NVIDIA DGX A100 (1x A100 40GB)](#inference-performance-nvidia-dgx-a100-gpu-1x-a100-40gb)
           * [Inference performance: NVIDIA DGX-1 (1x V100 16GB)](#inference-performance-nvidia-dgx-1-1x-v100-16gb)
           * [Inference performance: NVIDIA DGX-1 (1x V100 32GB)](#inference-performance-nvidia-dgx-1-1x-v100-32gb)
           * [Inference performance: NVIDIA DGX-2 (1x V100 32GB)](#inference-performance-nvidia-dgx-2-1x-v100-32gb)
           * [Inference performance: NVIDIA T4](#inference-performance-nvidia-t4)
- [Release notes](#release-notes)
   * [Changelog](#changelog)
   * [Known issues](#known-issues)

## Model overview
This repository provides an implementation of the Jasper model in PyTorch from the paper `Jasper: An End-to-End Convolutional Neural Acoustic Model` [https://arxiv.org/pdf/1904.03288.pdf](https://arxiv.org/pdf/1904.03288.pdf).
The Jasper model is an end-to-end neural acoustic model for automatic speech recognition (ASR) that provides near state-of-the-art results on LibriSpeech among end-to-end ASR models without any external data. The Jasper architecture of convolutional layers was designed to facilitate fast GPU inference, by allowing whole sub-blocks to be fused into a single GPU kernel. This is important for meeting strict real-time requirements of ASR systems in deployment.

The results of the acoustic model are combined with the results of external language models to get the top-ranked word sequences
corresponding to a given audio segment. This post-processing step is called decoding.

This repository is a PyTorch implementation of Jasper and provides scripts to train the Jasper 10x5 model with dense residuals from scratch on the [Librispeech](http://www.openslr.org/12) dataset to achieve the greedy decoding results of the original paper.
The original reference code provides Jasper as part of a research toolkit in TensorFlow [openseq2seq](https://github.com/NVIDIA/OpenSeq2Seq).
This repository provides a simple implementation of Jasper with scripts for training and replicating the Jasper paper results.
This includes data preparation scripts, training and inference scripts.
Both training and inference scripts offer the option to use Automatic Mixed Precision (AMP) to benefit from Tensor Cores for better performance.

In addition to providing the hyperparameters for training a model checkpoint, we publish a thorough inference analysis across different NVIDIA GPU platforms, for example, DGX A100, DGX-1, DGX-2 and T4.

This model is trained with mixed precision using Tensor Cores on Volta, Turing, and the NVIDIA Ampere GPU architectures. Therefore, researchers can get results 3x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.

The original paper takes the output of the Jasper acoustic model and shows results for 3 different decoding variations: greedy decoding, beam search with a 6-gram language model and beam search with further rescoring of the best ranked hypotheses with Transformer XL, which is a neural language model. Beam search and the rescoring with the neural language model scores are run on CPU and result in better word error rates compared to greedy decoding.
This repository provides instructions to reproduce greedy decoding results. To run beam search or rescoring with TransformerXL, use the following scripts from the [openseq2seq](https://github.com/NVIDIA/OpenSeq2Seq) repository:
https://github.com/NVIDIA/OpenSeq2Seq/blob/master/scripts/decode.py
https://github.com/NVIDIA/OpenSeq2Seq/tree/master/external_lm_rescore

### Model architecture
Details on the model architecture can be found in the paper [Jasper: An End-to-End Convolutional Neural Acoustic Model](https://arxiv.org/pdf/1904.03288.pdf).

|<img src="images/jasper_model.png" width="100%" height="40%"> | <img src="images/jasper_dense_residual.png" width="100%" height="40%">|
|:---:|:---:|
|Figure 1: Jasper BxR model: B- number of blocks, R- number of sub-blocks | Figure 2: Jasper Dense Residual |

Jasper is an end-to-end neural acoustic model that is based on convolutions.
In the audio processing stage, each frame is transformed into mel-scale spectrogram features, which the acoustic model takes as input and outputs a probability distribution over the vocabulary for each frame.
The acoustic model has a modular block structure and can be parametrized accordingly:
a Jasper BxR model has B blocks, each consisting of R repeating sub-blocks.

Each sub-block applies the following operations in sequence: 1D-Convolution, Batch Normalization, ReLU activation, and Dropout.

Each block input is connected directly to the last subblock of all following blocks via a residual connection, which is referred to as `dense residual` in the paper.
Every block differs in kernel size and number of filters, which are increasing in size from the bottom to the top layers.
Irrespective of the exact block configuration parameters B and R, every Jasper model has four additional convolutional blocks:
one immediately succeeding the input layer (Prologue) and three at the end of the B blocks (Epilogue).

The Prologue is to decimate the audio signal
in time in order to process a shorter time sequence for efficiency. The Epilogue with dilation captures a bigger context around an audio time step, which decreases the model word error rate (WER).
The paper achieves best results with Jasper 10x5 with dense residual connections, which is also the focus of this repository and is in the following referred to as Jasper Large.

### Default configuration
The following features were implemented in this model:

* GPU-supported feature extraction with data augmentation options [SpecAugment](https://arxiv.org/abs/1904.08779) and [Cutout](https://arxiv.org/pdf/1708.04552.pdf)
* offline and online [Speed Perturbation](https://www.danielpovey.com/files/2015_interspeech_augmentation.pdf)
* data-parallel multi-GPU training and evaluation
* AMP with dynamic loss scaling for Tensor Core training
* FP16 inference

Competitive training results and analysis is provided for the following Jasper model configuration

|    **Model** | **Number of Blocks** | **Number of Subblocks** | **Max sequence length** | **Number of Parameters** |
|--------------|----------------------|-------------------------|-------------------------|--------------------------|
| Jasper Large |                   10 |                       5 |                  16.7 s |                    333 M |


### Feature support matrix
The following features are supported by this model.

| **Feature**   | **Jasper**    |
|---------------|---------------|
|[Apex AMP](https://nvidia.github.io/apex/amp.html) | Yes |
|[Apex DistributedDataParallel](https://nvidia.github.io/apex/parallel.html#apex.parallel.DistributedDataParallel) | Yes |

#### Features
[Apex AMP](https://nvidia.github.io/apex/amp.html) - a tool that enables Tensor Core-accelerated training. Refer to the [Enabling mixed precision](#enabling-mixed-precision) section for more details.

[Apex
DistributedDataParallel](https://nvidia.github.io/apex/parallel.html#apex.parallel.DistributedDataParallel) -
a module wrapper that enables easy multiprocess distributed data parallel
training, similar to
[torch.nn.parallel.DistributedDataParallel](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel).
`DistributedDataParallel` is optimized for use with
[NCCL](https://github.com/NVIDIA/nccl). It achieves high performance by
overlapping communication with computation during `backward()` and bucketing
smaller gradient transfers to reduce the total number of transfers required.


### Mixed precision training
*Mixed precision* is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format, while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in Volta, and following with both the Turing and Ampere architectures, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using mixed precision training requires two steps:

1. Porting the model to use the FP16 data type where appropriate.
2. Adding loss scaling to preserve small gradient values.

The ability to train deep learning networks with lower precision was introduced in the Pascal architecture and first supported in [CUDA 8](https://devblogs.nvidia.com/parallelforall/tag/fp16/) in the NVIDIA Deep Learning SDK.

For information about:
* How to train using mixed precision, see the[Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html) documentation.
* Techniques used for mixed precision training, see the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
* APEX tools for mixed precision training, see the [NVIDIA Apex: Tools for Easy Mixed-Precision Training in PyTorch](https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/).


#### Enabling mixed precision
For training, mixed precision can be enabled by setting the flag: `train.py --amp`. When using bash helper scripts:  `scripts/train.sh` `scripts/inference.sh`, etc., mixed precision can be enabled with env variable `AMP=true`.

Mixed precision is enabled in PyTorch by using the Automatic Mixed Precision
(AMP) library from [APEX](https://github.com/NVIDIA/apex) that casts variables
to half-precision upon retrieval, while storing variables in single-precision
format. Furthermore, to preserve small gradient magnitudes in backpropagation,
a [loss
scaling](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#lossscaling)
step must be included when applying gradients. In PyTorch, loss scaling can be
easily applied by using `scale_loss()` method provided by AMP. The scaling
value to be used can be
[dynamic](https://nvidia.github.io/apex/amp.html#apex.amp.initialize) or fixed.

For an in-depth walk through on AMP, check out sample usage
[here](https://nvidia.github.io/apex/amp.html#). [APEX](https://github.com/NVIDIA/apex) is a PyTorch extension that contains
utility libraries, such as AMP, which require minimal network code changes to
leverage Tensor Cores performance.

The following steps were needed to enable mixed precision training in Jasper:

* Import AMP from APEX (file: `train.py`):
```bash
from apex import amp
```

* Initialize AMP and wrap the model and the optimizer
```bash
   model, optimizer = amp.initialize(
     min_loss_scale=1.0,
     models=model,
     optimizers=optimizer,
     opt_level=’O1’)

```

* Apply `scale_loss` context manager
```bash
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
```

#### Enabling TF32
TensorFloat-32 (TF32) is the new math mode in [NVIDIA A100](#https://www.nvidia.com/en-us/data-center/a100/) GPUs for handling the matrix math also called tensor operations. TF32 running on Tensor Cores in A100 GPUs can provide up to 10x speedups compared to single-precision floating-point math (FP32) on Volta GPUs. 

TF32 Tensor Cores can speed up networks using FP32, typically with no loss of accuracy. It is more robust than FP16 for models which require high dynamic range for weights or activations.

For more information, refer to the [TensorFloat-32 in the A100 GPU Accelerates AI Training, HPC up to 20x](#https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) blog post.

TF32 is supported in the NVIDIA Ampere GPU architecture and is enabled by default.


### Glossary
**Acoustic model**
Assigns a probability distribution over a vocabulary of characters given an audio frame.

**Language Model**
Assigns a probability distribution over a sequence of words. Given a sequence of words, it assigns a probability to the whole sequence.

**Pre-training**
Training a model on vast amounts of data on the same (or different) task to build general understandings.

**Automatic Speech Recognition (ASR)**
Uses both acoustic model and language model to output the transcript of an input audio signal.


## Setup
The following section lists the requirements in order to start training and evaluating the Jasper model.

### Requirements
This repository contains a `Dockerfile` which extends the PyTorch 20.06-py3 NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:

* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch 20.06-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
-   Supported GPUs:
    - [NVIDIA Volta architecture](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
    - [NVIDIA Turing architecture](https://www.nvidia.com/en-us/geforce/turing/)
    - [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)

Further required python packages are listed in `requirements.txt`, which are automatically installed with the Docker container built. To manually install them, run
```bash
pip install -r requirements.txt
```

For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:

* [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
* [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/dgx/user-guide/index.html#accessing_registry)
* [Running PyTorch](https://docs.nvidia.com/deeplearning/dgx/pytorch-release-notes/running.html#running)

For those unable to use the PyTorch NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/dgx/support-matrix/index.html).


## Quick Start Guide

To train your model using mixed or TF32 precision with Tensor Cores or using FP32, perform the following steps using the default parameters of the Jasper model on the Librispeech dataset. For details concerning training and inference, see [Advanced](#Advanced) section.

1. Clone the repository.
```bash
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/PyTorch/SpeechRecognition/Jasper
```
2. Build the Jasper PyTorch container.

Running the following scripts will build and launch the container which contains all the required dependencies for data download and processing as well as training and inference of the model.

```bash
bash scripts/docker/build.sh
```

3. Start an interactive session in the NGC container to run data download/training/inference

```bash
bash scripts/docker/launch.sh <DATA_DIR> <CHECKPOINT_DIR> <RESULT_DIR>
```
Within the container, the contents of this repository will be copied to the `/workspace/jasper` directory. The `/datasets`, `/checkpoints`, `/results` directories are mounted as volumes
and mapped to the corresponding directories `<DATA_DIR>`, `<CHECKPOINT_DIR>`, `<RESULT_DIR>` on the host.

4. Download and preprocess the dataset.

No GPU is required for data download and preprocessing. Therefore, if GPU usage is a limited resource, launch the container for this section on a CPU machine by following Steps 2 and 3.

Note: Downloading and preprocessing the dataset requires 500GB of free disk space and can take several hours to complete.

This repository provides scripts to download, and extract the following datasets:

* LibriSpeech [http://www.openslr.org/12](http://www.openslr.org/12)

LibriSpeech contains 1000 hours of 16kHz read English speech derived from public domain audiobooks from LibriVox project and has been carefully segmented and aligned. For more information, see the [LIBRISPEECH: AN ASR CORPUS BASED ON PUBLIC DOMAIN AUDIO BOOKS](http://www.danielpovey.com/files/2015_icassp_librispeech.pdf) paper.

Inside the container, download and extract the datasets into the required format for later training and inference:
```bash
bash scripts/download_librispeech.sh
```
Once the data download is complete, the following folders should exist:

* `/datasets/LibriSpeech/`
   * `train-clean-100/`
   * `train-clean-360/`
   * `train-other-500/`
   * `dev-clean/`
   * `dev-other/`
   * `test-clean/`
   * `test-other/`

Since `/datasets/` is mounted to `<DATA_DIR>` on the host (see Step 3),  once the dataset is downloaded it will be accessible from outside of the container at `<DATA_DIR>/LibriSpeech`.


Next, convert the data into WAV files and add speed perturbation with 0.9 and 1.1 to the training files:
```bash
bash scripts/preprocess_librispeech.sh
```
Once the data is converted, the following additional files and folders should exist:
* `datasets/LibriSpeech/`
   * `librispeech-train-clean-100-wav.json`
   * `librispeech-train-clean-360-wav.json`
   * `librispeech-train-other-500-wav.json`
   * `librispeech-dev-clean-wav.json`
   * `librispeech-dev-other-wav.json`
   * `librispeech-test-clean-wav.json`
   * `librispeech-test-other-wav.json`
   * `train-clean-100-wav/` containsWAV files with original speed, 0.9 and 1.1
   * `train-clean-360-wav/` contains WAV files with original speed, 0.9 and 1.1
   * `train-other-500-wav/` contains WAV files with original speed, 0.9 and 1.1
   * `dev-clean-wav/`
   * `dev-other-wav/`
   * `test-clean-wav/`
   * `test-other-wav/`


5. Start training.

Inside the container, use the following script to start training.
Make sure the downloaded and preprocessed dataset is located at `<DATA_DIR>/LibriSpeech` on the host (see Step 3), which corresponds to `/datasets/LibriSpeech` inside the container.

```bash
bash scripts/train.sh [OPTIONS]
```
By default automatic precision is disabled, batch size is 64 over two gradient accumulation steps, and the recipe is run on a total of 8 GPUs. The hyperparameters are tuned for a GPU with at least 32GB of memory and will require adjustment for 16GB GPUs (e.g., by lowering batch size and using more gradient accumulation steps).

More details on available [OPTIONS] can be found in [Parameters](#parameters) and [Training process](#training-process).

6. Start validation/evaluation.

Inside the container, use the following script to run evaluation.
 Make sure the downloaded and preprocessed dataset is located at `<DATA_DIR>/LibriSpeech` on the host (see Step 3), which corresponds to `/datasets/LibriSpeech` inside the container.
```bash
bash scripts/evaluation.sh [OPTIONS]
```
By default, this will use full precision, a batch size of 64 and run on a single GPU.

More details on available [OPTIONS] can be found in [Parameters](#parameters) and [Evaluation process](#evaluation-process).


7. Start inference/predictions.

Inside the container, use the following script to run inference.
 Make sure the downloaded and preprocessed dataset is located at `<DATA_DIR>/LibriSpeech` on the host (see Step 3), which corresponds to `/datasets/LibriSpeech` inside the container.
A pretrained model checkpoint can be downloaded from [NGC model repository](https://ngc.nvidia.com/catalog/models/nvidia:jasperpyt_fp16).

```bash
bash scripts/inference.sh [OPTIONS]
```
By default this will use single precision, a batch size of 64 and run on a single GPU.

More details on available [OPTIONS] can be found in [Parameters](#parameters) and [Inference process](#inference-process).


## Advanced

The following sections provide greater details of the dataset, running training and inference, and getting training and inference results.


### Scripts and sample code
In the `root` directory, the most important files are:
* `train.py` - Serves as entry point for training
* `inference.py` - Serves as entry point for inference and evaluation
* `model.py` - Contains the model architecture
* `dataset.py` - Contains the data loader and related functionality
* `optimizer.py` - Contains the optimizer
* `inference_benchmark.py` - Serves as inference benchmarking script that measures the latency of pre-processing and the acoustic model
* `requirements.py` - Contains the required dependencies that are installed when building the Docker container
* `Dockerfile` - Container with the basic set of dependencies to run Jasper

The `scripts/` folder encapsulates all the one-click scripts required for running various supported functionalities, such as:
* `train.sh` - Runs training using the `train.py` script
* `inference.sh` - Runs inference using the `inference.py` script
* `evaluation.sh` - Runs evaluation using the `inference.py` script
* `download_librispeech.sh` - Downloads LibriSpeech dataset
* `preprocess_librispeech.sh` - Preprocess LibriSpeech raw data files to be ready for training and inference
* `inference_benchmark.sh` - Runs the inference benchmark using the `inference_benchmark.py` script
* `train_benchmark.sh` - Runs the training performance benchmark using the `train.py` script
* `docker/` - Contains the scripts for building and launching the container


Other folders included in the `root` directory are:
* `notebooks/` - Jupyter notebooks and example audio files
* `configs/    - model configurations
* `utils/`     - data downloading and common routines
* `parts/`     - data pre-processing

### Parameters

Parameters could be set as env variables, or passed as positional arguments.

The complete list of available parameters for `scripts/train.sh` script contains:
```bash
 DATA_DIR: directory of dataset. (default: '/datasets/LibriSpeech')
 MODEL_CONFIG: relative path to model configuration. (default: 'configs/jasper10x5dr_sp_offline_specaugment.toml')
 RESULT_DIR: directory for results, logs, and created checkpoints. (default: '/results')
 CHECKPOINT: model checkpoint to continue training from. Model checkpoint is a dictionary object that contains apart from the model weights the optimizer state as well as the epoch number. If CHECKPOINT is set, training starts from scratch. (default: "")
 CREATE_LOGFILE: boolean that indicates whether to create a training log that will be stored in `$RESULT_DIR`. (default: true)
 CUDNN_BENCHMARK: boolean that indicates whether to enable cudnn benchmark mode for using more optimized kernels. (default: true)
 NUM_GPUS: number of GPUs to use. (default: 8)
 AMP: if set to `true`, enables automatic mixed precision (default: false)
 EPOCHS: number of training epochs. (default: 400)
 SEED: seed for random number generator and used for ensuring reproducibility. (default: 6)
 BATCH_SIZE: data batch size. (default: 64)
 LEARNING_RATE: Initial learning rate. (default: 0.015)
 GRADIENT_ACCUMULATION_STEPS: number of gradient accumulation steps until optimizer updates weights. (default: 2)
```

The complete list of available parameters for `scripts/inference.sh` script contains:
```bash
DATA_DIR: directory of dataset. (default: '/datasets/LibriSpeech')
DATASET: name of dataset to use. (default: 'dev-clean')
MODEL_CONFIG: model configuration. (default: 'configs/jasper10x5dr_sp_offline_specaugment.toml')
RESULT_DIR: directory for results and logs. (default: '/results')
CHECKPOINT: model checkpoint path. (required)
CREATE_LOGFILE: boolean that indicates whether to create a log file that will be stored in `$RESULT_DIR`. (default: true)
CUDNN_BENCHMARK: boolean that indicates whether to enable cudnn benchmark mode for using more optimized kernels. (default: false)
AMP: if set to `true`, enables FP16 inference with AMP (default: false)
NUM_STEPS: number of inference steps. If -1 runs inference on entire dataset. (default: -1)
SEED: seed for random number generator and useful for ensuring reproducibility. (default: 6)
BATCH_SIZE: data batch size.(default: 64)
LOGITS_FILE: destination path for serialized model output with binary protocol. If 'none' does not save model output. (default: 'none')
PREDICTION_FILE: destination path for saving predictions. If 'none' does not save predictions. (default: '${RESULT_DIR}/${DATASET}.predictions)
```

The complete list of available parameters for `scripts/evaluation.sh` script contains:
```bash
DATA_DIR: directory of dataset.(default: '/datasets/LibriSpeech')
DATASET: name of dataset to use.(default: 'dev-clean')
MODEL_CONFIG: model configuration.(default: 'configs/jasper10x5dr_sp_offline_specaugment.toml')
RESULT_DIR: directory for results and logs. (default: '/results')
CHECKPOINT: model checkpoint path. (required)
CREATE_LOGFILE: boolean that indicates whether to create a log file that will be stored in `$RESULT_DIR`. (default: true)
CUDNN_BENCHMARK: boolean that indicates whether to enable cudnn benchmark mde for using more optimized kernels. (default: false)
NUM_GPUS: number of GPUs to run evaluation on (default: 1)
AMP: if set to `true`, enables FP16 with AMP (default: false)
NUM_STEPS: number of inference steps per GPU. If -1 runs inference on entire dataset (default: -1)
SEED: seed for random number generator and useful for ensuring reproducibility. (default: 0)
BATCH_SIZE: data batch size.(default: 64)
```

The `scripts/inference_benchmark.sh` script pads all input to the same length and computes the mean, 90%, 95%, 99% percentile of latency for the specified number of inference steps. Latency is measured in millisecond per batch. The `scripts/inference_benchmark.sh`
measures latency for a single GPU and extends  `scripts/inference.sh` by :
```bash
 MAX_DURATION: filters out input audio data that exceeds a maximum number of seconds. This ensures that when all filtered audio samples are padded to maximum length that length will stay under this specified threshold (default: 36)
```

The `scripts/train_benchmark.sh` script pads all input to the same length according to the input argument `MAX_DURATION` and measures average training latency and throughput performance. Latency is measured in seconds per batch, throughput in sequences per second.
The complete list of available parameters for `scripts/train_benchmark.sh` script contains:
```bash
DATA_DIR: directory of dataset.(default: '/datasets/LibriSpeech')
MODEL_CONFIG: model configuration. (default: 'configs/jasper10x5dr_sp_offline_specaugment.toml')
RESULT_DIR: directory for results and logs. (default: '/results')
CREATE_LOGFILE: boolean that indicates whether to create a log file that will be stored in `$RESULT_DIR`. (default: true)
CUDNN_BENCHMARK: boolean that indicates whether to enable cudnn benchmark mode for using more optimized kernels. (default: true)
NUM_GPUS: number of GPUs to use. (default: 8)
AMP: if set to `true`, enables automatic mixed precision with AMP (default: false)
NUM_STEPS: number of training iterations. If -1 runs full training for  400 epochs. (default: -1)
MAX_DURATION: filters out input audio data that exceed a maximum number of seconds. This ensures that when all filtered audio samples are padded to maximum length that length will stay under this specified threshold (default: 16.7)
SEED: seed for random number generator and useful for ensuring reproducibility. (default: 0)
BATCH_SIZE: data batch size.(default: 32)
LEARNING_RATE: Initial learning rate. (default: 0.015)
GRADIENT_ACCUMULATION_STEPS: number of gradient accumulation steps until optimizer updates weights. (default: 1)
PRINT_FREQUENCY: number of iterations after which training progress is printed. (default: 1)
```

### Command-line options
To see the full list of available options and their descriptions, use the `-h` or `--help` command-line option with the Python file, for example:

```bash
python train.py --help
python inference.py --help
```

### Getting the data
The Jasper model was trained on LibriSpeech dataset. We use the concatenation of `train-clean-100`, `train-clean-360` and `train-other-500` for training and `dev-clean` for validation.

This repository contains the `scripts/download_librispeech.sh` and `scripts/preprocess_librispeech.sh` scripts which will automatically download and preprocess the training, test and development datasets. By default, data will be downloaded to the `/datasets/LibriSpeech` directory, a minimum of 500GB free space is required for download and preprocessing, the final preprocessed dataset is 320GB.


#### Dataset guidelines
The `scripts/preprocess_librispeech.sh` script converts the input audio files to WAV format with a sample rate of 16kHz, target transcripts are stripped from whitespace characters, then lower-cased. For `train-clean-100`, `train-clean-360` and `train-other-500` it also creates speed perturbed versions with rates of 0.9 and 1.1 for data augmentation.

After preprocessing, the script creates JSON files with output file paths, sample rate, target transcript and other metadata. These JSON files are used by the training script to identify training and validation datasets.

The Jasper model was tuned on audio signals with a sample rate of 16kHz, if you wish to use a different sampling rate then some hyperparameters might need to be changed - specifically window size and step size.


### Training process

The training is performed using `train.py` script along with parameters defined in  `scripts/train.sh`
The `scripts/train.sh` script runs a job on a single node that trains the Jasper model from scratch using LibriSpeech as training data. To make training more efficient, we discard audio samples longer than 16.7 seconds from the training dataset, the total number of these samples is less than 1%. Such filtering does not degrade accuracy, but it allows us to decrease the number of time steps in a batch, which requires less GPU memory and increases training speed.
Apart from the default arguments as listed in the [Parameters](#parameters) section, by default the training script:

* Runs on 8 GPUs with at least 32GB of memory and training/evaluation batch size 64, split over two gradient accumulation steps
* Uses TF32 precision (A100 GPU) or FP32 (other GPUs)
* Trains on the concatenation of all 3 LibriSpeech training datasets and evaluates on the LibriSpeech dev-clean dataset
* Maintains an exponential moving average of parameters for evaluation
* Has cudnn benchmark enabled
* Runs for 400 epochs
* Uses an initial learning rate of 0.015 and polynomial (quadratic) learning rate decay
* Saves a checkpoint every 10 epochs
* Runs evaluation on the development dataset every 100 iterations and at the end of training
* Prints out training progress every 25 iterations
* Creates a log file with training progress
* Uses offline speed perturbed data
* Uses SpecAugment in data pre-processing
* Filters out audio samples longer than 16.7 seconds
* Pads each sequence in a batch to the same length (smallest multiple of 16 that is at least the length of the longest sequence in the batch)
* Uses masked convolutions and dense residuals as described in the paper
* Uses weight decay of 0.001
* Uses [Novograd](https://arxiv.org/pdf/1905.11286.pdf) as optimizer with betas=(0.95, 0)

Enabling AMP permits batch size 64 with one gradient accumulation step. Such setup will match the greedy WER [Results](#results) of the Jasper paper on a DGX-1 with 32GB V100 GPUs.

### Inference process
Inference is performed using the `inference.py` script along with parameters defined in `scripts/inference.sh`.
The `scripts/inference.sh` script runs the job on a single GPU, taking a pre-trained Jasper model checkpoint and running it on the specified dataset.
Apart from the default arguments as listed in the [Parameters](#parameters) section by default the inference script:

* Evaluates on the LibriSpeech dev-clean dataset
* Uses a batch size of 64
* Runs for 1 epoch and prints out the final word error rate
* Creates a log file with progress and results which will be stored in the results folder
* Pads each sequence in a batch to the same length (smallest multiple of 16 that is at least the length of the longest sequence in the batch
* Does not use data augmentation
* Does greedy decoding and saves the transcription in the results folder
* Has the option to save the model output tensors for more complex decoding, for example, beam search
* Has cudnn benchmark disabled

### Evaluation process
Evaluation is performed using the `inference.py` script along with parameters defined in `scripts/evaluation.sh`.
The `scripts/evaluation.sh` script runs a job on a single GPU, taking a pre-trained Jasper model checkpoint and running it on the specified dataset.
Apart from the default arguments as listed in the [Parameters](#parameters) section, by default the evaluation script:

* Uses a batch size of 64
* Evaluates the LibriSpeech dev-clean dataset
* Runs for 1 epoch and prints out the final word error rate
* Creates a log file with progress and results which is saved in the results folder
* Pads each sequence in a batch to the same length (smallest multiple of 16 that is at least the length of the longest sequence in the batch)
* Does not use data augmentation
* Has cudnn benchmark disabled

### Deploying Jasper using TensorRT
NVIDIA TensorRT is a platform for high-performance deep learning inference. It includes a deep learning inference optimizer and runtime that delivers low latency and high-throughput for deep learning inference applications. Jasper’s architecture, which is of deep convolutional nature, is designed to facilitate fast GPU inference. After optimizing the compute-intensive acoustic model with NVIDIA TensorRT, inference throughput increased by up to 1.8x over native PyTorch. 
More information on how to perform inference using TensorRT and speed up comparison between TensorRT and native PyTorch can be found in the subfolder [./trt/README.md](trt/README.md)

### Deploying Jasper using Triton Inference Server
The NVIDIA Triton Inference Server provides a datacenter and cloud inferencing solution optimized for NVIDIA GPUs. The server provides an inference service via an HTTP or gRPC endpoint, allowing remote clients to request inferencing for any number of GPU or CPU models being managed by the server.
More information on how to perform inference using TensorRT Inference Server with different model backends can be found in the subfolder [./trtis/README.md](trtis/README.md)


## Performance

### Benchmarking
The following section shows how to run benchmarks measuring the model performance in training and inference modes.

#### Training performance benchmark
To benchmark the training performance on a specific batch size and audio length, for `NUM_STEPS` run:

```bash
export NUM_STEPS=<NUM_STEPS>
export MAX_DURATION=<DURATION>
export BATCH_SIZE=<BATCH_SIZE>
bash scripts/train_benchmark.sh
```

By default, this script runs 400 epochs on the configuration `configs/jasper10x5dr_sp_offline_specaugment.toml`
using batch size 32 on a single node with 8x GPUs with at least 32GB of memory.
By default, `NUM_STEPS=-1` means training is run for 400 EPOCHS. If `$NUM_STEPS > 0` is specified, training is only run for a user-defined number of iterations. Audio samples longer than `MAX_DURATION` are filtered out, the remaining ones are padded to this duration such that all batches have the same length. At the end of training the script saves the model checkpoint to the results folder, runs evaluation on LibriSpeech dev-clean dataset, and prints out information such as average training latency performance in seconds, average trng throughput in sequences per second, final training loss, final training WER, evaluation loss and evaluation WER.


#### Inference performance benchmark
To benchmark the inference performance on a specific batch size and audio length, run:

```bash
bash scripts/inference_benchmark.sh
```
By default, the script runs on a single GPU and evaluates on the entire dataset using the model configuration `configs/jasper10x5dr_sp_offline_specaugment.toml` and batch size 32.
By default, `MAX_DURATION` is set to 36 seconds, which covers the maximum audio length. All audio samples are padded to this length. The script prints out `MAX_DURATION`, `BATCH_SIZE` and latency performance in milliseconds per batch.

Adjustments can be made with env variables, e.g.,
```bash
export SEED=42
export BATCH_SIZE=1
bash scripts/inference_benchmark.sh
```

### Results
The following sections provide details on how we achieved our performance and accuracy in training and inference.
All results are trained on 960 hours of LibriSpeech with a maximum audio length of 16.7s. The training is evaluated
on LibriSpeech dev-clean, dev-other, test-clean, test-other.
The results for Jasper Large's word error rate from the original paper after greedy decoding are shown below:

| **Number of GPUs**    |  **dev-clean WER** | **dev-other WER**| **test-clean WER**| **test-other WER**
|---    |---    |---    |---    |---    |
|8  |   3.64|   11.89| 3.86 | 11.95


#### Training accuracy results

##### Training accuracy: NVIDIA DGX A100 (8x A100 40GB)
Our results were obtained by running the `scripts/train.sh` training script in the 20.06-py3 NGC container on NVIDIA DGX A100 (8x A100 40GB) GPUs.

| **Number of GPUs** | **Batch size per GPU** | **Precision** | **dev-clean WER** | **dev-other WER** | **test-clean WER** | **test-other WER** | **Time to train** | **Time to train speedup (TF32 to mixed precision)** |
|-----|-----|-------|-------|-------|------|-------|-------|-----|
|   8 |  64 | mixed |  3.53 | 11.11 | 3.75 | 11.07 | 60 h  | 1.9 |
|   8 |  64 |  TF32 |  3.55 | 11.30 | 3.81 | 11.17 | 115 h |  -  |

For each precision, we show the best of 8 runs chosen based on dev-clean WER. For TF32, two gradient accumulation steps have been used.

##### Training accuracy: NVIDIA DGX-1 (8x V100 32GB)
Our results were obtained by running the `scripts/train.sh` training script in the PyTorch 20.06-py3 NGC container with NVIDIA DGX-1 with (8x V100 32GB) GPUs.
The following tables report the word error rate(WER) of the acoustic model with greedy decoding on all LibriSpeech dev and test datasets for mixed precision training.

| **Number of GPUs** | **Batch size per GPU** | **Precision** | **dev-clean WER** | **dev-other WER** | **test-clean WER** | **test-other WER** | **Time to train** | **Time to train speedup (FP32 to mixed precision)** |
|-----|-----|-------|-------|-------|------|-------|-------|-----|
|   8 |  64 | mixed |  3.49 | 11.22 | 3.74 | 10.94 | 105 h | 3.1 |
|   8 |  64 |  FP32 |  3.65 | 11.47 | 3.86 | 11.30 | 330 h |  -  |

We show the best of 5 runs (mixed precision) and 2 runs (FP32) chosen based on dev-clean WER. For FP32, two gradient accumulation steps have been used.

##### Training stability test
The following table compares greedy decoding word error rates across 8 different training runs with different seeds for mixed precision training.

| **DGX A100, FP16, 8x GPU**   |   **Seed #1** |   **Seed #2** |   **Seed #3** |   **Seed #4** |   **Seed #5** |   **Seed #6** |   **Seed #7** |   **Seed #8** |   **Mean** |   **Std** |
|-----------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|-----:|
| dev-clean  |  3.69 |  3.71 |  3.64 |  3.53 |  3.71 |  3.66 |  3.77 |  3.70 |  3.68 | 0.07 |
| dev-other  | 11.39 | 11.65 | 11.46 | 11.11 | 11.23 | 11.18 | 11.43 | 11.60 | 11.38 | 0.19 |
| test-clean |  3.97 |  3.96 |  3.81 |  3.75 |  3.90 |  3.82 |  3.93 |  3.82 |  3.87 | 0.08 |
| test-other | 11.27 | 11.34 | 11.40 | 11.07 | 11.24 | 11.29 | 11.58 | 11.58 | 11.35 | 0.17 |

| **DGX A100, TF32, 8x GPU**   |   **Seed #1** |   **Seed #2** |   **Seed #3** |   **Seed #4** |   **Seed #5** |   **Seed #6** |   **Seed #7** |   **Seed #8** |   **Mean** |   **Std** |
|-----------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|-----:|
| dev-clean  |  3.56 |  3.60 |  3.60 |  3.55 |  3.65 |  3.57 |  3.89 |  3.67 |  3.64 | 0.11 |
| dev-other  | 11.27 | 11.41 | 11.65 | 11.30 | 11.51 | 11.11 | 12.18 | 11.50 | 11.49 | 0.32 |
| test-clean |  3.80 |  3.79 |  3.88 |  3.81 |  3.94 |  3.82 |  4.13 |  3.85 |  3.88 | 0.11 |
| test-other | 11.40 | 11.26 | 11.47 | 11.17 | 11.36 | 11.16 | 12.15 | 11.46 | 11.43 | 0.32 |

| **DGX-1 32GB, FP16, 8x GPU**   |   **Seed #1** |   **Seed #2** |   **Seed #3** |   **Seed #4** |   **Seed #5** |   **Mean** |   **Std** |
|-----------:|------:|------:|------:|------:|------:|------:|-----:|
| dev-clean  |  3.69 |  3.75 |  3.63 |  3.86 |  3.49 |  3.68 | 0.14 |
| dev-other  | 11.35 | 11.63 | 11.60 | 11.68 | 11.22 | 11.50 | 0.20 |
| test-clean |  3.90 |  3.84 |  3.94 |  3.96 |  3.74 |  3.88 | 0.09 |
| test-other | 11.17 | 11.45 | 11.31 | 11.60 | 10.94 | 11.29 | 0.26 |

#### Training performance results
Our results were obtained by running the `scripts/train.sh` training script in the PyTorch 20.06-py3 NGC container. Performance (in sequences per second) is the steady-state throughput.

##### Training performance: NVIDIA DGX A100 (8x A100 40GB)
| **GPUs** | **Batch size / GPU** | **Throughput - TF32** | **Throughput - mixed precision** | **Throughput speedup (TF32 to mixed precision)** | **Weak scaling - TF32** | **Weak scaling - mixed precision** |
|--:|---:|------:|-------:|-----:|-----:|-----:|
| 1 | 32 |  36.09 |  69.33 | 1.92 | 1.00 | 1.00 |
| 4 | 32 | 143.05 | 264.91 | 1.85 | 3.96 | 3.82 |
| 8 | 32 | 285.25 | 524.33 | 1.84 | 7.90 | 7.56 |

| **GPUs** | **Batch size / GPU** | **Throughput - TF32** | **Throughput - mixed precision** | **Throughput speedup (TF32 to mixed precision)** | **Weak scaling - TF32** | **Weak scaling - mixed precision** |
|--:|---:|------:|-------:|-----:|-----:|-----:|
| 1 | 64 |      - |  77.79 |    - |    - | 1.00 |
| 4 | 64 |      - | 304.32 |    - |    - | 3.91 |
| 8 | 64 |      - | 602.88 |    - |    - | 7.75 |

Note: Mixed precision permits higher batch sizes during training. We report the maximum batch sizes (as powers of 2), which are allowed without gradient accumulation.

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

##### Training performance: NVIDIA DGX-1 (8x V100 16GB)
| **GPUs** | **Batch size / GPU** | **Throughput - FP32** | **Throughput - mixed precision** | **Throughput speedup (FP32 to mixed precision)** | **Weak scaling - FP32** | **Weak scaling - mixed precision** |
|--:|---:|------:|-------:|-----:|-----:|-----:|
| 1 | 16 | 11.12 |  28.87 | 2.60 | 1.00 | 1.00 |
| 4 | 16 | 42.39 | 109.40 | 2.58 | 3.81 | 3.79 |
| 8 | 16 | 84.45 | 194.30 | 2.30 | 7.59 | 6.73 |

| **GPUs** | **Batch size / GPU** | **Throughput - FP32** | **Throughput - mixed precision** | **Throughput speedup (FP32 to mixed precision)** | **Weak scaling - FP32** | **Weak scaling - mixed precision** |
|--:|---:|------:|-------:|-----:|-----:|-----:|
| 1 | 32 |     - |  37.57 |    - |    - | 1.00 |
| 4 | 32 |     - | 134.80 |    - |    - | 3.59 |
| 8 | 32 |     - | 276.14 |    - |    - | 7.35 |

Note: Mixed precision permits higher batch sizes during training. We report the maximum batch sizes (as powers of 2), which are allowed without gradient accumulation.

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

##### Training performance: NVIDIA DGX-1 (8x V100 32GB)
| **GPUs** | **Batch size / GPU** | **Throughput - FP32** | **Throughput - mixed precision** | **Throughput speedup (FP32 to mixed precision)** | **Weak scaling - FP32** | **Weak scaling - mixed precision** |
|--:|---:|------:|-------:|-----:|-----:|-----:|
| 1 | 32 | 13.15 |  35.63 | 2.71 | 1.00 | 1.00 |
| 4 | 32 | 51.21 | 134.01 | 2.62 | 3.90 | 3.76 |
| 8 | 32 | 99.88 | 247.97 | 2.48 | 7.60 | 6.96 |

| **GPUs** | **Batch size / GPU** | **Throughput - FP32** | **Throughput - mixed precision** | **Throughput speedup (FP32 to mixed precision)** | **Weak scaling - FP32** | **Weak scaling - mixed precision** |
|--:|---:|------:|-------:|-----:|-----:|-----:|
| 1 | 64 |     - |  41.74 |    - |    - | 1.00 |
| 4 | 64 |     - | 158.44 |    - |    - | 3.80 |
| 8 | 64 |     - | 312.22 |    - |    - | 7.48 |

Note: Mixed precision permits higher batch sizes during training. We report the maximum batch sizes (as powers of 2), which are allowed without gradient accumulation.

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

##### Training performance: NVIDIA DGX-2 (16x V100 32GB)
| **GPUs** | **Batch size / GPU** | **Throughput - FP32** | **Throughput - mixed precision** | **Throughput speedup (FP32 to mixed precision)** | **Weak scaling - FP32** | **Weak scaling - mixed precision** |
|---:|---:|-------:|-------:|-----:|------:|------:|
|  1 | 32 |  14.13 |  41.05 | 2.90 |  1.00 |  1.00 |
|  4 | 32 |  54.32 | 156.47 | 2.88 |  3.84 |  3.81 |
|  8 | 32 | 110.26 | 307.13 | 2.79 |  7.80 |  7.48 |
| 16 | 32 | 218.14 | 561.85 | 2.58 | 15.44 | 13.69 |

| **GPUs** | **Batch size / GPU** | **Throughput - FP32** | **Throughput - mixed precision** | **Throughput speedup (FP32 to mixed precision)** | **Weak scaling - FP32** | **Weak scaling - mixed precision** |
|---:|---:|-------:|-------:|-----:|------:|------:|
|  1 | 64 |      - |  46.41 |    - |     - |  1.00 |
|  4 | 64 |      - | 147.90 |    - |     - |  3.19 |
|  8 | 64 |      - | 359.15 |    - |     - |  7.74 |
| 16 | 64 |      - | 703.13 |    - |     - | 15.15 |

Note: Mixed precision permits higher batch sizes during training. We report the maximum batch sizes (as powers of 2), which are allowed without gradient accumulation.

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.


#### Inference performance results
Our results were obtained by running the `scripts/inference_benchmark.sh` script in the PyTorch 20.06-py3 NGC container on NVIDIA DGX A100, DGX-1, DGX-2 and T4 on a single GPU. Performance numbers (latency in milliseconds per batch) were averaged over 1000 iterations.

##### Inference performance: NVIDIA DGX A100 (1x A100 40GB)
|    | |FP16 Latency (ms) Percentiles | | | | TF32 Latency (ms) Percentiles | | | | FP16/TF32 speed up |
|---:|-------------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-----:|
| BS | Duration (s) |   90% |   95% |   99% |   Avg |    90% |    95% |    99% |    Avg |  Avg |
|  1 |            2 | 36.31 | 36.85 | 43.18 | 35.96 |  41.16 |  41.63 |  47.90 |  40.89 | 1.14 |
|  2 |            2 | 37.56 | 43.32 | 45.23 | 37.11 |  42.53 |  47.79 |  49.62 |  42.07 | 1.13 |
|  4 |            2 | 43.10 | 44.85 | 47.22 | 41.43 |  47.88 |  49.75 |  51.55 |  43.25 | 1.04 |
|  8 |            2 | 44.02 | 44.30 | 45.21 | 39.51 |  50.14 |  50.47 |  51.50 |  45.63 | 1.16 |
| 16 |            2 | 48.04 | 48.38 | 49.12 | 42.76 |  70.90 |  71.22 |  72.50 |  60.78 | 1.42 |
|  1 |            7 | 37.74 | 37.88 | 38.92 | 37.02 |  41.53 |  42.17 |  44.75 |  40.79 | 1.10 |
|  2 |            7 | 40.91 | 41.11 | 42.35 | 40.02 |  46.44 |  46.80 |  49.67 |  45.67 | 1.14 |
|  4 |            7 | 43.94 | 44.32 | 46.71 | 43.00 |  54.39 |  54.80 |  56.63 |  53.53 | 1.24 |
|  8 |            7 | 50.01 | 50.19 | 52.92 | 48.62 |  68.55 |  69.25 |  72.28 |  67.61 | 1.39 |
| 16 |            7 | 60.38 | 60.76 | 62.44 | 57.92 |  93.17 |  94.15 |  98.84 |  92.21 | 1.59 |
|  1 |         16.7 | 41.39 | 41.75 | 43.62 | 40.73 |  45.79 |  46.10 |  47.76 |  45.21 | 1.11 |
|  2 |         16.7 | 46.43 | 46.76 | 47.72 | 45.81 |  52.53 |  53.13 |  55.60 |  51.71 | 1.13 |
|  4 |         16.7 | 50.88 | 51.68 | 54.74 | 50.11 |  66.29 |  66.96 |  70.45 |  65.00 | 1.30 |
|  8 |         16.7 | 62.09 | 62.76 | 65.08 | 61.40 |  94.16 |  94.67 |  97.46 |  93.00 | 1.51 |
| 16 |         16.7 | 75.22 | 76.86 | 80.76 | 73.99 | 139.51 | 140.88 | 144.10 | 137.94 | 1.86 |

##### Inference performance: NVIDIA DGX-1 (1x V100 16GB)
|    | |FP16 Latency (ms) Percentiles | | | | FP32 Latency (ms) Percentiles | | | | FP16/FP32 speed up |
|---:|-------------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-----:|
| BS | Duration (s) |   90% |   95% |   99% |   Avg |    90% |    95% |    99% |    Avg |  Avg |
|  1 |    2 |  52.26 |  59.93 |  66.62 |  50.34 |  70.90 |  76.47 |  79.84 |  68.61 | 1.36 |
|  2 |    2 |  62.04 |  67.68 |  70.91 |  58.65 |  75.72 |  80.15 |  83.50 |  71.33 | 1.22 |
|  4 |    2 |  75.12 |  77.12 |  82.80 |  66.55 |  80.88 |  82.60 |  86.63 |  73.65 | 1.11 |
|  8 |    2 |  71.62 |  72.99 |  81.10 |  66.39 |  99.57 | 101.43 | 107.16 |  92.34 | 1.39 |
| 16 |    2 |  78.51 |  80.33 |  87.31 |  72.91 | 104.79 | 107.22 | 114.21 |  96.18 | 1.32 |
|  1 |    7 |  52.67 |  54.40 |  64.27 |  50.47 |  73.86 |  75.61 |  84.93 |  72.08 | 1.43 |
|  2 |    7 |  60.49 |  62.41 |  72.87 |  58.45 |  93.07 |  94.51 | 102.40 |  91.55 | 1.57 |
|  4 |    7 |  70.55 |  72.95 |  82.59 |  68.43 | 131.48 | 137.60 | 149.06 | 129.23 | 1.89 |
|  8 |    7 |  83.91 |  85.28 |  93.08 |  76.40 | 152.49 | 157.92 | 166.80 | 150.49 | 1.97 |
| 16 |    7 | 100.21 | 103.12 | 109.00 |  96.31 | 178.45 | 181.46 | 187.20 | 174.33 | 1.81 |
|  1 | 16.7 |  56.84 |  60.05 |  66.54 |  54.69 | 109.55 | 111.19 | 120.40 | 102.25 | 1.87 |
|  2 | 16.7 |  69.39 |  70.97 |  75.34 |  67.39 | 149.93 | 150.79 | 154.06 | 147.45 | 2.19 |
|  4 | 16.7 |  87.48 |  93.96 | 102.73 |  85.09 | 211.78 | 219.66 | 232.99 | 208.38 | 2.45 |
|  8 | 16.7 | 106.91 | 111.92 | 116.55 | 104.13 | 246.92 | 250.94 | 268.44 | 243.34 | 2.34 |
| 16 | 16.7 | 149.08 | 153.86 | 166.17 | 146.28 | 292.84 | 298.02 | 313.04 | 288.54 | 1.97 |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

##### Inference performance: NVIDIA DGX-1 (1x V100 32GB)
|    | |FP16 Latency (ms) Percentiles | | | | FP32 Latency (ms) Percentiles | | | | FP16/FP32 speed up |
|---:|-------------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-----:|
| BS | Duration (s) |   90% |   95% |   99% |   Avg |    90% |    95% |    99% |    Avg |  Avg |
|  1 |    2 |  64.60 |  67.34 |  79.87 |  60.73 |  84.69 |  86.78 |  96.02 |  79.32 | 1.31 |
|  2 |    2 |  71.52 |  73.32 |  82.00 |  63.93 |  85.33 |  87.65 |  96.34 |  78.09 | 1.22 |
|  4 |    2 |  80.38 |  84.62 |  93.09 |  74.95 |  90.29 |  97.59 | 100.61 |  84.44 | 1.13 |
|  8 |    2 |  83.43 |  85.51 |  91.17 |  74.09 | 107.28 | 111.89 | 115.19 |  98.76 | 1.33 |
| 16 |    2 |  90.01 |  90.81 |  96.48 |  79.85 | 115.39 | 116.95 | 123.71 | 103.26 | 1.29 |
|  1 |    7 |  53.74 |  54.09 |  56.67 |  53.07 |  86.07 |  86.55 |  91.59 |  78.79 | 1.48 |
|  2 |    7 |  63.34 |  63.67 |  66.08 |  62.62 |  96.25 |  96.82 |  99.72 |  95.44 | 1.52 |
|  4 |    7 |  80.35 |  80.86 |  83.80 |  73.41 | 132.19 | 132.94 | 135.59 | 131.46 | 1.79 |
|  8 |    7 |  77.68 |  78.11 |  86.71 |  75.72 | 156.30 | 157.72 | 165.55 | 154.87 | 2.05 |
| 16 |    7 | 103.52 | 106.66 | 111.93 |  98.15 | 180.71 | 182.82 | 191.12 | 178.61 | 1.82 |
|  1 | 16.7 |  57.58 |  57.79 |  59.75 |  56.58 | 104.51 | 104.87 | 108.01 | 104.04 | 1.84 |
|  2 | 16.7 |  69.19 |  69.58 |  71.49 |  68.58 | 151.25 | 152.07 | 155.21 | 149.30 | 2.18 |
|  4 | 16.7 |  87.17 |  88.53 |  97.41 |  86.56 | 211.28 | 212.41 | 214.97 | 208.54 | 2.41 |
|  8 | 16.7 | 116.25 | 116.90 | 120.14 | 109.21 | 247.63 | 248.93 | 254.77 | 245.19 | 2.25 |
| 16 | 16.7 | 151.99 | 154.79 | 163.36 | 149.80 | 293.99 | 296.05 | 303.04 | 291.00 | 1.94 |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

##### Inference performance: NVIDIA DGX-2 (1x V100 32GB)
|    | |FP16 Latency (ms) Percentiles | | | | FP32 Latency (ms) Percentiles | | | | FP16/FP32 speed up |
|---:|-------------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-----:|
| BS | Duration (s) |   90% |   95% |   99% |   Avg |    90% |    95% |    99% |    Avg |  Avg |
|  1 |    2 |  47.25 |  48.24 |  50.28 |  41.53 |  67.03 |  68.15 |  70.17 |  61.82 | 1.49 |
|  2 |    2 |  54.11 |  55.20 |  60.44 |  48.82 |  69.11 |  70.38 |  75.93 |  64.45 | 1.32 |
|  4 |    2 |  63.82 |  67.64 |  71.58 |  61.47 |  71.51 |  74.55 |  79.31 |  67.85 | 1.10 |
|  8 |    2 |  64.78 |  65.86 |  67.68 |  59.07 |  90.84 |  91.99 |  94.10 |  84.28 | 1.43 |
| 16 |    2 |  70.59 |  71.49 |  73.58 |  63.85 |  96.92 |  97.58 |  99.98 |  87.73 | 1.37 |
|  1 |    7 |  42.35 |  42.55 |  43.50 |  41.08 |  63.87 |  64.02 |  64.73 |  62.54 | 1.52 |
|  2 |    7 |  47.82 |  48.04 |  49.43 |  46.79 |  81.17 |  81.43 |  82.28 |  80.02 | 1.71 |
|  4 |    7 |  58.27 |  58.54 |  59.69 |  56.96 | 116.00 | 116.46 | 118.79 | 114.82 | 2.02 |
|  8 |    7 |  62.88 |  63.62 |  67.16 |  61.47 | 143.90 | 144.34 | 147.36 | 139.54 | 2.27 |
| 16 |    7 |  88.04 |  88.57 |  90.96 |  82.84 | 163.04 | 164.04 | 167.30 | 161.36 | 1.95 |
|  1 | 16.7 |  44.54 |  44.86 |  45.86 |  43.53 |  88.10 |  88.41 |  89.37 |  87.21 | 2.00 |
|  2 | 16.7 |  55.21 |  55.55 |  56.92 |  54.33 | 134.99 | 135.69 | 137.87 | 132.97 | 2.45 |
|  4 | 16.7 |  72.93 |  73.58 |  74.95 |  72.02 | 193.50 | 194.21 | 196.04 | 191.24 | 2.66 |
|  8 | 16.7 |  96.94 |  97.66 |  99.58 |  92.73 | 227.70 | 228.74 | 231.59 | 225.35 | 2.43 |
| 16 | 16.7 | 138.25 | 139.75 | 143.71 | 133.82 | 273.69 | 274.53 | 279.50 | 269.13 | 2.01 |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

##### Inference performance: NVIDIA T4
|    | |FP16 Latency (ms) Percentiles | | | | FP32 Latency (ms) Percentiles | | | | FP16/FP32 speed up |
|---:|-------------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|-----:|
| BS | Duration (s) |   90% |   95% |   99% |   Avg |    90% |    95% |    99% |    Avg |  Avg |
|  1 |    2 |  64.13 |  65.25 |  76.11 |  59.08 |  94.69 |  98.23 | 109.86 |  89.00 | 1.51 |
|  2 |    2 |  67.59 |  70.77 |  84.06 |  57.47 | 103.88 | 105.37 | 114.59 |  93.30 | 1.62 |
|  4 |    2 |  75.19 |  81.05 |  87.01 |  65.79 | 120.73 | 128.29 | 146.83 | 112.96 | 1.72 |
|  8 |    2 |  74.15 |  77.69 |  84.96 |  62.77 | 161.97 | 163.46 | 170.25 | 153.07 | 2.44 |
| 16 |    2 | 100.62 | 105.08 | 113.00 |  82.06 | 216.18 | 217.92 | 222.46 | 188.57 | 2.30 |
|  1 |    7 |  77.88 |  79.61 |  81.90 |  70.22 | 110.37 | 113.93 | 121.39 | 107.17 | 1.53 |
|  2 |    7 |  81.09 |  83.94 |  87.28 |  78.06 | 148.30 | 151.21 | 158.55 | 141.26 | 1.81 |
|  4 |    7 |  99.85 | 100.83 | 104.24 |  96.81 | 229.94 | 232.34 | 238.11 | 225.43 | 2.33 |
|  8 |    7 | 147.38 | 150.37 | 153.66 | 142.64 | 394.26 | 396.35 | 398.89 | 390.77 | 2.74 |
| 16 |    7 | 280.32 | 281.37 | 282.74 | 278.01 | 484.20 | 485.74 | 499.89 | 482.67 | 1.74 |
|  1 | 16.7 |  76.97 |  79.78 |  81.61 |  75.55 | 171.45 | 176.90 | 179.18 | 167.95 | 2.22 |
|  2 | 16.7 |  96.48 |  99.42 | 101.21 |  92.74 | 276.12 | 278.67 | 282.06 | 270.05 | 2.91 |
|  4 | 16.7 | 129.63 | 131.67 | 134.42 | 124.55 | 522.23 | 524.79 | 527.32 | 509.75 | 4.09 |
|  8 | 16.7 | 209.64 | 211.36 | 214.66 | 204.83 | 706.84 | 709.21 | 715.57 | 697.97 | 3.41 |
| 16 | 16.7 | 342.23 | 344.62 | 350.84 | 337.42 | 848.02 | 849.83 | 858.22 | 834.38 | 2.47 |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

## Release notes

### Changelog
June 2020
- Updated performance tables to include A100 results

December 2019
* Inference support for TRT 6 with dynamic shapes
* Inference support for TensorRT Inference Server with acoustic model backends in ONNX, PyTorch JIT, TensorRT
* Jupyter notebook for inference with TensorRT Inference Server

November 2019
* Google Colab notebook for inference with native TensorRT

September 2019
* Inference support for TensorRT 6 with static shapes
* Jupyter notebook for inference

August 2019
* Initial release

### Known issues
There are no known issues in this release.

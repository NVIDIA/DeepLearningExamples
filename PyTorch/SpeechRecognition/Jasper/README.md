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
   * [Deploying Jasper using Triton Inference Server](#deploying-jasper-using-triton-inference)
- [Performance](#performance)
   * [Benchmarking](#benchmarking)
       * [Training performance benchmark](#training-performance-benchmark)
       * [Inference performance benchmark](#inference-performance-benchmark)
   * [Results](#results)
       * [Training accuracy results](#training-accuracy-results)
           * [Training accuracy: NVIDIA DGX A100 (8x A100 80GB)](#training-accuracy-nvidia-dgx-a100-8x-a100-80gb)
           * [Training accuracy: NVIDIA DGX-1 (8x V100 32GB)](#training-accuracy-nvidia-dgx-1-8x-v100-32gb)
           * [Training stability test](#training-stability-test)
       * [Training performance results](#training-performance-results)
         * [Training performance: NVIDIA DGX A100 (8x A100 80GB)](#training-performance-nvidia-dgx-a100-8x-a100-80gb)
         * [Training performance: NVIDIA DGX-1 (8x V100 16GB)](#training-performance-nvidia-dgx-1-8x-v100-16gb)
         * [Training performance: NVIDIA DGX-1 (8x V100 32GB)](#training-performance-nvidia-dgx-1-8x-v100-32gb)
         * [Training performance: NVIDIA DGX-2 (16x V100 32GB)](#training-performance-nvidia-dgx-2-16x-v100-32gb)
       * [Inference performance results](#inference-performance-results)
           * [Inference performance: NVIDIA DGX A100 (1x A100 80GB)](#inference-performance-nvidia-dgx-a100-gpu-1x-a100-80gb)
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
This repository contains a `Dockerfile` which extends the PyTorch 20.10-py3 NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:

* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch 20.10-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
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
bash scripts/docker/launch.sh <DATA_DIR> <CHECKPOINT_DIR> <OUTPUT_DIR>
```
Within the container, the contents of this repository will be copied to the `/workspace/jasper` directory. The `/datasets`, `/checkpoints`, `/results` directories are mounted as volumes
and mapped to the corresponding directories `<DATA_DIR>`, `<CHECKPOINT_DIR>`, `<OUTPUT_DIR>` on the host.

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
```bash
datasets/LibriSpeech/
├── dev-clean
├── dev-other
├── test-clean
├── test-other
├── train-clean-100
├── train-clean-360
└── train-other-500
```

Since `/datasets/` is mounted to `<DATA_DIR>` on the host (see Step 3),  once the dataset is downloaded it will be accessible from outside of the container at `<DATA_DIR>/LibriSpeech`.


Next, convert the data into WAV files:
```bash
bash scripts/preprocess_librispeech.sh
```
Once the data is converted, the following additional files and folders should exist:
```bash
datasets/LibriSpeech/
├── dev-clean-wav
├── dev-other-wav
├── librispeech-train-clean-100-wav.json
├── librispeech-train-clean-360-wav.json
├── librispeech-train-other-500-wav.json
├── librispeech-dev-clean-wav.json
├── librispeech-dev-other-wav.json
├── librispeech-test-clean-wav.json
├── librispeech-test-other-wav.json
├── test-clean-wav
├── test-other-wav
├── train-clean-100-wav
├── train-clean-360-wav
└── train-other-500-wav
```

The DALI data pre-processing pipeline, which is enabled by default, performs speed perturbation on-line during training.
Without DALI, on-line speed perturbation might slow down the training.
If you wish to disable DALI, speed perturbation can be computed off-line with:
```bash
SPEEDS="0.9 1.1" bash scripts/preprocess_librispeech.sh
```

5. Start training.

Inside the container, use the following script to start training.
Make sure the downloaded and preprocessed dataset is located at `<DATA_DIR>/LibriSpeech` on the host (see Step 3), which corresponds to `/datasets/LibriSpeech` inside the container.

```bash
[OPTION1=value1 OPTION2=value2 ...] bash scripts/train.sh
```
By default automatic precision is disabled, batch size is 64 over two gradient accumulation steps, and the recipe is run on a total of 8 GPUs. The hyperparameters are tuned for a GPU with at least 32GB of memory and will require adjustment for 16GB GPUs (e.g., by lowering batch size and using more gradient accumulation steps).

Options are being passed as environment variables. More details on available options can be found in [Parameters](#parameters) and [Training process](#training-process).

6. Start validation/evaluation.

Inside the container, use the following script to run evaluation.
 Make sure the downloaded and preprocessed dataset is located at `<DATA_DIR>/LibriSpeech` on the host (see Step 3), which corresponds to `/datasets/LibriSpeech` inside the container.
```bash
[OPTION1=value1 OPTION2=value2 ...] bash scripts/evaluation.sh [OPTIONS]
```
By default, this will use full precision, a batch size of 64 and run on a single GPU.

Options are being passed as environment variables. More details on available options can be found in [Parameters](#parameters) and [Evaluation process](#evaluation-process).


7. Start inference/predictions.

Inside the container, use the following script to run inference.
 Make sure the downloaded and preprocessed dataset is located at `<DATA_DIR>/LibriSpeech` on the host (see Step 3), which corresponds to `/datasets/LibriSpeech` inside the container.
A pretrained model checkpoint can be downloaded from [NGC model repository](https://ngc.nvidia.com/catalog/models/nvidia:jasperpyt_fp16).

```bash
[OPTION1=value1 OPTION2=value2 ...] bash scripts/inference.sh
```
By default this will use single precision, a batch size of 64 and run on a single GPU.

Options are being passed as environment variables. More details on available options can be found in [Parameters](#parameters) and [Inference process](#inference-process).


## Advanced

The following sections provide greater details of the dataset, running training and inference, and getting training and inference results.


### Scripts and sample code
In the `root` directory, the most important files are:
```
jasper
├── common        # data pre-processing, logging, etc.
├── configs       # model configurations
├── Dockerfile    # container with the basic set of dependencies to run Jasper
├── inference.py  # entry point for inference
├── jasper        # model-specific code
├── notebooks     # jupyter notebooks and example audio files
├── scripts       # one-click scripts required for running various supported functionalities
│   ├── docker                     # contains the scripts for building and launching the container
│   ├── download_librispeech.sh    # downloads LibriSpeech dataset
│   ├── evaluation.sh              # runs evaluation using the `inference.py` script
│   ├── inference_benchmark.sh     # runs the inference benchmark using the `inference_benchmark.py` script
│   ├── inference.sh               # runs inference using the `inference.py` script
│   ├── preprocess_librispeech.sh  # preprocess LibriSpeech raw data files for training and inference
│   ├── train_benchmark.sh         # runs the training performance benchmark using the `train.py` script
│   └── train.sh                   # runs training using the `train.py` script
├── train.py      # entry point for training
├── triton        # example of inference using Triton Inference Server
└── utils         # data downloading and common routines
```

### Parameters

Parameters could be set as env variables, or passed as positional arguments.

The complete list of available parameters for `scripts/train.sh` script contains:
```bash
DATA_DIR:                directory of dataset. (default: '/datasets/LibriSpeech')
MODEL_CONFIG:            relative path to model configuration. (default: 'configs/jasper10x5dr_speedp-online_speca.yaml')
OUTPUT_DIR:              directory for results, logs, and created checkpoints. (default: '/results')
CHECKPOINT:              a specific model checkpoint to continue training from. To resume training from the last checkpoint, see the RESUME option.
RESUME:                  resume training from the last checkpoint found in OUTPUT_DIR, or from scratch if there are no checkpoints (default: true)
CUDNN_BENCHMARK:         boolean that indicates whether to enable cudnn benchmark mode for using more optimized kernels. (default: true)
NUM_GPUS:                number of GPUs to use. (default: 8)
AMP:                     if set to `true`, enables automatic mixed precision (default: false)
BATCH_SIZE:              effective data batch size. The real batch size per GPU might be lower, if gradient accumulation is enabled (default: 64)
GRAD_ACCUMULATION_STEPS: number of gradient accumulation steps until optimizer updates weights. (default: 2)
LEARNING_RATE:           initial learning rate. (default: 0.01)
MIN_LEARNING_RATE:       minimum learning rate, despite LR scheduling (default: 1e-5)
LR_POLICY:               how to decay LR (default: exponential)
LR_EXP_GAMMA:            decay factor for the exponential LR schedule (default: 0.981)
EMA:                     decay factor for exponential averages of checkpoints (default: 0.999)
SEED:                    seed for random number generator and used for ensuring reproducibility. (default: 0)
EPOCHS:                  number of training epochs. (default: 440)
WARMUP_EPOCHS:           number of initial epoch of linearly increasing LR. (default: 2)
HOLD_EPOCHS:             number of epochs to hold maximum LR after warmup. (default: 140)
SAVE_FREQUENCY:          number of epochs between saving the model to disk. (default: 10)
EPOCHS_THIS_JOB:         run training for this number of epochs. Does not affect LR schedule like the EPOCHS parameter. (default: 0)
DALI_DEVICE:             device to run the DALI pipeline on for calculation of filterbanks. Valid choices: cpu, gpu, none. (default: gpu)
PAD_TO_MAX_DURATION:     pad all sequences with zeros to maximum length. (default: false)
EVAL_FREQUENCY:          number of steps between evaluations on the validation set. (default: 544)
PREDICTION_FREQUENCY:    the number of steps between writing a sample prediction to stdout. (default: 544)
TRAIN_MANIFESTS:         lists of .json training set files
VAL_MANIFESTS:           lists of .json validation set files

```

The complete list of available parameters for `scripts/inference.sh` script contains:
```bash
DATA_DIR:            directory of dataset. (default: '/datasets/LibriSpeech')
MODEL_CONFIG:        model configuration. (default: 'configs/jasper10x5dr_speedp-online_speca.yaml')
OUTPUT_DIR:          directory for results and logs. (default: '/results')
CHECKPOINT:          model checkpoint path. (required)
DATASET:             name of the LibriSpeech subset to use. (default: 'dev-clean')
LOG_FILE:            path to the DLLogger .json logfile. (default: '')
CUDNN_BENCHMARK:     enable cudnn benchmark mode for using more optimized kernels. (default: false)
MAX_DURATION:        filter out recordings shorter then MAX_DURATION seconds. (default: "")
PAD_TO_MAX_DURATION: pad all sequences with zeros to maximum length. (default: false)
PAD_LEADING:         pad every batch with leading zeros to counteract conv shifts of the field of view. (default: 16)
NUM_GPUS:            number of GPUs to use. Note that with > 1 GPUs WER results might be inaccurate due to the batching policy. (default: 1)
NUM_STEPS:           number of batches to evaluate, loop the dataset if necessary. (default: 0)
NUM_WARMUP_STEPS:    number of initial steps before measuring performance. (default: 0)
AMP:                 enable FP16 inference with AMP. (default: false)
BATCH_SIZE:          data batch size. (default: 64)
EMA:                 Attempt to load exponentially averaged weights from a checkpoint. (default: true)
SEED:                seed for random number generator and used for ensuring reproducibility. (default: 0)
DALI_DEVICE:         device to run the DALI pipeline on for calculation of filterbanks. Valid choices: cpu, gpu, none. (default: gpu)
CPU:                 run inference on CPU. (default: false)
LOGITS_FILE:         dump logit matrices to a file. (default: "")
PREDICTION_FILE:     save predictions to a file. (default: "${OUTPUT_DIR}/${DATASET}.predictions")
```

The complete list of available parameters for `scripts/evaluation.sh` is the same as for `scripts/inference.sh` except for the few default changes.
```bash
PREDICTION_FILE: (default: "")
DATASET:         (default: "test-other")
```

The `scripts/inference_benchmark.sh` script pads all input to a fixed duration and computes the mean, 90%, 95%, 99% percentile of latency for the specified number of inference steps. Latency is measured in milliseconds per batch. The `scripts/inference_benchmark.sh` measures latency for a single GPU and loops over a number of batch sizes and durations. It extends  `scripts/inference.sh`, and changes the defaults with:
```bash
BATCH_SIZE_SEQ:      batch sizes to measure on. (default: "1 2 4 8 16")
MAX_DURATION_SEQ:    input durations (in seconds) to measure on (default: "2 7 16.7")
CUDNN_BENCHMARK:     (default: true)
PAD_TO_MAX_DURATION: (default: true)
PAD_LEADING:         (default: 0)
NUM_WARMUP_STEPS:    (default: 10)
NUM_STEPS:           (default: 500)
DALI_DEVICE:         (default: cpu)
```

The `scripts/train_benchmark.sh` script pads all input to the same length according to the input argument `MAX_DURATION` and measures average training latency and throughput performance. Latency is measured in seconds per batch, throughput in sequences per second.
Training performance is measured with on-line speed perturbation and cuDNN benchmark mode enabled.
The script `scripts/train_benchmark.sh` loops over a number of batch sizes and GPU counts.
It extends `scripts/train.sh`, and the complete list of available parameters for `scripts/train_benchmark.sh` script contains:
```bash
BATCH_SIZE_SEQ:          batch sizes to measure on. (default: "1 2 4 8 16")
NUM_GPUS_SEQ:            number of GPUs to run the training on. (default: "1 4 8")
MODEL_CONFIG:            (default: "configs/jasper10x5dr_speedp-online_train-benchmark.yaml")
TRAIN_MANIFESTS:         (default: "$DATA_DIR/librispeech-train-clean-100-wav.json")
RESUME:                  (default: false)
EPOCHS_THIS_JOB:         (default: 2)
EPOCHS:                  (default: 100000)
SAVE_FREQUENCY:          (default: 100000)
EVAL_FREQUENCY:          (default: 100000)
GRAD_ACCUMULATION_STEPS: (default: 1)
PAD_TO_MAX_DURATION:     (default: true)
EMA:                     (default: 0)
```

### Command-line options
To see the full list of available options and their descriptions, use the `-h` or `--help` command-line option with the Python file, for example:

```bash
python train.py --help
python inference.py --help
```

### Getting the data
The Jasper model was trained on the LibriSpeech dataset. We use the concatenation of `train-clean-100`, `train-clean-360` and `train-other-500` for training and `dev-clean` for validation.

This repository contains the `scripts/download_librispeech.sh` and `scripts/preprocess_librispeech.sh` scripts which will automatically download and preprocess the training, test and development datasets. By default, data will be downloaded to the `/datasets/LibriSpeech` directory, a minimum of 250GB free space is required for download and preprocessing, the final preprocessed dataset is approximately 100GB. With offline speed perturbation, the dataset will be about 3x larger.

#### Dataset guidelines
The `scripts/preprocess_librispeech.sh` script converts the input audio files to WAV format with a sample rate of 16kHz, target transcripts are stripped from whitespace characters, then lower-cased. For `train-clean-100`, `train-clean-360` and `train-other-500`. It can optionally create speed perturbed versions with rates of 0.9 and 1.1 for data augmentation. In the current version, those augmentations are applied on-line with the DALI pipeline without any impact on training time.

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
* Runs for 440 epochs
* Uses an initial learning rate of 0.01 and an exponential learning rate decay
* Saves a checkpoint every 10 epochs
* Automatically removes old checkpoints and preserves milestone checkpoints
* Runs evaluation on the development dataset every 544 iterations and at the end of training
* Maintains a separate checkpoint with the lowest WER on development set
* Prints out training progress every iteration to stdout
* Creates a DLLogger logfile and a Tensorboard log
* Calculates speed perturbation on-line during training
* Uses SpecAugment in data pre-processing
* Filters out audio samples longer than 16.7 seconds
* Pads each batch so its length would be divisible by 16
* Uses masked convolutions and dense residuals as described in the paper
* Uses weight decay of 0.001
* Uses [Novograd](https://arxiv.org/pdf/1905.11286.pdf) as optimizer with betas=(0.95, 0)

Enabling AMP permits batch size 64 with one gradient accumulation step. In the current setup it will improve upon the greedy WER [Results](#results) of the Jasper paper on a DGX-1 with 32GB V100 GPUs.

### Inference process
Inference is performed using the `inference.py` script along with parameters defined in `scripts/inference.sh`.
The `scripts/inference.sh` script runs the job on a single GPU, taking a pre-trained Jasper model checkpoint and running it on the specified dataset.
Apart from the default arguments as listed in the [Parameters](#parameters) section by default the inference script:

* Evaluates on the LibriSpeech dev-clean dataset
* Uses a batch size of 64
* Runs for 1 epoch and prints out the final word error rate
* Creates a log file with progress and results which will be stored in the results folder
* Pads each batch so its length would be divisible by 16
* Does not use data augmentation
* Does greedy decoding and saves the transcription in the results folder
* Has the option to save the model output tensors for more complex decoding, for example, beam search
* Has cudnn benchmark disabled

### Evaluation process
Evaluation is performed using the `inference.py` script along with parameters defined in `scripts/evaluation.sh`.
The setup is similar to `scripts/inference.sh`, with two differences:

* Evaluates the LibriSpeech test-other dataset
* Model outputs are not saved

### Deploying Jasper using Triton Inference Server
The NVIDIA Triton Inference Server provides a datacenter and cloud inferencing solution optimized for NVIDIA GPUs. The server provides an inference service via an HTTP or gRPC endpoint, allowing remote clients to request inferencing for any number of GPU or CPU models being managed by the server.
More information on how to perform inference using Triton Inference Server with different model backends can be found in the subfolder [./triton/README.md](triton/README.md)


## Performance

The performance measurements in this document were conducted at the time of publication and may not reflect the performance achieved from NVIDIA’s latest software release. For the most up-to-date performance measurements, go to [NVIDIA Data Center Deep Learning Product Performance](https://developer.nvidia.com/deep-learning-performance-training-inference).

### Benchmarking
The following section shows how to run benchmarks measuring the model performance in training and inference modes.

#### Training performance benchmark
To benchmark the training performance in a specific setting on the `train-clean-100` subset of LibriSpeech, run:

```bash
BATCH_SIZE_SEQ=<BATCH_SIZES> NUM_GPUS_SEQ=<NUMS_OF_GPUS> bash scripts/train_benchmark.sh
```

By default, this script runs 2 epochs on the configuration `configs/jasper10x5dr_speedp-online_train-benchmark.yaml`,
which applies gentle speed perturbation that does not change the length of the output, enabling immediate stabilization of training step times in the cuDNN benchmark mode. The script runs benchmarks on batch sizes 32 on 1, 4, and 8 GPUs, and requires a 8x 32GB GPU machine.

#### Inference performance benchmark
To benchmark the inference performance on a specific batch size and audio length, run:

```bash
BATCH_SIZE_SEQ=<BATCH_SIZES> MAX_DURATION_SEQ=<DURATIONS> bash scripts/inference_benchmark.sh
```
By default, the script runs on a single GPU and evaluates on the dataset limited to utterances shorter than MAX_DURATION. It uses the model configuration `configs/jasper10x5dr_speedp-online_speca.yaml`.


### Results
The following sections provide details on how we achieved our performance and accuracy in training and inference.
All results are trained on 960 hours of LibriSpeech with a maximum audio length of 16.7s. The training is evaluated
on LibriSpeech dev-clean, dev-other, test-clean, test-other. Checkpoints for evaluation are being chosen based on their
word error rate on dev-clean.

#### Training accuracy results

##### Training accuracy: NVIDIA DGX A100 (8x A100 80GB)
Our results were obtained by running the `scripts/train.sh` training script in the PyTorch 20.10-py3 NGC container with NVIDIA DGX A100 with (8x A100 80GB) GPUs.
The following table reports the word error rate (WER) of the acoustic model with greedy decoding on all LibriSpeech dev and test datasets for mixed precision training.

| Number of GPUs | Batch size per GPU | Precision | dev-clean WER | dev-other WER | test-clean WER | test-other WER | Time to train |
|-----|-----|-------|-------|-------|------|-------|------|
|   8 |  64 | mixed |  3.20 |  9.78 | 3.41 |  9.71 | 70 h |

##### Training accuracy: NVIDIA DGX-1 (8x V100 32GB)
Our results were obtained by running the `scripts/train.sh` training script in the PyTorch 20.10-py3 NGC container with NVIDIA DGX-1 with (8x V100 32GB) GPUs.
The following table reports the word error rate (WER) of the acoustic model with greedy decoding on all LibriSpeech dev and test datasets for mixed precision training.

| Number of GPUs | Batch size per GPU | Precision | dev-clean WER | dev-other WER | test-clean WER | test-other WER | Time to train |
|-----|-----|-------|-------|-------|------|-------|-------|
|   8 |  64 | mixed |  3.26 | 10.00 | 3.54 |  9.80 | 130 h |

We show the best of 5 runs (mixed precision) and 2 runs (FP32) chosen based on dev-clean WER. For FP32, two gradient accumulation steps have been used.

##### Training stability test
The following table compares greedy decoding word error rates across 8 different training runs with different seeds for mixed precision training.

| DGX A100 80GB, FP16, 8x GPU |   Seed #1 |   Seed #2 |   Seed #3 |   Seed #4 |   Seed #5 |   Seed #6 |   Seed #7 |   Seed #8 |   Mean |   Std |
|-----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|-------:|------:|
| dev-clean  |      3.46 |      3.55 |      3.45 |      3.44 |      3.25 |      3.34 |      3.20 |      3.40 |   3.39 |  0.11 |
| dev-other  |     10.30 |     10.77 |     10.36 |     10.26 |      9.99 |     10.18 |      9.78 |     10.32 |  10.25 |  0.27 |
| test-clean |      3.84 |      3.81 |      3.66 |      3.64 |      3.58 |      3.55 |      3.41 |      3.73 |   3.65 |  0.13 |
| test-other |     10.61 |     10.52 |     10.49 |     10.47 |      9.89 |     10.09 |      9.71 |     10.26 |  10.26 |  0.31 |


| DGX-1 32GB, FP16, 8x GPU |   Seed #1 |   Seed #2 |   Seed #3 |   Seed #4 |   Seed #5 |   Seed #6 |   Seed #7 |   Seed #8 |   Mean |   Std |
|-----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|-------:|------:|
| dev-clean  |      3.31 |      3.31 |      3.26 |      3.44 |      3.40 |      3.35 |      3.36 |      3.28 |   3.34 |  0.06 |
| dev-other  |     10.02 |     10.01 |     10.00 |     10.06 |     10.05 |     10.03 |     10.10 |     10.04 |  10.04 |  0.03 |
| test-clean |      3.49 |      3.50 |      3.54 |      3.61 |      3.57 |      3.58 |      3.48 |      3.51 |   3.54 |  0.04 |
| test-other |     10.11 |     10.14 |      9.80 |     10.09 |     10.17 |      9.99 |      9.86 |     10.00 |  10.02 |  0.13 |

#### Training performance results
Our results were obtained by running the `scripts/train.sh` training script in the PyTorch 20.10-py3 NGC container. Performance (in sequences per second) is the steady-state throughput.

##### Training performance: NVIDIA DGX A100 (8x A100 80GB)
| Batch size / GPU | GPUs | Throughput - TF32 | Throughput - mixed precision | Throughput speedup (TF32 to mixed precision) | Weak scaling - TF32 | Weak scaling - mixed precision |
|----:|----:|-------:|-------:|-----:|-----:|-----:|
|  32 |   1 |  42.18 |  64.32 | 1.52 | 1.00 | 1.00 |
|  32 |   4 | 157.49 | 239.23 | 1.52 | 3.73 | 3.72 |
|  32 |   8 | 310.10 | 470.09 | 1.52 | 7.35 | 7.31 |
|  64 |   1 |  49.64 |  75.59 | 1.52 | 1.00 | 1.00 |
|  64 |   4 | 192.66 | 289.16 | 1.50 | 3.88 | 3.83 |
|  64 |   8 | 371.41 | 547.91 | 1.48 | 7.48 | 7.25 |

Note: Mixed precision permits higher batch sizes during training. We report the maximum batch sizes (as powers of 2), which are allowed without gradient accumulation.

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

##### Training performance: NVIDIA DGX-1 (8x V100 16GB)
| Batch size / GPU | GPUs | Throughput - FP32 | Throughput - mixed precision | Throughput speedup (FP32 to mixed precision) | Weak scaling - FP32 | Weak scaling - mixed precision |
|----:|----:|------:|-------:|-----:|-----:|-----:|
|  16 |   1 | 10.71 |  27.87 | 2.60 | 1.00 | 1.00 |
|  16 |   4 | 40.28 |  99.80 | 2.48 | 3.76 | 3.58 |
|  16 |   8 | 78.23 | 193.89 | 2.48 | 7.30 | 6.96 |

Note: Mixed precision permits higher batch sizes during training. We report the maximum batch sizes (as powers of 2), which are allowed without gradient accumulation.

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

##### Training performance: NVIDIA DGX-1 (8x V100 32GB)
| Batch size / GPU | GPUs | Throughput - FP32 | Throughput - mixed precision | Throughput speedup (FP32 to mixed precision) | Weak scaling - FP32 | Weak scaling - mixed precision |
|----:|----:|------:|-------:|-----:|-----:|-----:|
|  32 |   1 | 12.22 |  34.08 | 2.79 | 1.00 | 1.00 |
|  32 |   4 | 46.97 | 128.39 | 2.73 | 3.84 | 3.77 |
|  32 |   8 | 92.44 | 249.00 | 2.69 | 7.57 | 7.31 |
|  64 |   1 |   N/A |  39.30 |  N/A |  N/A | 1.00 |
|  64 |   4 |   N/A | 150.18 |  N/A |  N/A | 3.82 |
|  64 |   8 |   N/A | 282.68 |  N/A |  N/A | 7.19 |

Note: Mixed precision permits higher batch sizes during training. We report the maximum batch sizes (as powers of 2), which are allowed without gradient accumulation.

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

##### Training performance: NVIDIA DGX-2 (16x V100 32GB)
| Batch size / GPU | GPUs | Throughput - FP32 | Throughput - mixed precision | Throughput speedup (FP32 to mixed precision) | Weak scaling - FP32 | Weak scaling - mixed precision |
|----:|----:|-------:|-------:|-----:|------:|------:|
|  32 |   1 |  13.46 |  38.94 | 2.89 |  1.00 |  1.00 |
|  32 |   4 |  51.38 | 143.44 | 2.79 |  3.82 |  3.68 |
|  32 |   8 | 100.54 | 280.48 | 2.79 |  7.47 |  7.20 |
|  32 |  16 | 188.14 | 515.90 | 2.74 | 13.98 | 13.25 |
|  64 |   1 |    N/A |  43.86 |  N/A |   N/A |  1.00 |
|  64 |   4 |    N/A | 165.27 |  N/A |   N/A |  3.77 |
|  64 |   8 |    N/A | 318.10 |  N/A |   N/A |  7.25 |
|  64 |  16 |    N/A | 567.47 |  N/A |   N/A | 12.94 |

Note: Mixed precision permits higher batch sizes during training. We report the maximum batch sizes (as powers of 2), which are allowed without gradient accumulation.

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.


#### Inference performance results
Our results were obtained by running the `scripts/inference_benchmark.sh` script in the PyTorch 20.10-py3 NGC container on NVIDIA DGX A100, DGX-1, DGX-2 and T4 on a single GPU. Performance numbers (latency in milliseconds per batch) were averaged over 500 iterations.

##### Inference performance: NVIDIA DGX A100 (1x A100 80GB)
|  |  | FP16 Latency (ms) Percentiles |  |  |  | TF32 Latency (ms) Percentiles |  |  |  | FP16/TF32 speed up |
|-----:|---------------:|------:|------:|------:|------:|------:|------:|-------:|------:|------:|
|   BS |   Duration (s) |   90% |   95% |   99% |   Avg |   90% |   95% |    99% |   Avg |   Avg |
|    1 |            2.0 | 32.40 | 32.50 | 32.82 | 32.30 | 33.30 | 33.64 |  34.65 | 33.25 |  1.03 |
|    2 |            2.0 | 32.90 | 33.51 | 34.35 | 32.69 | 34.48 | 34.65 |  35.66 | 34.27 |  1.05 |
|    4 |            2.0 | 32.85 | 33.01 | 33.89 | 32.60 | 34.09 | 34.46 |  35.22 | 34.00 |  1.04 |
|    8 |            2.0 | 35.51 | 35.89 | 37.10 | 35.33 | 34.86 | 35.36 |  36.08 | 34.45 |  0.98 |
|   16 |            2.0 | 36.00 | 36.57 | 37.40 | 35.77 | 43.83 | 44.12 |  44.77 | 43.39 |  1.21 |
|    1 |            7.0 | 33.50 | 33.99 | 34.91 | 33.03 | 33.83 | 34.25 |  34.95 | 33.70 |  1.02 |
|    2 |            7.0 | 34.43 | 34.89 | 35.72 | 34.22 | 34.41 | 34.73 |  35.69 | 34.28 |  1.00 |
|    4 |            7.0 | 34.30 | 34.59 | 35.43 | 34.07 | 37.95 | 38.18 |  38.87 | 37.55 |  1.10 |
|    8 |            7.0 | 35.98 | 36.28 | 37.11 | 35.28 | 44.64 | 44.79 |  45.37 | 44.29 |  1.26 |
|   16 |            7.0 | 39.86 | 40.08 | 41.16 | 39.33 | 55.17 | 55.46 |  57.24 | 54.56 |  1.39 |
|    1 |           16.7 | 35.20 | 35.80 | 38.71 | 34.36 | 35.36 | 35.76 |  36.55 | 34.64 |  1.01 |
|    2 |           16.7 | 35.40 | 35.81 | 36.50 | 34.76 | 36.34 | 36.53 |  37.40 | 35.87 |  1.03 |
|    4 |           16.7 | 36.01 | 36.38 | 37.37 | 35.57 | 44.69 | 45.09 |  45.88 | 43.92 |  1.23 |
|    8 |           16.7 | 41.48 | 41.78 | 44.22 | 40.69 | 58.57 | 58.74 |  59.62 | 58.11 |  1.43 |
|   16 |           16.7 | 61.37 | 61.93 | 66.32 | 60.92 | 97.33 | 97.71 | 100.04 | 96.56 |  1.59 |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

##### Inference performance: NVIDIA DGX-1 (1x V100 16GB)
|  |  | FP16 Latency (ms) Percentiles |  |  |  | FP32 Latency (ms) Percentiles |  |  |  | FP16/FP32 speed up |
|-----:|---------------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|------:|
|   BS |   Duration (s) |    90% |    95% |    99% |    Avg |    90% |    95% |    99% |    Avg |   Avg |
|    1 |            2.0 |  45.42 |  45.62 |  49.54 |  45.02 |  48.83 |  48.99 |  51.66 |  48.44 |  1.08 |
|    2 |            2.0 |  50.31 |  50.53 |  53.66 |  49.10 |  49.87 |  50.04 |  52.99 |  49.41 |  1.01 |
|    4 |            2.0 |  49.17 |  49.48 |  52.13 |  48.73 |  52.92 |  53.21 |  55.28 |  52.31 |  1.07 |
|    8 |            2.0 |  51.20 |  51.40 |  52.32 |  49.01 |  73.02 |  73.30 |  75.00 |  71.99 |  1.47 |
|   16 |            2.0 |  51.75 |  52.24 |  56.36 |  51.27 |  83.99 |  84.57 |  86.69 |  83.24 |  1.62 |
|    1 |            7.0 |  48.13 |  48.53 |  50.95 |  46.78 |  48.52 |  48.75 |  50.89 |  48.01 |  1.03 |
|    2 |            7.0 |  49.52 |  50.10 |  52.35 |  48.00 |  65.27 |  65.41 |  66.59 |  64.79 |  1.35 |
|    4 |            7.0 |  51.75 |  52.01 |  54.39 |  50.38 |  93.75 |  94.77 |  97.04 |  92.27 |  1.83 |
|    8 |            7.0 |  54.80 |  56.27 |  66.23 |  52.95 | 130.65 | 131.09 | 132.91 | 129.82 |  2.45 |
|   16 |            7.0 |  73.02 |  73.42 |  75.83 |  71.96 | 157.53 | 158.20 | 160.73 | 155.51 |  2.16 |
|    1 |           16.7 |  48.10 |  48.52 |  52.71 |  47.20 |  73.34 |  73.56 |  74.19 |  72.69 |  1.54 |
|    2 |           16.7 |  64.21 |  64.52 |  65.56 |  56.06 | 129.48 | 129.97 | 131.78 | 126.36 |  2.25 |
|    4 |           16.7 |  60.38 |  61.03 |  63.18 |  58.87 | 183.33 | 183.85 | 185.53 | 181.90 |  3.09 |
|    8 |           16.7 |  85.88 |  86.34 |  87.70 |  84.46 | 227.42 | 228.21 | 229.63 | 225.71 |  2.67 |
|   16 |           16.7 | 135.62 | 136.40 | 137.69 | 131.58 | 276.90 | 277.59 | 281.16 | 275.08 |  2.09 |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

##### Inference performance: NVIDIA DGX-1 (1x V100 32GB)
|  |  | FP16 Latency (ms) Percentiles |  |  |  | FP32 Latency (ms) Percentiles |  |  |  | FP16/FP32 speed up |
|-----:|---------------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|------:|
|   BS |   Duration (s) |    90% |    95% |    99% |    Avg |    90% |    95% |    99% |    Avg |   Avg |
|    1 |            2.0 |  52.74 |  53.01 |  54.40 |  51.47 |  55.97 |  56.22 |  57.93 |  54.93 |  1.07 |
|    2 |            2.0 |  51.77 |  52.15 |  54.69 |  50.98 |  56.58 |  56.87 |  58.88 |  55.35 |  1.09 |
|    4 |            2.0 |  51.41 |  51.76 |  53.47 |  50.55 |  61.56 |  61.87 |  63.81 |  60.74 |  1.20 |
|    8 |            2.0 |  51.83 |  52.15 |  54.08 |  50.85 |  80.20 |  80.69 |  81.67 |  77.69 |  1.53 |
|   16 |            2.0 |  70.48 |  70.96 |  72.11 |  62.98 |  93.00 |  93.44 |  94.17 |  89.05 |  1.41 |
|    1 |            7.0 |  49.77 |  50.21 |  51.88 |  48.73 |  52.74 |  52.99 |  54.54 |  51.67 |  1.06 |
|    2 |            7.0 |  51.12 |  51.47 |  52.84 |  49.98 |  65.33 |  65.63 |  67.07 |  64.64 |  1.29 |
|    4 |            7.0 |  53.13 |  53.56 |  55.68 |  52.15 |  93.54 |  93.85 |  94.72 |  92.76 |  1.78 |
|    8 |            7.0 |  57.67 |  58.07 |  59.89 |  56.41 | 133.93 | 134.18 | 134.88 | 133.15 |  2.36 |
|   16 |            7.0 |  76.09 |  76.48 |  79.13 |  75.27 | 162.35 | 162.77 | 164.63 | 161.30 |  2.14 |
|    1 |           16.7 |  54.78 |  55.29 |  56.83 |  52.51 |  75.37 |  76.27 |  78.05 |  74.32 |  1.42 |
|    2 |           16.7 |  56.80 |  57.20 |  59.01 |  55.49 | 130.60 | 131.36 | 132.93 | 128.55 |  2.32 |
|    4 |           16.7 |  64.19 |  64.84 |  66.47 |  62.87 | 188.09 | 188.76 | 190.07 | 185.76 |  2.95 |
|    8 |           16.7 |  87.46 |  87.86 |  89.99 |  86.47 | 232.33 | 232.89 | 234.43 | 230.44 |  2.67 |
|   16 |           16.7 | 136.02 | 136.52 | 139.44 | 134.78 | 283.87 | 284.59 | 286.70 | 282.01 |  2.09 |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

##### Inference performance: NVIDIA DGX-2 (1x V100 32GB)
|  |  | FP16 Latency (ms) Percentiles |  |  |  | FP32 Latency (ms) Percentiles |  |  |  | FP16/FP32 speed up |
|-----:|---------------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|------:|
|   BS |   Duration (s) |    90% |    95% |    99% |    Avg |    90% |    95% |    99% |    Avg |   Avg |
|    1 |            2.0 |  35.88 |  36.12 |  39.80 |  35.20 |  42.95 |  43.67 |  46.65 |  42.23 |  1.20 |
|    2 |            2.0 |  36.36 |  36.57 |  40.97 |  35.60 |  41.83 |  42.21 |  45.60 |  40.97 |  1.15 |
|    4 |            2.0 |  36.69 |  36.89 |  41.25 |  36.05 |  48.35 |  48.52 |  52.35 |  47.80 |  1.33 |
|    8 |            2.0 |  37.49 |  37.70 |  41.37 |  36.88 |  65.41 |  65.64 |  66.50 |  64.96 |  1.76 |
|   16 |            2.0 |  41.35 |  41.79 |  45.58 |  40.91 |  77.22 |  77.51 |  79.48 |  76.54 |  1.87 |
|    1 |            7.0 |  36.07 |  36.55 |  40.31 |  35.62 |  39.52 |  39.84 |  43.07 |  38.93 |  1.09 |
|    2 |            7.0 |  37.42 |  37.66 |  41.36 |  36.79 |  55.94 |  56.19 |  58.33 |  55.60 |  1.51 |
|    4 |            7.0 |  38.51 |  38.95 |  42.55 |  37.98 |  86.62 |  87.08 |  87.50 |  86.20 |  2.27 |
|    8 |            7.0 |  42.82 |  43.00 |  47.11 |  42.55 | 122.05 | 122.29 | 122.70 | 121.59 |  2.86 |
|   16 |            7.0 |  67.74 |  67.92 |  69.05 |  65.69 | 149.92 | 150.16 | 151.03 | 149.49 |  2.28 |
|    1 |           16.7 |  39.28 |  39.78 |  43.34 |  38.35 |  66.73 |  67.16 |  69.80 |  66.01 |  1.72 |
|    2 |           16.7 |  43.05 |  43.42 |  47.18 |  42.43 | 120.04 | 121.12 | 123.32 | 118.14 |  2.78 |
|    4 |           16.7 |  52.18 |  52.49 |  56.11 |  51.63 | 176.09 | 176.51 | 178.70 | 174.60 |  3.38 |
|    8 |           16.7 |  78.55 |  78.79 |  81.66 |  78.04 | 216.19 | 216.68 | 217.63 | 214.48 |  2.75 |
|   16 |           16.7 | 125.57 | 125.92 | 128.78 | 124.33 | 264.11 | 264.49 | 266.14 | 262.80 |  2.11 |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

##### Inference performance: NVIDIA T4
|  |  | FP16 Latency (ms) Percentiles |  |  |  | FP32 Latency (ms) Percentiles |  |  |  | FP16/FP32 speed up |
|-----:|---------------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|------:|
|   BS |   Duration (s) |    90% |    95% |    99% |    Avg |    90% |    95% |    99% |    Avg |   Avg |
|    1 |            2.0 |  43.62 |  46.95 |  50.46 |  37.23 |  51.31 |  52.37 |  56.21 |  49.77 |  1.34 |
|    2 |            2.0 |  49.09 |  50.46 |  53.11 |  40.61 |  81.85 |  82.22 |  83.94 |  80.81 |  1.99 |
|    4 |            2.0 |  47.71 |  51.14 |  55.09 |  41.29 | 112.56 | 115.13 | 118.56 | 111.60 |  2.70 |
|    8 |            2.0 |  51.37 |  53.11 |  55.48 |  45.94 | 198.95 | 199.48 | 200.28 | 197.22 |  4.29 |
|   16 |            2.0 |  63.59 |  64.30 |  66.90 |  61.77 | 221.75 | 222.07 | 223.22 | 220.09 |  3.56 |
|    1 |            7.0 |  47.49 |  48.66 |  53.36 |  40.76 |  73.63 |  74.41 |  77.65 |  72.41 |  1.78 |
|    2 |            7.0 |  48.63 |  50.01 |  58.35 |  43.44 | 114.66 | 115.28 | 117.63 | 112.41 |  2.59 |
|    4 |            7.0 |  52.19 |  52.85 |  54.22 |  49.94 | 200.38 | 201.29 | 202.97 | 197.21 |  3.95 |
|    8 |            7.0 |  84.90 |  85.56 |  87.52 |  83.41 | 404.00 | 404.72 | 405.70 | 400.25 |  4.80 |
|   16 |            7.0 | 157.12 | 157.58 | 159.19 | 155.01 | 490.93 | 492.09 | 493.44 | 486.45 |  3.14 |
|    1 |           16.7 |  50.57 |  51.57 |  57.58 |  46.27 | 150.39 | 151.84 | 153.54 | 147.31 |  3.18 |
|    2 |           16.7 |  63.64 |  64.55 |  66.31 |  61.98 | 256.54 | 258.16 | 262.71 | 250.34 |  4.04 |
|    4 |           16.7 | 140.44 | 141.06 | 142.00 | 138.14 | 519.59 | 521.41 | 523.86 | 512.74 |  3.71 |
|    8 |           16.7 | 267.03 | 268.06 | 270.01 | 263.15 | 727.33 | 728.61 | 731.36 | 722.62 |  2.75 |
|   16 |           16.7 | 362.40 | 364.02 | 367.80 | 358.75 | 867.92 | 869.19 | 871.46 | 860.37 |  2.40 |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

## Release notes
We're constantly refining and improving our performance on AI and HPC workloads even on the same hardware with frequent updates to our software stack. For our latest performance data please refer to these pages for AI and HPC benchmarks.

### Changelog
February 2021
* Added DALI data-processing pipeline for on-the-fly data processing and augmentation on CPU or GPU
* Revised training recipe: ~10% relative improvement in Word Error Rate (WER)
* Updated Triton scripts for compatibility with Triton V2 API, updated Triton inference results
* Refactored codebase
* Updated performance results for the PyTorch 20.10-py3 NGC container

June 2020
* Updated performance tables to include A100 results

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

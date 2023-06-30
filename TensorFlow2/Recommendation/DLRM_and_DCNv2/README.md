# DLRM and DCNv2 for TensorFlow 2

This repository provides recipes to train and deploy two ranking models – DLRM and DCNv2.
This document provides instructions on how to run those models and a description of the features implemented.
Detailed instructions for reproducing, as well as benchmark results and descriptions of the respective architectures, can be found in:

* [doc/DLRM.md](doc/DLRM.md) for DLRM
* [doc/DCNv2.md](doc/DCNv2.md) for DCNv2


## Table Of Contents

   * [Overview](#overview)
      * [Default configuration](#default-configuration)
      * [Feature support matrix](#feature-support-matrix)
         * [Features](#features)
      * [Mixed precision training](#mixed-precision-training)
         * [Enabling mixed precision](#enabling-mixed-precision)
         * [Enabling TF32](#enabling-tf32)
      * [Hybrid-parallel training with Merlin Distributed Embeddings](#hybrid-parallel-training-with-merlin-distributed-embeddings)
         * [Training very large embedding tables](#training-very-large-embedding-tables)
         * [Multi-node training](#multi-node-training)
      * [Preprocessing on GPU with Spark 3](#preprocessing-on-gpu-with-spark-3)
      * [BYO dataset functionality overview](#byo-dataset-functionality-overview)
   * [Setup](#setup)
      * [Requirements](#requirements)
   * [Advanced](#advanced)
      * [Scripts and sample code](#scripts-and-sample-code)
      * [Parameters](#parameters)
      * [Command-line options](#command-line-options)
      * [Getting the Data](#getting-the-data)
      * [Inference deployment](#inference-deployment)
   * [Release notes](#release-notes)
      * [Changelog](#changelog)


## Overview

This directory contains Deep Learning Recommendation Model (DLRM) and Deep Cross Network version 2 (DCNv2).
Both are recommendation models designed to use categorical and numerical inputs.

Using the scripts provided here, you can efficiently train models too large to fit into a single GPU.
This is because we use a hybrid-parallel approach, which combines model parallelism with data parallelism for
different parts of the neural network.
This is explained in detail in the [next section](#hybrid-parallel-training-with-merlin-distributed-embeddings).

Using DLRM or DCNv2, you can train a high-quality general model for recommendations.

Both models in this directory are trained with mixed precision using Tensor Cores on NVIDIA Volta, NVIDIA Turing, and NVIDIA Ampere GPU architectures.
Therefore, researchers can get results 2x faster than training without Tensor Cores while experiencing the
benefits of mixed precision training. This model is tested against each NGC monthly container
release to ensure consistent accuracy and performance over time.


### Default configuration

The following features were implemented:
- general
    - static loss scaling for Tensor Cores (mixed precision) training
    - hybrid-parallel multi-GPU training using Merlin Distributed Embeddings
- inference
    - inference using Merlin HPS, Triton ensembles and TensorRT
- preprocessing
    - dataset preprocessing using Spark 3 on GPUs

### Feature support matrix

The following features are supported by this model:

| Feature               | DLRM and DCNv2
|----------------------|--------------------------
|Hybrid-parallel training with Merlin Distributed Embeddings | Yes
|Multi-node training | Yes
|Triton inference with TensorRT and Merlin Hierarchical Parameter Server | Yes
|Automatic mixed precision (AMP)   | Yes
|XLA | Yes
|Preprocessing on GPU with Spark 3| Yes
|Inference using NVIDIA Triton | Yes


#### Features

**Automatic Mixed Precision (AMP)**
Enables mixed precision training without any changes to the code-base by performing automatic graph rewrites and loss scaling controlled by an environmental variable.

**XLA**

The training script supports a `--xla` flag. It can be used to enable XLA JIT compilation. Currently, we use [XLA Lite](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html#xla-lite). It delivers a steady 10-30% performance boost depending on your hardware platform, precision, and the number of GPUs. It is turned off by default.

**Horovod**
Horovod is a distributed training framework for TensorFlow, Keras, PyTorch, and MXNet. The goal of Horovod is to make distributed deep learning fast and easy to use. For more information about how to get started with Horovod, refer tothe Horovod [official repository](https://github.com/horovod/horovod).

**Hybrid-parallel training with Merlin Distributed Embeddings**
Our model uses Merlin Distributed Embeddings to implement efficient multi-GPU training.
For details, refer to the example sources in this repository or refer to the TensorFlow tutorial.
For a detailed description of our multi-GPU approach, visit this [section](#hybrid-parallel-training-with-merlin-distributed-embeddings).

**Multi-node training**
This repository supports multi-node training. For more information, refer to the [multinode section](#multi-node-training)

**Merlin Hierarchical Parameter server (HPS)**
This repository supports inference with Merlin HPS. For more information, refer to [doc/inference.md](doc/inference.md).


### Mixed precision training

Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in NVIDIA Volta, and following with both the NVIDIA Turing and NVIDIA Ampere architectures, significant training speedups are experienced by switching to mixed precision – up to 3.4x overall speedup on the most arithmetically intense model architectures. Using mixed precision training requires two steps:
1.  Porting the model to use the FP16 data type where appropriate.
2.  Adding loss scaling to preserve small gradient values.

The ability to train deep learning networks with lower precision was introduced in the Pascal architecture and first supported in [CUDA 8](https://devblogs.nvidia.com/parallelforall/tag/fp16/) in the NVIDIA Deep Learning SDK.

For information about:
-   How to train using mixed precision, refer to the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) documentation.
-   Techniques used for mixed precision training, refer to the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.

#### Enabling mixed precision

Mixed precision training is turned off by default. To turn it on, issue the `--amp` flag to the `dlrm.py` or `dcnv2.py` script.


#### Enabling TF32

TensorFloat-32 (TF32) is the new math mode in [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) GPUs for handling the matrix math also called tensor operations. TF32 running on Tensor Cores in A100 GPUs can provide up to 10x speedups compared to single-precision floating-point math (FP32) on Volta GPUs.

TF32 Tensor Cores can speed up networks using FP32, typically with no loss of accuracy. It is more robust than FP16 for models which require high dynamic range for weights or activations.

For more information, refer to the [TensorFloat-32 in the A100 GPU Accelerates AI Training, HPC up to 20x](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) blog post.

TF32 is supported in the NVIDIA Ampere GPU architecture and is enabled by default.


### Hybrid-parallel training with Merlin Distributed Embeddings

Many recommendation models contain very large embedding tables. As a result, the model is often too large to fit onto a single device.
This could be easily solved by training in a model-parallel way, using either the CPU or other GPUs as "memory donors."
However, this approach is suboptimal as the "memory donor" devices' compute is not utilized.
In this repository, we use the model-parallel approach for the Embedding Tables while employing a usual data-parallel approach
for the more compute-intensive MLPs and Dot Interaction layer. This way, we can train models much larger than what would normally fit into
a single GPU while at the same time making the training faster by using multiple GPUs. We call this approach hybrid-parallel training.

To implement this approach, we use the [Merlin Distributed Embeddings](https://github.com/NVIDIA-Merlin/distributed-embeddings) library. 
It provides a scalable model parallel wrapper called `distributed_embeddings.dist_model_parallel`. This wrapper automatically distributes embedding tables to multiple GPUs.
This way, embeddings can be scaled beyond a single GPU’s memory capacity without
complex code to handle cross-worker communication.

Under the hood, Merlin Distributed Embeddings uses a
specific multi-GPU communication pattern called
[all-2-all](https://en.wikipedia.org/wiki/All-to-all_\(parallel_pattern\)) to transition from model-parallel to data-parallel
paradigm. In the [original DLRM whitepaper](https://arxiv.org/abs/1906.00091), this is referred to as "butterfly shuffle."

An example model using Hybrid Parallelism is shown in Figure 2. The compute-intensive dense layers are run in data-parallel
mode. The smaller embedding tables are run model-parallel, so each smaller table is placed entirely on a single device.
This is not suitable for larger tables that need more memory than can be provided by a single device. Therefore,
those large tables are split into multiple parts and each part is run on a different GPU.

<p align="center">
  <img width="100%" src="./doc/img/hybrid_parallel.svg" />
  <br>
Figure 2. Hybrid parallelism with Merlin Distributed Embeddings.
</p>

In this repository, for both DLRM and DCNv2,
we train models of three sizes: "small" (15.6 GiB), "large" (84.9 GiB), and "extra large" (421 GiB).
The "small" model can be trained on a single V100-32GB GPU. The "large" model needs at least 8xV100-32GB GPUs,
but each table  can fit on a single GPU. 

The "extra large" model, on the other hand, contains tables that do not fit into a single device and will be automatically
split and stored across multiple GPUs by Merlin Distributed Embeddings.

#### Training very large embedding tables

We tested this approach by training a DLRM model on the Criteo Terabyte dataset with the frequency limiting option turned off (set to zero).
The weights of the resulting model take 421 GiB. The largest table weighs 140 GiB.
Here are the commands you can use to reproduce this:

```
# build and run the preprocessing container as in the Quick Start Guide
# then when preprocessing set the frequency limit to 0:
./prepare_dataset.sh DGX2 0

# build and run the training container same as in the Quick Start Guide
# then append options necessary for training very large embedding tables:
horovodrun -np 8 -H localhost:8 --mpi-args=--oversubscribe numactl --interleave=all -- python -u dlrm.py --dataset_path /data/dlrm/ --amp --xla
```

When using this method on a DGX A100 with 8 A100-80GB GPUs and a large-enough dataset, it is possible to train a single embedding table of up to 600 GB. You can also use multi-node training (described below) to train even larger recommender systems.


#### Multi-node training

Multi-node training is supported. Depending on the exact interconnect hardware and model configuration,
you might experience only a modest speedup with multi-node.
Multi-node training can also be used to train larger models.
For example, to train a 1.68 TB variant of DLRM on multi-node, you can run:

```
cmd='numactl --interleave=all -- python -u dlrm.py --dataset_path /data/dlrm/full_criteo_data --amp --xla\
--embedding_dim 512 --bottom_mlp_dims 512,256,512' \
srun_flags='--mpi=pmix' \
cont=nvidia_dlrm_tf \
mounts=/data/dlrm:/data/dlrm \
sbatch -n 32 -N 4 -t 00:20:00 slurm_multinode.sh
```

### Preprocessing on GPU with Spark 3

Refer to the [preprocessing documentation](doc/criteo_dataset.md#advanced) for a detailed description of the Spark 3 GPU functionality.


### BYO dataset functionality overview

Refer to the [BYO Dataset summary](doc/multidataset.md) for details.

### Inference using NVIDIA Triton

The [deployment](deployment) directory contains two examples of deploying recommender models larger than single GPU memory. Both use the NVIDIA Triton Inference Server.
1. For the example with Merlin Hierarchical Parameter Server and TensorRT,
refer to [detailed documentation](doc/merlin_hps_inference.md)
2. For the example with TensorFlow SavedModel and TensorRT
3. Refer to [detailed documentation](doc/tensorflow_inference.md)

## Setup

The following section lists the requirements for training DLRM and DCNv2.

### Requirements

This repository contains Dockerfile that extends the TensorFlow 2 NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:
-   [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
-   [TensorFlow 2  23.02-py3](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow/tags) NGC container
-   Supported GPUs:
    - [NVIDIA Volta architecture](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
    - [NVIDIA Turing architecture](https://www.nvidia.com/en-us/geforce/turing/)
    - [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)


For more information about how to get started with NGC containers, refer to the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
-   [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
-   [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)
- [Running TensorFlow](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html#running)

For those unable to use the TensorFlow NGC container, to set up the required environment or create your own container, refer to the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## Advanced

The following sections provide more details of the dataset, running training and inference, and the training results.

### Scripts and sample code

These are the important modules in this repository:
- `dlrm.py` - The script for training DLRM. Wrapper around `main.py`.
- `dcnv2.py` - The script for training DCNv2. Wrapper around `main.py`.
- `main.py` - Contains common code for training and evaluating DLRM and DCNv2 (e.g., the training loop)
- `Dockerfile` - defines the docker image used for training DLRM and DCNv2.
- `nn/model.py` - Contains the definition of the full neural network, which can be used to create DLRM and DCNv2.
- `nn/dense_model.py` - Defines the "dense" part of DLRM and DCNv2 (Bottom MLP, Interaction, Top MLP).
- `nn/sparse_model.py` - Defines the "sparse" part of DLRM and DCNv2 (Embedding layers).
- `nn/trainer.py` - Defines a single training step (forward, backward, weight update).
- `nn/embedding.py` - Implementations of the embedding layers.
- `nn/lr_scheduler.py` - Defines a TensorFlow learning rate scheduler that supports learning rate warmup and polynomial decay.
- `deployment/deploy.py` - The script used for creating the Triton model store for inference.
- `deployment/evaluate_latency.py` - The script used to evaluate the latency of deployed Triton DLRM and DCNv2 models.
- `deployment/evaluate_accuracy.py` - The script used to evaluate the accuracy of deployed Triton DLRM and DCNv2 models.
- `dataloading/dataloader.py` - Handles defining the dataset objects based on command-line flags.
- `dataloading/datasets.py` - Defines the `TfRawBinaryDataset` class responsible for storing and loading the training data.
- `preproc` - directory containing source code for preprocessing the Criteo 1TB Dataset.
- `slurm_multinode.sh` - Example batch script for multi-node training on SLURM clusters.
- `tensorflow-dot-based-interact` - A directory with a set of custom CUDA kernels. They provide fast implementations of the dot-interaction operation for various precisions and hardware platforms.
- `utils.py` - General utilities, such as a timer used for taking performance measurements.


### Parameters

The table below lists the most important command-line parameters of the `main.py` script.

| Scope| parameter| Comment| Default Value |
| ----- | --- | ---- | ---- |
|datasets|dataset_path|Path to the JSON file with the sizes of embedding tables|
|function|mode| Choose "train" to train the model, "inference" to benchmark inference and "eval" to run validation| train|
|optimizations|amp| Enable automatic mixed precision| False
|optimizations|xla| Enable XLA| False|
|hyperparameters|batch_size| Batch size used for training|65536|
|hyperparameters|epochs| Number of epochs to train for|1|
|hyperparameters|optimizer| Optimization algorithm for training |SGD|
|hyperparameters|evals_per_epoch| Number of evaluations per epoch|1|
|hyperparameters|valid_batch_size| Batch size used for validation|65536|
|hyperparameters|max_steps| Stop the training/inference after this many optimization steps|-1|
|checkpointing|restore_checkpoint_path| Path from which to restore a checkpoint before training|None|
|checkpointing|save_checkpoint_path| Path to which to save a checkpoint file at the end of the training|None|
|debugging|run_eagerly| Disable all tf.function decorators for debugging|False|
|debugging|print_freq| Number of steps between debug prints|1000|
|debugging|max_steps| Exit early after performing a prescribed number of steps|None|


### Command-line options

The training script supports a number of command-line flags.
You can get the descriptions of those, for example, by running `python dlrm.py --help`.

### Getting the Data
Refer to:

* [doc/criteo_dataset.md](doc/criteo_dataset.md) for information on how to run on the Criteo 1TB dataset.
* [doc/multidataset.md](doc/multidataset.md) for information on training with your own dataset.


## Release notes
We’re constantly refining and improving our performance on AI and HPC workloads, even on the same hardware, with frequent updates to our software stack. For our latest performance data, refer to these pages for [AI](https://developer.nvidia.com/deep-learning-performance-training-inference) and [HPC](https://developer.nvidia.com/hpc-application-performance) benchmarks.

### Changelog
June 2023
- Support and performance numbers for DCNv2
- Support inference deployment using NVIDIA Merlin HPS, NVIDIA Triton, and NVIDIA TensorRT for DLRM and DCNv2
- Major refactoring and usability improvements

July 2022
- Start using Merlin Distributed Embeddings  

March 2022
- Major performance improvements
- Support for BYO dataset

March 2021
- Initial release

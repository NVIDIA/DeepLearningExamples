# nnU-Net For PyTorch

This repository provides a script and recipe to train the nnU-Net model to achieve state-of-the-art accuracy and is tested and maintained by NVIDIA.

## Table Of Contents

- [Model overview](#model-overview)
    * [Model architecture](#model-architecture)
    * [Default configuration](#default-configuration)
    * [Feature support matrix](#feature-support-matrix)
        * [Features](#features)
    * [Mixed precision training](#mixed-precision-training)
        * [Enabling mixed precision](#enabling-mixed-precision)
        * [TF32](#tf32)
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
        * [Multi-dataset](#multi-dataset)
    * [Training process](#training-process)
    * [Inference process](#inference-process)
- [Performance](#performance)
    * [Benchmarking](#benchmarking)
        * [Training performance benchmark](#training-performance-benchmark)
        * [Inference performance benchmark](#inference-performance-benchmark)
    * [Results](#results)
        * [Training accuracy results](#training-accuracy-results)             
            * [Training accuracy: NVIDIA DGX-1 (8x V100 16GB)](#training-accuracy-nvidia-dgx-1-8x-v100-16gb)
        * [Training performance results](#training-performance-results)
            * [Training performance: NVIDIA DGX A100 (8x A100 40GB)](#training-performance-nvidia-dgx-a100-8x-a100-40gb) 
            * [Training performance: NVIDIA DGX-1 (8x V100 16GB)](#training-performance-nvidia-dgx-1-8x-v100-16gb)
            * [Training performance: NVIDIA DGX-2 (16x V100 32GB)](#training-performance-nvidia-dgx-2-16x-v100-32gb)
        * [Inference performance results](#inference-performance-results)
            * [Inference performance: NVIDIA DGX A100 (1x A100 40GB)](#inference-performance-nvidia-dgx-a100-1x-a100-40gb)
            * [Inference performance: NVIDIA DGX-1 (1x V100 16GB)](#inference-performance-nvidia-dgx-1-1x-v100-16gb)
- [Release notes](#release-notes)
    * [Changelog](#changelog)
    * [Known issues](#known-issues)

## Model overview
    
The nnU-Net ("no-new-Net") refers to a robust and self-adapting framework for U-Net based medical image segmentation. This repository contains a nnU-Net implementation as described in the paper: [nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation](https://arxiv.org/abs/1809.10486).
    
The differences between this nnU-net and [original model](https://github.com/MIC-DKFZ/nnUNet) are:
  - Dynamic selection of patch size and spacings for low resolution U-Net are not supported and they need to be set in `data_preprocessing/configs.py` file.
  - Cascaded U-Net is not supported.
  - The following data augmentations are not used: rotation, simulation of low resolution, gamma augmentation.
    
This model is trained with mixed precision using Tensor Cores on Volta, Turing, and the NVIDIA Ampere GPU architectures. Therefore, researchers can get results 2x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.

### Model architecture
    
The nnU-Net allows training two types of networks: 2D U-Net and 3D U-Net to perform semantic segmentation of 3D images, with high accuracy and performance.
    
The following figure shows the architecture of the 3D U-Net model and its different components. U-Net is composed of a contractive and an expanding path, that aims at building a bottleneck in its center-most part through a combination of convolution, instance norm and leaky relu operations. After this bottleneck, the image is reconstructed through a combination of convolutions and upsampling. Skip connections are added with the goal of helping the backward flow of gradients in order to improve the training.

<img src="images/unet3d.png" width="900"/>
    
*Figure 1: The 3D U-Net architecture*

### Default configuration

All convolution blocks in U-Net in both encoder and decoder are using two convolution layers followed by instance normalization and a leaky ReLU nonlinearity. For downsampling we are using strided convolution whereas transposed convolution for upsampling.

All models were trained with RAdam optimizer, learning rate 0.001 and weight_decay 0.0001. For loss function we use the average of [cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy) and [dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient).

Early stopping is triggered if validation dice score wasn't improved during the last 100 epochs.

Used data augmentation: crop with oversampling the foreground class, mirroring, zoom, gaussian noise, gaussian blur, brightness.

### Feature support matrix

The following features are supported by this model: 

| Feature               | nnUNet               
|-----------------------|--------------------------   
|[DALI](https://docs.nvidia.com/deeplearning/dali/release-notes/index.html) | Yes
|Automatic mixed precision (AMP)   | Yes  
|Distributed data parallel (DDP)   | Yes
         
#### Features

**DALI**

NVIDIA DALI - DALI is a library accelerating data preparation pipeline. To accelerate your input pipeline, you only need to define your data loader
with the DALI library. For details, see example sources in this repository or see the [DALI documentation](https://docs.nvidia.com/deeplearning/dali/index.html)

**Automatic Mixed Precision (AMP)**

This implementation uses native PyTorch AMP implementation of mixed precision training. It allows us to use FP16 training with FP32 master weights by modifying just a few lines of code.

**DistributedDataParallel (DDP)**

The model uses PyTorch Lightning implementation of distributed data parallelism at the module level which can run across multiple machines.
    
### Mixed precision training

Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format, while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in Volta, and following with both the Turing and Ampere architectures, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using mixed precision training requires two steps:

1. Porting the model to use the FP16 data type where appropriate.
2. Adding loss scaling to preserve small gradient values.

The ability to train deep learning networks with lower precision was introduced in the Pascal architecture and first supported in [CUDA 8](https://devblogs.nvidia.com/parallelforall/tag/fp16/) in the NVIDIA Deep Learning SDK.

For information about:
* How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) documentation.
* Techniques used for mixed precision training, see the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
* APEX tools for mixed precision training, see the [NVIDIA Apex: Tools for Easy Mixed-Precision Training in PyTorch](https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/).

#### Enabling mixed precision
    
For training and inference, mixed precision can be enabled by adding the `--amp` flag. Mixed precision is using [native Pytorch implementation](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/).

#### TF32

TensorFloat-32 (TF32) is the new math mode in [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) GPUs for handling the matrix math also called tensor operations. TF32 running on Tensor Cores in A100 GPUs can provide up to 10x speedups compared to single-precision floating-point math (FP32) on Volta GPUs. 

TF32 Tensor Cores can speed up networks using FP32, typically with no loss of accuracy. It is more robust than FP16 for models which require high dynamic range for weights or activations.

For more information, refer to the [TensorFloat-32 in the A100 GPU Accelerates AI Training, HPC up to 20x](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) blog post.

TF32 is supported in the NVIDIA Ampere GPU architecture and is enabled by default.

### Glossary

**Test time augmentation**

Test time augmentation is an inference technique which averages predictions from augmented images with its prediction. As a result, predictions are more accurate, but with the cost of slower inference process. For nnU-Net, we use all possible flip combinations for image augmenting. Test time augmentation can be enabled by adding the `--tta` flag.

**Deep supervision**

Deep supervision is a technique which adds auxiliary loss in U-Net decoder. For nnU-Net, we add auxiliary losses to all but the lowest two decoder levels. Final loss is the weighted average of losses. Deep supervision can be enabled by adding the `--deep_supervision` flag.

## Setup

The following section lists the requirements that you need to meet in order to start training the nnU-Net model.

### Requirements

This repository contains Dockerfile which extends the PyTorch NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:
-   [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
-   PyTorch 20.12 NGC container
-   Supported GPUs:
    - [NVIDIA Volta architecture](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
    - [NVIDIA Turing architecture](https://www.nvidia.com/en-us/geforce/turing/)
    - [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)

For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
-   [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
-   [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)
-   Running [PyTorch](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html#running)
  
For those unable to use the PyTorch NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## Quick Start Guide

To train your model using mixed or TF32 precision with Tensor Cores or using FP32, perform the following steps using the default parameters of the nnUNet model on the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) dataset. For the specifics concerning training and inference, see the [Advanced](#advanced) section.

1. Clone the repository.

Executing this command will create your local repository with all the code to run nnU-Net.
```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/Pytorch/Segmentation/nnunet_pyt
```
    
2. Build the nnU-Net PyTorch NGC container.
    
This command will use the Dockerfile to create a Docker image named `nnunet_pyt`, downloading all the required components automatically.

```
docker build -t nnunet_pyt .
```
    
The NGC container contains all the components optimized for usage on NVIDIA hardware.
    
3. Start an interactive session in the NGC container to run preprocessing/training/inference.
    
The following command will launch the container and mount the `./data` directory as a volume to the `/data` directory inside the container, and `./results` directory to the `/results` directory in the container.
    
```
mkdir data results
docker run -it --runtime=nvidia --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 --rm -v ${PWD}/data:/data -v ${PWD}/results:/results nnunet_pyt:latest /bin/bash
```

4. Prepare BraTS dataset.

To download dataset run:

```
python download.py --task 01
```

then to preprocess 2D or 3D dataset version run:

```
python preprocess.py --task 01 --dim {2,3}
```

If you have prepared both 2D and 3D datasets then `ls /data` should print:
```
01_3d 01_2d Task01_BrainTumour
```

For the specifics concerning data preprocessing, see the [Getting the data](#getting-the-data) section.
    
5. Start training.
   
Training can be started with:
```
python scripts/train.py --gpus <gpus> --fold <fold> --dim <dim> [--amp]
```

Where:
```
--gpus             number of gpus
--fold             fold number, possible choices: `0, 1, 2, 3, 4`
--dim              U-Net dimension, possible choices: `2, 3`
--amp              enable automatic mixed precision
```
You can customize the training process. For details, see the [Training process](#training-process) section.

6. Start benchmarking.

The training and inference performance can be evaluated by using benchmarking scripts, such as:
 
```
python scripts/benchmark.py --mode {train, predict} --gpus <ngpus> --dim {2,3} --batch_size <bsize> [--amp] 
```

 which will make the model run and report the performance.


7. Start inference/predictions.
   
Inference can be started with:
```
python scripts/inference.py --dim <dim> --fold <fold> --ckpt_path <path/to/checkpoint> [--amp] [--tta] [--save_preds]
```

Where:
```
--dim              U-Net dimension. Possible choices: `2, 3`
--fold             fold number. Possible choices: `0, 1, 2, 3, 4`
--val_batch_size   batch size (default: 4)
--ckpt_path        path to checkpoint
--amp              enable automatic mixed precision
--tta              enable test time augmentation
--save_preds       enable saving prediction masks
```
You can customize the inference process. For details, see the [Inference process](#inference-process) section.

Now that you have your model trained and evaluated, you can choose to compare your training results with our [Training accuracy results](#training-accuracy-results). You can also choose to benchmark yours performance to [Training performance benchmark](#training-performance-results), or [Inference performance benchmark](#inference-performance-results). Following the steps in these sections will ensure that you achieve the same accuracy and performance results as stated in the [Results](#results) section.
    
## Advanced

The following sections provide greater details of the dataset, running training and inference, and the training results.

### Scripts and sample code

In the root directory, the most important files are:

* `main.py`: Entry point to the application. Runs training, evaluation, inference or benchmarking.
* `preprocess.py`: Entry point to data preprocessing.
* `download.py`: Downloads given dataset from [Medical Segmentation Decathlon](http://medicaldecathlon.com/).
* `Dockerfile`: Container with the basic set of dependencies to run nnU-Net.
* `requirements.txt:` Set of extra requirements for running nnU-Net.
    
The `data_preprocessing/` folder contains information about the data preprocessing used by nnU-Net. Its contents are:
    
* `configs.py`: Defines dataset configuration like patch size or spacing.
* `preprocessor.py`: Implements data preprocessing pipeline.
* `convert2tfrec.py`: Implements conversion from numpy files to tfrecords.
    
The `data_loading/` folder contains information about the data pipeline used by nnU-Net. Its contents are:
    
* `data_module.py`: Defines `LightningDataModule` used by PyTorch Lightning.
* `dali_loader.py`: Implements DALI data loader.
    
The `model/` folder contains information about the building blocks of nnU-Net and the way they are assembled. Its contents are:
    
* `layers.py`: Implements convolution blocks used by U-Net template.
* `metrics.py`: Implements metrics and loss function.
* `nn_unet.py`: Implements training/validation/test logic and dynamic creation of U-Net architecture used by nnU-Net.
* `unet.py`: Implements the U-Net template.
    
The `utils/` folder includes:
* `utils.py`: Defines some utility functions e.g. parser initialization.
* `logger.py`: Defines logging callback for performance benchmarking.
    
Other folders included in the root directory are:

* `images/`: Contains a model diagram.
* `scripts/`: Provides scripts for data preprocessing, training, benchmarking and inference of nnU-Net.

### Parameters

The complete list of the available parameters for the `main.py` script contains:

  * `--exec_mode`: Select the execution mode to run the model (default: `train`). Modes available:
    - `train` - Trains model with validation evaluation after every epoch.
    - `evaluate` - Loads checkpoint and performs evaluation on validation set (requires `--fold`).
    - `predict` - Loads checkpoint and runs inference on the validation set. If flag `--save_preds` is also provided then stores the predictions in the `--results_dir` directory.
  * `--data`: Path to data directory (default: `/data`)
  * `--results` Path to results directory (default: `/results`)
  * `--logname` Name of dlloger output (default: `None`)
  * `--task` Task number. MSD uses numbers 01-10"
  * `--gpus`: Number of GPUs (default: `1`)
  * `--dim`: U-Net dimension (default: `3`)
  * `--amp`: Enable automatic mixed precision (default: `False`)
  * `--negative_slope` Negative slope for LeakyReLU (default: `0.01`)
  * `--fold`: Fold number (default: `0`)
  * `--nfolds`: Number of cross-validation folds (default: `5`)
  * `--patience`: Early stopping patience (default: `50`)
  * `--min_epochs`: Force training for at least these many epochs (default: `100`)
  * `--max_epochs`: Stop training after this number of epochs (default: `10000`)
  * `--batch_size`: Batch size (default: `2`)
  * `--val_batch_size`: Validation batch size (default: `4`)
  * `--tta`: Enable test time augmentation (default: `False`)
  * `--deep_supervision`: Enable deep supervision (default: `False`)
  * `--benchmark`: Run model benchmarking (default: `False`)
  * `--norm`: Normalization layer, one from: {`instance,batch,group`} (default: `instance`)
  * `--oversampling`: Probability of cropped area to have foreground pixels (default: `0.33`)
  * `--optimizer`: Optimizer, one from: {`sgd,adam,adamw,radam,fused_adam`} (default: `radam`)
  * `--learning_rate`: Learning rate (default: `0.001`)
  * `--momentum`: Momentum factor (default: `0.99`)
  * `--scheduler`: Learning rate scheduler, one from: {`none,multistep,cosine,plateau`} (default: `none`)
  * `--steps`: Steps for multi-step scheduler (default: `None`)
  * `--factor`: Factor used by `multistep` and `reduceLROnPlateau` schedulers (default: `0.1`)
  * `--lr_patience`: Patience for ReduceLROnPlateau scheduler (default: `75`)
  * `--weight_decay`: Weight decay (L2 penalty) (default: `0.0001`)
  * `--seed`: Random seed (default: `1`)
  * `--num_workers`: Number of subprocesses to use for data loading (default: `8`)
  * `--resume_training`: Resume training from the last checkpoint (default: `False`)
  * `--overlap`: Amount of overlap between scans during sliding window inference (default: `0.25`)
  * `--val_mode`: How to blend output of overlapping windows one from: {`gaussian,constant`} (default: `gaussian`)
  * `--ckpt_path`: Path to checkpoint
  * `--save_preds`: Enable prediction saving (default: `False`)
  * `--warmup`:  Warmup iterations before collecting statistics for model benchmarking. (default: `5`)
  * `--train_batches`: Limit number of batches for training (default: 0)
  * `--test_batches`: Limit number of batches for evaluation/inference (default: 0)
  * `--affinity`: Type of CPU affinity (default: `socket_unique_interleaved`)
  * `--save_ckpt`: Enable saving checkpoint (default: `False`)
  * `--gradient_clip_val`: Gradient clipping value (default: `0`)

### Command-line options

To see the full list of available options and their descriptions, use the `-h` or `--help` command-line option, for example:

`python main.py --help`

The following example output is printed when running the model:

```
usage: main.py [-h] [--exec_mode {train,evaluate,predict}] [--data DATA] [--results RESULTS] [--logname LOGNAME] [--task TASK] [--gpus GPUS] [--num_nodes NUM_NODES] [--learning_rate LEARNING_RATE] [--gradient_clip_val GRADIENT_CLIP_VAL] [--accumulate_grad_batches ACCUMULATE_GRAD_BATCHES] [--negative_slope NEGATIVE_SLOPE] [--tta] [--amp] [--benchmark] [--deep_supervision] [--sync_batchnorm] [--save_ckpt] [--nfolds NFOLDS] [--seed SEED] [--ckpt_path CKPT_PATH] [--fold FOLD] [--patience PATIENCE] [--lr_patience LR_PATIENCE] [--batch_size BATCH_SIZE] [--val_batch_size VAL_BATCH_SIZE] [--steps STEPS [STEPS ...]] [--create_idx] [--profile] [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY] [--save_preds] [--dim {2,3}] [--resume_training] [--factor FACTOR] [--num_workers NUM_WORKERS] [--min_epochs MIN_EPOCHS] [--max_epochs MAX_EPOCHS] [--warmup WARMUP] [--oversampling OVERSAMPLING] [--norm {instance,batch,group}] [--overlap OVERLAP] [--affinity {socket,single,single_unique,socket_unique_interleaved,socket_unique_continuous,disabled}] [--scheduler {none,multistep,cosine,plateau}] [--optimizer {sgd,adam,adamw,radam,fused_adam}] [--val_mode {gaussian,constant}] [--train_batches TRAIN_BATCHES] [--test_batches TEST_BATCHES]

optional arguments:
  -h, --help            show this help message and exit
  --exec_mode {train,evaluate,predict}
                        Execution mode to run the model (default: train)
  --data DATA           Path to data directory (default: /data)
  --results RESULTS     Path to results directory (default: /results)
  --logname LOGNAME     Name of dlloger output (default: None)
  --task TASK           Task number. MSD uses numbers 01-10 (default: None)
  --gpus GPUS           Number of gpus (default: 1)
  --learning_rate LEARNING_RATE
                        Learning rate (default: 0.001)
  --gradient_clip_val GRADIENT_CLIP_VAL
                        Gradient clipping norm value (default: 0)
  --negative_slope NEGATIVE_SLOPE
                        Negative slope for LeakyReLU (default: 0.01)
  --tta                 Enable test time augmentation (default: False)
  --amp                 Enable automatic mixed precision (default: False)
  --benchmark           Run model benchmarking (default: False)
  --deep_supervision    Enable deep supervision (default: False)
  --sync_batchnorm      Enable synchronized batchnorm (default: False)
  --save_ckpt           Enable saving checkpoint (default: False)
  --nfolds NFOLDS       Number of cross-validation folds (default: 5)
  --seed SEED           Random seed (default: 1)
  --ckpt_path CKPT_PATH
                        Path to checkpoint (default: None)
  --fold FOLD           Fold number (default: 0)
  --patience PATIENCE   Early stopping patience (default: 100)
  --lr_patience LR_PATIENCE
                        Patience for ReduceLROnPlateau scheduler (default: 70)
  --batch_size BATCH_SIZE
                        Batch size (default: 2)
  --val_batch_size VAL_BATCH_SIZE
                        Validation batch size (default: 4)
  --steps STEPS [STEPS ...]
                        Steps for multistep scheduler (default: None)
  --create_idx          Create index files for tfrecord (default: False)
  --profile             Run dlprof profiling (default: False)
  --momentum MOMENTUM   Momentum factor (default: 0.99)
  --weight_decay WEIGHT_DECAY
                        Weight decay (L2 penalty) (default: 0.0001)
  --save_preds          Enable prediction saving (default: False)
  --dim {2,3}           UNet dimension (default: 3)
  --resume_training     Resume training from the last checkpoint (default: False)
  --factor FACTOR       Scheduler factor (default: 0.3)
  --num_workers NUM_WORKERS
                        Number of subprocesses to use for data loading (default: 8)
  --min_epochs MIN_EPOCHS
                        Force training for at least these many epochs (default: 100)
  --max_epochs MAX_EPOCHS
                        Stop training after this number of epochs (default: 10000)
  --warmup WARMUP       Warmup iterations before collecting statistics (default: 5)
  --oversampling OVERSAMPLING
                        Probability of crop to have some region with positive label (default: 0.33)
  --norm {instance,batch,group}
                        Normalization layer (default: instance)
  --overlap OVERLAP     Amount of overlap between scans during sliding window inference (default: 0.25)
  --affinity {socket,single,single_unique,socket_unique_interleaved,socket_unique_continuous,disabled}
                        type of GPU affinity (default: socket_unique_interleaved)
  --scheduler {none,multistep,cosine,plateau}
                        Learning rate scheduler (default: none)
  --optimizer {sgd,adam,adamw,radam,fused_adam}
                        Optimizer (default: radam)
  --val_mode {gaussian,constant}
                        How to blend output of overlapping windows (default: gaussian)
  --train_batches TRAIN_BATCHES
                        Limit number of batches for training (used for benchmarking mode only) (default: 0)
  --test_batches TEST_BATCHES
                        Limit number of batches for inference (used for benchmarking mode only) (default: 0)
```

### Getting the data

The nnU-Net model was trained on the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) datasets. All datasets are in Neuroimaging Informatics Technology Initiative (NIfTI) format.

#### Dataset guidelines

To train nnU-Net you will need to preprocess your dataset as a first step with `preprocess.py` script.

The `preprocess.py` script is using the following command-line options:

```
  --data               Path to data directory (default: `/data`)
  --results            Path to directory for saving preprocessed data (default: `/data`)
  --exec_mode          Mode for data preprocessing
  --task               Number of tasks to be run. MSD uses numbers 01-10
  --dim                Data dimension to prepare (default: `3`)
  --n_jobs             Number of parallel jobs for data preprocessing (default: `-1`) 
  --vpf                Number of volumes per tfrecord (default: `1`) 
```

To preprocess data for 3D U-Net run: `python preprocess.py --task 01 --dim 3`

In `data_preprocessing/configs.py` for each [Medical Segmentation Decathlon](http://medicaldecathlon.com/) task there are defined: patch size, precomputed spacings and statistics for CT datasets.

The preprocessing pipeline consists of the following steps:

1. Cropping to the region of nonzero values.
2. Resampling to the median voxel spacing of their respective dataset (exception for anisotropic datasets where the lowest resolution axis is selected to be the 10th percentile of the spacings).
3. Padding volumes so that dimensions are at least as patch size.
4. Normalizing
    * For CT modalities the voxel values are clipped to 0.5 and 99.5 percentiles of the foreground voxels and then data is normalized with mean and standard deviation from collected from foreground voxels.
    * For MRI modalities z-score normalization is applied.

#### Multi-dataset

Adding your dataset is possible, however, your data should correspond to [Medical Segmentation Decathlon](http://medicaldecathlon.com/) (i.e data should be `NIfTi` format and there should be `dataset.json` file where you need to provide fields: modality, labels and at least one of training, test).

To add your dataset, perform the following:

1. Mount your dataset to `/data` directory.
 
2. In `data_preprocessing/config.py`:
    - Add to the `task_dir` dictionary your dataset directory name. For example, for Brain Tumour dataset, it corresponds to `"01": "Task01_BrainTumour"`.
    - Add the patch size that you want to use for training to the `patch_size` dictionary. For example, for Brain Tumour dataset it corresponds to `"01_3d": [128, 128, 128]` for 3D U-Net and `"01_2d": [192, 160]` for 2D U-Net. There are three types of suffixes `_3d, _2d` they correspond to 3D UNet and 2D U-Net.

3. Preprocess your data with `preprocess.py` scripts. For example, to preprocess Brain Tumour dataset for 2D U-Net you should run `python preprocess.py --task 01 --dim 2`.

### Training process

The model trains for at least `--min_epochs` and at most `--max_epochs` epochs. After each epoch evaluation, the validation set is done and validation loss is monitored for early stopping (see `--patience` flag). Default training settings are:
* RAdam optimizer with learning rate of 0.001 and weight decay 0.0001.
* Training batch size is set to 2 for 3D U-Net and 16 for 2D U-Net.
    
This default parametrization is applied when running scripts from the `./scripts` directory and when running `main.py` without explicitly overriding these parameters. By default, the training is in full precision. To enable AMP, pass the `--amp` flag. AMP can be enabled for every mode of execution.

The default configuration minimizes a function `L = 0.5 * (1 - dice) + 0.5 * cross entropy` during training and reports achieved convergence as [dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) per class. The training, with a combination of dice and cross entropy has been proven to achieve better convergence than a training using only dice.

The training can be run directly without using the predefined scripts. The name of the training script is `main.py`. For example:

```
python main.py --exec_mode train --task 01 --fold 0 --gpus 1 --amp --deep_supervision
```
  
Training artifacts will be saved to `/results` (you can override it with `--results <path/to/results/>`) in the container. Some important artifacts are:
* `/results/logs.json`: Collected dice scores and loss values evaluated after each epoch during training on validation set.
* `/results/train_logs.json`: Selected best dice scores achieved during training.
* `/results/checkpoints`: Saved checkpoints. By default, two checkpoints are saved - one after each epoch ('last.ckpt') and one with the highest validation dice (e.g 'epoch=5.ckpt' for if highest dice was at 5th epoch).

To load the pretrained model provide `--ckpt_path <path/to/checkpoint>`.

### Inference process

Inference can be launched by passing the `--exec_mode predict` flag. For example:

```
python main.py --exec_mode predict --task 01 --fold 0 --gpus 1 --amp --tta --save_preds --ckpt_path <path/to/checkpoint>
```

The script will then:

* Load the checkpoint from the directory specified by the `<path/to/checkpoint>` directory
* Run inference on the preprocessed validation dataset corresponding to fold 0
* Print achieved score to the console
* If `--save_preds` is provided then resulting masks in the NumPy format will be saved in the `/results` directory
                       
## Performance

### Benchmarking

The following section shows how to run benchmarks to measure the model performance in training and inference modes.

#### Training performance benchmark

To benchmark training, run one of the scripts in `./scripts`:

```
python scripts/benchmark.py --mode train --gpus <ngpus> --dim {2,3} --batch_size <bsize> [--amp] 
```

For example, to benchmark 3D U-Net training using mixed-precision on 8 GPUs with batch size of 2 for 80 batches, run:

```
python scripts/benchmark.py --mode train --gpus 8 --dim 3 --batch_size 2 --train_batches 80 --amp
```

Each of these scripts will by default run 10 warm-up iterations and benchmark the performance during the next 70 iterations. To modify these values provide: `--warmup <warmup> --train_batches <number/of/train/batches>`.

At the end of the script, a line reporting the best train throughput and latency will be printed.

#### Inference performance benchmark

To benchmark inference, run one of the scripts in `./scripts`:

```
python scripts/benchmark.py --mode predict --dim {2,3} --batch_size <bsize> --test_batches <number/of/test/batches> [--amp]
```

For example, to benchmark inference using mixed-precision for 3D U-Net, with batch size of 4 for 80 batches, run:

```
python scripts/benchmark.py --mode predict --dim 3 --amp --batch_size 4 --test_batches 80
```

Each of these scripts will by default run 10 warm-up iterations and benchmark the performance during the next 70 iterations. To modify these values provide: `--warmup <warmup> --test_batches <number/of/test/batches>`.

At the end of the script, a line reporting the inference throughput and latency will be printed.

### Results

The following sections provide details on how to achieve the same performance and accuracy in training and inference.

#### Training accuracy results

##### Training accuracy: NVIDIA DGX-1 (8x A100 16GB)

Our results were obtained by running the `python scripts/train.py --gpus {1,8} --fold {0,1,2,3,4} --dim {2,3} --batch_size <bsize> [--amp]` training scripts and averaging results in the PyTorch 20.12 NGC container on NVIDIA DGX-1 with (8x V100 16GB) GPUs.

| Dimension | GPUs | Batch size / GPU  | Accuracy - mixed precision | Accuracy - FP32 | Time to train - mixed precision | Time to train - TF32|  Time to train speedup (TF32 to mixed precision)        
|:-:|:-:|:--:|:-----:|:-----:|:-----:|:-----:|:----:|
| 2 | 1 | 16 |0.7021 |0.7051 |89min  | 104min| 1.17 |
| 2 | 8 | 16 |0.7316 |0.7316 |13 min | 17 min| 1.31 |
| 3 | 1 | 2  |0.7436 |0.7433 |241 min|342 min| 1.42 |
| 3 | 8 | 2  |0.7443 |0.7443 |36 min | 44 min| 1.22 |

##### Training accuracy: NVIDIA DGX-1 (8x V100 16GB)

Our results were obtained by running the `python scripts/train.py --gpus {1,8} --fold {0,1,2,3,4} --dim {2,3} --batch_size <bsize> [--amp]` training scripts and averaging results in the PyTorch 20.12 NGC container on NVIDIA DGX-1 with (8x V100 16GB) GPUs.

| Dimension | GPUs | Batch size / GPU | Accuracy - mixed precision |  Accuracy - FP32 |  Time to train - mixed precision | Time to train - FP32  | Time to train speedup (FP32 to mixed precision)        
|:-:|:-:|:--:|:-----:|:-----:|:-----:|:-----:|:----:|
| 2 | 1 | 16 |0.7034 |0.7033 |144 min|180 min| 1.25 |
| 2 | 8 | 16 |0.7319 |0.7315 |37 min |44 min | 1.19 |
| 3 | 1 | 2  |0.7439 |0.7436 |317 min|738 min| 2.32 |
| 3 | 8 | 2  |0.7440 |0.7441 |58 min |121 min| 2.09 |

#### Training performance results

##### Training performance: NVIDIA DGX-2 A100 (8x A100 80GB)

Our results were obtained by running the `python scripts/benchmark.py --mode train --gpus {1,8} --dim {2,3} --batch_size <bsize> [--amp]` training script in the NGC container on NVIDIA DGX-2 A100 (8x A100 80GB) GPUs. Performance numbers (in volumes per second) were averaged over an entire training epoch.

| Dimension | GPUs | Batch size / GPU  | Throughput - mixed precision [img/s] | Throughput - TF32 [img/s] | Throughput speedup (TF32 - mixed precision) | Weak scaling - mixed precision | Weak scaling - TF32 |
|:-:|:-:|:--:|:------:|:------:|:-----:|:-----:|:-----:|
| 2 | 1 | 32 | 674.34 |  489.3 | 1.38  |  N/A  |  N/A  |
| 2 | 1 | 64 | 856.34 | 565.62 | 1.51  |  N/A  |  N/A  |
| 2 | 1 | 128| 926.64 | 600.34 | 1.54  |  N/A  |  N/A  |
| 2 | 8 | 32 | 3957.33 | 3275.88 | 1.21| 5.868 | 6.695 |
| 2 | 8 | 64 | 5667.14 | 4037.82 | 1.40 | 6.618 | 7.139 |
| 2 | 8 | 128| 6310.97 | 4568.13 | 1.38 | 6.811 | 7.609 |
| 3 | 1 | 1  | 4.24 | 3.57 | 1.19 |  N/A  |  N/A  |
| 3 | 1 | 2  | 6.74 | 5.21 | 1.29 |  N/A  |  N/A  |
| 3 | 1 | 4  | 9.52 | 4.16 | 2.29 |  N/A  |  N/A  |
| 3 | 8 | 1  | 32.48 | 27.79 | 1.17 | 7.66 | 7.78 |
| 3 | 8 | 2  | 51.50 | 40.67 | 1.27 | 7.64 | 7.81 |
| 3 | 8 | 4  | 74.29 | 31.50 | 2.36 | 7.80 | 7.57 |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).


##### Training performance: NVIDIA DGX-1 (8x V100 16GB)

Our results were obtained by running the `python scripts/benchmark.py --mode train --gpus {1,8} --dim {2,3} --batch_size <bsize> [--amp]` training script in the PyTorch 20.10 NGC container on NVIDIA DGX-1 with (8x V100 16GB) GPUs. Performance numbers (in volumes per second) were averaged over an entire training epoch.

| Dimension | GPUs | Batch size / GPU | Throughput - mixed precision [img/s] | Throughput - FP32 [img/s] | Throughput speedup (FP32 - mixed precision) | Weak scaling - mixed precision | Weak scaling - FP32 |
|:-:|:-:|:---:|:---------:|:-----------:|:--------:|:---------:|:-------------:|
| 2 | 1 | 32 | 416.68  | 275.99  | 1.51 |  N/A |  N/A |
| 2 | 1 | 64 | 524.13  | 281.84  | 1.86 |  N/A |  N/A |
| 2 | 1 | 128| 557.48  | 272.68  | 2.04 |  N/A |  N/A |
| 2 | 8 | 32 | 2731.22 | 2005.49 | 1.36 | 6.56 | 7.27 |
| 2 | 8 | 64 | 3604.83 | 2088.58 | 1.73 |  6.88 | 7.41 |
| 2 | 8 | 128| 4202.35 | 2094.63 |  2.01 | 7.54 | 7.68 |
| 3 | 1 | 1  | 3.97    | 1.77    |  2.24 |  N/A |  N/A |
| 3 | 1 | 2  | 5.49    | 2.32    |  2.37 |  N/A |  N/A |
| 3 | 1 | 4  | 6.78    | OOM     |  N/A  |  N/A |  N/A |
| 3 | 8 | 1  | 29.98   | 13.78   |  2.18 | 7.55 | 7.79 |
| 3 | 8 | 2  | 41.31   | 18.11   |  2.28 | 7.53 | 7.81 |
| 3 | 8 | 4  | 50.26   | OOM     |  N/A  | 7.41 | N/A  |
 

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

#### Inference performance results

##### Inference performance: NVIDIA DGX-2 A100 (1x A100 80GB)

Our results were obtained by running the `python scripts/benchmark.py --mode predict --dim {2,3} --batch_size <bsize> [--amp]` inferencing benchmarking script in the PyTorch 20.10 NGC container on NVIDIA DGX-2 A100 (1x A100 80GB) GPU.


FP16

| Dimension | Batch size |   Resolution  | Throughput Avg [img/s] | Latency Avg [ms] | Latency 90% [ms] | Latency 95% [ms] | Latency 99% [ms] |
|:----------:|:---------:|:-------------:|:----------------------:|:----------------:|:----------------:|:----------------:|:----------------:|
| 2 | 32 | 4x192x160 | 3281.91  | 9.75  | 9.88  | 10.14 | 10.17 |
| 2 | 64 | 4x192x160 | 3625.3   | 17.65 | 18.13 | 18.16 | 18.24 |
| 2 |128 | 4x192x160 | 3867.24  | 33.10 | 33.29 | 33.29 | 33.35 |
| 3 | 1  | 4x128x128x128 | 10.93| 91.52 | 91.30 | 92,68 | 111.87|
| 3 | 2  | 4x128x128x128 | 18.85| 106.08| 105.12| 106.05| 127.95|
| 3 | 4  | 4x128x128x128 | 27.4 | 145.98| 164.05| 165.58| 183.43|


TF32

| Dimension | Batch size |   Resolution  | Throughput Avg [img/s] | Latency Avg [ms] | Latency 90% [ms] | Latency 95% [ms] | Latency 99% [ms] |
|:----------:|:---------:|:-------------:|:----------------------:|:----------------:|:----------------:|:----------------:|:----------------:|
| 2 | 32 | 4x192x160 | 2002.66  | 15.98 | 16.14 | 16.24 | 16.37|
| 2 | 64 | 4x192x160 | 2180.54  | 29.35 | 29.50 | 29.51 | 29.59|
| 2 |128 | 4x192x160 | 2289.12  | 55.92 | 56.08 | 56.13 | 56.36|
| 3 | 1  | 4x128x128x128 | 10.05| 99.55 | 99.17 | 99.82 |120.39|
| 3 | 2  | 4x128x128x128 | 16.29|122.78 |123.06 |124.02 |143.47|
| 3 | 4  | 4x128x128x128 | 15.99|250.16 |273.67 |274.85 |297.06|

Throughput is reported in images per second. Latency is reported in milliseconds per batch.
To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).


##### Inference performance: NVIDIA DGX-1 (1x V100 16GB)

Our results were obtained by running the `python scripts/benchmark.py --mode predict --dim {2,3} --batch_size <bsize> [--amp]` inferencing benchmarking script in the PyTorch 20.10 NGC container on NVIDIA DGX-1 with (1x V100 16GB) GPU.

FP16
 
| Dimension | Batch size |   Resolution  | Throughput Avg [img/s] | Latency Avg [ms] | Latency 90% [ms] | Latency 95% [ms] | Latency 99% [ms] |
|:----------:|:---------:|:-------------:|:----------------------:|:----------------:|:----------------:|:----------------:|:----------------:|
| 2 | 32 | 4x192x160 | 1697.16 | 18.86 | 18.89 | 18.95 | 18.99 |
| 2 | 64 | 4x192x160 | 2008.81 | 31.86 | 31.95 | 32.01 | 32.08 |
| 2 |128 | 4x192x160 | 2221.44 | 57.62 | 57.83 | 57.88 | 57.96 |
| 3 | 1  | 4x128x128x128 | 11.01 |  90.76 |  89.96 |  90.53 | 116.67 |
| 3 | 2  | 4x128x128x128 | 16.60 | 120.49 | 119.69 | 120.72 | 146.42 |
| 3 | 4  | 4x128x128x128 | 21.18 | 188.85 | 211.92 | 214.17 | 238.19 |

FP32
 
| Dimension | Batch size |   Resolution  | Throughput Avg [img/s] | Latency Avg [ms] | Latency 90% [ms] | Latency 95% [ms] | Latency 99% [ms] |
|:----------:|:---------:|:-------------:|:----------------------:|:----------------:|:----------------:|:----------------:|:----------------:|
| 2 | 32 | 4x192x160 | 1106.22 | 28.93 | 29.06 | 29.10 | 29.15 |
| 2 | 64 | 4x192x160 | 1157.24 | 55.30 | 55.39 | 55.44 | 55.50 |
| 2 |128 | 4x192x160 | 1171.24 | 109.29 | 109.83 | 109.98 | 110.58 |
| 3 | 1  | 4x128x128x128 | 6.8 | 147.10 | 147.51 | 148.15 | 170.46 |
| 3 | 2  | 4x128x128x128 | 8.53| 234.46 | 237.00 | 238.43 | 258.92 |
| 3 | 4  | 4x128x128x128 | 9.6 | 416.83 | 439.97 | 442.12 | 454.69 |

Throughput is reported in images per second. Latency is reported in milliseconds per batch.
To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

## Release notes

### Changelog

January 2021
- Initial release

### Known issues

There are no known issues in this release.

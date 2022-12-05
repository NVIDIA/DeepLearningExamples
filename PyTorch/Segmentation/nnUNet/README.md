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
            * [Training accuracy: NVIDIA DGX A100 (8x A100 80G)](#training-accuracy-nvidia-dgx-a100-8x-a100-80g)
            * [Training accuracy: NVIDIA DGX-1 (8x V100 32G)](#training-accuracy-nvidia-dgx-1-8x-v100-32g)
        * [Training performance results](#training-performance-results)
            * [Training performance: NVIDIA DGX A100 (8x A100 80G)](#training-performance-nvidia-dgx-a100-8x-a100-80g) 
            * [Training performance: NVIDIA DGX-1 (8x V100 32G)](#training-performance-nvidia-dgx-1-8x-v100-32g)
        * [Inference performance results](#inference-performance-results)
            * [Inference performance: NVIDIA DGX A100 (1x A100 80G)](#inference-performance-nvidia-dgx-a100-1x-a100-80g)
            * [Inference performance: NVIDIA DGX-1 (1x V100 32G)](#inference-performance-nvidia-dgx-1-1x-v100-32g)
- [Release notes](#release-notes)
    * [Changelog](#changelog)
    * [Known issues](#known-issues)

## Model overview
 
The nnU-Net ("no-new-Net") refers to a robust and self-adapting framework for U-Net based medical image segmentation. This repository contains a nnU-Net implementation as described in the paper: [nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation](https://arxiv.org/abs/1809.10486). 

The differences between this nnU-net and [original model](https://github.com/MIC-DKFZ/nnUNet) are:
- Dynamic selection of patch size is not supported, and it has to be set in `data_preprocessing/configs.py` file.
- Cascaded U-Net is not supported.
- The following data augmentations are not used: rotation, simulation of low resolution, gamma augmentation.

This model is trained with mixed precision using Tensor Cores on Volta, Turing, and the NVIDIA Ampere GPU architectures. Therefore, researchers can get results 2x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.

We developed the model using [PyTorch Lightning](https://www.pytorchlightning.ai), a new easy-to-use framework that ensures code readability and reproducibility without the boilerplate.

### Model architecture
 
The nnU-Net allows the training of two types of networks: 2D U-Net and 3D U-Net to perform semantic segmentation of 3D images, with high accuracy and performance.

The following figure shows the architecture of the 3D U-Net model and its different components. U-Net is composed of a contractive and an expanding path, that aims at building a bottleneck in its centremost part through a combination of convolution, instance norm, and leaky ReLU operations. After this bottleneck, the image is reconstructed through a combination of convolutions and upsampling. Skip connections are added with the goal of helping the backward flow of gradients to improve the training.

<img src="images/unet3d.png" width="900"/>
 
*Figure 1: The 3D U-Net architecture*

### Default configuration

All convolution blocks in U-Net in both encoder and decoder are using two convolution layers followed by instance normalization and a leaky ReLU nonlinearity. For downsampling, we are using stride convolution whereas transposed convolution is used for upsampling.

All models were trained with an Adam optimizer. For loss function we use the average of [cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy) and [dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient).

Early stopping is triggered if the validation dice score wasn't improved during the last 100 epochs.

Used data augmentation: crop with oversampling the foreground class, mirroring, zoom, Gaussian noise, Gaussian blur, brightness, and contrast.

### Feature support matrix

The following features are supported by this model: 

| Feature               | nnUNet               
|-----------------------|--------------------------   
|[DALI](https://docs.nvidia.com/deeplearning/dali/release-notes/index.html) | Yes
|Automatic mixed precision (AMP)   | Yes  
|Distributed data-parallel (DDP)   | Yes
 
#### Features

**DALI**

NVIDIA DALI - DALI is a library-accelerating data preparation pipeline. To speed up your input pipeline, you only need to define your data loader
with the DALI library. For details, see example sources in this repository or see the [DALI documentation](https://docs.nvidia.com/deeplearning/dali/index.html)

**Automatic Mixed Precision (AMP)**

This implementation uses native PyTorch AMP implementation of mixed precision training. It allows us to use FP16 training with FP32 master weights by modifying a few lines of code.

**DistributedDataParallel (DDP)**

The model uses PyTorch Lightning implementation of distributed data parallelism at the module level which can run across multiple machines.
 
### Mixed precision training

Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format while storing minimal information in single-precision to keep as much information as possible in critical parts of the network. Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in Volta, and following with both the Turing and Ampere architectures, significant training speedups are experienced by switching to mixed precision -- up to 3x speedup on the most intense model architectures. Using mixed precision training requires two steps:

1. Porting the model to use the FP16 data type where appropriate.
2. Adding loss scaling to preserve small gradient values.

The ability to train deep learning networks with lower precision was introduced in the Pascal architecture and first supported in [CUDA 8](https://devblogs.nvidia.com/parallelforall/tag/fp16/) in the NVIDIA Deep Learning SDK.

For information about:
* How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) documentation.
* Techniques used for mixed precision training, see the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
* APEX tools for mixed precision training, see the [NVIDIA Apex: Tools for Easy Mixed-Precision Training in PyTorch](https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/).

#### Enabling mixed precision
 
For training and inference, mixed precision can be enabled by adding the `--amp` flag. Mixed precision is using [native PyTorch implementation](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/).

#### TF32

TensorFloat-32 (TF32) is the new math mode in [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) GPUs for handling the matrix math also called tensor operations. TF32 running on Tensor Cores in A100 GPUs can provide up to 10x speedups compared to single-precision floating-point math (FP32) on Volta GPUs. 

TF32 Tensor Cores can speed up networks using FP32, typically with no loss of accuracy. It is more robust than FP16 for models which require a high dynamic range for weights or activations.

For more information, refer to the [TensorFloat-32 in the A100 GPU Accelerates AI Training, HPC up to 20x](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) blog post.

TF32 is supported in the NVIDIA Ampere GPU architecture and is enabled by default.

### Glossary

**Test time augmentation**

Test time augmentation is an inference technique that averages predictions from augmented images with its prediction. As a result, predictions are more accurate, but with the cost of a slower inference process. For nnU-Net, we use all possible flip combinations for image augmenting. Test time augmentation can be enabled by adding the `--tta` flag.

## Setup

The following section lists the requirements that you need to meet to start training the nnU-Net model.

### Requirements

This repository contains Dockerfile which extends the PyTorch NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
-   PyTorch 22.11 NGC container
-   Supported GPUs:
 - [NVIDIA Volta architecture](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
 - [NVIDIA Turing architecture](https://www.nvidia.com/en-us/geforce/turing/)
 - [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)

For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
- [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
- [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)
-   Running [PyTorch](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html#running)
 
For those unable to use the PyTorch NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## Quick Start Guide

To train your model using mixed or TF32 precision with Tensor Cores or using FP32, perform the following steps using the default parameters of the nnUNet model on the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) dataset. For the specifics on training and inference, see the [Advanced](#advanced) section.

1. Clone the repository.

Executing this command will create your local repository with all the code to run nnU-Net.
```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/PyTorch/Segmentation/nnUNet
```
 
2. Build the nnU-Net PyTorch NGC container.
 
This command will use the Dockerfile to create a Docker image named `nnunet`, downloading all the required components automatically.

```
docker build -t nnunet .
```
 
The NGC container contains all the components optimized for usage on NVIDIA hardware.
 
3. Start an interactive session in the NGC container to run preprocessing/training/inference.
 
The following command will launch the container and mount the `./data` directory as a volume to the `/data` directory inside the container, and `./results` directory to the `/results` directory in the container.
 
```
mkdir data results
docker run -it --privileged --runtime=nvidia --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 --rm -v ${PWD}/data:/data -v ${PWD}/results:/results nnunet:latest /bin/bash
```

4. Prepare the BraTS dataset.

To download and preprocess the data run:
```
python download.py --task 01
python preprocess.py --task 01 --dim 3
python preprocess.py --task 01 --dim 2
```

Then `ls /data` should print:
```
01_3d 01_2d Task01_BrainTumour
```

For the specifics on data preprocessing, see the [Getting the data](#getting-the-data) section.
 
5. Start training.
 
Training can be started with:
```
python scripts/train.py --gpus <gpus> --fold <fold> --dim <dim> [--amp] [--bind]
```

To see descriptions of the train script arguments run `python scripts/train.py --help`. You can customize the training process. For details, see the [Training process](#training-process) section.

6. Start benchmarking.

The training and inference performance can be evaluated by using benchmarking scripts, such as:
 
```
python scripts/benchmark.py --mode {train,predict} --gpus <ngpus> --dim {2,3} --batch_size <bsize> [--amp] [--bind]
```

To see descriptions of the benchmark script arguments run `python scripts/benchmark.py --help`.


7. Start inference/predictions.
 
Inference can be started with:
```
python scripts/inference.py --data <path/to/data> --dim <dim> --fold <fold> --ckpt_path <path/to/checkpoint> [--amp] [--tta] [--save_preds]
```

Note: You have to prepare either validation or test dataset to run this script by running `python preprocess.py --task 01 --dim {2,3} --exec_mode {val,test}`. After preprocessing inside given task directory (e.g. `/data/01_3d/` for task 01 and dim 3) it will create `val` or `test` directory with preprocessed data ready for inference. Possible workflow:

```
python preprocess.py --task 01 --dim 3 --exec_mode val
python scripts/inference.py --data /data/01_3d/val --dim 3 --fold 0 --ckpt_path <path/to/checkpoint> --amp --tta --save_preds
```

Then if you have labels for predicted images you can evaluate them with `evaluate.py` script. For example:

```
python evaluate.py --preds /results/preds_task_01_dim_3_fold_0_tta --lbls /data/Task01_BrainTumour/labelsTr
```

To see descriptions of the inference script arguments run `python scripts/inference.py --help`. You can customize the inference process. For details, see the [Inference process](#inference-process) section.

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
* `evaluate.py`: Compare predictions with ground truth and get the final score.
 
The `data_preprocessing` folder contains information about the data preprocessing used by nnU-Net. Its contents are:
 
* `configs.py`: Defines dataset configuration like patch size or spacing.
* `preprocessor.py`: Implements data preprocessing pipeline.
 
The `data_loading` folder contains information about the data pipeline used by nnU-Net. Its contents are:
 
* `data_module.py`: Defines `LightningDataModule` used by PyTorch Lightning.
* `dali_loader.py`: Implements DALI data loader.
 
The `nnunet` folder contains information about the building blocks of nnU-Net and the way they are assembled. Its contents are:
 
* `metrics.py`: Implements dice metric
* `loss.py`: Implements loss function.
* `nn_unet.py`: Implements training/validation/test logic and dynamic creation of U-Net architecture used by nnU-Net.
 
The `utils` folder includes:

* `args.py`: Defines command line arguments.
* `utils.py`: Defines utility functions.
* `logger.py`: Defines logging callback for performance benchmarking.

The `notebooks` folder includes:

* `BraTS21.ipynb`: Notebook with our solution ranked 3 for the BraTS21 challenge.
* `BraTS22.ipynb`: Notebook with our solution ranked 2 for the BraTS22 challenge.
* `custom_dataset.ipynb`: Notebook which demonstrates how to use nnU-Net with the custom dataset.

Other folders included in the root directory are:

* `images/`: Contains a model diagram.
* `scripts/`: Provides scripts for training, benchmarking, and inference of nnU-Net.

### Command-line options
To see the full list of available options and their descriptions, use the `-h` or `--help` command-line option, for example:

`python main.py --help`

The following example output is printed when running the model:

```
usage: main.py [-h] [--exec_mode {train,evaluate,predict}] [--data DATA] [--results RESULTS] [--logname LOGNAME] [--task TASK] [--gpus GPUS] [--learning_rate LEARNING_RATE] [--gradient_clip_val GRADIENT_CLIP_VAL] [--negative_slope NEGATIVE_SLOPE] [--tta] [--brats] [--deep_supervision] [--more_chn] [--invert_resampled_y] [--amp] [--benchmark] [--focal] [--sync_batchnorm] [--save_ckpt] [--nfolds NFOLDS] [--seed SEED] [--skip_first_n_eval SKIP_FIRST_N_EVAL] [--ckpt_path CKPT_PATH] [--fold FOLD] [--patience PATIENCE] [--batch_size BATCH_SIZE] [--val_batch_size VAL_BATCH_SIZE] [--profile] [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY] [--save_preds] [--dim {2,3}] [--resume_training] [--num_workers NUM_WORKERS] [--epochs EPOCHS] [--warmup WARMUP] [--norm {instance,batch,group}] [--nvol NVOL] [--depth DEPTH] [--min_fmap MIN_FMAP] [--deep_supr_num DEEP_SUPR_NUM] [--res_block] [--filters FILTERS [FILTERS ...]] [--data2d_dim {2,3}] [--oversampling OVERSAMPLING] [--overlap OVERLAP] [--affinity {socket,single_single,single_single_unique,socket_unique_interleaved,socket_unique_continuous,disabled}] [--scheduler] [--optimizer {sgd,adam}] [--blend {gaussian,constant}] [--train_batches TRAIN_BATCHES] [--test_batches TEST_BATCHES]

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
                        Learning rate (default: 0.0008)
  --gradient_clip_val GRADIENT_CLIP_VAL
                        Gradient clipping norm value (default: 0)
  --negative_slope NEGATIVE_SLOPE
                        Negative slope for LeakyReLU (default: 0.01)
  --tta                 Enable test time augmentation (default: False)
  --brats               Enable BraTS specific training and inference (default: False)
  --deep_supervision    Enable deep supervision (default: False)
  --more_chn            Create encoder with more channels (default: False)
  --invert_resampled_y  Resize predictions to match label size before resampling (default: False)
  --amp                 Enable automatic mixed precision (default: False)
  --benchmark           Run model benchmarking (default: False)
  --focal               Use focal loss instead of cross entropy (default: False)
  --sync_batchnorm      Enable synchronized batchnorm (default: False)
  --save_ckpt           Enable saving checkpoint (default: False)
  --nfolds NFOLDS       Number of cross-validation folds (default: 5)
  --seed SEED           Random seed (default: 1)
  --skip_first_n_eval SKIP_FIRST_N_EVAL
                        Skip the evaluation for the first n epochs. (default: 0)
  --ckpt_path CKPT_PATH
                        Path to checkpoint (default: None)
  --fold FOLD           Fold number (default: 0)
  --patience PATIENCE   Early stopping patience (default: 100)
  --batch_size BATCH_SIZE
                        Batch size (default: 2)
  --val_batch_size VAL_BATCH_SIZE
                        Validation batch size (default: 4)
  --profile             Run dlprof profiling (default: False)
  --momentum MOMENTUM   Momentum factor (default: 0.99)
  --weight_decay WEIGHT_DECAY
                        Weight decay (L2 penalty) (default: 0.0001)
  --save_preds          Enable prediction saving (default: False)
  --dim {2,3}           UNet dimension (default: 3)
  --resume_training     Resume training from the last checkpoint (default: False)
  --num_workers NUM_WORKERS
                        Number of subprocesses to use for data loading (default: 8)
  --epochs EPOCHS       Number of training epochs (default: 1000)
  --warmup WARMUP       Warmup iterations before collecting statistics (default: 5)
  --norm {instance,batch,group}
                        Normalization layer (default: instance)
  --nvol NVOL           Number of volumes which come into single batch size for 2D model (default: 4)
  --depth DEPTH         The depth of the encoder (default: 5)
  --min_fmap MIN_FMAP   Minimal dimension of feature map in the bottleneck (default: 4)
  --deep_supr_num DEEP_SUPR_NUM
                        Number of deep supervision heads (default: 2)
  --res_block           Enable residual blocks (default: False)
  --filters FILTERS [FILTERS ...]
                        [Optional] Set U-Net filters (default: None)
  --data2d_dim {2,3}    Input data dimension for 2d model (default: 3)
  --oversampling OVERSAMPLING
                        Probability of crop to have some region with positive label (default: 0.4)
  --overlap OVERLAP     Amount of overlap between scans during sliding window inference (default: 0.5)
  --affinity {socket,single_single,single_single_unique,socket_unique_interleaved,socket_unique_continuous,disabled}
                        type of CPU affinity (default: socket_unique_contiguous)
  --scheduler           Enable cosine rate scheduler with warmup (default: False)
  --optimizer {sgd,adam}
                        Optimizer (default: adam)
  --blend {gaussian,constant}
                        How to blend output of overlapping windows (default: gaussian)
  --train_batches TRAIN_BATCHES
                        Limit number of batches for training (used for benchmarking mode only) (default: 0)
  --test_batches TEST_BATCHES
                        Limit number of batches for inference (used for benchmarking mode only) (default: 0)
```

### Getting the data

The nnU-Net model was trained on the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) datasets. All datasets are in Neuroimaging Informatics Technology Initiative (NIfTI) format.

#### Dataset guidelines

To train nnU-Net you will need to preprocess your dataset as the first step with `preprocess.py` script. Run `python scripts/preprocess.py --help` to see descriptions of the preprocess script arguments.

For example to preprocess data for 3D U-Net run: `python preprocess.py --task 01 --dim 3`.

In `data_preprocessing/configs.py` for each [Medical Segmentation Decathlon](http://medicaldecathlon.com/) task, there are defined: patch sizes, precomputed spacings and statistics for CT datasets.

The preprocessing pipeline consists of the following steps:

1. Cropping to the region of non-zero values.
2. Resampling to the median voxel spacing of their respective dataset (exception for anisotropic datasets where the lowest resolution axis is selected to be the 10th percentile of the spacings).
3. Padding volumes so that dimensions are at least as patch size.
4. Normalizing:
 * For CT modalities the voxel values are clipped to 0.5 and 99.5 percentiles of the foreground voxels and then data is normalized with mean and standard deviation collected from foreground voxels.
 * For MRI modalities z-score normalization is applied.

#### Multi-dataset

It is possible to run nnUNet on a custom dataset. If your dataset corresponds to [Medical Segmentation Decathlon](http://medicaldecathlon.com/) (i.e. data should be in `NIfTi` format and there should be `dataset.json` file where you need to provide fields: modality, labels, and at least one of training, test) you need to perform the following:

1. Mount your dataset to the `/data` directory.
 
2. In `data_preprocessing/config.py`:
 - Add to the `task_dir` dictionary your dataset directory name. For example, for the Brain Tumour dataset, it corresponds to `"01": "Task01_BrainTumour"`.
 - Add the patch size that you want to use for training to the `patch_size` dictionary. For example, for Brain Tumour dataset it corresponds to `"01_3d": [128, 128, 128]` for 3D U-Net and `"01_2d": [192, 160]` for 2D U-Net. There are three types of suffixes `_3d, _2d` they correspond to 3D UNet and 2D U-Net.

3. Preprocess your data with `preprocess.py` scripts. For example, to preprocess the Brain Tumour dataset for 2D U-Net you should run `python preprocess.py --task 01 --dim 2`.

If you have a dataset in another format or you want to customize data preprocessing or data loading see `notebooks/custom_dataset.ipynb`.

### Training process

The model trains for at least `--min_epochs` and at most `--max_epochs` epochs. After each epoch evaluation, the validation set is done and validation loss is monitored for early stopping (see `--patience` flag). Default training settings are:
* Adam optimizer with a learning rate of 0.0008 and weight decay of 0.0001.
* Training batch size is set to 2 for 3D U-Net and 16 for 2D U-Net.
 
This default parametrization is applied when running scripts from the `scripts` directory and when running `main.py` without overriding these parameters. By default, the training is in full precision. To enable AMP, pass the `--amp` flag. AMP can be enabled for every mode of execution.

The default configuration minimizes a function `L = (1 - dice_coefficient) + cross_entropy` during training and reports achieved convergence as [dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) per class. The training, with a combination of dice and cross-entropy has been proven to achieve better convergence than training using only dice.

The training can be run without using the predefined scripts. The name of the training script is `main.py`. For example:

```
python main.py --exec_mode train --task 01 --fold 0 --gpus 1 --amp
```
 
Training artifacts will be saved to `/results` in the container. Some important artifacts are:
* `/results/logs.json`: Collected dice scores and loss values evaluated after each epoch during training on the validation set.
* `/results/checkpoints`: Saved checkpoints. By default, two checkpoints are saved - one after each epoch ('last.ckpt') and one with the highest validation dice (e.g 'epoch=5.ckpt' for if the highest dice was at the 5th epoch).

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

To benchmark training, run the `scripts/benchmark.py` script with `--mode train`:

```
python scripts/benchmark.py --mode train --gpus <ngpus> --dim {2,3} --batch_size <bsize> [--amp] [--bind]
```

For example, to benchmark 3D U-Net training using mixed-precision on 8 GPUs with a batch size of 2, run:

```
python scripts/benchmark.py --mode train --gpus 8 --dim 3 --batch_size 2 --amp
```

Each of these scripts will by default run 1 warm-up epoch and start performance benchmarking during the second epoch.

At the end of the script, a line reporting the best train throughput and latency will be printed.

#### Inference performance benchmark

To benchmark inference, run the `scripts/benchmark.py` script with `--mode predict`:

```
python scripts/benchmark.py --mode predict --dim {2,3} --batch_size <bsize> [--amp]
```

For example, to benchmark inference using mixed-precision for 3D U-Net, with a batch size of 4, run:

```
python scripts/benchmark.py --mode predict --dim 3 --amp --batch_size 4
```

Each of these scripts will by default run a warm-up for 1 data pass and start inference benchmarking during the second pass.

At the end of the script, a line reporting the inference throughput and latency will be printed.

*Note that this benchmark reports performance numbers for iterations over samples with fixed patch sizes.
The real inference process uses sliding window for input images with arbitrary resolution and performance may vary for images with different resolutions.*

### Results

The following sections provide details on how to achieve the same performance and accuracy in training and inference.

#### Training accuracy results

##### Training accuracy: NVIDIA DGX A100 (8x A100 80G)

Our results were obtained by running the `python scripts/train.py --gpus {1,8} --fold {0,1,2,3,4} --dim {2,3} [--amp] [--bind] --learning_rate lr --seed n` training scripts and averaging results in the PyTorch 22.11 NGC container on NVIDIA DGX with (8x A100 80G) GPUs.

Note: We recommend using `--bind` flag for multi-GPU settings to increase the throughput. To launch multi-GPU with `--bind` use PyTorch distributed launcher, e.g., `python -m torch.distributed.launch --use_env --nproc_per_node=8 scripts/benchmark.py --mode train --gpus 8 --dim 3 --amp --batch_size 2 --bind` for the interactive session, or use regular command when launching with SLURM's sbatch.

| Dimension | GPUs | Batch size / GPU  | Dice - mixed precision | Dice - TF32 | Time to train - mixed precision | Time to train - TF32|  Time to train speedup (TF32 to mixed precision)        
|:-:|:-:|:--:|:-----:|:-----:|:-----:|:-----:|:----:|
| 2 | 1 | 2  | 73.21 | 73.11 | 33 min| 48 min| 1.46 |
| 2 | 8 | 2  | 73.15 | 73.16 |  9 min| 13 min| 1.44 |
| 3 | 1 | 2  | 74.35 | 74.34 |104 min|167 min| 1.61 |
| 3 | 8 | 2  | 74.30 | 74.32 |  23min| 36 min| 1.57 |

Reported dice score is the average over 5 folds from the best run for grid search over learning rates {1e-4, 2e-4, ..., 9e-4} and seed {1, 3, 5}.

##### Training accuracy: NVIDIA DGX-1 (8x V100 32G)

Our results were obtained by running the `python scripts/train.py --gpus {1,8} --fold {0,1,2,3,4} --dim {2,3} [--amp] [--bind] --seed n ` training scripts and averaging results in the PyTorch 22.11 NGC container on NVIDIA DGX-1 with (8x V100 32G) GPUs.

Note: We recommend using `--bind` flag for multi-GPU settings to increase the throughput. To launch multi-GPU with `--bind` use PyTorch distributed launcher, e.g., `python -m torch.distributed.launch --use_env --nproc_per_node=8 scripts/benchmark.py --mode train --gpus 8 --dim 3 --amp --batch_size 2 --bind` for the interactive session, or use regular command when launching with SLURM's sbatch.

| Dimension | GPUs | Batch size / GPU | Dice - mixed precision |  Dice - FP32 |  Time to train - mixed precision | Time to train - FP32  | Time to train speedup (FP32 to mixed precision)        
|:-:|:-:|:--:|:-----:|:-----:|:-----:|:-----:|:----:|
| 2 | 1 | 2  | 73.18 | 73.22 | 60 min|114 min| 1.90 |
| 2 | 8 | 2  | 73.15 | 73.18 | 13 min| 19 min| 1.46 |
| 3 | 1 | 2  | 74.31 | 74.33 |201 min|680 min| 3.38 |
| 3 | 8 | 2  | 74.35 | 74.39 | 41 min|153 min| 3.73 |

Reported dice score is the average over 5 folds from the best run for grid search over learning rates {1e-4, 2e-4, ..., 9e-4} and seed {1, 3, 5}.

#### Training performance results

##### Training performance: NVIDIA DGX A100 (8x A100 80G)

Our results were obtained by running the `python scripts/benchmark.py --mode train --gpus {1,8} --dim {2,3} --batch_size <bsize> [--amp]` training script in the NGC container on NVIDIA DGX A100 (8x A100 80G) GPUs. Performance numbers (in volumes per second) were averaged over an entire training epoch.

Note: We recommend using `--bind` flag for multi-gpu settings to increase the througput. To launch multi-GPU with `--bind` use `python -m torch.distributed.launch --use_env --nproc_per_node=<npgus> scripts/train.py --bind ...` for the interactive session, or use regular command when launching with SLURM's sbatch.

| Dimension | GPUs | Batch size / GPU  | Throughput - mixed precision [img/s] | Throughput - TF32 [img/s] | Throughput speedup (TF32 - mixed precision) | Weak scaling - mixed precision | Weak scaling - TF32 |
|:-:|:-:|:--:|:------:|:------:|:-----:|:-----:|:-----:|
|	2	|	1	|	32	|	1040.58	|	732.22	|	1.42	|	-	|	-	|
|	2	|	1	|	64	|	1238.68	|	797.37	|	1.55	|	-	|	-	|
|	2	|	1	|	128	|	1345.29	|	838.38	|	1.60	|	-	|	-	|
|	2	|	8	|	32	|	7747.27	|	5588.2	|	1.39	|	7.45	|	7.60	|
|	2	|	8	|	64	|	9417.27	|	6246.95	|	1.51	|	7.60	|	8.04	|
|	2	|	8	|	128	|	10694.1	|	6631.08	|	1.61	|	7.95	|	7.83	|
|	3	|	1	|	1	|	24.61	|	9.66	|	2.55	|	-	|	-	|
|	3	|	1	|	2	|	27.48	|	11.27	|	2.44	|	-	|	-	|
|	3	|	1	|	4	|	29.96	|	12.22	|	2.45	|	-	|	-	|
|	3	|	8	|	1	|	187.07	|	76.44	|	2.45	|	7.63	|	7.91	|
|	3	|	8	|	2	|	220.83	|	88.67	|	2.49	|	7.83	|	7.87	|
|	3	|	8	|	4	|	234.5	|	96.61	|	2.43	|	7.91	|	7.91	|

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

##### Training performance: NVIDIA DGX-1 (8x V100 32G)

Our results were obtained by running the `python scripts/benchmark.py --mode train --gpus {1,8} --dim {2,3} --batch_size <bsize> [--amp] [--bind]` training script in the PyTorch 22.11 NGC container on NVIDIA DGX-1 with (8x V100 32G) GPUs. Performance numbers (in volumes per second) were averaged over an entire training epoch.

Note: We recommend using `--bind` flag for multi-gpu settings to increase the througput. To launch multi-GPU with `--bind` use `python -m torch.distributed.launch --use_env --nproc_per_node=<npgus> scripts/train.py --bind ...` for the interactive session, or use regular command when launching with SLURM's sbatch.

| Dimension | GPUs | Batch size / GPU | Throughput - mixed precision [img/s] | Throughput - FP32 [img/s] | Throughput speedup (FP32 - mixed precision) | Weak scaling - mixed precision | Weak scaling - FP32 |
|:-:|:-:|:---:|:---------:|:-----------:|:--------:|:---------:|:-------------:|
|	2	|	1	|	32	|	561.6	|	310.21	|	1.81	|	-	|	-	|
|	2	|	1	|	64	|	657.91	|	326.02	|	2.02	|	-	|	-	|
|	2	|	1	|	128	|	706.92	|	332.81	|	2.12	|	-	|	-	|
|	2	|	8	|	32	|	3903.88	|	2396.88	|	1.63	|	6.95	|	7.73	|
|	2	|	8	|	64	|	4922.76	|	2590.66	|	1.90	|	7.48	|	7.95	|
|	2	|	8	|	128	|	5597.87	|	2667.56	|	2.10	|	7.92	|	8.02	|
|	3	|	1	|	1	|	11.38	|	2.07	|	5.50	|	-	|	-	|
|	3	|	1	|	2	|	12.34	|	2.51	|	4.92	|	-	|	-	|
|	3	|	8	|	1	|	84.38	|	16.55	|	5.10	|	7.41	|	8.00	|
|	3	|	8	|	2	|	98.17	|	20.15	|	4.87	|	7.96	|	8.03	|

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

#### Inference performance results

##### Inference performance: NVIDIA DGX A100 (1x A100 80G)

Our results were obtained by running the `python scripts/benchmark.py --mode predict --dim {2,3} --batch_size <bsize> [--amp]` inferencing benchmarking script in the PyTorch 22.11 NGC container on NVIDIA DGX A100 (1x A100 80G) GPU.

FP16

| Dimension | Batch size |  Resolution  | Throughput Avg [img/s] | Latency Avg [ms] | Latency 90% [ms] | Latency 95% [ms] | Latency 99% [ms] |
|:----------:|:---------:|:-------------:|:----------------------:|:----------------:|:----------------:|:----------------:|:----------------:|
|	2	|	32	|	192x160	|	1818.05	|	17.6	| 19.86 | 20.38 | 20.98 |
|	2	|	64	|	192x160	|	3645.16	|	17.56	| 19.86 | 20.82 | 23.66 |
|	2	|	128	|	192x160	|	3850.35	|	33.24	| 34.72 | 61.4 | 63.58 |
|	3	|	1	|	128x128x128	|	68.45	|	14.61	| 17.02 | 17.41 | 19.27 |
|	3	|	2	|	128x128x128	|	56.9	|	35.15	| 40.9 | 43.15 | 57.94 |
|	3	|	4	|	128x128x128	|	76.39	|	52.36	| 57.9 | 59.52 | 70.24 |

TF32

| Dimension | Batch size |  Resolution  | Throughput Avg [img/s] | Latency Avg [ms] | Latency 90% [ms] | Latency 95% [ms] | Latency 99% [ms] |
|:----------:|:---------:|:-------------:|:----------------------:|:----------------:|:----------------:|:----------------:|:----------------:|
|	2	|	32	|	192x160	|	1868.56	|	17.13	| 51.75 | 53.07 | 54.92 |
|	2	|	64	|	192x160	|	2508.57	|	25.51	| 56.83 | 90.08 | 96.87 |
|	2	|	128	|	192x160	|	2609.6	|	49.05	| 191.48 | 201.8 | 205.29 |
|	3	|	1	|	128x128x128	|	35.02	|	28.55	| 51.75 | 53.07 | 54.92 |
|	3	|	2	|	128x128x128	|	39.88	|	50.15	| 56.83 | 90.08 | 96.87 |
|	3	|	4	|	128x128x128	|	41.32	|	96.8	| 191.48 | 201.8 | 205.29 |

Throughput is reported in images per second. Latency is reported in milliseconds per batch.
To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

##### Inference performance: NVIDIA DGX-1 (1x V100 32G)

Our results were obtained by running the `python scripts/benchmark.py --mode predict --dim {2,3} --batch_size <bsize> [--amp]` inferencing benchmarking script in the PyTorch 22.11 NGC container on NVIDIA DGX-1 with (1x V100 32G) GPU.

FP16
 
| Dimension | Batch size |Resolution| Throughput Avg [img/s] | Latency Avg [ms] | Latency 90% [ms] | Latency 95% [ms] | Latency 99% [ms] |
|:----------:|:---------:|:-------------:|:----------------------:|:----------------:|:----------------:|:----------------:|:----------------:|
|	2	|	32	|	192x160	|	1254.38	|	25.51	| 29.07 | 30.07 | 31.23 |
|	2	|	64	|	192x160	|	2024.13	|	31.62	| 71.51 | 71.78 | 72.44 |
|	2	|	128	|	192x160	|	2136.95	|	59.9	| 61.23 | 61.63 | 110.13 |
|	3	|	1	|	128x128x128	|	36.93	|	27.08	| 28.6 | 31.43 | 48.3 |
|	3	|	2	|	128x128x128	|	38.86	|	51.47	| 53.3 | 54.77 | 92.49 |
|	3	|	4	|	128x128x128	|	39.15	|	102.18	| 104.62 | 112.17 | 180.47 |

FP32
 
| Dimension | Batch size |Resolution| Throughput Avg [img/s] | Latency Avg [ms] | Latency 90% [ms] | Latency 95% [ms] | Latency 99% [ms] |
|:----------:|:---------:|:-------------:|:----------------------:|:----------------:|:----------------:|:----------------:|:----------------:|
|	2	|	32	|	192x160	|	1019.97	|	31.37	| 32.93 | 55.58 | 69.14 |
|	2	|	64	|	192x160	|	1063.59	|	60.17	| 62.32 | 63.11 | 111.01 |
|	2	|	128	|	192x160	|	1069.81	|	119.65	| 123.48 | 123.83 | 225.46 |
|	3	|	1	|	128x128x128	|	9.92	|	100.78	| 103.2 | 103.62 | 111.97 |
|	3	|	2	|	128x128x128	|	10.14	|	197.33	| 201.05 | 201.4 | 201.79 |
|	3	|	4	|	128x128x128	|	10.25	|	390.33	| 398.21 | 399.34 | 401.05 |

Throughput is reported in images per second. Latency is reported in milliseconds per batch.
To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

## Release notes

### Changelog

November 2022
- Container updated to 22.11
- Add support for 3D channel last convolutions
- Add support for nvFuser Instance Normalization
- Add support for GPU binding

October 2022
- Add Jupyter Notebook with BraTS'22 solution (ranked 2)

December 2021
- Container updated to 22.11
- Use MONAI DynUNet instead of custom U-Net implementation
- Add balanced multi-GPU evaluation
- Support for evaluation with resampled volumes to original shape

October 2021
- Add Jupyter Notebook with BraTS'21 solution (ranked 3)

May 2021
- Add Triton Inference Server support
- Removed deep supervision, attention, and drop block

March 2021
- Container updated to 21.02
- Change data format from tfrecord to npy and data loading for 2D 

January 2021
- Initial release
- Add notebook with custom dataset loading

### Known issues

There are no known issues in this release.

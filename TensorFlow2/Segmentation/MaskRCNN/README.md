# Mask R-CNN For Tensorflow

This repository provides a script and recipe to train the Mask R-CNN model for Tensorflow to achieve state-of-the-art accuracy, and is tested and maintained by NVIDIA.

## Table Of Contents

- [Model overview](#model-overview)
    * [Model architecture](#model-architecture)
    * [Default configuration](#default-configuration)
    * [Feature support matrix](#feature-support-matrix)
        * [Features](#features)
    * [Mixed precision training](#mixed-precision-training)
        * [Enabling mixed precision](#enabling-mixed-precision)
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

- [Performance](#performance)
    * [Benchmarking](#benchmarking)
        * [Training performance benchmark](#training-performance-benchmark)
        * [Inference performance benchmark](#inference-performance-benchmark)
    * [Results](#results)
        * [Training accuracy results in TensorFlow 1.1x](#training-accuracy-results-in-tensorflow-11x)
            * [Training accuracy: NVIDIA DGX-1 (8x V100 16G)](#training-accuracy-nvidia-dgx-1-8x-v100-16g)
            * [Training stability test](#training-stability-test)
            * [Training performance results](#training-performance-results)
            * [Training performance: NVIDIA DGX-1 (8x V100 16G)](#training-performance-nvidia-dgx-1-8x-v100-16G)
        * [Training accuracy results in TensorFlow 2.0](#training-accuracy-results-in-tensorflow-20)
            * [Training accuracy: NVIDIA DGX-1 (8x V100 16G)](#training-accuracy-nvidia-dgx-1-8x-v100-16g_1)
            * [Training stability test](#training-stability-test_1)
            * [Training performance results](#training-performance-results_1)
            * [Training performance: NVIDIA DGX-1 (8x V100 16G)](#training-performance-nvidia-dgx-1-8x-v100-16G_1)
        * [Inference performance results in TensorFlow 1.1x](#inference-performance-results-in-tensorflow-11x)
            * [Inference performance: NVIDIA DGX-1 (1x V100 16G)](#inference-performance-nvidia-dgx-1-1x-v100-16g)
        * [Inference performance results in TensorFlow 2.0](#inference-performance-results-in-tensorflow-2x)
            * [Inference performance: NVIDIA DGX-1 (1x V100 16G)](#inference-performance-nvidia-dgx-1-1x-v100-16g_1)
- [Release notes](#release-notes)
    * [Changelog](#changelog)
    * [Known issues](#known-issues)

## Model overview

Mask R-CNN is a convolution-based neural network for the task of object instance segmentation. The paper describing the model can be found [here](https://arxiv.org/abs/1703.06870). NVIDIA’s Mask R-CNN 19.12 is an optimized version of [Google's TPU implementation](https://github.com/tensorflow/tpu/tree/master/models/official/mask_rcnn), leveraging mixed precision arithmetic using Tensor Cores on NVIDIA Volta and Turing GPUs while maintaining target accuracy. 
Because this model trains with mixed precision using Tensor Cores on Volta, researchers can get results much faster than training without Tensor Cores. This model is tested against each NGC monthly container release to ensure consistent 
accuracy and performance over time.

This repository also contains scripts to interactively launch training, 
benchmarking and inference routines in a Docker container.

The major differences between the official implementation of the paper and our version of Mask R-CNN are as follows:

- Mixed precision support with [TensorFlow AMP](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html#tfamp).
- Gradient accumulation to simulate larger batches.
- Custom fused CUDA kernels for faster computations.

There are other publicly NVIDIA available implementations of Mask R-CNN:

- [NVIDIA PyTorch implementation](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/MaskRCNN)
- [Matterport](https://github.com/matterport/Mask_RCNN)
- [Tensorpack](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN)

### Model architecture

Mask R-CNN builds on top of Faster R-CNN adding an additional mask head for the task of image segmentation.

The architecture consists of the following:

- ResNet-50 backbone with Feature Pyramid Network (FPN)
- Region proposal network (RPN) head
- RoI Align
- Bounding and classification box head
- Mask head

### Default configuration

The Mask R-CNN configuration and the hyper-parameters for training and testing purposes are in separate files.
The default configuration of this model can be found at `mask-rcnn/hyperparameters/mask_rcnn_params.py`. 

The default configuration is as follows:

  - Feature extractor:
    - Images resized with aspect ratio maintained and smaller side length between [832,1344]
    - Ground Truth mask size 112
    - Backbone network weights are frozen after second epoch

  - RPN:
    - Anchor stride set to 16
    - Anchor sizes set to (32, 64, 128, 256, 512)
    - Foreground IOU Threshold set to 0.7, Background IOU Threshold set to 0.3
    - RPN target fraction of positive proposals set to 0.5
    - Train Pre-NMS Top proposals set to 2000 per FPN layer
    - Train Post-NMS Top proposals set to 1000
    - Test Pre-NMS Top proposals set to 1000 per FPN layer
    - Test Post-NMS Top proposals set to 1000
    - RPN NMS Threshold set to 0.7

  - RoI heads:
    - Foreground threshold set to 0.5
    - Batch size per image set to 512
    - Positive fraction of batch set to 0.25

The default hyper-parameters can be found at `mask-rcnn/hyperparameters/cmdline_utils.py`. 
These hyperparameters can be overridden through the command-line options, in the launch scripts.

### Feature support matrix

The following features are supported by this model:

| **Feature** | **Mask R-CNN** |
|:---------:|:----------:|
|Horovod Multi-GPU|Yes|
|Automatic mixed precision (AMP)|Yes|        

#### Features

The following features are supported by this model.

**Horovod**

Horovod is a distributed training framework for TensorFlow, Keras, PyTorch and MXNet. The goal of Horovod is to make distributed deep learning fast and easy to use. For more information about how to get started with Horovod, see the [Horovod: Official repository](https://github.com/horovod/horovod).

**Multi-GPU training with Horovod**

Our model uses Horovod to implement efficient multi-GPU training with NCCL. For details, see example sources in this repository or see the [TensorFlow tutorial](https://github.com/horovod/horovod/#usage).

**Automatic Mixed Precision (AMP)**

Automatic Mixed Precision (TF-AMP) enables mixed precision training without any changes to the code-base by 
performing automatic graph rewrites and loss scaling controlled by an environmental variable.

### Mixed precision training

Mixed precision is the combined use of different numerical precision in a computational method. 
[Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of 
[Tensor Cores](https://developer.nvidia.com/tensor-cores) in the Volta and Turing architecture, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using mixed precision training requires two steps:

1.  Porting the model to use the FP16 data type where appropriate.
2.  Adding loss scaling to preserve small gradient values.

The ability to train deep learning networks with lower precision was introduced in the Pascal architecture and first 
supported in [CUDA 8](https://devblogs.nvidia.com/parallelforall/tag/fp16/) in the NVIDIA Deep Learning SDK.

For information about:
-   How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and 
[Training With Mixed Precision](https://docs.nvidia.com/deeplearning/sdk/Mixed-Precision-training/index.html) documentation.
-   Techniques used for mixed precision training, see the 
[Mixed Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
-   How to access and enable AMP for TensorFlow, see 
[Using TF-AMP](https://docs.nvidia.com/deeplearning/dgx/tensorflow-user-guide/index.html#tfamp) from the TensorFlow User Guide.

#### Enabling mixed precision

AMP for TensorFlow enables the full 
[mixed precision methodology](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#tensorflow) 
in your existing TensorFlow model code.  AMP enables mixed precision training on Volta and Turing GPUs automatically. 
The TensorFlow framework code makes all necessary model changes internally.

In TF-AMP, the computational graph is optimized to use as few casts as necessary and maximizes the use of FP16, and the loss scaling is automatically applied inside of supported optimizers. AMP can be configured to work with the existing experimental loss scaling optimizer: `tf.compat.v1.train.experimental.MixedPrecisionLossScaleOptimizer` by disabling the AMP scaling with a single environment variable to perform only the automatic mixed precision optimization. It accomplishes this by automatically rewriting all 
computation graphs with the necessary operations to enable mixed precision training and automatic loss scaling.

## Setup

The following section lists the requirements that you need to meet in order to start training the Mask R-CNN model.

### Requirements

This repository contains Dockerfile which extends the TensorFlow NGC container and encapsulates some dependencies. 
Aside from these dependencies, ensure you have the following components:

-   [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
-   [TensorFlow 20.02-tf2-py3 NGC Container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)
-   [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or 
[Turing](https://www.nvidia.com/en-us/geforce/turing/) based GPU

For more information about how to get started with NGC containers, see the following sections from the 
NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:

-   [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
-   [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)
-   Running [TensorFlow](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html#running)

For those unable to use the TensorFlow NGC container, to set up the required environment or create your own 
container, see the versioned 
[NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## Quick Start Guide

To train your model using mixed precision with Tensor Cores or using FP32, perform the following steps using 
the default parameters of the Mask R-CNN model on the COCO 2014 dataset.

1. Clone the repository.

    ```bash
    git clone https://github.com/NVIDIA/DeepLearningExamples.git
    cd DeepLearningExamples/TensorFlow/Segmentation/MaskRCNN
    ```

2.  Build the Mask R-CNN TensorFlow NGC container.

    **For TensorFlow 1.1x:** `bash ./scripts/docker/build_tf1.sh`

    **For TensorFlow 2.x:** `bash ./scripts/docker/build_tf2.sh`

3.  Start an interactive session in the NGC container to run training/inference.

    Run the following command to launch the Docker container, the only argument is the *absolute path* to the 
    `data directory` which holds or will hold the `tfrecords` data. If data has not already been downloaded in the `data directory` then download it in step 4, else step 4 can be skipped.
    
    **For TensorFlow 1.1x:** `bash ./scripts/docker/launch_tf1.sh [data directory]`    
    
    **For TensorFlow 2.x:** `bash ./scripts/docker/launch_tf2.sh [data directory]`

4.  Download and preprocess the dataset.

    This repository provides scripts to download and extract the [COCO 2017 dataset](http://cocodataset.org/#download).  
    If you already have the data then you do not need to run the following script, proceed to downloading the pre-trained weights. 
    Data will be downloaded to the `data directory` provided in step 3.
    
    ```bash
    cd dataset
    bash download_and_preprocess_coco.sh /data
    ```

    By default, the data is organized into the following structure:

    ```bash
    <data/dir>
    annotations/
      instances_train2017.json
      instances_val2017.json
    train2017/
      COCO_train2017_*.jpg
    val2017/
      COCO_val2017_*.jpg
    ```

    This repository also provides scripts to download the pre-trained weights of ResNet-50 backbone. 
    The script will make a new directory with the name `weights` in the current directory and 
    download the pre-trained weights in it.

    ```bash
    ./download_and_process_pretrained_weights.sh
    ```

    Ensure that the `weights` folder created has a `resnet` folder in it. Inside the `resnet` folder there 
    should be 3 folders for checkpoints and weights: `extracted_from_maskrcnn`, `resnet-nhwc-2018-02-07` and 
    `resnet-nhwc-2018-10-14`. Before moving to the next step, ensure the above folders are not empty.


5. Start training.
    
    To run training for a default configuration (on 1/4/8 GPUs, AMP/FP32), run one of the scripts in the 
    `./scripts` directory called `./scripts/train_{AMP,FP32}_{1,4,8}GPU{_XLA}.sh`. For example: 
    
    `bash ./scripts/train_AMP_8GPU.sh`

    The above script trains a model and performs an evaluation on the COCO 2017 dataset. By default, this training script:

    -  Uses 8 GPUs.
    -  Saves a checkpoint every 3696 iterations and at the end of training. All checkpoints, evaluation results and training logs are saved to the `/results` directory (in the container which can be mounted to a local directory).
    -  Mixed precision training with Tensor Cores.

6. Start validation/evaluation.

    - For evaluation with AMP precision: `bash ./scripts/evaluation_AMP.sh`
    - For evaluation with FP32 precision: `bash ./scripts/evaluation_FP32.sh`

## Advanced

The following sections provide greater details of the dataset, running training and inference, and the training results.

### Scripts and sample code

Descriptions of the key scripts and folders are provided below.

-  `mask_rcnn` - Contains codes to build individual components of the model such as 
backbone, FPN, RPN, mask and bbox heads etc.
-  `download_and_process_pretrained_weights.sh` - Can be used to download backbone pre-trained weights.
-  `scripts/` - A folder that contains shell scripts to train the model and perform inferences.
    -   `train_{AMP,FP32}_{1,4,8}GPU{_XLA}.sh` - Training script on 1, 4, 8 GPUs with AMP or FP32 precision and either with XLA (Accelerated Linear Algebra) of TensorFlow enabled or disabled.
    -   `evaluation_{AMP,FP32}.sh` - Evaluations script on either AMP precision or FP32 precision.
-  `dataset/` - A folder that contains shell scripts and Python files to download the dataset.
-  `mask_rcnn_main.py` - Is the main function that is the starting point for the training and evaluation process.
-  `docker/` - A folder that contains scripts to build a Docker image and start an interactive session.

### Parameters

#### `mask_rcnn_main.py` script parameters

You can modify the training behavior through the various flags in both the `train_net.py` script and through overriding specific parameters in the config files. Flags in the `mask_rcnn_main.py` script are as follows:

-   `--mode` - Specifies the action to take like `train`, `train_and_eval` or `eval`.
-   `--checkpoint` - The checkpoint of the backbone.
-   `--eval_samples` - Number of samples to evaluate.
-   `--init_learning_rate` - Initial learning rate.
-   `--learning_rate_steps` - Specifies at which steps to reduce the learning rate.
-   `--num_steps_per_eval` - Specifies after how many steps of training evaluation should be performed.
-   `--total_steps` - Specifies the total number of steps for which training should be run.
-   `--train_batch_size` - Training batch size per GPU.
-   `--eval_batch_size` - Evaluation batch size per GPU.
-   `--use_amp` - Specifies to use AMP precision or FP32.
-   `--use_xla` - Specifies to use XLA (Accelerated Linear Algebra) of TensorFlow or not.

### Command-line options

To see the full list of available options and their descriptions, use the `-h` or `--help` command-line option, for example:
`python mask_rcnn_main.py --helpfull`

### Getting the data

The Mask R-CNN model was trained on the COCO 2017 dataset.  This dataset comes with a training and validation set.

This repository contains the `./dataset/download_and_preprocess_coco.sh` script which automatically downloads and preprocesses the training and validation sets. The helper scripts are also present in the `dataset/` folder.

#### Dataset guidelines

The data should be organized into the following structure:

```bash
<data/dir>
annotations/
  instances_train2017.json
  instances_val2017.json
train2017/
  COCO_train2017_*.jpg
val2017/
  COCO_val2017_*.jpg
```

### Training process

Training is performed using the `mask_rcnn_main.py` script along with parameters defined in the config files. 
The default config files can be found in the 
`mask_rcnn_tf/mask_rcnn/mask_rcnn_params.py, mask_rcnn_tf/mask_rcnn/cmd_utils.py` files. To specify which GPUs to train on, `CUDA_VISIBLE_DEVICES` variable can be changed in the training scripts
provided in the `scripts` folder. 

This script outputs results to the `/results` directory by default. The training log will contain information about:

-   Loss, time per iteration, learning rate and memory metrics
-   Performance values such as throughput per step
-   Test accuracy and test performance values after evaluation

### Inference process

To run inference run `mask_rcnn_main.py` with commandline parameter 
`mode=eval`. To run inference with a checkpoint, set the commandline 
parameter `--model_dir` to `[absolute path of checkpoint folder]`.

The inference log will contain information about:

-   Inference time per step
-   Inference throughput per step
-   Evaluation accuracy and performance values


## Performance

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in training and inference modes.

#### Training performance benchmark

Training benchmarking can be performed by running the script:

To run training on a single GPU with either AMP or FP32 precision with or without XLA, run the following script:

```bash
bash scripts/train_{AMP,FP32}_1GPU{_XLA}.sh
```

To run training on 8 GPUs with either AMP or FP32 precision with or without XLA, run the following script:

```bash
bash scripts/train_{AMP,FP32}_8GPU{_XLA}.sh
```

#### Inference performance benchmark

Inference benchmarking can be performed by running the script:

To run on a single GPU with either AMP or FP32 precision, run the following script:

```
bash scripts/evaluation_{AMP, FP32}.sh
```

### Results

The following sections provide details on how we achieved our performance and accuracy in training and inference.

#### Training accuracy results in Tensorflow 1.1x

##### Training accuracy: NVIDIA DGX-1 (8x V100 16G)

Our results were obtained by building and launching the docker containers for TensorFlow 1.1x `./scripts/docker/build_tf1.sh`, `bash ./scripts/docker/launch_tf1.sh [data directory]` respectively and running the `scripts/train_{AMP,FP32}_{1,4,8}GPU{_XLA}.sh`  training script on NVIDIA DGX-1 with 8x V100 16G GPUs.

| **Number of GPUs** | **Batch Size** | **Training time with AMP (hours)** | **Training time with FP32 (hours)** |
| --- | --- | ----- | ----- |
| 8 | 4 | 9.43 | 13.02 |


| **Precision** | **Number of GPUs** | **Batch size/GPU** | **Final AP BBox** | **Final AP Segm** |
| --- | --- | ----- | ----- | ----- |
| **AMP** | 8 | 4 | 0.378 | 0.343 |
| **FP32** | 8 | 4 | 0.377 | 0.343 |


##### Training stability test

The following tables compare the mAP scores across 5 different training runs with different seeds, for both AMP and FP32 respectively.  The runs showcase consistent convergence on all 5 seeds with very little deviation.

| **Config** | **Seed #1** | **Seed #2** | **Seed #3** |  **Seed #4** | **Seed #5** | **mean** | **std** |
| --- | --- | ----- | ----- | --- | --- | ----- | ----- |
|  8 GPUs, AMP, final AP BBox  | 0.377 | 0.378 | 0.379 | 0.376  | 0.379 | 0.378 | 0.001 |
| 8 GPUs, AMP, final AP Segm | 0.342 | 0.342 | 0.344 | 0.341  | 0.342 | 0.342 | 0.001 |


| **Config** | **Seed #1** | **Seed #2** | **Seed #3** |  **Seed #4** | **Seed #5** | **mean** | **std** |
| --- | --- | ----- | ----- | --- | --- | ----- | ----- |
|  8 GPUs, FP32, final AP BBox  | 0.379 | 0.378 | 0.376 | 0.376  | 0.378 | 0.377 | 0.001 |
| 8 GPUs, FP32, final AP Segm | 0.343 | 0.343 | 0.342 | 0.343  | 0.343 | 0.343 | 0.0004 |


##### Training performance results

##### Training performance: NVIDIA DGX-1 (8x V100 16G)

| **Number of GPUs** | **Batch size/GPU** | **FP 32 items/sec** | **AMP items/sec** | **Speed-up with mixed precision** |
| --- | --- | ----- | ----- | --- |
| 1 | 2 | 3.2 | 4.2 | 1.315 |
| 4 | 2 | 15.1 | 22.4 | 1.48 |
| 8 | 2 | 27.8 | 47.3 | 1.701 |

| **Number of GPUs** | **Batch size/GPU** | **FP 32 items/sec** | **AMP items/sec** | **Speed-up with mixed precision** |
| --- | --- | ----- | ----- | --- |
| 1 | 4 | 3.4 | 4.7 | 1.38 |
| 4 | 4 | 21.7 | 31.03 | 1.42 |
| 8 | 4 | 38.4 | 46.5 | 1.21 |

Model performances can be improved upon by using XLA. 

Note: This feature is still experimental and can be unstable.

| **Number of GPUs** | **Batch size/GPU** | **FP 32 items/sec** | **AMP items/sec** | **Speed-up with mixed precision** |
| --- | --- | ----- | ----- | ---- |
| 8   |   4 |  48.9 |  57.7 | 1.18 |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

#### Training accuracy results in Tensorflow 2.1

##### Training accuracy: NVIDIA DGX-1 (8x V100 16G)

Our results were obtained by running the `scripts/train_{AMP,FP32}_{1,4,8}GPU{_XLA}.sh`  training script in the 
TensorFlow 20.02-py3 NGC container on NVIDIA DGX-1 with 8x V100 16G GPUs.

| **Number of GPUs** | **Batch Size** | **Training time with AMP (hours)** | **Training time with FP32 (hours)** |
| --- | --- | ----- | ----- |
| 8 | 4 | 9.4 | 13.08 |


| **Precision** | **Number of GPUs** | **Batch size/GPU** | **Final AP BBox** | **Final AP Segm** |
| --- | --- | ----- | ----- | ----- |
| **AMP** | 8 | 4 | 0.378 | 0.340 |
| **FP32** | 8 | 4 | 0.378 | 0.341 |


##### Training stability test

The following tables compare the mAP scores across 5 different training runs with different seeds, for both 
AMP and FP32 respectively.  The runs showcase consistent convergence on all 5 seeds with very little deviation.

| **Config** | **Seed #1** | **Seed #2** | **Seed #3** |  **Seed #4** | **Seed #5** | **mean** | **std** |
| --- | --- | ----- | ----- | --- | --- | ----- | ----- |
|  8 GPUs, AMP, final AP BBox  | 0.378 | 0.376 | 0.377 | 0.379  | 0.377 | 0.3774 | 0.001 |
| 8 GPUs, AMP, final AP Segm | 0.341 | 0.339 | 0.339 | 0.342  | 0.341 | 0.3402 | 0.001 |


| **Config** | **Seed #1** | **Seed #2** | **Seed #3** |  **Seed #4** | **Seed #5** | **mean** | **std** |
| --- | --- | ----- | ----- | --- | --- | ----- | ----- |
|  8 GPUs, FP32, final AP BBox  | 0.378 | 0.379 | 0.377 | 0.379  | 0.378 | 0.3782 | 0.001 |
| 8 GPUs, FP32, final AP Segm | 0.341 | 0.339 | 0.339 | 0.342  | 0.339 | 0.3401 | 0.002 |


##### Training performance results

##### Training performance: NVIDIA DGX-1 (8x V100 16G)

| **Number of GPUs** | **Batch size/GPU** | **FP 32 items/sec** | **AMP items/sec** | **Speed-up with mixed precision** |
| --- | --- | ----- | ----- | --- |
| 1 | 2 | 4.5 | 6.6 | 1.466 |
| 4 | 2 | 21.0 | 34.8 | 1.657 |
| 8 | 2 | 38.3 | 49.8 | 1.300 |

| **Number of GPUs** | **Batch size/GPU** | **FP 32 items/sec** | **AMP items/sec** | **Speed-up with mixed precision** |
| --- | --- | ----- | ----- | --- |
| 1 | 4 | 4.7 | 7.0 | 1.489 |
| 4 | 4 | 22.2 | 35.8 | 1.612 |
| 8 | 4 | 38.9 | 54.1 | 1.390 |

Model performances can be improved upon by using XLA. 

Note: This feature is still experimental and can be unstable.

| **Number of GPUs** | **Batch size/GPU** | **FP 32 items/sec** | **AMP items/sec** | **Speed-up with mixed precision** |
| --- | --- | ----- | ----- | ---- |
| 8   |   4 |  37.4 |  43.5 | 1.16 |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

#### Inference performance results in TensorFlow 1.1x

##### Inference performance: NVIDIA DGX-1 (1x V100 16G)

Our results were obtained by building and launching the docker containers for TensorFlow 1.1x `./scripts/docker/build_tf1.sh`, `bash ./scripts/docker/launch_tf1.sh [data directory]` respectively and by running the `./scripts/evaluation_{AMP,FP32}.sh` script on NVIDIA DGX-1 with 1x V100 16G GPUs. Performance numbers (in items/images per second)
were averaged over an entire training epoch.

| **Number of GPUs** | **Batch size/GPU** | **FP 32 items/sec** | **AMP items/sec** | **Speedup** |
| --- | --- | ----- | ----- | ----- |
|  1  | 8 | 5.2 | 6.6 | 1.269 |

Latency is computed as the time taken for a batch to process as they are fed in one after another in the model ie no pipelining

Precision AMP

| Batch Size (per GPU) | Throughput-Average (images/sec) | Latency-Average (sec) | Latency-90% (sec) | Latency-95% (sec) | Latency-99% (sec) |
|------------|------------------------------|---------------------|-----------------|-----------------|-----------------|
| 2 | 5.5 | 0.4407  | 0.4134 | 0.4234 | 0.4510 |
| 4 | 6.2 | 0.7716  | 0.7241 | 0.7361 | 0.8878 |
| 8 | 6.6 | 1.3771  | 1.2011 | 1.2340 | 1.6586 |


Precision FP32

| Batch Size (per GPU) | Throughput-Average (images/sec) | Latency-Average (sec) | Latency-90% (sec) | Latency-95% (sec) | Latency-99% (sec) |
|------------|------------------------------|---------------------|-----------------|-----------------|-----------------|
| 2 | 4.9 | 0.4638  | 0.4062 | 0.4528 | 0.5144 |++
| 4 | 4.7 | 0.9632  | 0.9055 | 0.9293 | 0.9868 |
| 8 | 6.0 | 2.1580  | 2.0107 | 2.0368 | 2.4152 |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

#### Inference performance results in TensorFlow 2.x

##### Inference performance: NVIDIA DGX-1 (1x V100 16G)

Our results were obtained by running the `./scripts/evaluation_{AMP,FP32}.sh` training script in the TensorFlow 20.02-py3 
NGC container on NVIDIA DGX-1 with 1x V100 16G GPUs. Performance numbers (in items/images per second)
were averaged over an entire training epoch.

| **Number of GPUs** | **Batch size/GPU** | **FP 32 items/sec** | **AMP items/sec** | **Speedup** |
| --- | --- | ----- | ----- | ----- |
|  1  | 8 | 5.4 | 5.5 | 1.018 |

Latency is computed as the time taken for a batch to process as they are fed in one after another in the model ie no pipelining

Precision AMP

| Batch Size (per GPU) | Throughput-Average (images/sec) | Latency-Average (sec) | Latency-90% (sec) | Latency-95% (sec) | Latency-99% (sec) |
|------------|------------------------------|---------------------|-----------------|-----------------|-----------------|
| 2 | 6.8 | 0.39  | 0.38 | 0.39 | 0.42 |
| 4 | 7.4 | 0.59  | 0.58 | 0.59 | 0.64 |
| 8 | 5.5 | 1.71  | 1.69 | 1.7 | 1.83 |


Precision FP32

| Batch Size (per GPU) | Throughput-Average (images/sec) | Latency-Average (sec) | Latency-90% (sec) | Latency-95% (sec) | Latency-99% (sec) |
|------------|------------------------------|---------------------|-----------------|-----------------|-----------------|
| 2 | 5.4 | 0.46  | 0.41 | 0.42 | 0.43 |++
| 4 | 5.3 | 0.86  | 0.82 | 0.84 | 0.89 |
| 8 | 5.4 | 1.69  | 1.65 | 1.68 | 1.83 |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

## Release notes

### Changelog

March 2020

- Initial release

### Known issues

-  The behavior of the model can be unstable when running with TensorFlow XLA enabled.




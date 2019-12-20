# ResNet50 v1.5 For PyTorch

This repository provides a script and recipe to train the ResNet50 model to
achieve state-of-the-art accuracy, and is tested and maintained by NVIDIA.

## Table Of Contents

* [Model overview](#model-overview)
  * [Model architecture](#model-architecture)
  * [Default configuration](#default-configuration)
    * [Optimizer](#optimizer)
    * [Data augmentation](#data-augmentation)
  * [DALI](#dali)
  * [Feature support matrix](#feature-support-matrix)
    * [Features](#features)
  * [Mixed precision training](#mixed-precision-training)
    * [Enabling mixed precision](#enabling-mixed-precision)
* [Setup](#setup)
  * [Requirements](#requirements)
* [Quick Start Guide](#quick-start-guide)
* [Advanced](#advanced)
  * [Scripts and sample code](#scripts-and-sample-code)
    * [Parameters](#parameters)
    * [Command-line options](#command-line-options)
    * [Getting the data](#getting-the-data)
        * [Dataset guidelines](#dataset-guidelines)
        * [Multi-dataset](#multi-dataset)
    * [Training process](#training-process)
    * [Inference process](#inference-process)

* [Performance](#performance)
  * [Benchmarking](#benchmarking)
    * [Training performance benchmark](#training-performance-benchmark)
    * [Inference performance benchmark](#inference-performance-benchmark)
  * [Results](#results)
    * [Training accuracy results](#training-accuracy-results)
      * [Training accuracy: NVIDIA DGX-1 (8x V100 16G)](#training-accuracy-nvidia-dgx-1-(8x-v100-16G))
      * [Example plots](*example-plots)
    * [Training performance results](#training-performance-results)
      * [Training performance: NVIDIA DGX-1 (8x V100 16G)](#training-performance-nvidia-dgx-1-(8x-v100-16G))
    * [Training time for 90 epochs](#training-time-for-90-epochs)
      * [Training time: NVIDIA DGX-1 (8x V100 16G)](#training-time-nvidia-dgx-1-(8x-v100-16G))
  * [Inference performance results](#inference-performance-results)
      * [Inference performance: NVIDIA DGX-1 (1x V100 16G)](#inference-performance-nvidia-dgx-1-(1x-v100-16G))
      * [Inference performance: NVIDIA T4](#inference-performance-nvidia-t4)
* [Release notes](#release-notes)
  * [Changelog](#changelog)
  * [Known issues](#known-issues)

## Model overview
The ResNet50 v1.5 model is a modified version of the [original ResNet50 v1 model](https://arxiv.org/abs/1512.03385).

The difference between v1 and v1.5 is that, in the bottleneck blocks which requires
downsampling, v1 has stride = 2 in the first 1x1 convolution, whereas v1.5 has stride = 2 in the 3x3 convolution.

This difference makes ResNet50 v1.5 slightly more accurate (~0.5% top1) than v1, but comes with a smallperformance drawback (~5% imgs/sec).

The model is initialized as described in [Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification](https://arxiv.org/pdf/1502.01852.pdf)

### Default configuration

The following sections highlight the default configurations for the ResNet50 model.

#### Optimizer

This model uses SGD with momentum optimizer with the following hyperparameters:

* Momentum (0.875)

* Learning rate (LR) = 0.256 for 256 batch size, for other batch sizes we lineary
scale the learning rate.

* Learning rate schedule - we use cosine LR schedule

* For bigger batch sizes (512 and up) we use linear warmup of the learning rate
during the first couple of epochs
according to [Training ImageNet in 1 hour](https://arxiv.org/abs/1706.02677).
Warmup length depends on the total training length.

* Weight decay (WD)= 3.0517578125e-05 (1/32768).

* We do not apply WD on Batch Norm trainable parameters (gamma/bias)

* Label smoothing = 0.1

* We train for:

    * 50 Epochs -> configuration that reaches 75.9% top1 accuracy

    * 90 Epochs -> 90 epochs is a standard for ImageNet networks

    * 250 Epochs -> best possible accuracy.

* For 250 epoch training we also use [MixUp regularization](https://arxiv.org/pdf/1710.09412.pdf).


#### Data augmentation

This model uses the following data augmentation:

* For training:
  * Normalization
  * Random resized crop to 224x224
    * Scale from 8% to 100%
    * Aspect ratio from 3/4 to 4/3
  * Random horizontal flip

* For inference:
  * Normalization
  * Scale to 256x256
  * Center crop to 224x224

#### Other training recipes

This script does not target any specific benchmark.
There are changes that others have made which can speed up convergence and/or increase accuracy.

One of the more popular training recipes is provided by [fast.ai](https://github.com/fastai/imagenet-fast).

The fast.ai recipe introduces many changes to the training procedure, one of which is progressive resizing of the training images.

The first part of training uses 128px images, the middle part uses 224px images, and the last part uses 288px images.
The final validation is performed on 288px images.

Training script in this repository performs validation on 224px images, just like the original paper described.

These two approaches can't be directly compared, since the fast.ai recipe requires validation on 288px images,
and this recipe keeps the original assumption that validation is done on 224px images.

Using 288px images means that a lot more FLOPs are needed during inference to reach the same accuracy.

### Feature support matrix

The following features are supported by this model:

| Feature               | ResNet50
|-----------------------|--------------------------
|[DALI](https://docs.nvidia.com/deeplearning/sdk/dali-release-notes/index.html)   |   Yes
|[APEX AMP](https://nvidia.github.io/apex/amp.html) | Yes |

#### Features

- NVIDIA DALI - DALI is a library accelerating data preparation pipeline. To accelerate your input pipeline, you only need to define your data loader
with the DALI library. For more information about DALI, refer to the [DALI product documentation](https://docs.nvidia.com/deeplearning/sdk/index.html#data-loading).

- [APEX](https://github.com/NVIDIA/apex) is a PyTorch extension that contains utility libraries, such as [Automatic Mixed Precision (AMP)](https://nvidia.github.io/apex/amp.html), which require minimal network code changes to leverage Tensor Cores performance. Refer to the [Enabling mixed precision](#enabling-mixed-precision) section for more details.

### DALI

We use [NVIDIA DALI](https://github.com/NVIDIA/DALI),
which speeds up data loading when CPU becomes a bottleneck.
DALI can use CPU or GPU, and outperforms the PyTorch native dataloader.

Run training with `--data-backends dali-gpu` or `--data-backends dali-cpu` to enable DALI.
For DGX1 we recommend `--data-backends dali-cpu`, for DGX2 we recommend `--data-backends dali-gpu`.

### Mixed precision training

Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format, while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in the Volta and Turing architecture, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using mixed precision training requires two steps:
1.  Porting the model to use the FP16 data type where appropriate.
2.  Adding loss scaling to preserve small gradient values.

The ability to train deep learning networks with lower precision was introduced in the Pascal architecture and first supported in [CUDA 8](https://devblogs.nvidia.com/parallelforall/tag/fp16/) in the NVIDIA Deep Learning SDK.

For information about:
-   How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html) documentation.
-   Techniques used for mixed precision training, see the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
-   How to access and enable AMP for TensorFlow, see [Using TF-AMP](https://docs.nvidia.com/deeplearning/dgx/tensorflow-user-guide/index.html#tfamp) from the TensorFlow User Guide.
-   APEX tools for mixed precision training, see the [NVIDIA Apex: Tools for Easy Mixed-Precision Training in PyTorch](https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/).

#### Enabling mixed precision

Mixed precision is enabled in PyTorch by using the Automatic Mixed Precision (AMP), a library from [APEX](https://github.com/NVIDIA/apex) that casts variables to half-precision upon retrieval,
while storing variables in single-precision format. Furthermore, to preserve small gradient magnitudes in backpropagation, a [loss scaling](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#lossscaling) step must be included when applying gradients.
In PyTorch, loss scaling can be easily applied by using scale_loss() method provided by AMP. The scaling value to be used can be [dynamic](https://nvidia.github.io/apex/fp16_utils.html#apex.fp16_utils.DynamicLossScaler) or fixed.

For an in-depth walk through on AMP, check out sample usage [here](https://github.com/NVIDIA/apex/tree/master/apex/amp#usage-and-getting-started). [APEX](https://github.com/NVIDIA/apex) is a PyTorch extension that contains utility libraries, such as AMP, which require minimal network code changes to leverage tensor cores performance.

To enable mixed precision, you can:
- Import AMP from APEX, for example:

  ```
  from apex import amp
  ```
- Initialize an AMP handle, for example:

  ```
  amp_handle = amp.init(enabled=True, verbose=True)
  ```
- Wrap your optimizer with the AMP handle, for example:

  ```
  optimizer = amp_handle.wrap_optimizer(optimizer)
  ```
- Scale loss before backpropagation (assuming loss is stored in a variable called losses)
  - Default backpropagate for FP32:

    ```
    losses.backward()
    ```
  - Scale loss and backpropagate with AMP:

    ```
    with optimizer.scale_loss(losses) as scaled_losses:
       scaled_losses.backward()
    ```

## Setup

The following section lists the requirements that you need to meet in order to start training the ResNet50 model.

### Requirements

This repository contains Dockerfile which extends the PyTorch NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:

* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch 19.10-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch) or newer
* [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) based GPU

For more information about how to get started with NGC containers, see the
following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning
DGX Documentation:
* [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
* [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/dgx/user-guide/index.html#accessing_registry)
* [Running PyTorch](https://docs.nvidia.com/deeplearning/dgx/pytorch-release-notes/running.html#running)

For those unable to use the PyTorch NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## Quick Start Guide

### 1. Clone the repository.
```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/PyTorch/Classification/
```

### 2. Download and preprocess the dataset.

The ResNet50 script operates on ImageNet 1k, a widely popular image classification dataset from the ILSVRC challenge.

PyTorch can work directly on JPEGs, therefore, preprocessing/augmentation is not needed.

1. [Download the images](http://image-net.org/download-images).

2. Extract the training data:
  ```bash
  mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
  tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
  find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
  cd ..
  ```

3. Extract the validation data and move the images to subfolders:
  ```bash
  mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
  wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
  ```

The directory in which the `train/` and `val/` directories are placed, is referred to as `<path to imagenet>` in this document.

### 3. Build the RN50v1.5 PyTorch NGC container.

```
docker build . -t nvidia_rn50
```

### 4. Start an interactive session in the NGC container to run training/inference.
```
nvidia-docker run --rm -it -v <path to imagenet>:/data/imagenet --ipc=host nvidia_rn50
```

### 5. Start training

To run training for a standard configuration (DGX1V/DGX2V, FP16/FP32, 50/90/250 Epochs),
run one of the scripts in the `./resnet50v1.5/training` directory
called `./resnet50v1.5/training/{DGX1, DGX2}_RN50_{AMP, FP16, FP32}_{50,90,250}E.sh`.

Ensure ImageNet is mounted in the `/data/imagenet` directory.

Example:
    `bash ./resnet50v1.5/training/DGX1_RN50_FP16_250E.sh <path were to store checkpoints and logs>`

### 6. Start inference

To run inference on ImageNet on a checkpointed model, run:

`python ./main.py --arch resnet50 --evaluate --epochs 1 --resume <path to checkpoint> -b <batch size> <path to imagenet>`

To run inference on JPEG image, you have to first extract the model weights from checkpoint:

`python checkpoint2model.py --checkpoint-path <path to checkpoint> --weight-path <path where weights will be stored>`

Then run classification script:

`python classify.py --arch resnet50 -c fanin --weights <path to weights from previous step> --precision AMP|FP16|FP32 --image <path to JPEG image>`


## Advanced

The following sections provide greater details of the dataset, running training and inference, and the training results.

### Scripts and sample code

To run a non standard configuration use:

* For 1 GPU
    * FP32
        `python ./main.py --arch resnet50 -c fanin --label-smoothing 0.1 <path to imagenet>`
        `python ./main.py --arch resnet50 -c fanin --label-smoothing 0.1 --amp --static-loss-scale 256 <path to imagenet>`

* For multiple GPUs
    * FP32
        `python ./multiproc.py --nproc_per_node 8 ./main.py --arch resnet50 -c fanin --label-smoothing 0.1 <path to imagenet>`
    * AMP
        `python ./multiproc.py --nproc_per_node 8 ./main.py --arch resnet50 -c fanin --label-smoothing 0.1 --amp --static-loss-scale 256 <path to imagenet>`

Use `python ./main.py -h` to obtain the list of available options in the `main.py` script.


### Commmand-line options:

To see the full list of available options and their descriptions, use the `-h` or `--help` command-line option, for example:

`python main.py -h`


```
usage: main.py [-h] [--data-backend BACKEND] [--arch ARCH]
               [--model-config CONF] [-j N] [--epochs N] [-b N]
               [--optimizer-batch-size N] [--lr LR] [--lr-schedule SCHEDULE]
               [--warmup E] [--label-smoothing S] [--mixup ALPHA]
               [--momentum M] [--weight-decay W] [--bn-weight-decay]
               [--nesterov] [--print-freq N] [--resume PATH]
               [--pretrained-weights PATH] [--fp16]
               [--static-loss-scale STATIC_LOSS_SCALE] [--dynamic-loss-scale]
               [--prof N] [--amp] [--local_rank LOCAL_RANK] [--seed SEED]
               [--gather-checkpoints] [--raport-file RAPORT_FILE] [--evaluate]
               [--training-only] [--no-checkpoints] [--workspace DIR]
               DIR

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --data-backend BACKEND
                        data backend: pytorch | syntetic | dali-gpu | dali-cpu
                        (default: dali-cpu)
  --arch ARCH, -a ARCH  model architecture: resnet18 | resnet34 | resnet50 |
                        resnet101 | resnet152 | resnet50 | se-
                        resnet50 (default: resnet50)
  --model-config CONF, -c CONF
                        model configs: classic | fanin | grp-fanin | grp-
                        fanout(default: classic)
  -j N, --workers N     number of data loading workers (default: 5)
  --epochs N            number of total epochs to run
  -b N, --batch-size N  mini-batch size (default: 256) per gpu
  --optimizer-batch-size N
                        size of a total batch size, for simulating bigger
                        batches using gradient accumulation
  --lr LR, --learning-rate LR
                        initial learning rate
  --lr-schedule SCHEDULE
                        Type of LR schedule: step, linear, cosine
  --warmup E            number of warmup epochs
  --label-smoothing S   label smoothing
  --mixup ALPHA         mixup alpha
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --bn-weight-decay     use weight_decay on batch normalization learnable
                        parameters, (default: false)
  --nesterov            use nesterov momentum, (default: false)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  --pretrained-weights PATH
                        load weights from here
  --fp16                Run model fp16 mode.
  --static-loss-scale STATIC_LOSS_SCALE
                        Static loss scale, positive power of 2 values can
                        improve fp16 convergence.
  --dynamic-loss-scale  Use dynamic loss scaling. If supplied, this argument
                        supersedes --static-loss-scale.
  --prof N              Run only N iterations
  --amp                 Run model AMP (automatic mixed precision) mode.
  --local_rank LOCAL_RANK
                        Local rank of python process. Set up by distributed
                        launcher
  --seed SEED           random seed used for numpy and pytorch
  --gather-checkpoints  Gather checkpoints throughout the training, without
                        this flag only best and last checkpoints will be
                        stored
  --raport-file RAPORT_FILE
                        file in which to store JSON experiment raport
  --evaluate            evaluate checkpoint/model
  --training-only       do not evaluate
  --no-checkpoints      do not store any checkpoints, useful for benchmarking
  --workspace DIR       path to directory where checkpoints will be stored
```


### Dataset guidelines

To use your own dataset, divide it in directories as in the following scheme:

 - Training images - `train/<class id>/<image>`
 - Validation images - `val/<class id>/<image>`

If your dataset's has number of classes different than 1000, you need to add a custom config
in the `image_classification/resnet.py` file.

```python
resnet_versions = {
    ...
    'resnet50-custom' : {
       'net' : ResNet,
       'block' : Bottleneck,
       'layers' : [3, 4, 6, 3],
       'widths' : [64, 128, 256, 512],
       'expansion' : 4,
       'num_classes' : <custom number of classes>,
       }
}
```

After adding the config, run the training script with `--arch resnet50-custom` flag.

### Training process

All the results of the training will be stored in the directory specified with `--workspace` argument.
Script will store:
 - most recent checkpoint - `checkpoint.pth.tar` (unless `--no-checkpoints` flag is used).
 - checkpoint with best validation accuracy - `model_best.pth.tar` (unless `--no-checkpoints` flag is used).
 - JSON log - in the file specified with `--raport-file` flag.

Metrics gathered through training:

 - `train.loss` - training loss
 - `train.total_ips` - training speed measured in images/second
 - `train.compute_ips` - training speed measured in images/second, not counting data loading
 - `train.data_time` - time spent on waiting on data
 - `train.compute_time` - time spent in forward/backward pass

### Inference process

Validation is done every epoch, and can be also run separately on a checkpointed model.

`python ./main.py --arch resnet50 --evaluate --epochs 1 --resume <path to checkpoint> -b <batch size> <path to imagenet>`

Metrics gathered through training:

 - `val.loss` - validation loss
 - `val.top1` - validation top1 accuracy
 - `val.top5` - validation top5 accuracy
 - `val.total_ips` - inference speed measured in images/second
 - `val.compute_ips` - inference speed measured in images/second, not counting data loading
 - `val.data_time` - time spent on waiting on data
 - `val.compute_time` - time spent on inference


To run inference on JPEG image, you have to first extract the model weights from checkpoint:

`python checkpoint2model.py --checkpoint-path <path to checkpoint> --weight-path <path where weights will be stored>`

Then run classification script:

`python classify.py --arch resnet50 -c fanin --weights <path to weights from previous step> --precision AMP|FP16|FP32 --image <path to JPEG image>`

Example output:



## Performance

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in training and inference modes.

#### Training performance benchmark

To benchmark training, run:

* For 1 GPU
    * FP32
`python ./main.py --arch resnet50 --training-only -p 1 --raport-file benchmark.json --epochs 1 --prof 100 <path to imagenet>`
    * FP16
`python ./main.py --arch resnet50 --training-only -p 1 --raport-file benchmark.json --epochs 1 --prof 100 --fp16 --static-loss-scale 256 <path to imagenet>`
    * AMP
`python ./main.py --arch resnet50 --training-only -p 1 --raport-file benchmark.json --epochs 1 --prof 100 --amp --static-loss-scale 256 <path to imagenet>`
* For multiple GPUs
    * FP32
`python ./multiproc.py --nproc_per_node 8 ./main.py --arch resnet50 --training-only -p 1 --raport-file benchmark.json --epochs 1 --prof 100 <path to imagenet>`
    * FP16
`python ./multiproc.py --nproc_per_node 8 ./main.py --arch resnet50 --training-only -p 1 --raport-file benchmark.json --fp16 --static-loss-scale 256 --epochs 1 --prof 100 <path to imagenet>`
    * AMP
`python ./multiproc.py --nproc_per_node 8 ./main.py --arch resnet50 --training-only -p 1 --raport-file benchmark.json --amp --static-loss-scale 256 --epochs 1 --prof 100 <path to imagenet>`

Each of these scripts will run 100 iterations and save results in the `benchmark.json` file.

#### Inference performance benchmark

To benchmark inference, run:

* FP32

`python ./main.py --arch resnet50 -p 1 --raport-file benchmark.json --epochs 1 --prof 100 --evaluate <path to imagenet>`

* FP16

`python ./main.py --arch resnet50 -p 1 --raport-file benchmark.json --epochs 1 --prof 100 --evaluate --fp16 <path to imagenet>`

* AMP

`python ./main.py --arch resnet50 -p 1 --raport-file benchmark.json --epochs 1 --prof 100 --evaluate --amp <path to imagenet>`

Each of these scripts will run 100 iterations and save results in the `benchmark.json` file.


### Results

Our results were obtained by running the applicable training script     in the pytorch-19.10 NGC container.

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

#### Training accuracy results

##### Training accuracy: NVIDIA DGX-1 (8x V100 16G)

| **epochs** | **Mixed Precision Top1** | **FP32 Top1** |
|:-:|:-:|:-:|
| 50 | 76.25 +/- 0.04 | 76.26 +/- 0.07 |
| 90 | 77.23 +/- 0.04 | 77.08 +/- 0.08 |
| 250 | 78.42 +/- 0.04 | 78.30 +/- 0.16 |

##### Training accuracy: NVIDIA DGX-2 (16x V100 32G)

| **epochs** | **Mixed Precision Top1** | **FP32 Top1** |
|:-:|:-:|:-:|
| 50 | 75.81 +/- 0.08 | 76.04 +/- 0.05 |
| 90 | 77.10 +/- 0.06 | 77.23 +/- 0.04 |
| 250 | 78.59 +/- 0.13 | 78.46 +/- 0.03 |



##### Example plots

The following images show a 250 epochs configuration on a DGX-1V.

![ValidationLoss](./img/loss_plot.png)

![ValidationTop1](./img/top1_plot.png)

![ValidationTop5](./img/top5_plot.png)

#### Training performance results

##### Traininig performance: NVIDIA DGX1-16G (8x V100 16G)

| **GPUs** | **Mixed Precision** | **FP32** | **Mixed Precision speedup** | **Mixed Precision Strong Scaling** | **FP32 Strong Scaling** |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 893.09 img/s | 380.44 img/s | 2.35x | 1.00x | 1.00x |
| 8 | 6888.75 img/s | 2945.37 img/s | 2.34x | 7.71x | 7.74x |

##### Traininig performance: NVIDIA DGX1-32G (8x V100 32G)

| **GPUs** | **Mixed Precision** | **FP32** | **Mixed Precision speedup** | **Mixed Precision Strong Scaling** | **FP32 Strong Scaling** |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 849.63 img/s | 373.93 img/s | 2.27x | 1.00x | 1.00x |
| 8 | 6614.15 img/s | 2911.22 img/s | 2.27x | 7.78x | 7.79x |

##### Traininig performance: NVIDIA DGX2 (16x V100 32G)

| **GPUs** | **Mixed Precision** | **FP32** | **Mixed Precision speedup** | **Mixed Precision Strong Scaling** | **FP32 Strong Scaling** |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 894.41 img/s | 402.23 img/s | 2.22x | 1.00x | 1.00x |
| 16 | 13443.82 img/s | 6263.41 img/s | 2.15x | 15.03x | 15.57x |

#### Training Time for 90 Epochs

##### Training time: NVIDIA DGX-1 (8x V100 16G)

| **GPUs** | **Mixed Precision training time** | **FP32 training time** |
|:-:|:-:|:-:|
| 1 | ~ 41 h | ~ 95 h |
| 8 | ~ 7 h | ~ 14 h |

##### Training time: NVIDIA DGX-2 (16x V100 32G)

| **GPUs** | **Mixed Precision training time** | **FP32 training time** |
|:-:|:-:|:-:|
| 1 | ~ 41 h | ~ 90 h |
| 16 | ~ 5 h | ~ 8 h |



#### Inference performance results

##### Inference performance: NVIDIA DGX-1 (1x V100 16G)

###### FP32 Inference Latency

| **batch size** | **Throughput Avg** | **Latency Avg** | **Latency 90%** | **Latency 95%** | **Latency 99%** |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 136.82 img/s | 7.12ms | 7.25ms | 8.36ms | 10.92ms |
| 2 | 266.86 img/s | 7.27ms | 7.41ms | 7.85ms | 9.11ms |
| 4 | 521.76 img/s | 7.44ms | 7.58ms | 8.14ms | 10.09ms |
| 8 | 766.22 img/s | 10.18ms | 10.46ms | 10.97ms | 12.75ms |
| 16 | 976.36 img/s | 15.79ms | 15.88ms | 15.95ms | 16.63ms |
| 32 | 1092.27 img/s | 28.63ms | 28.71ms | 28.76ms | 29.30ms |
| 64 | 1161.55 img/s | 53.69ms | 53.86ms | 53.90ms | 54.23ms |
| 128 | 1209.12 img/s | 104.24ms | 104.68ms | 104.80ms | 105.00ms |
| 256 | N/A | N/A | N/A | N/A | N/A |

###### Mixed Precision Inference Latency

| **batch size** | **Throughput Avg** | **Latency Avg** | **Latency 90%** | **Latency 95%** | **Latency 99%** |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 114.97 img/s | 8.56ms | 9.32ms | 11.43ms | 12.79ms |
| 2 | 238.70 img/s | 8.20ms | 8.75ms | 9.49ms | 12.31ms |
| 4 | 448.69 img/s | 8.67ms | 9.20ms | 9.97ms | 10.60ms |
| 8 | 875.00 img/s | 8.88ms | 9.31ms | 9.80ms | 10.82ms |
| 16 | 1746.07 img/s | 8.89ms | 9.05ms | 9.56ms | 12.81ms |
| 32 | 2004.28 img/s | 14.07ms | 14.14ms | 14.31ms | 14.92ms |
| 64 | 2254.60 img/s | 25.93ms | 26.05ms | 26.07ms | 26.17ms |
| 128 | 2360.14 img/s | 50.14ms | 50.28ms | 50.34ms | 50.68ms |
| 256 | 2342.13 img/s | 96.74ms | 96.91ms | 96.99ms | 97.14ms |



##### Inference performance: NVIDIA T4

###### FP32 Inference Latency

| **batch size** | **Throughput Avg** | **Latency Avg** | **Latency 90%** | **Latency 95%** | **Latency 99%** |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 179.85 img/s | 5.51ms | 5.65ms | 7.34ms | 10.97ms |
| 2 | 348.12 img/s | 5.67ms | 5.95ms | 6.33ms | 9.81ms |
| 4 | 556.27 img/s | 7.03ms | 7.34ms | 8.13ms | 9.65ms |
| 8 | 740.43 img/s | 10.32ms | 10.33ms | 10.60ms | 13.87ms |
| 16 | 909.17 img/s | 17.19ms | 17.15ms | 18.13ms | 21.06ms |
| 32 | 999.07 img/s | 31.07ms | 31.12ms | 31.17ms | 32.41ms |
| 64 | 1090.47 img/s | 57.62ms | 57.84ms | 57.91ms | 58.05ms |
| 128 | 1142.46 img/s | 110.94ms | 111.15ms | 111.23ms | 112.16ms |
| 256 | N/A | N/A | N/A | N/A | N/A |

###### Mixed Precision Inference Latency

| **batch size** | **Throughput Avg** | **Latency Avg** | **Latency 90%** | **Latency 95%** | **Latency 99%** |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 163.78 img/s | 6.05ms | 5.92ms | 7.98ms | 11.58ms |
| 2 | 333.43 img/s | 5.91ms | 6.05ms | 6.63ms | 11.52ms |
| 4 | 645.45 img/s | 6.04ms | 6.33ms | 7.01ms | 8.90ms |
| 8 | 1164.15 img/s | 6.73ms | 7.31ms | 8.04ms | 12.41ms |
| 16 | 1606.42 img/s | 9.53ms | 9.86ms | 10.52ms | 17.01ms |
| 32 | 1857.29 img/s | 15.67ms | 15.61ms | 16.14ms | 18.66ms |
| 64 | 2011.62 img/s | 28.64ms | 28.69ms | 28.82ms | 31.06ms |
| 128 | 2083.90 img/s | 54.87ms | 54.96ms | 54.99ms | 55.27ms |
| 256 | 2043.72 img/s | 106.51ms | 106.62ms | 106.68ms | 107.03ms |





## Release notes

### Changelog

1. September 2018
  * Initial release
2. January 2019
  * Added options Label Smoothing, fan-in initialization, skipping weight decay on batch norm gamma and bias.
3. May 2019
  * Cosine LR schedule
  * MixUp regularization
  * DALI support
  * DGX2 configurations
  * gradients accumulation
4. July 2019
  * DALI-CPU dataloader
  * Updated README

### Known issues

There are no known issues with this model.



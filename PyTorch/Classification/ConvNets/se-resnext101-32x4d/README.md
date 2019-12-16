# SE-ResNeXt101-32x4d For PyTorch

## Table Of Contents
* [Model overview](#model-overview)
  * [Default configuration](#default-configuration)
  * [Mixed precision training](#mixed-precision-training)
    * [Enabling mixed precision](#enabling-mixed-precision)
* [Setup](#setup)
  * [Requirements](#requirements)
* [Quick Start Guide](#quick-start-guide)
* [Advanced](#advanced)
* [Performance](#performance)
  * [Benchmarking](#benchmarking)
    * [Training performance benchmark](#training-performance-benchmark)
    * [Inference performance benchmark](#inference-performance-benchmark)
  * [Results](#results)
    * [Training accuracy results](#training-accuracy-results)
      * [NVIDIA DGX-1 (8x V100 16G)](#nvidia-dgx-1-(8x-v100-16G))
      * [NVIDIA DGX-2 (16x V100 32G)](#nvidia-dgx-2-(16x-v100-32G))
    * [Training performance results](#training-performance-results)
      * [NVIDIA DGX-1 (8x V100 16G)](#nvidia-dgx-1-(8x-v100-16G))
      * [NVIDIA DGX-2 (16x V100 32G)](#nvidia-dgx-2-(16x-v100-32G))
    * [Inference performance results](#inference-performance-results)
* [Release notes](#release-notes)
  * [Changelog](#changelog)
  * [Known issues](#known-issues)

## Model overview

The SE-ResNeXt101-32x4d is a [ResNeXt101-32x4d](https://arxiv.org/pdf/1611.05431.pdf)
model with added Squeeze-and-Excitation module introduced
in [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf) paper.

Squeeze and Excitation module architecture for ResNet-type models:

![SEArch](./img/SEArch.png)

_ Image source: [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf) _

### Default configuration

#### Optimizer

This model uses SGD with momentum optimizer with the following hyperparameters:

* momentum (0.875)

* Learning rate = 0.256 for 256 batch size, for other batch sizes we lineary
scale the learning rate.

* Learning rate schedule - we use cosine LR schedule

* For bigger batch sizes (512 and up) we use linear warmup of the learning rate
during first couple of epochs
according to [Training ImageNet in 1 hour](https://arxiv.org/abs/1706.02677).
Warmup length depends on total training length.

* Weight decay: 6.103515625e-05 (1/16384).

* We do not apply WD on Batch Norm trainable parameters (gamma/bias)

* Label Smoothing: 0.1

* We train for:

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

### DALI

For DGX2 configurations we use [NVIDIA DALI](https://github.com/NVIDIA/DALI),
which speeds up data loading when CPU becomes a bottleneck.
DALI can also use CPU, and it outperforms the pytorch native dataloader.

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

Mixed precision is enabled in PyTorch by using the Automatic Mixed Precision (AMP),  library from [APEX](https://github.com/NVIDIA/apex) that casts variables to half-precision upon retrieval,
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

### Requirements

Ensure you meet the following requirements:

* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch 19.09-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch) or newer
* (optional) NVIDIA Volta GPU (see section below) - for best training performance using mixed precision

For more information about how to get started with NGC containers, see the
following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning
DGX Documentation:
* [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
* [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/dgx/user-guide/index.html#accessing_registry)
* [Running PyTorch](https://docs.nvidia.com/deeplearning/dgx/pytorch-release-notes/running.html#running)

## Quick Start Guide

### 1. Clone the repository.
```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/PyTorch/Classification/SE-RNXT101-32x4d/
```

### 2. Download and preprocess the dataset.

The SE-ResNeXt101-32x4d script operates on ImageNet 1k, a widely popular image classification dataset from ILSVRC challenge.

PyTorch can work directly on JPEGs, therefore, preprocessing/augmentation is not needed.

1. Download the images from http://image-net.org/download-images

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

### 3. Build the SE-RNXT101-32x4d PyTorch NGC container.

```
docker build . -t nvidia_rnxt101-32x4d
```

### 4. Start an interactive session in the NGC container to run training/inference.
```
nvidia-docker run --rm -it -v <path to imagenet>:/imagenet --ipc=host nvidia_rnxt101-32x4d
```

### 5. Running training

To run training for a standard configuration (DGX1V/DGX2V, AMP/FP32, 90/250 Epochs),
run one of the scripts in the `./se-resnext101-32x4d/training` directory
called `./se-resnext101-32x4d/training/{DGX1, DGX2}_SE-RNXT101-32x4d_{AMP, FP32}_{90,250}E.sh`.

Ensure imagenet is mounted in the `/imagenet` directory.

Example:
    `bash ./se-resnext101-32x4d/training/DGX1_SE-RNXT101-32x4d_FP16_250E.sh`
   
To run a non standard configuration use:

* For 1 GPU
    * FP32
        `python ./main.py --arch se-resnext101-32x4d -c fanin --label-smoothing 0.1 <path to imagenet>`
    * AMP
        `python ./main.py --arch se-resnext101-32x4d -c fanin --label-smoothing 0.1 --amp --static-loss-scale 256 <path to imagenet>`

* For multiple GPUs
    * FP32
        `python ./multiproc.py --nproc_per_node 8 ./main.py --arch se-resnext101-32x4d -c fanin --label-smoothing 0.1 <path to imagenet>`
    * AMP
        `python ./multiproc.py --nproc_per_node 8 ./main.py --arch se-resnext101-32x4d -c fanin --label-smoothing 0.1 --amp --static-loss-scale 256 <path to imagenet>`

Use `python ./main.py -h` to obtain the list of available options in the `main.py` script.

### 6. Running inference

To run inference on a checkpointed model run:

`python ./main.py --arch se-resnext101-32x4d --evaluate --epochs 1 --resume <path to checkpoint> -b <batch size> <path to imagenet>`

## Advanced

### Commmand-line options:

```
```

## Performance

### Benchmarking

#### Training performance benchmark

To benchmark training, run:

* For 1 GPU
    * FP32
`python ./main.py --arch se-resnext101-32x4d --training-only -p 1 --raport-file benchmark.json --epochs 1 --prof 100 <path to imagenet>`
    * AMP
`python ./main.py --arch se-resnext101-32x4d --training-only -p 1 --raport-file benchmark.json --epochs 1 --prof 100 --amp --static-loss-scale 256 <path to imagenet>`
* For multiple GPUs
    * FP32
`python ./multiproc.py --nproc_per_node 8 ./main.py --arch se-resnext101-32x4d --training-only -p 1 --raport-file benchmark.json --epochs 1 --prof 100 <path to imagenet>`
    * AMP
`python ./multiproc.py --nproc_per_node 8 ./main.py --arch se-resnext101-32x4d --training-only -p 1 --raport-file benchmark.json --amp --static-loss-scale 256 --epochs 1 --prof 100 <path to imagenet>`

Each of this scripts will run 100 iterations and save results in benchmark.json file

#### Inference performance benchmark

To benchmark inference, run:

* FP32

`python ./main.py --arch se-resnext101-32x4d -p 1 --raport-file benchmark.json --epochs 1 --prof 100 --evaluate <path to imagenet>`

* AMP

`python ./main.py --arch se-resnext101-32x4d -p 1 --raport-file benchmark.json --epochs 1 --prof 100 --evaluate --amp <path to imagenet>`

Each of this scripts will run 100 iterations and save results in benchmark.json file



### Results

#### Training Accuracy Results

##### NVIDIA DGX-1 (8x V100 16G)

| **epochs** | **Mixed Precision Top1** | **FP32 Top1** |
|:-:|:-:|:-:|
| 90 | 80.03 +/- 0.10 | 79.86 +/- 0.13 |
| 250 | 80.96 +/- 0.04 | 80.97 +/- 0.09 |

##### NVIDIA DGX-2 (16x V100 32G)

No Data



##### Example plots (90 Epochs configuration on DGX1V)

![ValidationLoss](./img/loss_plot.png)

![ValidationTop1](./img/top1_plot.png)

![ValidationTop5](./img/top5_plot.png)

#### Training Performance Results

##### NVIDIA DGX1-16G (8x V100 16G)

| **# of GPUs** | **Mixed Precision** | **FP32** | **Mixed Precision speedup** | **Mixed Precision Strong Scaling** | **FP32 Strong Scaling** |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 266.65 img/s | 128.23 img/s | 2.08x | 1.00x | 1.00x |
| 8 | 2031.17 img/s | 977.45 img/s | 2.08x | 7.62x | 7.62x |

##### NVIDIA DGX1-32G (8x V100 32G)

| **# of GPUs** | **Mixed Precision** | **FP32** | **Mixed Precision speedup** | **Mixed Precision Strong Scaling** | **FP32 Strong Scaling** |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 255.22 img/s | 125.13 img/s | 2.04x | 1.00x | 1.00x |
| 8 | 1959.35 img/s | 963.21 img/s | 2.03x | 7.68x | 7.70x |

##### NVIDIA DGX2 (16x V100 32G)

| **# of GPUs** | **Mixed Precision** | **FP32** | **Mixed Precision speedup** | **Mixed Precision Strong Scaling** | **FP32 Strong Scaling** |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 261.58 img/s | 130.85 img/s | 2.00x | 1.00x | 1.00x |
| 16 | 3776.03 img/s | 1953.13 img/s | 1.93x | 14.44x | 14.93x |

#### Training Time for 90 Epochs

##### NVIDIA DGX-1 (8x V100 16G)

| **# of GPUs** | **Mixed Precision training time** | **FP32 training time** |
|:-:|:-:|:-:|
| 1 | ~ 134 h | ~ 277 h |
| 8 | ~ 19 h | ~ 38 h |

##### NVIDIA DGX-2 (16x V100 32G)

| **# of GPUs** | **Mixed Precision training time** | **FP32 training time** |
|:-:|:-:|:-:|
| 1 | ~ 137 h | ~ 271 h |
| 16 | ~ 11 h | ~ 20 h |



#### Inference Performance Results

##### NVIDIA VOLTA V100 16G on DGX1V

###### FP32 Inference Latency

| **batch_size** | **FP32 50.0%** | **FP32 90.0%** | **FP32 99.0%** | **FP32 100.0%** |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 28.92ms | 30.92ms | 34.65ms | 40.01ms |
| 2 | 29.38ms | 31.30ms | 34.79ms | 40.65ms |
| 4 | 28.97ms | 29.78ms | 33.90ms | 42.28ms |
| 8 | 29.75ms | 32.73ms | 35.61ms | 39.83ms |
| 16 | 44.52ms | 44.93ms | 46.90ms | 48.84ms |
| 32 | 80.63ms | 81.28ms | 82.69ms | 85.82ms |
| 64 | 142.57ms | 142.99ms | 145.01ms | 148.87ms |
| 128 | N/A | N/A | N/A | N/A |

###### Mixed Precision Inference Latency

| **batch_size** | **AMP @50.0%** | **speedup** | **AMP @90.0%** | **speedup** | **AMP @99.0%** | **speedup** | **AMP @100.0%** | **speedup** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 27.85ms | 1.0x | 29.75ms | 1.0x | 35.85ms | 1.0x | 42.54ms | 0.9x |
| 2 | 28.37ms | 1.0x | 30.24ms | 1.0x | 37.07ms | 0.9x | 42.81ms | 0.9x |
| 4 | 29.73ms | 1.0x | 31.39ms | 0.9x | 37.17ms | 0.9x | 43.78ms | 1.0x |
| 8 | 30.19ms | 1.0x | 31.20ms | 1.0x | 34.46ms | 1.0x | 42.87ms | 0.9x |
| 16 | 30.92ms | 1.4x | 32.48ms | 1.4x | 36.49ms | 1.3x | 42.76ms | 1.1x |
| 32 | 40.61ms | 2.0x | 40.90ms | 2.0x | 43.67ms | 1.9x | 44.88ms | 1.9x |
| 64 | 72.04ms | 2.0x | 72.29ms | 2.0x | 76.46ms | 1.9x | 77.46ms | 1.9x |
| 128 | 130.12ms | N/A | 130.34ms | N/A | 131.12ms | N/A | 140.27ms | N/A |

###### FP32 Inference throughput

| **batch_size** | **FP32 @50.0%** | **FP32 @90.0%** | **FP32 @99.0%** | **FP32 @100.0%** |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 34 img/s | 32 img/s | 29 img/s | 25 img/s |
| 2 | 68 img/s | 63 img/s | 57 img/s | 48 img/s |
| 4 | 137 img/s | 133 img/s | 117 img/s | 93 img/s |
| 8 | 267 img/s | 243 img/s | 223 img/s | 198 img/s |
| 16 | 357 img/s | 354 img/s | 331 img/s | 325 img/s |
| 32 | 392 img/s | 389 img/s | 381 img/s | 361 img/s |
| 64 | 444 img/s | 442 img/s | 434 img/s | 426 img/s |
| 128 | N/A | N/A | N/A | N/A |

###### Mixed Precision Inference throughput

| **batch_size** | **AMP @50.0%** | **speedup** | **AMP @90.0%** | **speedup** | **AMP @99.0%** | **speedup** | **AMP @100.0%** | **speedup** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 36 img/s | 1.0x | 33 img/s | 1.0x | 28 img/s | 1.0x | 23 img/s | 0.9x |
| 2 | 70 img/s | 1.0x | 66 img/s | 1.0x | 53 img/s | 0.9x | 46 img/s | 1.0x |
| 4 | 133 img/s | 1.0x | 126 img/s | 0.9x | 107 img/s | 0.9x | 90 img/s | 1.0x |
| 8 | 263 img/s | 1.0x | 254 img/s | 1.0x | 226 img/s | 1.0x | 184 img/s | 0.9x |
| 16 | 513 img/s | 1.4x | 488 img/s | 1.4x | 435 img/s | 1.3x | 369 img/s | 1.1x |
| 32 | 781 img/s | 2.0x | 775 img/s | 2.0x | 723 img/s | 1.9x | 680 img/s | 1.9x |
| 64 | 882 img/s | 2.0x | 878 img/s | 2.0x | 818 img/s | 1.9x | 777 img/s | 1.8x |
| 128 | 978 img/s | N/A | 976 img/s | N/A | 969 img/s | N/A | 891 img/s | N/A |

##### NVIDIA T4

###### FP32 Inference Latency

| **batch_size** | **FP32 50.0%** | **FP32 90.0%** | **FP32 99.0%** | **FP32 100.0%** |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 23.73ms | 26.94ms | 33.03ms | 35.92ms |
| 2 | 23.20ms | 24.53ms | 29.42ms | 36.79ms |
| 4 | 23.82ms | 24.59ms | 27.57ms | 31.07ms |
| 8 | 29.73ms | 30.51ms | 33.07ms | 34.98ms |
| 16 | 48.49ms | 48.91ms | 51.01ms | 54.54ms |
| 32 | 86.81ms | 87.15ms | 90.74ms | 90.89ms |
| 64 | 155.01ms | 156.07ms | 164.74ms | 167.99ms |
| 128 | N/A | N/A | N/A | N/A |

###### Mixed Precision Inference Latency

| **batch_size** | **AMP @50.0%** | **speedup** | **AMP @90.0%** | **speedup** | **AMP @99.0%** | **speedup** | **AMP @100.0%** | **speedup** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 25.22ms | 0.9x | 26.10ms | 1.0x | 31.72ms | 1.0x | 34.56ms | 1.0x |
| 2 | 25.18ms | 0.9x | 25.83ms | 0.9x | 33.07ms | 0.9x | 37.80ms | 1.0x |
| 4 | 24.94ms | 1.0x | 25.58ms | 1.0x | 27.93ms | 1.0x | 30.55ms | 1.0x |
| 8 | 26.29ms | 1.1x | 27.59ms | 1.1x | 32.69ms | 1.0x | 35.78ms | 1.0x |
| 16 | 27.63ms | 1.8x | 28.36ms | 1.7x | 34.44ms | 1.5x | 39.55ms | 1.4x |
| 32 | 44.43ms | 2.0x | 44.69ms | 1.9x | 47.99ms | 1.9x | 51.38ms | 1.8x |
| 64 | 79.17ms | 2.0x | 79.40ms | 2.0x | 84.34ms | 2.0x | 84.64ms | 2.0x |
| 128 | 147.41ms | N/A | 149.02ms | N/A | 151.90ms | N/A | 159.28ms | N/A |

###### FP32 Inference throughput

| **batch_size** | **FP32 @50.0%** | **FP32 @90.0%** | **FP32 @99.0%** | **FP32 @100.0%** |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 42 img/s | 37 img/s | 30 img/s | 27 img/s |
| 2 | 86 img/s | 81 img/s | 68 img/s | 54 img/s |
| 4 | 167 img/s | 161 img/s | 143 img/s | 128 img/s |
| 8 | 267 img/s | 261 img/s | 240 img/s | 226 img/s |
| 16 | 328 img/s | 325 img/s | 296 img/s | 289 img/s |
| 32 | 367 img/s | 365 img/s | 350 img/s | 343 img/s |
| 64 | 411 img/s | 408 img/s | 380 img/s | 373 img/s |
| 128 | N/A | N/A | N/A | N/A |

###### Mixed Precision Inference throughput

| **batch_size** | **AMP @50.0%** | **speedup** | **AMP @90.0%** | **speedup** | **AMP @99.0%** | **speedup** | **AMP @100.0%** | **speedup** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 39 img/s | 0.9x | 38 img/s | 1.0x | 31 img/s | 1.0x | 29 img/s | 1.0x |
| 2 | 79 img/s | 0.9x | 77 img/s | 1.0x | 60 img/s | 0.9x | 52 img/s | 1.0x |
| 4 | 159 img/s | 1.0x | 155 img/s | 1.0x | 142 img/s | 1.0x | 130 img/s | 1.0x |
| 8 | 302 img/s | 1.1x | 288 img/s | 1.1x | 243 img/s | 1.0x | 222 img/s | 1.0x |
| 16 | 575 img/s | 1.8x | 560 img/s | 1.7x | 458 img/s | 1.5x | 402 img/s | 1.4x |
| 32 | 713 img/s | 1.9x | 708 img/s | 1.9x | 619 img/s | 1.8x | 549 img/s | 1.6x |
| 64 | 804 img/s | 2.0x | 801 img/s | 2.0x | 712 img/s | 1.9x | 636 img/s | 1.7x |
| 128 | 857 img/s | N/A | 855 img/s | N/A | 840 img/s | N/A | 783 img/s | N/A |



## Release notes

### Changelog

1. October 2019
  * Initial release

### Known issues

There are no known issues with this model.



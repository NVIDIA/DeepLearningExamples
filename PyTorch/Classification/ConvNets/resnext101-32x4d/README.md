# ResNeXt101-32x4d For PyTorch

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

The ResNeXt101-32x4d is a model introduced in [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf) paper.

It is based on regular ResNet model, substituting 3x3 Convolutions inside the bottleneck block for 3x3 Grouped Convolutions.

![ResNextArch](./img/ResNeXtArch.png)

_ Image source: [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf) _

ResNeXt101-32x4d model's cardinality equals to 32 and bottleneck width equals to 4

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
cd DeepLearningExamples/PyTorch/Classification/RNXT101-32x4d/
```

### 2. Download and preprocess the dataset.

The ResNeXt101-32x4d script operates on ImageNet 1k, a widely popular image classification dataset from ILSVRC challenge.

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

### 3. Build the RNXT101-32x4d PyTorch NGC container.

```
docker build . -t nvidia_rnxt101-32x4d
```

### 4. Start an interactive session in the NGC container to run training/inference.
```
nvidia-docker run --rm -it -v <path to imagenet>:/imagenet --ipc=host nvidia_rnxt101-32x4d
```

### 5. Running training

To run training for a standard configuration (DGX1V/DGX2V, AMP/FP32, 90/250 Epochs),
run one of the scripts in the `./resnext101-32x4d/training` directory
called `./resnext101-32x4d/training/{DGX1, DGX2}_RNXT101-32x4d_{AMP, FP32}_{90,250}E.sh`.

Ensure imagenet is mounted in the `/imagenet` directory.

Example:
    `bash ./resnext101-32x4d/training/DGX1_RNXT101-32x4d_FP16_250E.sh`
   
To run a non standard configuration use:

* For 1 GPU
    * FP32
        `python ./main.py --arch resnext101-32x4d -c fanin --label-smoothing 0.1 <path to imagenet>`
    * AMP
        `python ./main.py --arch resnext101-32x4d -c fanin --label-smoothing 0.1 --amp --static-loss-scale 256 <path to imagenet>`

* For multiple GPUs
    * FP32
        `python ./multiproc.py --nproc_per_node 8 ./main.py --arch resnext101-32x4d -c fanin --label-smoothing 0.1 <path to imagenet>`
    * AMP
        `python ./multiproc.py --nproc_per_node 8 ./main.py --arch resnext101-32x4d -c fanin --label-smoothing 0.1 --amp --static-loss-scale 256 <path to imagenet>`

Use `python ./main.py -h` to obtain the list of available options in the `main.py` script.

### 6. Running inference

To run inference on a checkpointed model run:

`python ./main.py --arch resnext101-32x4d --evaluate --epochs 1 --resume <path to checkpoint> -b <batch size> <path to imagenet>`

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
`python ./main.py --arch resnext101-32x4d --training-only -p 1 --raport-file benchmark.json --epochs 1 --prof 100 <path to imagenet>`
    * AMP
`python ./main.py --arch resnext101-32x4d --training-only -p 1 --raport-file benchmark.json --epochs 1 --prof 100 --amp --static-loss-scale 256 <path to imagenet>`
* For multiple GPUs
    * FP32
`python ./multiproc.py --nproc_per_node 8 ./main.py --arch resnext101-32x4d --training-only -p 1 --raport-file benchmark.json --epochs 1 --prof 100 <path to imagenet>`
    * AMP
`python ./multiproc.py --nproc_per_node 8 ./main.py --arch resnext101-32x4d --training-only -p 1 --raport-file benchmark.json --amp --static-loss-scale 256 --epochs 1 --prof 100 <path to imagenet>`

Each of this scripts will run 100 iterations and save results in benchmark.json file

#### Inference performance benchmark

To benchmark inference, run:

* FP32

`python ./main.py --arch resnext101-32x4d -p 1 --raport-file benchmark.json --epochs 1 --prof 100 --evaluate <path to imagenet>`

* AMP

`python ./main.py --arch resnext101-32x4d -p 1 --raport-file benchmark.json --epochs 1 --prof 100 --evaluate --amp <path to imagenet>`

Each of this scripts will run 100 iterations and save results in benchmark.json file



### Results

#### Training Accuracy Results

##### NVIDIA DGX-1 (8x V100 16G)

| **epochs** | **Mixed Precision Top1** | **FP32 Top1** |
|:-:|:-:|:-:|
| 90 | 79.23 +/- 0.09 | 79.23 +/- 0.09 |
| 250 | 79.92 +/- 0.13 | 80.06 +/- 0.06 |

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
| 1 | 313.43 img/s | 146.66 img/s | 2.14x | 1.00x | 1.00x |
| 8 | 2384.85 img/s | 1116.58 img/s | 2.14x | 7.61x | 7.61x |

##### NVIDIA DGX1-32G (8x V100 32G)

| **# of GPUs** | **Mixed Precision** | **FP32** | **Mixed Precision speedup** | **Mixed Precision Strong Scaling** | **FP32 Strong Scaling** |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 297.83 img/s | 143.27 img/s | 2.08x | 1.00x | 1.00x |
| 8 | 2270.85 img/s | 1104.62 img/s | 2.06x | 7.62x | 7.71x |

##### NVIDIA DGX2 (16x V100 32G)

| **# of GPUs** | **Mixed Precision** | **FP32** | **Mixed Precision speedup** | **Mixed Precision Strong Scaling** | **FP32 Strong Scaling** |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 308.42 img/s | 151.67 img/s | 2.03x | 1.00x | 1.00x |
| 16 | 4473.37 img/s | 2261.97 img/s | 1.98x | 14.50x | 14.91x |

#### Training Time for 90 Epochs

##### NVIDIA DGX-1 (8x V100 16G)

| **# of GPUs** | **Mixed Precision training time** | **FP32 training time** |
|:-:|:-:|:-:|
| 1 | ~ 114 h | ~ 242 h |
| 8 | ~ 17 h | ~ 34 h |

##### NVIDIA DGX-2 (16x V100 32G)

| **# of GPUs** | **Mixed Precision training time** | **FP32 training time** |
|:-:|:-:|:-:|
| 1 | ~ 116 h | ~ 234 h |
| 16 | ~ 10 h | ~ 18 h |



#### Inference Performance Results

##### NVIDIA VOLTA V100 16G on DGX1V

###### FP32 Inference Latency

| **batch_size** | **FP32 50.0%** | **FP32 90.0%** | **FP32 99.0%** | **FP32 100.0%** |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 20.53ms | 23.41ms | 26.00ms | 28.14ms |
| 2 | 21.94ms | 22.90ms | 26.59ms | 30.53ms |
| 4 | 22.08ms | 24.96ms | 26.03ms | 26.91ms |
| 8 | 24.03ms | 25.17ms | 28.52ms | 32.59ms |
| 16 | 39.73ms | 40.01ms | 40.32ms | 44.05ms |
| 32 | 73.53ms | 74.05ms | 74.26ms | 78.31ms |
| 64 | 130.88ms | 131.38ms | 131.81ms | 134.32ms |
| 128 | N/A | N/A | N/A | N/A |

###### Mixed Precision Inference Latency

| **batch_size** | **AMP @50.0%** | **speedup** | **AMP @90.0%** | **speedup** | **AMP @99.0%** | **speedup** | **AMP @100.0%** | **speedup** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 22.64ms | 0.9x | 25.19ms | 0.9x | 26.63ms | 1.0x | 28.43ms | 1.0x |
| 2 | 23.35ms | 0.9x | 25.11ms | 0.9x | 27.29ms | 1.0x | 27.95ms | 1.1x |
| 4 | 22.35ms | 1.0x | 24.38ms | 1.0x | 25.92ms | 1.0x | 27.09ms | 1.0x |
| 8 | 23.35ms | 1.0x | 26.45ms | 1.0x | 27.74ms | 1.0x | 28.22ms | 1.2x |
| 16 | 24.77ms | 1.6x | 26.93ms | 1.5x | 28.73ms | 1.4x | 29.07ms | 1.5x |
| 32 | 35.70ms | 2.1x | 35.96ms | 2.1x | 36.13ms | 2.1x | 36.36ms | 2.2x |
| 64 | 63.40ms | 2.1x | 63.63ms | 2.1x | 63.96ms | 2.1x | 64.74ms | 2.1x |
| 128 | 117.52ms | N/A | 118.02ms | N/A | 118.35ms | N/A | 118.43ms | N/A |

###### FP32 Inference throughput

| **batch_size** | **FP32 @50.0%** | **FP32 @90.0%** | **FP32 @99.0%** | **FP32 @100.0%** |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 48 img/s | 42 img/s | 38 img/s | 35 img/s |
| 2 | 90 img/s | 87 img/s | 74 img/s | 64 img/s |
| 4 | 179 img/s | 158 img/s | 152 img/s | 147 img/s |
| 8 | 329 img/s | 314 img/s | 279 img/s | 243 img/s |
| 16 | 399 img/s | 395 img/s | 389 img/s | 361 img/s |
| 32 | 433 img/s | 429 img/s | 423 img/s | 403 img/s |
| 64 | 487 img/s | 485 img/s | 475 img/s | 436 img/s |
| 128 | N/A | N/A | N/A | N/A |

###### Mixed Precision Inference throughput

| **batch_size** | **AMP @50.0%** | **speedup** | **AMP @90.0%** | **speedup** | **AMP @99.0%** | **speedup** | **AMP @100.0%** | **speedup** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 44 img/s | 0.9x | 39 img/s | 0.9x | 37 img/s | 1.0x | 35 img/s | 1.0x |
| 2 | 85 img/s | 0.9x | 78 img/s | 0.9x | 73 img/s | 1.0x | 71 img/s | 1.1x |
| 4 | 177 img/s | 1.0x | 163 img/s | 1.0x | 153 img/s | 1.0x | 145 img/s | 1.0x |
| 8 | 339 img/s | 1.0x | 299 img/s | 1.0x | 286 img/s | 1.0x | 282 img/s | 1.2x |
| 16 | 640 img/s | 1.6x | 589 img/s | 1.5x | 551 img/s | 1.4x | 547 img/s | 1.5x |
| 32 | 887 img/s | 2.0x | 879 img/s | 2.1x | 846 img/s | 2.0x | 731 img/s | 1.8x |
| 64 | 1001 img/s | 2.1x | 996 img/s | 2.1x | 978 img/s | 2.1x | 797 img/s | 1.8x |
| 128 | 1081 img/s | N/A | 1078 img/s | N/A | 1068 img/s | N/A | 767 img/s | N/A |

##### NVIDIA T4

###### FP32 Inference Latency

| **batch_size** | **FP32 50.0%** | **FP32 90.0%** | **FP32 99.0%** | **FP32 100.0%** |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 17.77ms | 19.21ms | 22.29ms | 24.47ms |
| 2 | 17.83ms | 19.00ms | 22.51ms | 29.68ms |
| 4 | 18.02ms | 18.88ms | 21.74ms | 26.49ms |
| 8 | 26.14ms | 27.35ms | 28.93ms | 29.46ms |
| 16 | 45.40ms | 45.72ms | 47.43ms | 48.93ms |
| 32 | 79.07ms | 79.37ms | 81.83ms | 82.45ms |
| 64 | 140.12ms | 140.73ms | 143.57ms | 149.46ms |
| 128 | N/A | N/A | N/A | N/A |

###### Mixed Precision Inference Latency

| **batch_size** | **AMP @50.0%** | **speedup** | **AMP @90.0%** | **speedup** | **AMP @99.0%** | **speedup** | **AMP @100.0%** | **speedup** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 18.99ms | 0.9x | 20.16ms | 1.0x | 26.21ms | 0.9x | 29.29ms | 0.8x |
| 2 | 19.24ms | 0.9x | 19.77ms | 1.0x | 24.51ms | 0.9x | 30.18ms | 1.0x |
| 4 | 18.81ms | 1.0x | 19.52ms | 1.0x | 21.95ms | 1.0x | 23.45ms | 1.1x |
| 8 | 18.96ms | 1.4x | 21.12ms | 1.3x | 25.77ms | 1.1x | 28.05ms | 1.1x |
| 16 | 23.27ms | 2.0x | 25.19ms | 1.8x | 27.31ms | 1.7x | 28.11ms | 1.7x |
| 32 | 39.22ms | 2.0x | 39.43ms | 2.0x | 41.96ms | 2.0x | 44.25ms | 1.9x |
| 64 | 71.70ms | 2.0x | 71.87ms | 2.0x | 72.78ms | 2.0x | 77.22ms | 1.9x |
| 128 | 134.17ms | N/A | 134.40ms | N/A | 134.81ms | N/A | 135.26ms | N/A |

###### FP32 Inference throughput

| **batch_size** | **FP32 @50.0%** | **FP32 @90.0%** | **FP32 @99.0%** | **FP32 @100.0%** |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 56 img/s | 51 img/s | 44 img/s | 40 img/s |
| 2 | 111 img/s | 103 img/s | 87 img/s | 66 img/s |
| 4 | 220 img/s | 210 img/s | 182 img/s | 149 img/s |
| 8 | 301 img/s | 290 img/s | 275 img/s | 270 img/s |
| 16 | 351 img/s | 348 img/s | 336 img/s | 325 img/s |
| 32 | 402 img/s | 401 img/s | 389 img/s | 376 img/s |
| 64 | 451 img/s | 446 img/s | 429 img/s | 398 img/s |
| 128 | N/A | N/A | N/A | N/A |

###### Mixed Precision Inference throughput

| **batch_size** | **AMP @50.0%** | **speedup** | **AMP @90.0%** | **speedup** | **AMP @99.0%** | **speedup** | **AMP @100.0%** | **speedup** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 52 img/s | 0.9x | 49 img/s | 1.0x | 38 img/s | 0.8x | 34 img/s | 0.8x |
| 2 | 103 img/s | 0.9x | 100 img/s | 1.0x | 81 img/s | 0.9x | 65 img/s | 1.0x |
| 4 | 210 img/s | 1.0x | 203 img/s | 1.0x | 181 img/s | 1.0x | 169 img/s | 1.1x |
| 8 | 418 img/s | 1.4x | 375 img/s | 1.3x | 304 img/s | 1.1x | 283 img/s | 1.0x |
| 16 | 673 img/s | 1.9x | 630 img/s | 1.8x | 581 img/s | 1.7x | 565 img/s | 1.7x |
| 32 | 807 img/s | 2.0x | 800 img/s | 2.0x | 705 img/s | 1.8x | 681 img/s | 1.8x |
| 64 | 887 img/s | 2.0x | 884 img/s | 2.0x | 799 img/s | 1.9x | 696 img/s | 1.7x |
| 128 | 950 img/s | N/A | 948 img/s | N/A | 918 img/s | N/A | 779 img/s | N/A |



## Release notes

### Changelog

1. October 2019
  * Initial release

### Known issues

There are no known issues with this model.



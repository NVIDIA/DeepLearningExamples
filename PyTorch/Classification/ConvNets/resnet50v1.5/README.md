# ResNet50 v1.5 For PyTorch

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
The ResNet50 v1.5 model is a modified version of the [original ResNet50 v1 model](https://arxiv.org/abs/1512.03385).

The difference between v1 and v1.5 is that, in the bottleneck blocks which requires
downsampling, v1 has stride = 2 in the first 1x1 convolution, whereas v1.5 has stride = 2 in the 3x3 convolution.

This difference makes ResNet50 v1.5 slightly more accurate (~0.5% top1) than v1, but comes with a smallperformance drawback (~5% imgs/sec).

The model is initialized as described in [Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification](https://arxiv.org/pdf/1502.01852.pdf)

### Default configuration

#### Optimizer

This model trains for 90 epochs, with standard ResNet v1.5 setup:

* SGD with momentum (0.875)

* Learning rate = 0.256 for 256 batch size, for other batch sizes we lineary
scale the learning rate.

* Learning rate schedule - we use cosine LR schedule

* For bigger batch sizes (512 and up) we use linear warmup of the learning rate
during first couple of epochs
according to [Training ImageNet in 1 hour](https://arxiv.org/abs/1706.02677).
Warmup length depends on total training length.

* Weight decay: 3.0517578125e-05 (1/32768).

* We do not apply WD on Batch Norm trainable parameters (gamma/bias)

* Label Smoothing: 0.1

* We train for:

    * 50 Epochs -> configuration that reaches 75.9% top1 accuracy

    * 90 Epochs -> 90 epochs is a standard for ResNet50

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
* [PyTorch 19.05-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch) or newer
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
cd DeepLearningExamples/PyTorch/Classification/RN50v1.5/
```

### 2. Download and preprocess the dataset.

The ResNet50 v1.5 script operates on ImageNet 1k, a widely popular image classification dataset from ILSVRC challenge.

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

### 3. Build the RN50v1.5 PyTorch NGC container.

```
docker build . -t nvidia_rn50
```

### 4. Start an interactive session in the NGC container to run training/inference.
```
nvidia-docker run --rm -it -v <path to imagenet>:/imagenet --ipc=host nvidia_rn50
```

### 5. Running training

To run training for a standard configuration (DGX1V/DGX2V, FP16/FP32, 50/90/250 Epochs),
run one of the scripts in the `./resnet50v1.5/training` directory
called `./resnet50v1.5/training/{DGX1, DGX2}_RN50_{AMP, FP16, FP32}_{50,90,250}E.sh`.

Ensure imagenet is mounted in the `/imagenet` directory.

Example:
    `bash ./resnet50v1.5/training/DGX1_RN50_FP16_250E.sh`
   
To run a non standard configuration use:

* For 1 GPU
    * FP32
        `python ./main.py --arch resnet50 -c fanin --label-smoothing 0.1 <path to imagenet>`
    * FP16
        `python ./main.py --arch resnet50 -c fanin --label-smoothing 0.1 --fp16 --static-loss-scale 256 <path to imagenet>`
    * AMP
        `python ./main.py --arch resnet50 -c fanin --label-smoothing 0.1 --amp --static-loss-scale 256 <path to imagenet>`

* For multiple GPUs
    * FP32
        `python ./multiproc.py --nproc_per_node 8 ./main.py --arch resnet50 -c fanin --label-smoothing 0.1 <path to imagenet>`
    * FP16
        `python ./multiproc.py --nproc_per_node 8 ./main.py --arch resnet50 -c fanin --label-smoothing 0.1 --fp16 --static-loss-scale 256 <path to imagenet>`
    * AMP
        `python ./multiproc.py --nproc_per_node 8 ./main.py --arch resnet50 -c fanin --label-smoothing 0.1 --amp --static-loss-scale 256 <path to imagenet>`

Use `python ./main.py -h` to obtain the list of available options in the `main.py` script.

### 6. Running inference

To run inference on a checkpointed model run:

`python ./main.py --arch resnet50 --evaluate --epochs 1 --resume <path to checkpoint> -b <batch size> <path to imagenet>`

## Advanced

### Commmand-line options:

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
                        data backend: pytorch | dali-gpu | dali-cpu (default:
                        pytorch)
  --arch ARCH, -a ARCH  model architecture: resnet18 | resnet34 | resnet50 |
                        resnet101 | resnet152 (default: resnet50)
  --model-config CONF, -c CONF
                        model configs: classic | fanin(default: classic)
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

## Performance

### Benchmarking

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

Each of this scripts will run 100 iterations and save results in benchmark.json file

#### Inference performance benchmark

To benchmark inference, run:

* FP32

`python ./main.py --arch resnet50 -p 1 --raport-file benchmark.json --epochs 1 --prof 100 --evaluate <path to imagenet>`

* FP16

`python ./main.py --arch resnet50 -p 1 --raport-file benchmark.json --epochs 1 --prof 100 --evaluate --fp16 <path to imagenet>`

* AMP

`python ./main.py --arch resnet50 -p 1 --raport-file benchmark.json --epochs 1 --prof 100 --evaluate --amp <path to imagenet>`

Each of this scripts will run 100 iterations and save results in benchmark.json file



### Results

#### Training Accuracy Results

##### NVIDIA DGX-1 (8x V100 16G)

| **epochs** | **Mixed Precision Top1** | **FP32 Top1** |
|:-:|:-:|:-:|
| 50 | 76.25 +/- 0.04 | 76.26 +/- 0.07 |
| 90 | 77.23 +/- 0.04 | 77.08 +/- 0.08 |
| 250 | 78.42 +/- 0.04 | 78.30 +/- 0.16 |

##### NVIDIA DGX-2 (16x V100 32G)

| **epochs** | **Mixed Precision Top1** | **FP32 Top1** |
|:-:|:-:|:-:|
| 50 | 75.81 +/- 0.08 | 76.04 +/- 0.05 |
| 90 | 77.10 +/- 0.06 | 77.23 +/- 0.04 |
| 250 | 78.59 +/- 0.13 | 78.46 +/- 0.03 |



##### Example plots (90 Epochs configuration on DGX1V)

![ValidationLoss](./img/loss_plot.png)

![ValidationTop1](./img/top1_plot.png)

![ValidationTop5](./img/top5_plot.png)

#### Training Performance Results

##### NVIDIA DGX1-16G (8x V100 16G)

| **# of GPUs** | **Mixed Precision** | **FP32** | **Mixed Precision speedup** | **Mixed Precision Strong Scaling** | **FP32 Strong Scaling** |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 893.09 img/s | 380.44 img/s | 2.35x | 1.00x | 1.00x |
| 8 | 6888.75 img/s | 2945.37 img/s | 2.34x | 7.71x | 7.74x |

##### NVIDIA DGX1-32G (8x V100 32G)

| **# of GPUs** | **Mixed Precision** | **FP32** | **Mixed Precision speedup** | **Mixed Precision Strong Scaling** | **FP32 Strong Scaling** |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 849.63 img/s | 373.93 img/s | 2.27x | 1.00x | 1.00x |
| 8 | 6614.15 img/s | 2911.22 img/s | 2.27x | 7.78x | 7.79x |

##### NVIDIA DGX2 (16x V100 32G)

| **# of GPUs** | **Mixed Precision** | **FP32** | **Mixed Precision speedup** | **Mixed Precision Strong Scaling** | **FP32 Strong Scaling** |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 894.41 img/s | 402.23 img/s | 2.22x | 1.00x | 1.00x |
| 16 | 13443.82 img/s | 6263.41 img/s | 2.15x | 15.03x | 15.57x |

#### Training Time for 90 Epochs

##### NVIDIA DGX-1 (8x V100 16G)

| **# of GPUs** | **Mixed Precision training time** | **FP32 training time** |
|:-:|:-:|:-:|
| 1 | ~ 41 h | ~ 95 h |
| 8 | ~ 7 h | ~ 14 h |

##### NVIDIA DGX-2 (16x V100 32G)

| **# of GPUs** | **Mixed Precision training time** | **FP32 training time** |
|:-:|:-:|:-:|
| 1 | ~ 41 h | ~ 90 h |
| 16 | ~ 5 h | ~ 8 h |



#### Inference Performance Results

##### NVIDIA VOLTA V100 16G on DGX1V

###### FP32 Inference Latency

| **batch_size** | **FP32 50.0%** | **FP32 90.0%** | **FP32 99.0%** | **FP32 100.0%** |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 6.91ms | 7.25ms | 10.92ms | 11.70ms |
| 2 | 7.16ms | 7.41ms | 9.11ms | 14.58ms |
| 4 | 7.29ms | 7.58ms | 10.09ms | 13.75ms |
| 8 | 9.81ms | 10.46ms | 12.75ms | 15.36ms |
| 16 | 15.76ms | 15.88ms | 16.63ms | 17.49ms |
| 32 | 28.60ms | 28.71ms | 29.30ms | 30.81ms |
| 64 | 53.68ms | 53.86ms | 54.23ms | 54.86ms |
| 128 | 104.21ms | 104.68ms | 105.00ms | 106.19ms |
| 256 | N/A | N/A | N/A | N/A |

###### Mixed Precision Inference Latency

| **batch_size** | **AMP @50.0%** | **speedup** | **AMP @90.0%** | **speedup** | **AMP @99.0%** | **speedup** | **AMP @100.0%** | **speedup** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 8.25ms | 0.8x | 9.32ms | 0.8x | 12.79ms | 0.9x | 14.29ms | 0.8x |
| 2 | 7.95ms | 0.9x | 8.75ms | 0.8x | 12.31ms | 0.7x | 15.92ms | 0.9x |
| 4 | 8.52ms | 0.9x | 9.20ms | 0.8x | 10.60ms | 1.0x | 11.23ms | 1.2x |
| 8 | 8.78ms | 1.1x | 9.31ms | 1.1x | 10.82ms | 1.2x | 12.54ms | 1.2x |
| 16 | 8.77ms | 1.8x | 9.05ms | 1.8x | 12.81ms | 1.3x | 14.05ms | 1.2x |
| 32 | 14.03ms | 2.0x | 14.14ms | 2.0x | 14.92ms | 2.0x | 15.06ms | 2.0x |
| 64 | 25.91ms | 2.1x | 26.05ms | 2.1x | 26.17ms | 2.1x | 27.17ms | 2.0x |
| 128 | 50.11ms | 2.1x | 50.28ms | 2.1x | 50.68ms | 2.1x | 51.43ms | 2.1x |
| 256 | 96.70ms | N/A | 96.91ms | N/A | 97.14ms | N/A | 98.04ms | N/A |

###### FP32 Inference throughput

| **batch_size** | **FP32 @50.0%** | **FP32 @90.0%** | **FP32 @99.0%** | **FP32 @100.0%** |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 140 img/s | 133 img/s | 89 img/s | 70 img/s |
| 2 | 271 img/s | 259 img/s | 208 img/s | 132 img/s |
| 4 | 531 img/s | 506 img/s | 325 img/s | 248 img/s |
| 8 | 782 img/s | 729 img/s | 523 img/s | 513 img/s |
| 16 | 992 img/s | 970 img/s | 832 img/s | 624 img/s |
| 32 | 1101 img/s | 1093 img/s | 963 img/s | 871 img/s |
| 64 | 1179 img/s | 1161 img/s | 1102 img/s | 1090 img/s |
| 128 | 1220 img/s | 1213 img/s | 1159 img/s | 1148 img/s |
| 256 | N/A | N/A | N/A | N/A |

###### Mixed Precision Inference throughput

| **batch_size** | **AMP @50.0%** | **speedup** | **AMP @90.0%** | **speedup** | **AMP @99.0%** | **speedup** | **AMP @100.0%** | **speedup** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 118 img/s | 0.8x | 104 img/s | 0.8x | 76 img/s | 0.9x | 66 img/s | 0.9x |
| 2 | 244 img/s | 0.9x | 220 img/s | 0.8x | 153 img/s | 0.7x | 123 img/s | 0.9x |
| 4 | 455 img/s | 0.9x | 423 img/s | 0.8x | 367 img/s | 1.1x | 275 img/s | 1.1x |
| 8 | 886 img/s | 1.1x | 836 img/s | 1.1x | 626 img/s | 1.2x | 471 img/s | 0.9x |
| 16 | 1771 img/s | 1.8x | 1713 img/s | 1.8x | 1100 img/s | 1.3x | 905 img/s | 1.5x |
| 32 | 2217 img/s | 2.0x | 1949 img/s | 1.8x | 1619 img/s | 1.7x | 1385 img/s | 1.6x |
| 64 | 2416 img/s | 2.0x | 2212 img/s | 1.9x | 1993 img/s | 1.8x | 1985 img/s | 1.8x |
| 128 | 2524 img/s | 2.1x | 2287 img/s | 1.9x | 2046 img/s | 1.8x | 1503 img/s | 1.3x |
| 256 | 2626 img/s | N/A | 2149 img/s | N/A | 1533 img/s | N/A | 1346 img/s | N/A |

##### NVIDIA T4

###### FP32 Inference Latency

| **batch_size** | **FP32 50.0%** | **FP32 90.0%** | **FP32 99.0%** | **FP32 100.0%** |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 5.19ms | 5.65ms | 10.97ms | 14.06ms |
| 2 | 5.39ms | 5.95ms | 9.81ms | 13.17ms |
| 4 | 6.65ms | 7.34ms | 9.65ms | 13.26ms |
| 8 | 10.13ms | 10.33ms | 13.87ms | 16.51ms |
| 16 | 16.76ms | 17.15ms | 21.06ms | 25.66ms |
| 32 | 31.02ms | 31.12ms | 32.41ms | 34.93ms |
| 64 | 57.60ms | 57.84ms | 58.05ms | 59.69ms |
| 128 | 110.91ms | 111.15ms | 112.16ms | 112.20ms |
| 256 | N/A | N/A | N/A | N/A |

###### Mixed Precision Inference Latency

| **batch_size** | **AMP @50.0%** | **speedup** | **AMP @90.0%** | **speedup** | **AMP @99.0%** | **speedup** | **AMP @100.0%** | **speedup** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 5.75ms | 0.9x | 5.92ms | 1.0x | 11.58ms | 0.9x | 15.17ms | 0.9x |
| 2 | 5.66ms | 1.0x | 6.05ms | 1.0x | 11.52ms | 0.9x | 18.27ms | 0.7x |
| 4 | 5.86ms | 1.1x | 6.33ms | 1.2x | 8.90ms | 1.1x | 11.89ms | 1.1x |
| 8 | 6.43ms | 1.6x | 7.31ms | 1.4x | 12.41ms | 1.1x | 13.12ms | 1.3x |
| 16 | 8.85ms | 1.9x | 9.86ms | 1.7x | 17.01ms | 1.2x | 19.01ms | 1.3x |
| 32 | 15.42ms | 2.0x | 15.61ms | 2.0x | 18.66ms | 1.7x | 29.76ms | 1.2x |
| 64 | 28.50ms | 2.0x | 28.69ms | 2.0x | 31.06ms | 1.9x | 34.26ms | 1.7x |
| 128 | 54.82ms | 2.0x | 54.96ms | 2.0x | 55.27ms | 2.0x | 60.48ms | 1.9x |
| 256 | 106.47ms | N/A | 106.62ms | N/A | 107.03ms | N/A | 111.27ms | N/A |

###### FP32 Inference throughput

| **batch_size** | **FP32 @50.0%** | **FP32 @90.0%** | **FP32 @99.0%** | **FP32 @100.0%** |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 186 img/s | 171 img/s | 79 img/s | 57 img/s |
| 2 | 359 img/s | 320 img/s | 122 img/s | 113 img/s |
| 4 | 570 img/s | 529 img/s | 391 img/s | 204 img/s |
| 8 | 757 img/s | 707 img/s | 479 img/s | 432 img/s |
| 16 | 918 img/s | 899 img/s | 750 img/s | 615 img/s |
| 32 | 1017 img/s | 1011 img/s | 932 img/s | 756 img/s |
| 64 | 1101 img/s | 1096 img/s | 1034 img/s | 1015 img/s |
| 128 | 1148 img/s | 1145 img/s | 1096 img/s | 874 img/s |
| 256 | N/A | N/A | N/A | N/A |

###### Mixed Precision Inference throughput

| **batch_size** | **AMP @50.0%** | **speedup** | **AMP @90.0%** | **speedup** | **AMP @99.0%** | **speedup** | **AMP @100.0%** | **speedup** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 169 img/s | 0.9x | 163 img/s | 1.0x | 79 img/s | 1.0x | 55 img/s | 1.0x |
| 2 | 343 img/s | 1.0x | 311 img/s | 1.0x | 122 img/s | 1.0x | 107 img/s | 0.9x |
| 4 | 662 img/s | 1.2x | 612 img/s | 1.2x | 430 img/s | 1.1x | 215 img/s | 1.1x |
| 8 | 1207 img/s | 1.6x | 1055 img/s | 1.5x | 601 img/s | 1.3x | 384 img/s | 0.9x |
| 16 | 1643 img/s | 1.8x | 1552 img/s | 1.7x | 908 img/s | 1.2x | 824 img/s | 1.3x |
| 32 | 1919 img/s | 1.9x | 1674 img/s | 1.7x | 1393 img/s | 1.5x | 1021 img/s | 1.4x |
| 64 | 2201 img/s | 2.0x | 1772 img/s | 1.6x | 1569 img/s | 1.5x | 1342 img/s | 1.3x |
| 128 | 2311 img/s | 2.0x | 1833 img/s | 1.6x | 1261 img/s | 1.2x | 1107 img/s | 1.3x |
| 256 | 2389 img/s | N/A | 1841 img/s | N/A | 1280 img/s | N/A | 1164 img/s | N/A |



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



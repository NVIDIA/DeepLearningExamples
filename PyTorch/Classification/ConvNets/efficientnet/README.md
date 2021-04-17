# EfficientNet For PyTorch

This repository provides a script and recipe to train the EfficientNet model to
achieve state-of-the-art accuracy, and is tested and maintained by NVIDIA.

## Table Of Contents

* [Model overview](#model-overview)
  * [Default configuration](#default-configuration)
  * [Feature support matrix](#feature-support-matrix)
    * [Features](#features)
  * [Mixed precision training](#mixed-precision-training)
    * [Enabling mixed precision](#enabling-mixed-precision)
    * [Enabling TF32](#enabling-tf32)
  * [Quantization](#quantization)
    * [Quantization-aware training](#qat)
* [Setup](#setup)
  * [Requirements](#requirements)
* [Quick Start Guide](#quick-start-guide)
* [Advanced](#advanced)
  * [Scripts and sample code](#scripts-and-sample-code)
  * [Command-line options](#command-line-options)
  * [Dataset guidelines](#dataset-guidelines)
  * [Training process](#training-process)
  * [Inference process](#inference-process)
    * [NGC pretrained weights](#ngc-pretrained-weights)
  * [QAT process](#qat-process)
* [Performance](#performance)
  * [Benchmarking](#benchmarking)
    * [Training performance benchmark](#training-performance-benchmark)
    * [Inference performance benchmark](#inference-performance-benchmark)
  * [Results](#results)
    * [Training accuracy results](#training-accuracy-results)
      * [Training accuracy: NVIDIA A100 (8x A100 80GB)](#training-accuracy-nvidia-a100-8x-a100-80gb)
      * [Training accuracy: NVIDIA DGX-1 (8x V100 16GB)](#training-accuracy-nvidia-dgx-1-8x-v100-16gb)
      * [Example plots](#example-plots)
    * [Training performance results](#training-performance-results)
      * [Training performance: NVIDIA A100 (8x A100 80GB)](#training-performance-nvidia-a100-8x-a100-80gb)
      * [Training performance: NVIDIA DGX-1 (8x V100 16GB)](#training-performance-nvidia-dgx-1-8x-v100-16gb)
      * [Training performance: NVIDIA DGX-1 (8x V100 32GB)](#training-performance-nvidia-dgx-1-8x-v100-32gb)
  * [Inference performance results](#inference-performance-results)
      * [Inference performance: NVIDIA A100 (1x A100 80GB)](#inference-performance-nvidia-a100-1x-a100-80gb)
      * [Inference performance: NVIDIA V100 (1x V100 16GB)](#inference-performance-nvidia-v100-1x-v100-16gb)
  * [QAT results](#qat-results)
      * [QAT Training performance: NVIDIA DGX-1 (8x V100 32GB)](#qat-training-performance-nvidia-dgx-1-8x-v100-32gb))
      * [QAT Inference accuracy](#qat-inference-accuracy)
* [Release notes](#release-notes)
  * [Changelog](#changelog)
  * [Known issues](#known-issues)

## Model overview

EfficientNet is an image classification model family. It was first described in [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946). The scripts provided enable you to train the EfficientNet-B0, EfficientNet-B4, EfficientNet-WideSE-B0 and, EfficientNet-WideSE-B4 models.

EfficientNet-WideSE models use Squeeze-and-Excitation layers wider than original EfficientNet models, the width of SE module is proportional to the width of Depthwise Separable Convolutions instead of block width.

WideSE models are slightly more accurate than original models.

This model is trained with mixed precision using Tensor Cores on Volta and the NVIDIA Ampere GPU architectures. Therefore, researchers can get results over 2x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.

We use [NHWC data layout](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html) when training using Mixed Precision.

### Default configuration

The following sections highlight the default configurations for the EfficientNet models.

**Optimizer**

This model uses RMSprop with the following hyperparameters:
* Momentum (0.9)
* Learning rate (LR):
  * 0.08 for 4096 batch size for B0 models
  * 0.16 for 4096 batch size for B4 models
scale the learning rate.
* Learning rate schedule - we use cosine LR schedule
* We use linear warmup of the learning rate during the first 16 epochs
* Weight decay (WD):
  * 1e-5 for B0 models
  * 5e-6 for B4 models
* We do not apply WD on Batch Norm trainable parameters (gamma/bias)
* Label smoothing = 0.1
* [MixUp](https://arxiv.org/pdf/1710.09412.pdf) = 0.2
* We train for 400 epochs

**Optimizer for QAT**

This model uses SGD optimizer for B0 models and RMSPROP optimizer alpha=0.853  epsilon=0.00422 for B4 models. Other hyperparameters we used are:
* Momentum:
  * 0.89 for B0 models
  * 0.9 for B4 models
* Learning rate (LR):
  * 0.0125 for 128 batch size for B0 models
  * 4.09e-06 for 32 batch size for B4 models
scale the learning rate.
* Learning rate schedule: 
  * cosine LR schedule for B0 models
  * linear LR schedule for B4 models
* Weight decay (WD):
  * 4.50e-05 for B0 models
  * 9.714e-04 for B4 models
* We do not apply WD on Batch Norm trainable parameters (gamma/bias)
* We train for:
	*10 epochs for B0 models
	*2 epochs for B4 models


**Data augmentation**

This model uses the following data augmentation:
* For training:
  * Auto-augmentation
  * Basic augmentation:
      * Normalization
      * Random resized crop to target images size (depending on model version)
        * Scale from 8% to 100%
        * Aspect ratio from 3/4 to 4/3
      * Random horizontal flip

* For inference:
  * Normalization
  * Scale to target image size + 32
  * Center crop to target image size


### Feature support matrix

The following features are supported by this model:

| Feature               | EfficientNet
|-----------------------|--------------------------
|[DALI](https://docs.nvidia.com/deeplearning/dali/release-notes/index.html)   |   Yes (without autoaugmentation)
|[APEX AMP](https://nvidia.github.io/apex/amp.html) | Yes
|[QAT](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization)  |   Yes

#### Features

**NVIDIA DALI**

DALI is a library accelerating data preparation pipeline. To accelerate your input pipeline, you only need to define your data loader
with the DALI library. For more information about DALI, refer to the [DALI product documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html).

We use [NVIDIA DALI](https://github.com/NVIDIA/DALI),
which speeds up data loading when CPU becomes a bottleneck.
DALI can use CPU or GPU, and outperforms the PyTorch native dataloader.

Run training with `--data-backends dali-gpu` or `--data-backends dali-cpu` to enable DALI.
For DGXA100 and DGX1 we recommend `--data-backends dali-cpu`.

DALI currently does not support Autoaugmentation, so for best accuracy it has to be disabled.


**[APEX](https://github.com/NVIDIA/apex)**

A PyTorch extension that contains utility libraries, such as [Automatic Mixed Precision (AMP)](https://nvidia.github.io/apex/amp.html), which require minimal network code changes to leverage Tensor Cores performance. Refer to the [Enabling mixed precision](#enabling-mixed-precision) section for more details.

**[QAT](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization)**

Quantization aware training (QAT) is a method for changing precision to INT8 which speeds up the inference process at the price of a slight decrease of network accuracy. Refer to the [Quantization](#quantization) section for more details.


### Mixed precision training

Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format, while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in Volta, and following with both the Turing and Ampere architectures, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using mixed precision training requires two steps:
1.  Porting the model to use the FP16 data type where appropriate.
2.  Adding loss scaling to preserve small gradient values.

The ability to train deep learning networks with lower precision was introduced in the Pascal architecture and first supported in CUDA 8 in the NVIDIA Deep Learning SDK.

For information about:
-   How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) documentation.
-   Techniques used for mixed precision training, see the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
-   APEX tools for mixed precision training, see the [NVIDIA Apex: Tools for Easy Mixed-Precision Training in PyTorch](https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/).

#### Enabling mixed precision

Mixed precision is enabled in PyTorch by using the Automatic Mixed Precision (AMP), a library from [APEX](https://github.com/NVIDIA/apex) that casts variables to half-precision upon retrieval,
while storing variables in single-precision format. Furthermore, to preserve small gradient magnitudes in backpropagation, a [loss scaling](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#lossscaling) step must be included when applying gradients.
In PyTorch, loss scaling can be easily applied by using `scale_loss()` method provided by AMP. The scaling value to be used can be [dynamic](https://nvidia.github.io/apex/fp16_utils.html#apex.fp16_utils.DynamicLossScaler) or fixed.

For an in-depth walk through on AMP, check out sample usage [here](https://github.com/NVIDIA/apex/tree/master/apex/amp#usage-and-getting-started). [APEX](https://github.com/NVIDIA/apex) is a PyTorch extension that contains utility libraries, such as AMP, which require minimal network code changes to leverage Tensor Cores performance.

To enable mixed precision, you can:
- Import AMP from APEX:

  ```python
  from apex import amp
  ```

- Wrap model and optimizer in `amp.initialize`:
  ```python
  model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale="dynamic")
  ```

- Scale loss before backpropagation:
  ```python
  with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
  ```

#### Enabling TF32

TensorFloat-32 (TF32) is the new math mode in [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) GPUs for handling the matrix math also called tensor operations. TF32 running on Tensor Cores in A100 GPUs can provide up to 10x speedups compared to single-precision floating-point math (FP32) on Volta GPUs. 

TF32 Tensor Cores can speed up networks using FP32, typically with no loss of accuracy. It is more robust than FP16 for models which require high dynamic range for weights or activations.

For more information, refer to the [TensorFloat-32 in the A100 GPU Accelerates AI Training, HPC up to 20x](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) blog post.

TF32 is supported in the NVIDIA Ampere GPU architecture and is enabled by default.

### Quantization

Quantization is the process of transforming deep learning models to use parameters and computations at a lower precision. Traditionally, DNN training and inference have relied on the IEEE single-precision floating-point format, using 32 bits to represent the floating-point model weights and activation tensors.

This compute budget may be acceptable at training as most DNNs are trained in data centers or in the cloud with NVIDIA V100 or A100 GPUs that have significantly large compute capability and much larger power budgets. However, during deployment, these models are most often required to run on devices with much smaller computing resources and lower power budgets at the edge. Running a DNN inference using the full 32-bit representation is not practical for real-time analysis given the compute, memory, and power constraints of the edge.

To help reduce the compute budget, while not compromising on the structure and number of parameters in the model, you can run inference at a lower precision. Initially, quantized inferences were run at half-point precision with tensors and weights represented as 16-bit floating-point numbers. While this resulted in compute savings of about 1.2–1.5x, there was still some compute budget and memory bandwidth that could be leveraged. In lieu of this, models are now quantized to an even lower precision, with an 8-bit integer representation for weights and tensors. This results in a model that is 4x smaller in memory and about 2–4x faster in throughput.

While 8-bit quantization is appealing to save compute and memory budgets, it is a lossy process. During quantization, a small range of floating-point numbers are squeezed to a fixed number of information buckets. This results in loss of information.

The minute differences which could originally be resolved using 32-bit representations are now lost because they are quantized to the same bucket in 8-bit representations. This is similar to rounding errors that one encounters when representing fractional numbers as integers. To maintain accuracy during inferences at a lower precision, it is important to try and mitigate errors arising due to this loss of information.

#### Quantization-aware training

In QAT, the quantization error is considered when training the model. The training graph is modified to simulate the lower precision behavior in the forward pass of the training process. This introduces the quantization errors as part of the training loss, which the optimizer tries to minimize during the training. Thus, QAT helps in modeling the quantization errors during training and mitigates its effects on the accuracy of the model at deployment.

However, the process of modifying the training graph to simulate lower precision behavior is intricate. To run QAT, it is necessary to insert FakeQuantization nodes for the weights of the DNN Layers and Quantize-Dequantize (QDQ) nodes to the intermediate activation tensors to compute their dynamic ranges.

For more information, see this [Quantization paper](https://arxiv.org/abs/2004.09602) and [Quantization-Aware Training](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#quantization-training) documentation.
Tutorial for `pytoch-quantization` library can be found here [`pytorch-quantization` tutorial](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/tutorials/quant_resnet50.html).

It is important to mention that EfficientNet is NN, which is hard to quantize because the activation function all across the network is the SiLU (called also the Swish), whose negative values lie in very short range, which introduce a large quantization error. More details can be found in Appendix D of the [Quantization paper](https://arxiv.org/abs/2004.09602).

## Setup

The following section lists the requirements that you need to meet in order to start training the EfficientNet model.

### Requirements

This repository contains Dockerfile which extends the PyTorch NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:

* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch 21.03-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch) or newer
* Supported GPUs:
    * [NVIDIA Volta architecture](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
    * [NVIDIA Turing architecture](https://www.nvidia.com/en-us/geforce/turing/)
    * [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)

For more information about how to get started with NGC containers, see the
following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning
DGX Documentation:
* [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
* [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/dgx/user-guide/index.html#accessing_registry)
* [Running PyTorch](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html#running)

To set up the required environment or create your own container, as an alternative to the use of the PyTorch NGC container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## Quick Start Guide

To train your model using mixed or TF32 precision with Tensor Cores or using FP32,
perform the following steps using the default parameters of the efficientnet model on the ImageNet dataset.
For the specifics concerning training and inference, see the [Advanced](#advanced) section.

1. Clone the repository.
```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/PyTorch/Classification/
```

2. Download and pre-process the dataset.

The EfficientNet script operates on ImageNet 1k, a widely popular image classification dataset from the ILSVRC challenge.

PyTorch can work directly on JPEGs, therefore, pre-processing/augmentation is not needed.


3. [Download the images](http://image-net.org/download-images).

4. Extract the training data:
  ```bash
  mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
  tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
  find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
  cd ..
  ```

5. Extract the validation data and move the images to subfolders:
  ```bash
  mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
  wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
  ```

The directory in which the `train/` and `val/` directories are placed, is referred to as `<path to imagenet>` in this document.

6. Build the EfficientNet PyTorch NGC container.

```
docker build . -t nvidia_efficientnet
```

7. Start an interactive session in the NGC container to run training/inference.
```
nvidia-docker run --rm -it -v <path to imagenet>:/imagenet --ipc=host nvidia_efficientnet
```


8. Start training

To run training for a standard configuration (DGX A100/DGX-1V, AMP/TF32/FP32, 400 Epochs),
run one of the scripts in the `./efficientnet/training` directory
called `./efficientnet/training/{AMP, TF32, FP32}/{ DGX A100, DGX-1V }_efficientnet-<version>_{AMP, TF32, FP32}_{ 400 }E.sh`.

Ensure ImageNet is mounted in the `/imagenet` directory.

For example:
    `bash ./efficientnet/training/AMP/DGXA100_efficientnet-b0_AMP.sh <path were to store checkpoints and logs>`

9. Start inference

You can download pre-trained weights from NGC:

```bash
wget --content-disposition  -O 

unzip 
```

To run inference on ImageNet, run:

`python ./main.py --arch efficientnet-<version> --evaluate --epochs 1 --pretrained-from-file  -b <batch size> <path to imagenet>`

To run inference on JPEG image using pre-trained weights, run:

`python classify.py --arch efficientnet-<version> -c fanin --weights  --precision AMP|FP32 --image <path to JPEG image>`


## Advanced

The following sections provide greater details of the dataset, running training and inference, and the training results.

### Scripts and sample code

For a non-standard configuration, run:

* For 1 GPU
    * FP32
        `python ./main.py --arch efficientnet-<version> --label-smoothing 0.1 <path to imagenet>`
        `python ./main.py --arch efficientnet-<version> --label-smoothing 0.1 --amp --static-loss-scale 256 <path to imagenet>`

* For multiple GPUs
    * FP32
        `python ./multiproc.py --nproc_per_node 8 ./main.py --arch efficientnet-<version> --label-smoothing 0.1 <path to imagenet>`
    * AMP
        `python ./multiproc.py --nproc_per_node 8 ./main.py --arch efficientnet-<version> --label-smoothing 0.1 --amp --static-loss-scale 256 <path to imagenet>`

Use `python ./main.py -h` to obtain the list of available options in the `main.py` script.


### Command-line options

To see the full list of available options and their descriptions, use the `-h` or `--help` command-line option, for example:

`python main.py -h`


### Dataset guidelines

To use your own dataset, divide it into directories. For example:

 - Training images - `train/<class id>/<image>`
 - Validation images - `val/<class id>/<image>`

If your dataset has a number of classes different than 1000, you need to pass the `--num-classes N` flag to the training script.

### Training process

All the results of the training will be stored in the directory specified with `--workspace` argument.

The script will store:
 - the most recent checkpoint - `checkpoint.pth.tar` (unless `--no-checkpoints` flag is used).
 - the checkpoint with the best validation accuracy - `model_best.pth.tar` (unless `--no-checkpoints` flag is used).
 - the JSON log - in the file specified with the `--raport-file` flag.

Metrics gathered through training:
 - `train.loss` - training loss
 - `train.total_ips` - training speed measured in images/second
 - `train.compute_ips` - training speed measured in images/second, not counting data loading
 - `train.data_time` - time spent on waiting on data
 - `train.compute_time` - time spent in forward/backward pass

To restart training from the checkpoint use the `--resume` option.

To start training from pretrained weights (for example, downloaded from NGC) use the `--pretrained-from-file` option.

The difference between `--resume` and `--pretrained-from-file` flags is that the pretrained weights contain only model weights,
and checkpoints, apart from model weights, contain optimizer state, LR scheduler state.

Checkpoints are suitable for dividing the training into parts, for example, in order
to divide the training job into shorter stages, or restart training after an infrastructure failure.

Pretrained weights can be used as a base for fine tuning the model to a different dataset,
or as a backbone to detection models.

### Inference process

Validation is done every epoch, and can be also run separately on a checkpointed model.

`python ./main.py --arch efficientnet-<version> --evaluate --epochs 1 --resume <path to checkpoint> -b <batch size> <path to imagenet>`

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

Then, run the classification script:

`python classify.py --arch efficientnet-<version> --weights <path to weights from previous step> --precision AMP|FP32 --image <path to JPEG image>`

You can also run the ImageNet validation on pretrained weights:

`python ./main.py --arch efficientnet-<version> --evaluate --epochs 1 --pretrained-from-file <path to pretrained weights> -b <batch size> <path to imagenet>`

#### NGC pretrained weights

Pretrained weights can be downloaded from NGC:

```bash
wget --content-disposition <ngc weights url>
```




To run inference on ImageNet, run:

`python ./main.py --arch efficientnet-<version> --evaluate --epochs 1 --pretrained-from-file  -b <batch size> <path to imagenet>`

To run inference on JPEG images using pretrained weights, run:

`python classify.py --arch efficientnet-<version> --weights  --precision AMP|FP32 --image <path to JPEG image>`


### Quantization process

EfficientNet-b0 and EfficientNet-b4 models can be quantized using the QAT process from running the `quant_main.py` script.

`python ./quant_main.py <path to imagenet> --arch efficientnet-quant-<version> --epochs <# of QAT epochs> --pretrained-from-file <path to non-quantized model weights> <any other parameters for training such as batch, momentum etc.>`

During the QAT process, evaluation is done in the same way as during standard training. `quant_main.py`  works in the same way as the original `main.py` script, but with quantized models. It means that `quant_main.py` can be used to resume the QAT process with the flag `--resume`:
`python ./quant_main.py <path to imagenet> --arch efficientnet-quant-<version> --resume <path to mid-training checkpoint> ...`
or to evaluate a created checkpoint with the flag `--evaluate`:
`python ./quant_main.py --arch efficientnet-quant-<version> --evaluate --epochs 1 --resume <path to checkpoint> -b <batch size> <path to imagenet>` 
It also can run on multi-GPU in an identical way as the standard `main.py` script:
`python ./multiproc.py --nproc_per_node 8 ./quant_main.py --arch efficientnet-quant-<version> ... <path to imagenet>`

There is also a possibility to transform trained models (quantized or not) into ONNX format, which is needed to convert it later into TensorRT, where quantized networks are much faster during inference. Conversion to TensorRT will be supported in the next release. The conversion to ONNX consists of two steps:
* translate checkpoint to pure weights:
`python checkpoint2model.py --checkpoint-path <path to quant checkpoint> --weight-path <path where quant weights will be stored>` 
* translate pure weights to ONNX:
`python model2onnx.py --arch efficientnet-quant-<version> --pretrained-from-file <path to model quant weights> -b <batch size>`

Quantized models could also be used to classify new images using the `classify.py`  flag. For example:
`python classify.py --arch efficientnet-quant-<version> -c fanin --pretrained-from-file <path to quant weights> --image <path to JPEG image>`


## Performance

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in training and inference modes.

#### Training performance benchmark

To benchmark training, run:

* For 1 GPU
    * FP32 (V100 GPUs only)
        `python ./launch.py --model efficientnet-<version> --precision FP32 --mode benchmark_training --platform DGX1V <path to imagenet> --raport-file benchmark.json --epochs 1 --prof 100`
    * TF32 (A100 GPUs only)
        `python ./launch.py --model efficientnet-<version> --precision TF32 --mode benchmark_training --platform DGXA100 <path to imagenet> --raport-file benchmark.json --epochs 1 --prof 100`
    * AMP
        `python ./launch.py --model efficientnet-<version> --precision AMP --mode benchmark_training --platform <DGX1V|DGXA100> <path to imagenet> --raport-file benchmark.json --epochs 1 --prof 100`

* For multiple GPUs
    * FP32 (V100 GPUs only)
        `python ./launch.py --model efficientnet-<version> --precision FP32 --mode benchmark_training --platform DGX1V <path to imagenet> --raport-file benchmark.json --epochs 1 --prof 100`
    * TF32 (A100 GPUs only)
        `python ./multiproc.py --nproc_per_node 8 ./launch.py --model efficientnet-<version> --precision TF32 --mode benchmark_training --platform DGXA100 <path to imagenet> --raport-file benchmark.json --epochs 1 --prof 100`
    * AMP
        `python ./multiproc.py --nproc_per_node 8 ./launch.py --model efficientnet-<version> --precision AMP --mode benchmark_training --platform <DGX1V|DGXA100> <path to imagenet> --raport-file benchmark.json --epochs 1 --prof 100`

Each of these scripts will run 100 iterations and save results in the `benchmark.json` file.

#### Inference performance benchmark

To benchmark inference, run:

* FP32 (V100 GPUs only)

`python ./launch.py --model efficientnet-<version> --precision FP32 --mode benchmark_inference --platform DGX1V <path to imagenet> --raport-file benchmark.json --epochs 1 --prof 100`

* TF32 (A100 GPUs only)

`python ./launch.py --model efficientnet-<version> --precision TF32 --mode benchmark_inference --platform DGXA100 <path to imagenet> --raport-file benchmark.json --epochs 1 --prof 100`

* AMP

`python ./launch.py --model efficientnet-<version> --precision AMP --mode benchmark_inference --platform <DGX1V|DGXA100> <path to imagenet> --raport-file benchmark.json --epochs 1 --prof 100`

Each of these scripts will run 100 iterations and save results in the `benchmark.json` file.

### Results

Our results were obtained by running the applicable training script in the pytorch-20.12 NGC container.

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

#### Training accuracy results

##### Training accuracy: NVIDIA DGX A100 (8x A100 80GB)

Our results were obtained by running the applicable `efficientnet/training/<AMP|TF32>/*.sh` training script in the PyTorch 20.12 NGC container on NVIDIA DGX A100 (8x A100 80GB) GPUs.

|       **Model**        | **Epochs** | **GPUs** | **Top1 accuracy - TF32** | **Top1 accuracy - mixed precision** | **Time to train - TF32** | **Time to train - mixed precision** | **Time to train speedup (TF32 to mixed precision)** |
|:----------------------:|:----------:|:--------:|:------------------------:|:-----------------------------------:|:------------------------:|:-----------------------------------:|:---------------------------------------------------:|
|    efficientnet-b0     |    400     |    8     |      77.16 +/- 0.07      |           77.42 +/- 0.11            |            19            |                 11                  |                        1.727                        |
|    efficientnet-b4     |    400     |    8     |      82.82 +/- 0.04      |           82.85 +/- 0.09            |           126            |                 66                  |                        1.909                        |
| efficientnet-widese-b0 |    400     |    8     |      77.84 +/- 0.08      |           77.84 +/- 0.02            |            19            |                 10                  |                        1.900                        |
| efficientnet-widese-b4 |    400     |    8     |      83.13 +/- 0.11      |            83.1 +/- 0.09            |           126            |                 66                  |                        1.909                        |


##### Training accuracy: NVIDIA DGX-1 (8x V100 16GB)

Our results were obtained by running the applicable `efficientnet/training/<AMP|FP32>/*.sh` training script in the PyTorch 20.12 NGC container on NVIDIA DGX-1 (8x V100 16GB) GPUs.

|       **Model**        | **Epochs** | **GPUs** | **Top1 accuracy - FP32** | **Top1 accuracy - mixed precision** | **Time to train - FP32** | **Time to train - mixed precision** | **Time to train speedup (FP32 to mixed precision)** |
|:----------------------:|:----------:|:--------:|:------------------------:|:-----------------------------------:|:------------------------:|:-----------------------------------:|:---------------------------------------------------:|
|    efficientnet-b0     |    400     |    8     |      77.02 +/- 0.04      |           77.17 +/- 0.08            |            34            |                 24                  |                        1.417                        |
| efficientnet-widese-b0 |    400     |    8     |      77.59 +/- 0.16      |           77.69 +/- 0.12            |            35            |                 24                  |                        1.458                        |


##### Example plots

The following images show an A100 run.

![ValidationLoss](./img/loss_plot.png)

![ValidationTop1](./img/top1_plot.png)

![ValidationTop5](./img/top5_plot.png)

#### Training performance results

##### Training performance: NVIDIA A100 (8x A100 80GB)

Our results were obtained by running the applicable `efficientnet/training/<AMP|TF32>/*.sh` training script in the PyTorch 20.12 NGC container on NVIDIA DGX A100 (8x A100 80GB) GPUs.

|       **Model**        | **GPUs** |  **TF32**  | **Throughput - mixed precision** | **Throughput speedup (TF32 to mixed precision)** | **TF32 Strong Scaling** | **Mixed Precision Strong Scaling** |
|:----------------------:|:--------:|:----------:|:--------------------------------:|:------------------------------------------------:|:-----------------------:|:----------------------------------:|
|    efficientnet-b0     |    1     | 1082 img/s |            2364 img/s            |                      2.18 x                      |          1.0 x          |               1.0 x                |
|    efficientnet-b0     |    8     | 8225 img/s |           14391 img/s            |                      1.74 x                      |         7.59 x          |               6.08 x               |
|    efficientnet-b4     |    1     | 154 img/s  |            300 img/s             |                      1.94 x                      |          1.0 x          |               1.0 x                |
|    efficientnet-b4     |    8     | 1204 img/s |            2341 img/s            |                      1.94 x                      |          7.8 x          |               7.8 x                |
| efficientnet-widese-b0 |    1     | 1081 img/s |            2368 img/s            |                      2.19 x                      |          1.0 x          |               1.0 x                |
| efficientnet-widese-b0 |    8     | 8233 img/s |           15053 img/s            |                      1.82 x                      |         7.61 x          |               6.35 x               |
| efficientnet-widese-b4 |    1     | 154 img/s  |            299 img/s             |                      1.94 x                      |          1.0 x          |               1.0 x                |
| efficientnet-widese-b4 |    8     | 1202 img/s |            2339 img/s            |                      1.94 x                      |          7.8 x          |               7.81 x               |


##### Training performance: NVIDIA DGX-1 (8x V100 16GB)

Our results were obtained by running the applicable `efficientnet/training/<AMP|FP32>/*.sh` training script in the PyTorch 20.12 NGC container on NVIDIA DGX-1 (8x V100 16GB) GPUs.

|       **Model**        | **GPUs** |  **FP32**  | **Throughput - mixed precision** | **Throughput speedup (FP32 to mixed precision)** | **FP32 Strong Scaling** | **Mixed Precision Strong Scaling** |
|:----------------------:|:--------:|:----------:|:--------------------------------:|:------------------------------------------------:|:-----------------------:|:----------------------------------:|
|    efficientnet-b0     |    1     | 652 img/s  |            1254 img/s            |                      1.92 x                      |          1.0 x          |               1.0 x                |
|    efficientnet-b0     |    8     | 4571 img/s |            7664 img/s            |                      1.67 x                      |          7.0 x          |               6.1 x                |
|    efficientnet-b4     |    1     |  80 img/s  |            199 img/s             |                      2.47 x                      |          1.0 x          |               1.0 x                |
|    efficientnet-b4     |    8     | 598 img/s  |            1330 img/s            |                      2.22 x                      |         7.42 x          |               6.67 x               |
| efficientnet-widese-b0 |    1     | 654 img/s  |            1255 img/s            |                      1.91 x                      |          1.0 x          |               1.0 x                |
| efficientnet-widese-b0 |    8     | 4489 img/s |            7694 img/s            |                      1.71 x                      |         6.85 x          |               6.12 x               |
| efficientnet-widese-b4 |    1     |  79 img/s  |            198 img/s             |                      2.51 x                      |          1.0 x          |               1.0 x                |
| efficientnet-widese-b4 |    8     | 590 img/s  |            1323 img/s            |                      2.24 x                      |         7.46 x          |               6.65 x               |


##### Training performance: NVIDIA DGX-1 (8x V100 32GB)

Our results were obtained by running the applicable `efficientnet/training/<AMP|FP32>/*.sh` training script in the PyTorch 20.12 NGC container on NVIDIA DGX-1 (8x V100 16GB) GPUs.

|       **Model**        | **GPUs** |  **FP32**  | **Throughput - mixed precision** | **Throughput speedup (FP32 to mixed precision)** | **FP32 Strong Scaling** | **Mixed Precision Strong Scaling** |
|:----------------------:|:--------:|:----------:|:--------------------------------:|:------------------------------------------------:|:-----------------------:|:----------------------------------:|
|    efficientnet-b0     |    1     | 637 img/s  |            1352 img/s            |                      2.12 x                      |          1.0 x          |               1.0 x                |
|    efficientnet-b0     |    8     | 4834 img/s |            8645 img/s            |                      1.78 x                      |         7.58 x          |               6.39 x               |
|    efficientnet-b4     |    1     |  84 img/s  |            200 img/s             |                      2.38 x                      |          1.0 x          |               1.0 x                |
|    efficientnet-b4     |    8     | 632 img/s  |            1519 img/s            |                      2.4 x                       |         7.53 x          |               7.58 x               |
| efficientnet-widese-b0 |    1     | 637 img/s  |            1349 img/s            |                      2.11 x                      |          1.0 x          |               1.0 x                |
| efficientnet-widese-b0 |    8     | 4841 img/s |            8693 img/s            |                      1.79 x                      |         7.59 x          |               6.43 x               |
| efficientnet-widese-b4 |    1     |  83 img/s  |            200 img/s             |                      2.38 x                      |          1.0 x          |               1.0 x                |
| efficientnet-widese-b4 |    8     | 627 img/s  |            1508 img/s            |                      2.4 x                       |         7.47 x          |               7.53 x               |


#### Inference performance results

##### Inference performance: NVIDIA A100 (1x A100 80GB)

Our results were obtained by running the applicable `efficientnet/inference/<AMP|FP32>/*.sh` inference script in the PyTorch 20.12 NGC container on NVIDIA DGX-1 (8x V100 16GB) GPUs.

###### TF32 Inference Latency

|       **Model**        | **Batch Size** | **Throughput Avg** | **Latency Avg** | **Latency 95%** | **Latency 99%** |
|:----------------------:|:--------------:|:------------------:|:---------------:|:---------------:|:---------------:|
|    efficientnet-b0     |       1        |     122 img/s      |    10.04 ms     |     8.59 ms     |     10.2 ms     |
|    efficientnet-b0     |       2        |     249 img/s      |     9.91 ms     |     9.08 ms     |    10.84 ms     |
|    efficientnet-b0     |       4        |     472 img/s      |    10.31 ms     |     9.67 ms     |    11.25 ms     |
|    efficientnet-b0     |       8        |     922 img/s      |    10.67 ms     |    10.76 ms     |    12.13 ms     |
|    efficientnet-b0     |       16       |     1796 img/s     |    10.86 ms     |     11.1 ms     |    13.01 ms     |
|    efficientnet-b0     |       32       |     3235 img/s     |    12.05 ms     |    13.28 ms     |    15.07 ms     |
|    efficientnet-b0     |       64       |     4658 img/s     |    16.27 ms     |    14.56 ms     |    16.18 ms     |
|    efficientnet-b0     |      128       |     4911 img/s     |    31.51 ms     |    26.24 ms     |    27.29 ms     |
|    efficientnet-b0     |      256       |     5015 img/s     |    62.64 ms     |    50.81 ms     |     55.6 ms     |
|    efficientnet-b4     |       1        |      63 img/s      |    17.64 ms     |    16.29 ms     |    17.92 ms     |
|    efficientnet-b4     |       2        |     122 img/s      |    18.27 ms     |    18.12 ms     |    22.32 ms     |
|    efficientnet-b4     |       4        |     247 img/s      |    18.25 ms     |    17.79 ms     |    21.02 ms     |
|    efficientnet-b4     |       8        |     469 img/s      |    19.03 ms     |    18.94 ms     |    22.49 ms     |
|    efficientnet-b4     |       16       |     572 img/s      |    29.95 ms     |    28.14 ms     |    28.99 ms     |
|    efficientnet-b4     |       32       |     638 img/s      |    52.25 ms     |    50.24 ms     |     50.5 ms     |
|    efficientnet-b4     |       64       |     680 img/s      |    96.93 ms     |     94.1 ms     |     94.3 ms     |
|    efficientnet-b4     |      128       |     672 img/s      |    197.49 ms    |    189.69 ms    |    189.91 ms    |
|    efficientnet-b4     |      256       |     679 img/s      |    392.15 ms    |    374.18 ms    |    386.85 ms    |
| efficientnet-widese-b0 |       1        |     120 img/s      |    10.21 ms     |     8.61 ms     |    11.37 ms     |
| efficientnet-widese-b0 |       2        |     242 img/s      |    10.16 ms     |     9.98 ms     |    11.36 ms     |
| efficientnet-widese-b0 |       4        |     493 img/s      |     9.97 ms     |     8.92 ms     |    10.23 ms     |
| efficientnet-widese-b0 |       8        |     913 img/s      |    10.77 ms     |    10.58 ms     |    12.11 ms     |
| efficientnet-widese-b0 |       16       |     1864 img/s     |    10.54 ms     |    10.34 ms     |    11.69 ms     |
| efficientnet-widese-b0 |       32       |     3218 img/s     |    12.06 ms     |    13.17 ms     |    15.69 ms     |
| efficientnet-widese-b0 |       64       |     4625 img/s     |     16.4 ms     |    15.35 ms     |    17.86 ms     |
| efficientnet-widese-b0 |      128       |     4904 img/s     |    31.84 ms     |    26.22 ms     |    28.69 ms     |
| efficientnet-widese-b0 |      256       |     5013 img/s     |     63.1 ms     |    50.95 ms     |    52.44 ms     |
| efficientnet-widese-b4 |       1        |      64 img/s      |    17.51 ms     |     16.5 ms     |    20.03 ms     |
| efficientnet-widese-b4 |       2        |     125 img/s      |    17.86 ms     |    17.24 ms     |    19.27 ms     |
| efficientnet-widese-b4 |       4        |     248 img/s      |    18.09 ms     |    17.36 ms     |    21.34 ms     |
| efficientnet-widese-b4 |       8        |     472 img/s      |    18.92 ms     |    18.33 ms     |    20.68 ms     |
| efficientnet-widese-b4 |       16       |     569 img/s      |    30.11 ms     |    28.18 ms     |    28.45 ms     |
| efficientnet-widese-b4 |       32       |     628 img/s      |    53.05 ms     |    51.11 ms     |    51.29 ms     |
| efficientnet-widese-b4 |       64       |     679 img/s      |    97.17 ms     |    94.22 ms     |    94.43 ms     |
| efficientnet-widese-b4 |      128       |     672 img/s      |    197.74 ms    |    189.93 ms    |    190.95 ms    |
| efficientnet-widese-b4 |      256       |     679 img/s      |    392.7 ms     |    373.84 ms    |    378.35 ms    |


###### Mixed Precision Inference Latency

|       **Model**        | **Batch Size** | **Throughput Avg** | **Latency Avg** | **Latency 95%** | **Latency 99%** |
|:----------------------:|:--------------:|:------------------:|:---------------:|:---------------:|:---------------:|
|    efficientnet-b0     |       1        |      99 img/s      |    11.89 ms     |    10.83 ms     |    13.04 ms     |
|    efficientnet-b0     |       2        |     208 img/s      |    11.43 ms     |    10.15 ms     |    10.87 ms     |
|    efficientnet-b0     |       4        |     395 img/s      |     12.0 ms     |    11.01 ms     |     12.8 ms     |
|    efficientnet-b0     |       8        |     763 img/s      |    12.33 ms     |    11.62 ms     |    13.94 ms     |
|    efficientnet-b0     |       16       |     1499 img/s     |    12.58 ms     |    12.57 ms     |     14.4 ms     |
|    efficientnet-b0     |       32       |     2875 img/s     |    13.19 ms     |    13.76 ms     |    15.29 ms     |
|    efficientnet-b0     |       64       |     5841 img/s     |     13.7 ms     |    14.91 ms     |    18.73 ms     |
|    efficientnet-b0     |      128       |     7850 img/s     |    21.53 ms     |    16.58 ms     |    18.94 ms     |
|    efficientnet-b0     |      256       |     8285 img/s     |    42.07 ms     |    30.87 ms     |    38.03 ms     |
|    efficientnet-b4     |       1        |      51 img/s      |     21.2 ms     |    19.73 ms     |    21.47 ms     |
|    efficientnet-b4     |       2        |     103 img/s      |    21.17 ms     |    20.91 ms     |    24.17 ms     |
|    efficientnet-b4     |       4        |     205 img/s      |    21.34 ms     |    20.32 ms     |    23.46 ms     |
|    efficientnet-b4     |       8        |     376 img/s      |    23.11 ms     |    22.64 ms     |    24.77 ms     |
|    efficientnet-b4     |       16       |     781 img/s      |    22.42 ms     |    23.03 ms     |    25.37 ms     |
|    efficientnet-b4     |       32       |     1048 img/s     |    32.52 ms     |    30.76 ms     |    31.65 ms     |
|    efficientnet-b4     |       64       |     1156 img/s     |    58.31 ms     |    55.45 ms     |    56.89 ms     |
|    efficientnet-b4     |      128       |     1197 img/s     |    112.92 ms    |    106.69 ms    |    107.84 ms    |
|    efficientnet-b4     |      256       |     1229 img/s     |    220.5 ms     |    206.68 ms    |    223.16 ms    |
| efficientnet-widese-b0 |       1        |     100 img/s      |    11.75 ms     |    10.62 ms     |    13.67 ms     |
| efficientnet-widese-b0 |       2        |     200 img/s      |    11.86 ms     |    11.38 ms     |    14.32 ms     |
| efficientnet-widese-b0 |       4        |     400 img/s      |    11.81 ms     |     10.8 ms     |     13.8 ms     |
| efficientnet-widese-b0 |       8        |     770 img/s      |    12.17 ms     |     11.2 ms     |    12.38 ms     |
| efficientnet-widese-b0 |       16       |     1501 img/s     |    12.62 ms     |    12.12 ms     |    14.94 ms     |
| efficientnet-widese-b0 |       32       |     2901 img/s     |    13.06 ms     |    13.28 ms     |    15.23 ms     |
| efficientnet-widese-b0 |       64       |     5853 img/s     |    13.69 ms     |    14.38 ms     |    16.91 ms     |
| efficientnet-widese-b0 |      128       |     7807 img/s     |    21.43 ms     |    16.63 ms     |     21.8 ms     |
| efficientnet-widese-b0 |      256       |     8270 img/s     |    42.01 ms     |    30.97 ms     |    34.55 ms     |
| efficientnet-widese-b4 |       1        |      52 img/s      |    21.03 ms     |     19.9 ms     |    22.23 ms     |
| efficientnet-widese-b4 |       2        |     102 img/s      |    21.34 ms     |     21.6 ms     |    24.23 ms     |
| efficientnet-widese-b4 |       4        |     200 img/s      |    21.76 ms     |    21.19 ms     |    23.69 ms     |
| efficientnet-widese-b4 |       8        |     373 img/s      |    23.31 ms     |    22.99 ms     |    28.33 ms     |
| efficientnet-widese-b4 |       16       |     763 img/s      |    22.93 ms     |    23.75 ms     |     26.6 ms     |
| efficientnet-widese-b4 |       32       |     1043 img/s     |     32.7 ms     |    31.03 ms     |    33.52 ms     |
| efficientnet-widese-b4 |       64       |     1152 img/s     |    58.27 ms     |    55.64 ms     |    55.86 ms     |
| efficientnet-widese-b4 |      128       |     1197 img/s     |    112.86 ms    |    106.72 ms    |    108.65 ms    |
| efficientnet-widese-b4 |      256       |     1229 img/s     |    221.11 ms    |    206.5 ms     |    221.37 ms    |


##### Inference performance: NVIDIA V100 (1x V100 16GB)

Our results were obtained by running the applicable `efficientnet/inference/<AMP|FP32>/*.sh` inference script in the PyTorch 20.12 NGC container on NVIDIA DGX-1 (8x V100 16GB) GPUs.

###### FP32 Inference Latency

|       **Model**        | **Batch Size** | **Throughput Avg** | **Latency Avg** | **Latency 95%** | **Latency 99%** |
|:----------------------:|:--------------:|:------------------:|:---------------:|:---------------:|:---------------:|
|    efficientnet-b0     |       1        |      77 img/s      |    14.23 ms     |    13.31 ms     |    14.68 ms     |
|    efficientnet-b0     |       2        |     153 img/s      |    14.46 ms     |    13.67 ms     |    14.69 ms     |
|    efficientnet-b0     |       4        |     317 img/s      |    14.06 ms     |    15.77 ms     |    17.28 ms     |
|    efficientnet-b0     |       8        |     646 img/s      |    13.88 ms     |    14.32 ms     |    15.05 ms     |
|    efficientnet-b0     |       16       |     1217 img/s     |    14.74 ms     |    15.89 ms     |    18.03 ms     |
|    efficientnet-b0     |       32       |     2162 img/s     |    16.51 ms     |     17.9 ms     |    20.06 ms     |
|    efficientnet-b0     |       64       |     2716 img/s     |    25.74 ms     |    23.64 ms     |    24.08 ms     |
|    efficientnet-b0     |      128       |     2816 img/s     |    50.21 ms     |    45.43 ms     |     46.3 ms     |
|    efficientnet-b0     |      256       |     2955 img/s     |    96.46 ms     |    85.96 ms     |    92.74 ms     |
|    efficientnet-b4     |       1        |      38 img/s      |    27.73 ms     |    27.98 ms     |    29.45 ms     |
|    efficientnet-b4     |       2        |      84 img/s      |     25.1 ms     |     24.6 ms     |    26.29 ms     |
|    efficientnet-b4     |       4        |     170 img/s      |    25.01 ms     |    24.84 ms     |    26.52 ms     |
|    efficientnet-b4     |       8        |     304 img/s      |    27.75 ms     |    26.28 ms     |    27.71 ms     |
|    efficientnet-b4     |       16       |     334 img/s      |    49.51 ms     |    47.98 ms     |    48.46 ms     |
|    efficientnet-b4     |       32       |     353 img/s      |    92.42 ms     |    90.81 ms     |     91.0 ms     |
|    efficientnet-b4     |       64       |     380 img/s      |    170.58 ms    |    168.32 ms    |    168.8 ms     |
|    efficientnet-b4     |      128       |     381 img/s      |    343.03 ms    |    334.58 ms    |    334.94 ms    |
| efficientnet-widese-b0 |       1        |      83 img/s      |    13.38 ms     |    13.14 ms     |    13.58 ms     |
| efficientnet-widese-b0 |       2        |     149 img/s      |    14.82 ms     |    15.09 ms     |    16.03 ms     |
| efficientnet-widese-b0 |       4        |     319 img/s      |    13.91 ms     |    13.06 ms     |    13.96 ms     |
| efficientnet-widese-b0 |       8        |     566 img/s      |    15.62 ms     |     16.3 ms     |     17.5 ms     |
| efficientnet-widese-b0 |       16       |     1211 img/s     |    14.85 ms     |    15.97 ms     |     18.8 ms     |
| efficientnet-widese-b0 |       32       |     2055 img/s     |    17.33 ms     |    19.54 ms     |    21.59 ms     |
| efficientnet-widese-b0 |       64       |     2707 img/s     |    25.66 ms     |    23.72 ms     |    23.93 ms     |
| efficientnet-widese-b0 |      128       |     2811 img/s     |    49.93 ms     |    45.46 ms     |    45.51 ms     |
| efficientnet-widese-b0 |      256       |     2953 img/s     |    96.43 ms     |    86.11 ms     |    87.33 ms     |
| efficientnet-widese-b4 |       1        |      44 img/s      |    24.16 ms     |    23.16 ms     |    25.41 ms     |
| efficientnet-widese-b4 |       2        |      89 img/s      |    23.95 ms     |    23.39 ms     |    25.93 ms     |
| efficientnet-widese-b4 |       4        |     169 img/s      |    25.35 ms     |    25.15 ms     |    30.58 ms     |
| efficientnet-widese-b4 |       8        |     279 img/s      |    30.27 ms     |    31.76 ms     |    33.37 ms     |
| efficientnet-widese-b4 |       16       |     331 img/s      |    49.84 ms     |    48.32 ms     |    48.75 ms     |
| efficientnet-widese-b4 |       32       |     353 img/s      |    92.31 ms     |    90.81 ms     |    90.95 ms     |
| efficientnet-widese-b4 |       64       |     375 img/s      |    172.79 ms    |    170.49 ms    |    170.69 ms    |
| efficientnet-widese-b4 |      128       |     381 img/s      |    342.33 ms    |    334.91 ms    |    335.23 ms    |


###### Mixed Precision Inference Latency

|       **Model**        | **Batch Size** | **Throughput Avg** | **Latency Avg** | **Latency 95%** | **Latency 99%** |
|:----------------------:|:--------------:|:------------------:|:---------------:|:---------------:|:---------------:|
|    efficientnet-b0     |       1        |      66 img/s      |    16.38 ms     |    15.63 ms     |    17.01 ms     |
|    efficientnet-b0     |       2        |     120 img/s      |     18.0 ms     |    18.39 ms     |    19.35 ms     |
|    efficientnet-b0     |       4        |     244 img/s      |    17.77 ms     |    18.98 ms     |     21.4 ms     |
|    efficientnet-b0     |       8        |     506 img/s      |    17.26 ms     |    18.23 ms     |    20.24 ms     |
|    efficientnet-b0     |       16       |     912 img/s      |    19.07 ms     |    20.33 ms     |    22.59 ms     |
|    efficientnet-b0     |       32       |     1758 img/s     |     20.3 ms     |     22.2 ms     |     24.7 ms     |
|    efficientnet-b0     |       64       |     3720 img/s     |    19.18 ms     |    20.09 ms     |    21.48 ms     |
|    efficientnet-b0     |      128       |     4942 img/s     |    30.53 ms     |     26.0 ms     |    27.54 ms     |
|    efficientnet-b0     |      256       |     5339 img/s     |    57.82 ms     |    47.63 ms     |    51.61 ms     |
|    efficientnet-b4     |       1        |      32 img/s      |    31.83 ms     |    32.51 ms     |    34.09 ms     |
|    efficientnet-b4     |       2        |      65 img/s      |    31.82 ms     |    34.53 ms     |    36.95 ms     |
|    efficientnet-b4     |       4        |     127 img/s      |    32.77 ms     |    32.87 ms     |    35.95 ms     |
|    efficientnet-b4     |       8        |     255 img/s      |     32.9 ms     |    34.56 ms     |    37.01 ms     |
|    efficientnet-b4     |       16       |     486 img/s      |    34.46 ms     |    36.56 ms     |     39.1 ms     |
|    efficientnet-b4     |       32       |     681 img/s      |    48.48 ms     |    46.98 ms     |    48.55 ms     |
|    efficientnet-b4     |       64       |     738 img/s      |    88.55 ms     |    86.55 ms     |    87.31 ms     |
|    efficientnet-b4     |      128       |     757 img/s      |    174.13 ms    |    168.73 ms    |    168.92 ms    |
|    efficientnet-b4     |      256       |     770 img/s      |    343.04 ms    |    329.95 ms    |    330.66 ms    |
| efficientnet-widese-b0 |       1        |      63 img/s      |    17.08 ms     |    16.36 ms     |     17.8 ms     |
| efficientnet-widese-b0 |       2        |     123 img/s      |    17.48 ms     |    16.74 ms     |    18.17 ms     |
| efficientnet-widese-b0 |       4        |     241 img/s      |    17.95 ms     |    17.29 ms     |    18.76 ms     |
| efficientnet-widese-b0 |       8        |     486 img/s      |    17.92 ms     |    19.42 ms     |     22.3 ms     |
| efficientnet-widese-b0 |       16       |     898 img/s      |     19.3 ms     |    20.57 ms     |    22.41 ms     |
| efficientnet-widese-b0 |       32       |     1649 img/s     |    21.06 ms     |    23.14 ms     |    24.83 ms     |
| efficientnet-widese-b0 |       64       |     3360 img/s     |    21.22 ms     |    22.89 ms     |    25.07 ms     |
| efficientnet-widese-b0 |      128       |     4934 img/s     |    30.35 ms     |    26.48 ms     |     30.3 ms     |
| efficientnet-widese-b0 |      256       |     5340 img/s     |    57.83 ms     |    47.59 ms     |     54.7 ms     |
| efficientnet-widese-b4 |       1        |      31 img/s      |    33.37 ms     |    34.12 ms     |    35.95 ms     |
| efficientnet-widese-b4 |       2        |      63 img/s      |     33.0 ms     |    33.73 ms     |    35.15 ms     |
| efficientnet-widese-b4 |       4        |     133 img/s      |    31.43 ms     |    31.72 ms     |    33.93 ms     |
| efficientnet-widese-b4 |       8        |     244 img/s      |    34.35 ms     |    36.98 ms     |    39.72 ms     |
| efficientnet-widese-b4 |       16       |     454 img/s      |     36.8 ms     |     39.8 ms     |    42.41 ms     |
| efficientnet-widese-b4 |       32       |     680 img/s      |    48.63 ms     |     48.1 ms     |    50.57 ms     |
| efficientnet-widese-b4 |       64       |     738 img/s      |    88.64 ms     |    86.56 ms     |     86.7 ms     |
| efficientnet-widese-b4 |      128       |     756 img/s      |    174.52 ms    |    168.98 ms    |    169.13 ms    |
| efficientnet-widese-b4 |      256       |     771 img/s      |    344.05 ms    |    329.69 ms    |    330.7 ms     |



#### Quantization results

##### QAT Training performance: NVIDIA DGX-1 (8x V100 32GB)

|       **Model**       | **GPUs** | **Calibration** |  **QAT model**  |  **FP32**  | **QAT ratio** |
|:---------------------:|:---------|:---------------:|:---------------:|:----------:|:-------------:|
| efficientnet-quant-b0 |    8     |   14.71 img/s   |  2644.62 img/s  | 3798 img/s |    0.696 x    |
| efficientnet-quant-b4 |    8     |    1.85 img/s   |   310.41 img/s  | 666 img/s  |    0.466 x    |


###### Quant Inference accuracy
The best checkpoints generated during training were used as a base for the QAT.

|       **Model**       | **QAT Epochs** | **QAT Top1** | **Gap between FP32 Top1 and QAT Top1** |
|:---------------------:|:--------------:|:------------:|:--------------------------------------:|
| efficientnet-quant-b0 |        10      |     77.12    |                  0.51                  |
| efficientnet-quant-b4 |         2      |     82.54    |                  0.44                  |


## Release notes
### Changelog

1. April 2020
  * Initial release

### Known issues

There are no known issues with this model.


# SSD300 v1.1 For PyTorch

This repository provides a script and recipe to train the SSD300 v1.1 model to achieve state of the art accuracy, and is tested and maintained by NVIDIA.

## Table Of Contents
- [Model overview](#model-overview)
    * [Model architecture](#model-architecture)
    * [Default configuration](#default-configuration)
    * [Feature support matrix](#feature-support-matrix)
        * [Features](#features)
    * [Mixed precision training](#mixed-precision-training)
        * [Enabling mixed precision](#enabling-mixed-precision)
        * [Enabling TF32](#enabling-tf32)
- [Setup](#setup)
    * [Requirements](#requirements)
- [Quick Start Guide](#quick-start-guide)
- [Advanced](#advanced)
    * [Scripts and sample code](#scripts-and-sample-code)
    * [Parameters](#parameters)
    * [Command-line options](#command-line-options)
    * [Getting the data](#getting-the-data)
        * [Dataset guidelines](#dataset-guidelines)
            * [Data preprocessing](#data-preprocessing)
            * [Data augmentation](#data-augmentation)
    * [Training process](#training-process)
    * [Evaluation process](#evaluation-process)
    * [Inference process](#inference-process)
- [Performance](#performance)
    * [Benchmarking](#benchmarking)
        * [Training performance benchmark](#training-performance-benchmark)
        * [Inference performance benchmark](#inference-performance-benchmark)
    * [Results](#results)
        * [Training accuracy results](#training-accuracy-results)
            * [Training accuracy: NVIDIA DGX A100 (8x A100 80GB)](#training-accuracy-nvidia-dgx-a100-8x-a100-80gb)
            * [Training accuracy: NVIDIA DGX-1 (8x V100 16GB)](#training-accuracy-nvidia-dgx-1-8x-v100-16gb)
            * [Training loss plot](#training-loss-plot)
            * [Training stability test](#training-stability-test)
        * [Training performance results](#training-performance-results)
            * [Training performance: NVIDIA DGX A100 (8x A100 80GB)](#training-performance-nvidia-dgx-a100-8x-a100-80gb)
            * [Training performance: NVIDIA DGX-1 (8x V100 16G)](#training-performance-nvidia-dgx-1-8x-v100-16gb)
        * [Inference performance results](#inference-performance-results)
            * [Inference performance: NVIDIA DGX A100 (1x A100 80GB)](#inference-performance-nvidia-dgx-a100-1x-a100-80gb)
            * [Inference performance: NVIDIA DGX-1 (1x V100 16GB)](#inference-performance-nvidia-dgx-1-1x-v100-16gb)
- [Release notes](#release-notes)
    * [Changelog](#changelog)
    * [Known issues](#known-issues)


## Model overview
The SSD300 v1.1 model is based on the
[SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) paper, which
describes SSD as “a method for detecting objects in images using a single deep neural network".
The input size is fixed to 300x300.

The main difference between this model and the one described in the paper is in the backbone.
Specifically, the VGG model is obsolete and is replaced by the ResNet-50 model.

From the
[Speed/accuracy trade-offs for modern convolutional object detectors](https://arxiv.org/abs/1611.10012)
paper, the following enhancements were made to the backbone:
*   The conv5_x, avgpool, fc and softmax layers were removed from the original classification model.
*   All strides in conv4_x are set to 1x1.

Detector heads are similar to the ones referenced in the paper, however,
they are enhanced by additional BatchNorm layers after each convolution.

Additionally, we removed weight decay on every bias parameter and
all the BatchNorm layer parameters as described in the
[Highly Scalable Deep Learning Training System with Mixed-Precision:
Training ImageNet in Four Minutes](https://arxiv.org/abs/1807.11205) paper.

Training of SSD requires computational costly augmentations.
To fully utilize GPUs during training we are using the
[NVIDIA DALI](https://github.com/NVIDIA/DALI) library
to accelerate data preparation pipelines.

This model is trained with mixed precision using Tensor Cores on Volta, Turing,
and the NVIDIA Ampere GPU architectures. Therefore, researchers can get results
2x faster than training without Tensor Cores, while experiencing the benefits of
mixed precision training. This model is tested against each NGC monthly
container release to ensure consistent accuracy and performance over time.

### Model architecture

Despite the changes described in the previous section,
the overall architecture, as described in the following diagram, has not changed.

<p align="center">
  <img width="90%" src="./img/ssd_diagram.png" />
  <br>
Figure 1. The architecture of a Single Shot MultiBox Detector model. Image has been taken from the <a href="https://arxiv.org/abs/1512.02325">Single Shot MultiBox Detector paper</a>.
</p>

The backbone is followed by 5 additional convolutional layers.
In addition to the convolutional layers, we attached 6 detection heads:
*   The first detection head is attached to the last conv4_x layer.
*   The other five detection heads are attached to the corresponding 5 additional layers.


### Default configuration
We trained the model for 65 epochs with the following setup:
*    SGD with momentum (0.9)
*    Learning rate = 2.6e-3 * number of GPUs * (batch_size / 32)
*    Learning rate decay – multiply by 0.1 before 43 and 54 epochs
*    We use linear warmup of the learning rate during the first epoch.

For more information, see the
 [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) paper.

To enable warmup provide argument the `--warmup 300`
*    Weight decay:
    *   0 for BatchNorms and biases
    *   5e-4 for other layers

**Note**: The learning rate is automatically scaled (in other words, multiplied
by the number of GPUs and multiplied by the batch size divided by 32).

### Feature support matrix

The following features are supported by this model.

| **Feature** | **SSD300 v1.1 PyTorch** |
|:---------:|:----------:|
|[AMP](https://pytorch.org/docs/stable/amp.html)                                        |  Yes |
|[APEX DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)               |  Yes |
|[NVIDIA DALI](https://docs.nvidia.com/deeplearning/sdk/dali-release-notes/index.html)  |  Yes |

#### Features

[AMP](https://pytorch.org/docs/stable/amp.html) is an abbreviation used for automatic mixed precision training.

[DDP](https://nvidia.github.io/apex/parallel.html) stands for DistributedDataParallel and is used for multi-GPU training.

[NVIDIA DALI](https://docs.nvidia.com/deeplearning/sdk/dali-release-notes/index.html) - DALI is a library accelerating data preparation pipeline.
To accelerate your input pipeline, you only need to define your data loader
with the DALI library.
For details, see example sources in this repo or see
the [DALI documentation](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html)

### Mixed precision training

Mixed precision is the combined use of different numerical precisions in
a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740)
training offers significant computational speedup by performing operations
in half-precision format, while storing minimal information in single-precision
to retain as much information as possible in critical parts of the network.
Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores)
in Volta, and following with both the Turing and Ampere architectures, significant training speedups are
experienced by switching to mixed precision -- up to 3x overall speedup
on the most arithmetically intense model architectures. Using mixed precision
training requires two steps:
1.  Porting the model to use the FP16 data type where appropriate.
2.  Adding loss scaling to preserve small gradient values.

The ability to train deep learning networks with lower precision was introduced
in the Pascal architecture and first supported in [CUDA 8](https://devblogs.nvidia.com/parallelforall/tag/fp16/)
in the NVIDIA Deep Learning SDK.

For information about:
-   How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html)
documentation.
-   Techniques used for mixed precision training, see the [Mixed-Precision
Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/)
blog.
-   PyTorch AMP, see the [PyTorch Automatic Mixed Precision package](https://pytorch.org/docs/stable/amp.html).


#### Enabling mixed precision

Mixed precision is enabled in PyTorch by using the Automatic Mixed Precision (AMP)
autocast [torch.cuda.amp.autocast](https://pytorch.org/docs/stable/amp.html#autocasting) which casts variables
to half-precision upon retrieval, while storing variables in single-precision format.
Furthermore, to preserve small gradient magnitudes in backpropagation,
a [gradient scaling](https://pytorch.org/docs/stable/amp.html#gradient-scaling)
step must be included.

For an in-depth walk through on AMP, check out sample usage
[here](https://pytorch.org/docs/stable/amp.html).


#### Enabling TF32

TensorFloat-32 (TF32) is the new math mode in [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) GPUs for handling the matrix math also called tensor operations. TF32 running on Tensor Cores in A100 GPUs can provide up to 10x speedups compared to single-precision floating-point math (FP32) on Volta GPUs.

TF32 Tensor Cores can speed up networks using FP32, typically with no loss of accuracy. It is more robust than FP16 for models which require high dynamic range for weights or activations.

For more information, refer to the [TensorFloat-32 in the A100 GPU Accelerates AI Training, HPC up to 20x](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) blog post.

TF32 is supported in the NVIDIA Ampere GPU architecture and is enabled by default.

### Glossary

backbone
: a part of a many object detection architectures, usually pre-trained for a different,
simpler task, like classification.

input pipeline
: set of operations performed for every item in input data before feeding the neural
network. Especially for object detection task, the input pipeline can be complex
and computationally significant. For that reason, solutions like NVIDIA DALI emerged.

object detection
: a subset of Computer Vision problem. The task of object detection is to localize
possibly multiple objects on the image and classify them. The difference between
Object Detection, Image Classification, and Localization are clearly explained in the
video published as a part of the [C4W3L01 course](https://www.youtube.com/watch?v=GSwYGkTfOKk).

SSD (Single Shot MultiBox Detector)
: a name for the detection model described in a [paper authored by Liu at al.](https://arxiv.org/abs/1512.02325)

ResNet (ResNet-50)
: a name for the classification model described in a [paper authored by He et al.](https://arxiv.org/abs/1512.03385)
In this repo, it is used as a backbone for SSD.

## Setup
The following section lists the requirements in order to start training the SSD300 v1.1 model.


### Requirements
This repository contains `Dockerfile` which extends the PyTorch 22.10 NGC container
and encapsulates some dependencies.  Aside from these dependencies,
ensure you have the following software:
* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch 22.10 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch)
* GPU-based architecture:
    * [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
    * [NVIDIA Turing](https://www.nvidia.com/en-us/geforce/turing/)
    * [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)

For more information about how to get started with NGC containers, see the
following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning
Documentation:
* [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
* [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/dgx/user-guide/index.html#accessing_registry)
* [Running PyTorch](https://docs.nvidia.com/deeplearning/dgx/pytorch-release-notes/running.html#running)

For those unable to use the [PyTorch 22.10 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch),
to set up the required environment or create your own container,
see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).


## Quick Start Guide

To train your model using mixed or TF32 precision with Tensor Cores or using FP32,
perform the following steps using the default parameters of the SSD v1.1 model
on the [COCO 2017](http://cocodataset.org/#download) dataset.
For the specifics concerning training and inference,
see the [Advanced](#advanced) section.


1. Clone the repository.
```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/PyTorch/Detection/SSD
```

2. Download and preprocess the dataset.

Extract the COCO 2017 dataset with `download_dataset.sh $COCO_DIR`.
Data will be downloaded to the `$COCO_DIR` directory (on the host).

3. Build the SSD300 v1.1 PyTorch NGC container.

```
docker build . -t nvidia_ssd
```

4. Start an interactive session in the NGC container to run training/inference.
```
docker run --rm -it --gpus=all --ipc=host -v $COCO_DIR:/coco nvidia_ssd
```

**Note**: the default mount point in the container is `/coco`.

5. Start training.

The `./examples` directory provides several sample scripts for various GPU settings
and act as wrappers around the `main.py` script.
The example scripts need two arguments:
- A path to the root SSD directory.
- A path to the COCO 2017 dataset.

Remaining arguments are passed to the `main.py` script.

The `--save save_dir` flag, saves the model after each epoch in `save_dir` directory.
The checkpoints are stored as `<save_dir>/epoch_*.pt`.

Use `python main.py -h` to obtain the list of available options in the `main.py` script.
For example, if you want to run 8 GPU training with Tensor Core acceleration and
save checkpoints after each epoch, run:

```
bash ./examples/SSD300_FP16_8GPU.sh . /coco --save $SSD_CHECKPINT_PATH
```

6. Start validation/evaluation.

The `main.py` training script automatically runs validation during training.
The results from the validation are printed to `stdout`.

To evaluate a checkpointed model saved in the previous point, run:

```
python ./main.py --backbone resnet50 --mode evaluation --checkpoint ./models/epoch_*.pt --data /coco
```

7. Optionally, resume training from a checkpointed model.

```
python ./main.py --backbone resnet50 --checkpoint ./models/epoch_*.pt --data /coco
```

8. Start inference/predictions.

You can check your trained model with a Jupyter notebook provided in the examples directory.
Start with running a Docker container with a Jupyter notebook server:
```
docker run --rm -it --gpus=all --ipc=host -v $SSD_CHECKPOINT_PATH:/checkpoints/SSD300v1.1.pt -v $COCO_PATH:/datasets/coco2017 -p 8888:8888 nvidia_ssd jupyter-notebook --ip 0.0.0.0 --allow-root
```

## Advanced

The following sections provide greater details of the dataset,
running training and inference, and the training results.

### Scripts and sample code

In the root directory, the most important files are:
 - `main.py`:               the script that controls the logic of training and validation of the SSD300 v1.1 model;
 - `Dockerfile`:            Instructions for docker to build a container with the basic set of dependencies to run SSD300 v1.1;
 - `requirements.txt`:      a set of extra Python requirements for running SSD300 v1.1;
 - `download_dataset.py`:   automatically downloads the COCO dataset for training.

The `ssd/` directory contains modules used to train and evaluate the SSD300 v1.1 model
 - `model.py`: the definition of SSD300 v1.1 model
 - `data.py`: definition of input pipelines used in training and evaluation
 - `train.py`: functions used to train the SSD300 v1.1 model
 - `evaluate.py`: functions used to evaluate the SSD300 v1.1 model
 - `coco_pipeline.py`: definition of input pipeline using NVIDIA DALI
 - `coco.py`: code specific for the COCO dataset
 - `logger.py`: utilities for logging
 - `utils.py`: extra utility functions

The `examples/` directory contains scripts wrapping common scenarios.

### Parameters

#### The script `main.py`
The script for training end evaluating the SSD300 v1.1 model have a variety
of parameters that control these processes.

##### Common parameters
`--data`
: use it to specify, where your dataset is. By default, the script will look for it
under the `/coco` directory.

`--checkpoint`
: allows you to specify the path to the pre-trained model.

`--save`
: when the flag is turned on, the script will save the trained model checkpoints in the specified directory

`--seed`
: Use it to specify the seed for RNGs.

`--amp`
: when the flag is turned on, the AMP features will be enabled.

##### Training related

`--epochs`
: a number of times the model will see every example from the training dataset.

`--evaluation`
: after this parameter, list the number of epochs after which evaluation should
be performed.

`--learning-rate`
: initial learning rate.

`--multistep`
: after this parameter, list the epochs after which learning rate should be decayed.

`--warmup`
: allows you to specify the number of iterations for which a linear learning-rate
warmup will be performed.

`--momentum`
: momentum argument for SGD optimizer.

`--weight-decay`
: weight decay argument for SGD optimizer.

`--batch-size`
: a number of inputs processed at once for each iteration.

`--backbone-path`
: the path to the checkpointed backbone. When it is not provided, a pre-trained model from torchvision
will be downloaded.

##### Evaluation related

`--eval-batch-size`
: a number of inputs processed at once for each iteration.

##### Utility parameters
`--help`
: displays a short description of all parameters accepted by the script.

### Command-line options

All these parameters can be controlled by passing command-line arguments
to the `main.py` script. To get a complete list of all command-line arguments
with descriptions and default values you can run:

```
python main.py --help
```

### Getting the data

The SSD model was trained on the COCO 2017 dataset. The [val2017](http://cocodataset.org/#download) validation set
was used as a validation dataset. PyTorch can work directly on JPEGs,
therefore, preprocessing/augmentation is not needed.

This repository contains the `download_dataset.sh` download script which will automatically
download and preprocess the training, validation and test datasets. By default,
data will be downloaded to the `/coco` directory.

#### Dataset guidelines

Our model expects input data aligned in a way a COCO dataset is aligned by the `download_dataset.sh` script.
`train2017` and `val2017` directories should contain images in JPEG format.
Annotation format is described in [the COCO documentation](http://cocodataset.org/#format-data).

The preprocessing of the data is defined in the `ssd/coco_pipeline.py` module.

##### Data preprocessing

Before we feed data to the model, both during training and inference, we perform:
* JPEG decoding
* normalization with a mean =` [0.485, 0.456, 0.406]` and std dev = `[0.229, 0.224, 0.225]`
* encoding bounding boxes
* resizing to 300x300

Additionally, during training, data is:
* randomly shuffled
* samples without annotations are skipped

##### Data augmentation

During training we perform the following augmentation techniques:
* Random crop using the algorithm described in the [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) paper
* Random horizontal flip
* Color jitter

### Training process

Training the SSD model is implemented in the `main.py` script.

By default, training is running for 65 epochs. Because evaluation is relatively time consuming,
it is not running every epoch. With default settings, evaluation is executed after epochs:
21, 31, 37, 42, 48, 53, 59, 64. The model is evaluated using pycocotools distributed with
the COCO dataset.
 Which epochs should be evaluated can be reconfigured with the `--evaluation` argument.

To run training with Tensor Cores, use the `--amp` flag when running the `main.py` script.
The flag `--save ./models` flag enables storing checkpoints after each epoch under `./models/epoch_*.pt`.

### Evaluation process

Pycocotools’ open-sourced scripts provides a consistent way
to evaluate models on the COCO dataset. We are using these scripts
during validation to measure a models performance in AP metric.
Metrics below are evaluated using pycocotools’ methodology, in the following format:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.27205
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.45869
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.27884
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.08275
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.29840
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.42722
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.25092
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.36528
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.38262
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.13577
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.42287
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.57277
```
The metric reported in our results is present in the first row.

### Inference process

Our scripts for SSD300 v1.1 presents two ways to run inference.
To get meaningful results, you need a pre-trained model checkpoint.

One way is to run an interactive session on Jupyter notebook, as described in a 8th step of the [Quick Start Guide](#quick-start-guide).

The container prints Jupyter notebook logs like this:
```
[I 16:17:58.935 NotebookApp] Writing notebook server cookie secret to /root/.local/share/jupyter/runtime/notebook_cookie_secret
[I 16:17:59.769 NotebookApp] JupyterLab extension loaded from /opt/conda/lib/python3.6/site-packages/jupyterlab
[I 16:17:59.769 NotebookApp] JupyterLab application directory is /opt/conda/share/jupyter/lab
[I 16:17:59.770 NotebookApp] Serving notebooks from local directory: /workspace
[I 16:17:59.770 NotebookApp] The Jupyter Notebook is running at:
[I 16:17:59.770 NotebookApp] http://(65935d756c71 or 127.0.0.1):8888/?token=04c78049c67f45a4d759c8f6ddd0b2c28ac4eab60d81be4e
[I 16:17:59.770 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[W 16:17:59.774 NotebookApp] No web browser found: could not locate runnable browser.
[C 16:17:59.774 NotebookApp]

    To access the notebook, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/nbserver-1-open.html
    Or copy and paste one of these URLs:
        http://(65935d756c71 or 127.0.0.1):8888/?token=04c78049c67f45a4d759c8f6ddd0b2c28ac4eab60d81be4e
```

Use the token printed in the last line to start your notebook session.
The notebook is in `examples/inference.ipynb`, for example:

http://127.0.0.1:8888/notebooks/examples/inference.ipynb?token=04c78049c67f45a4d759c8f6ddd0b2c28ac4eab60d81be4e

Another way is to run a script `examples/SSD300_inference.py`. It contains the logic from the notebook, wrapped into a Python script. The script contains sample usage.

To use the inference example script in your own code, you can call the `main` function, providing input image URIs as an argument. The result will be a list of detections for each input image.





## Performance

The performance measurements in this document were conducted at the time of publication and may not reflect the performance achieved from NVIDIA’s latest software release. For the most up-to-date performance measurements, go to [NVIDIA Data Center Deep Learning Product Performance](https://developer.nvidia.com/deep-learning-performance-training-inference).

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in training and inference modes.

#### Training performance benchmark

The training benchmark was run in various scenarios on A100 80GB and V100 16G GPUs. The benchmark does not require a checkpoint from a fully trained model.

To benchmark training, run:
```
torchrun --nproc_per_node={NGPU} \
       main.py --batch-size {bs} \
               --mode benchmark-training \
               --benchmark-warmup 100 \
               --benchmark-iterations 200 \
               {AMP} \
               --data {data}
```

Where the `{NGPU}` selects number of GPUs used in benchmark, the `{bs}` is the desired
batch size, the `{AMP}` is set to `--amp` if you want to benchmark training with
Tensor Cores, and the `{data}` is the location of the COCO 2017 dataset.

`--benchmark-warmup` is specified to omit the first iteration of the first epoch.
`--benchmark-iterations` is a number of iterations used to measure performance.

#### Inference performance benchmark

Inference benchmark was run on 1x A100 80GB GPU and 1x V100 16G GPU. To benchmark inference, run:
```
python main.py --eval-batch-size {bs} \
               --mode benchmark-inference \
               --benchmark-warmup 100 \
               --benchmark-iterations 200 \
               {AMP} \
               --data {data}
```

Where the `{bs}` is the desired batch size, the `{AMP}` is set to `--amp` if you want to benchmark inference with Tensor Cores, and the `{data}` is the location of the COCO 2017 dataset.

`--benchmark-warmup` is specified to omit the first iterations of the first epoch. `--benchmark-iterations` is a number of iterations used to measure performance.

### Results

The following sections provide details on how we achieved our performance and accuracy in training and inference.

#### Training accuracy results

##### Training accuracy: NVIDIA DGX A100 (8x A100 80GB)

Our results were obtained by running the `./examples/SSD300_A100_{FP16,TF32}_{1,4,8}GPU.sh`
script in the `pytorch-22.10-py3` NGC container on NVIDIA DGX A100 (8x A100 80GB) GPUs.

|GPUs       |Batch size / GPU|Accuracy - TF32|Accuracy  - mixed precision|Time to train - TF32|Time to train  - mixed precision|Time to train speedup  (TF32 to mixed precision)|
|-----------|----------------|---------------|---------------------------|--------------------|--------------------------------|------------------------------------------------|
|1          |64              |0.271          |0.272                      |03:19:59            |03:18:35                        |100%                                            |
|4          |64              |0.270          |0.270                      |00:51:22            |00:51:31                        | 99%                                            |
|8          |64              |0.270          |0.269                      |00:26:10            |00:26:10                        | 99%                                            |
|1          |128             |0.274          |0.271                      |03:03:56            |03:03:50                        |100%                                            |
|4          |128             |0.272          |0.270                      |00:46:51            |00:47:01                        | 99%                                            |
|8          |128             |0.267          |0.267                      |00:23:44            |00:23:46                        | 99%                                            |
|1          |256             |0.272          |0.272                      |02:56:37            |02:56:44                        | 99%                                            |
|4          |256             |0.271          |0.267                      |00:45:05            |00:45:07                        | 99%                                            |
|8          |256             |0.260          |0.258                      |00:22:49            |00:22:56                        |100%                                            |

##### Training accuracy: NVIDIA DGX-1 (8x V100 16GB)

Our results were obtained by running the `./examples/SSD300_FP{16,32}_{1,4,8}GPU.sh`
script in the `pytorch-22.10-py3` NGC container on NVIDIA DGX-1 with 8x
V100 16GB GPUs.

|GPUs       |Batch size / GPU|Accuracy - FP32|Accuracy  - mixed precision|Time to train - FP32|Time to train  - mixed precision|Time to train speedup  (FP32 to mixed precision)|
|-----------|----------------|---------------|---------------------------|--------------------|--------------------------------|------------------------------------------------|
|1          |32              |0.269          |0.271                      |20:04:48            |07:25:27                        |270%                                            |
|4          |32              |0.270          |0.269                      |05:08:56            |01:58:41                        |260%                                            |
|8          |32              |0.271          |0.269                      |02:35:00            |01:00:27                        |256%                                            |
|1          |64              |<N/A>          |0.272                      |<N/A>               |06:47:58                        |<N/A>                                           |
|4          |64              |<N/A>          |0.270                      |<N/A>               |01:46:34                        |<N/A>                                           |
|8          |64              |<N/A>          |0.269                      |<N/A>               |00:53:52                        |<N/A>                                           |

Due to smaller size, mixed precision models can be trained with bigger batches. In such cases mixed precision speedup is calculated versus FP32 training with maximum batch size for that precision

##### Training loss plot

Here are example graphs of FP32, TF32 and AMP training on 8 GPU configuration:

![TrainingLoss](./img/training_loss.png)

##### Training stability test

The SSD300 v1.1 model was trained for 65 epochs, starting
from 15 different initial random seeds. The training was performed in the `pytorch-22.10-py3` NGC container on
NVIDIA DGX A100 8x A100 80GB GPUs with batch size per GPU = 128.
After training, the models were evaluated on the test dataset. The following
table summarizes the final mAP on the test set.

|**Precision**|**Average mAP**|**Standard deviation**|**Minimum**|**Maximum**|**Median**|
|------------:|--------------:|---------------------:|----------:|----------:|---------:|
| AMP         | 0.2679503039  | 0.001360494012       | 0.26201   | 0.27013   | 0.26529  |
| TF32        | 0.2670691823  | 0.001639394102       | 0.26181   | 0.27274   | 0.26492  |

#### Training performance results

##### Training performance: NVIDIA DGX A100 (8x A100 80GB)

Our results were obtained by running the `main.py` script with the `--mode
benchmark-training` flag in the `pytorch-22.10-py3` NGC container on NVIDIA
DGX A100 (8x A100 80GB) GPUs. Performance numbers (in items/images per second)
were averaged over an entire training epoch.

|GPUs       |Batch size / GPU|Throughput - TF32|Throughput  - mixed precision|Throughput speedup (TF32 - mixed precision)|Weak scaling - TF32             |Weak scaling  - mixed precision                 |
|-----------|----------------|-----------------|-----------------------------|-------------------------------------------|--------------------------------|------------------------------------------------|
|1          |64              | 364.27          | 662.91                      |181%                                       |100%                            |100%                                            |
|4          |64              |1432.73          |2581.24                      |180%                                       |393%                            |389%                                            |
|8          |64              |2838.76          |5252.84                      |185%                                       |779%                            |792%                                            |
|1          |128             | 377.18          | 724.41                      |192%                                       |100%                            |100%                                            |
|4          |128             |1493.13          |2885.55                      |193%                                       |395%                            |398%                                            |
|8          |128             |2967.23          |5733.98                      |193%                                       |786%                            |791%                                            |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

##### Training performance: NVIDIA DGX-1 (8x V100 16GB)

Our results were obtained by running the `main.py` script with the `--mode
benchmark-training` flag in the `pytorch-22.10-py3` NGC container on NVIDIA
DGX-1 with 8x V100 16GB GPUs. Performance numbers (in items/images per second)
were averaged over an entire training epoch.

|GPUs       |Batch size / GPU|Throughput - FP32|Throughput  - mixed precision|Throughput speedup (FP32 - mixed precision)|Weak scaling - FP32             |Weak scaling  - mixed precision                 |
|-----------|----------------|-----------------|-----------------------------|-------------------------------------------|--------------------------------|------------------------------------------------|
|1          |32              |107.22           | 296.80                      |276%                                       |100%                            |100%                                            |
|4          |32              |419.54           |1115.59                      |265%                                       |391%                            |375%                                            |
|8          |32              |840.35           |2153.96                      |256%                                       |783%                            |725%                                            |
|1          |64              |<N/A>            | 322.81                      |<N/A>                                      |<N/A>                           |100%                                            |
|4          |64              |<N/A>            |1238.27                      |<N/A>                                      |<N/A>                           |383%                                            |
|8          |64              |<N/A>            |2520.50                      |<N/A>                                      |<N/A>                           |780%                                            |

Due to smaller size, mixed precision models can be trained with bigger batches. In such cases mixed precision speedup is calculated versus FP32 training with maximum batch size for that precision

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

#### Inference performance results

##### Inference performance: NVIDIA DGX A100 (1x A100 80GB)

Our results were obtained by running the `main.py` script with `--mode
benchmark-inference` flag in the pytorch-22.10-py3 NGC container on NVIDIA
DGX A100 (1x A100 80GB) GPU.

|Batch size |Throughput - TF32|Throughput  - mixed precision|Throughput speedup (TF32 - mixed precision)|Weak scaling - TF32 |Weak scaling  - mixed precision |
|-----------|-----------------|-----------------------------|-------------------------------------------|--------------------|--------------------------------|
|1          |158.83           | 142.67                      | 89%                                       |100%                |100%                           |
|2          |308.31           | 261.21                      | 84%                                       |194%                |183%                           |
|4          |481.69           | 454.95                      | 94%                                       |303%                |318%                           |
|8          |597.72           | 742.05                      |124%                                       |376%                |520%                           |
|16         |590.44           | 887.01                      |150%                                       |371%                |621%                           |
|32         |708.97           | 970.27                      |136%                                       |446%                |680%                           |
|64         |798.16           |1057.51                      |132%                                       |502%                |741%                           |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

##### Inference performance: NVIDIA DGX-1 (1x V100 16GB)

Our results were obtained by running the `main.py` script with `--mode
benchmark-inference` flag in the pytorch-22.10-py3 NGC container on NVIDIA
DGX-1 with (1x V100 16GB) GPU.

|Batch size |Throughput - FP32|Throughput  - mixed precision|Throughput speedup (FP32 - mixed precision)|Weak scaling - FP32 |Weak scaling  - mixed precision |
|-----------|-----------------|-----------------------------|-------------------------------------------|--------------------|--------------------------------|
|1          | 93.21           | 84.59                       | 90%                                       |100%                |100%                            |
|2          |148.61           |165.30                       |111%                                       |159%                |195%                            |
|4          |206.82           |304.77                       |147%                                       |221%                |360%                            |
|8          |242.55           |447.25                       |184%                                       |260%                |528%                            |
|16         |292.44           |541.05                       |185%                                       |313%                |639%                            |
|32         |311.61           |605.30                       |194%                                       |334%                |715%                            |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

## Release notes

### Changelog

October 2022
 * upgrade the PyTorch container to 22.10
 * switched to using torchvision IMAGENET1K_V2 backbone weights
 * added a flag to control for torchvision weight enums
 * added a flag to control TF32 computations
 * fixed various depreciation warnings
 * set `TORCH_CUDNN_V8_API_ENABLED` environment variable which replaces `CUDNN_V8_API_ENABLED` from older containers
 * updated [nv-cocoapi](https://github.com/NVIDIA/cocoapi/) from 0.6.0 to 0.7.3
 * updated python dependencies

June 2022
 * upgrade the PyTorch container to 22.05
 * fixed DALI depreciation warnings

January 2022
 * upgrade the PyTorch container to 22.01
 * made AMP the default data precision
 * added --data-layout option (channels_first is the recommended layout with --no-amp)
 * updated README with new performance numbers

November 2021
 * upgrade the PyTorch container to 21.11
 * switched data layout from NCHW (channels first) to NHWC (channels last)
 * replaced `torch.distributed.launch` with `torchrun`
 * updated README with new performance numbers

May 2021
 * upgrade the PyTorch container to 21.05
 * replaced APEX AMP with native PyTorch AMP
 * updated [nv-cocoapi](https://github.com/NVIDIA/cocoapi/) from 0.4.0 to 0.6.0
 * code updated to use DALI 1.2.0

April 2021
 * upgrade the PyTorch container to 21.04
 * changed python package naming

March 2021
 * upgrade the PyTorch container to 21.03
 * code updated to use DALI 0.30.0
 * use DALI [BoxEncoder](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops.html#nvidia.dali.ops.BoxEncoder) instead of a CUDA extension
 * replaced [cocoapi](https://github.com/cocodataset/cocoapi) with [nv-cocoapi](https://github.com/NVIDIA/cocoapi/)

June 2020
 * upgrade the PyTorch container to 20.06
 * update performance tables to include A100 results
 * update examples with A100 configs

August 2019
 * upgrade the PyTorch container to 19.08
 * update Results section in the README
 * code updated to use DALI 0.12.0
 * checkpoint loading fix
 * fixed links in the README

July 2019
 * script and notebook for inference
 * use AMP instead of hand-crafted FP16 support
 * README update
 * introduced a parameter with a path to the custom backbone checkpoint
 * minor enchantments of `example/*` scripts
 * alignment to changes in PyTorch 19.06

March 2019
 * Initial release

## Known issues

There are no known issues with this model.


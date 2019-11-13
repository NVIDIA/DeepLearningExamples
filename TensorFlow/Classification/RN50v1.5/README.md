# ResNet-50 v1.5 for TensorFlow

This repository provides a script and recipe to train the ResNet-50 v1.5 model to achieve state of the art accuracy, and is tested and maintained by NVIDIA.

## Table Of Contents
* [Model overview](#model-overview)
    * [Default configuration](#default-configuration)
    * [Data augmentation](#data-augmentation)
    * [Other training recipes](#other-training-recipes)
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
    * [Command line options](#command-line-options)
    * [Getting the data](#getting-the-data)
        * [Data augmentation](#data-augmentation)
    * [Training process](#training-process)
    * [Inference process](#inference-process)
* [Performance](#performance)
    * [Benchmarking](#benchmarking)
        * [Training performance benchmark](#training-performance-benchmark)
        * [Inference performance benchmark](#inference-performance-benchmark)
    * [Results](#results)
        * [Training accuracy results](#training-accuracy-results)
            * [NVIDIA DGX-1 (8x V100 16G)](#nvidia-dgx-1-8x-v100-16g)
            * [NVIDIA DGX-1 (8x V100 32G)](#nvidia-dgx-1-8x-v100-32g)
        * [Training performance results](#training-performance-results)
            * [NVIDIA DGX-1 (8x V100 16G)](#nvidia-dgx-1-8x-v100-16g)
            * [NVIDIA DGX-2 (16x V100 32G)](#nvidia-dgx-2-16x-v100-32g)
        * [Inference performance results](#inference-performance-results)
            * [NVIDIA DGX-1 (8x V100 16G)](#nvidia-dgx-1-8x-v100-16g)
            * [NVIDIA DGX-2 (16x V100 32G)](#nvidia-dgx-2-16x-v100-32g)
            * [NVIDIA T4 (1x T4)](#nvidia-t4-1x-t4-16g)
* [Release notes](#release-notes)
    * [Changelog](#changelog)
    * [Known issues](#known-issues)

## Model overview
The ResNet50 v1.5 model is a modified version of the [original ResNet50 v1 model](https://arxiv.org/abs/1512.03385).

The difference between v1 and v1.5 is in the bottleneck blocks which requires
downsampling, for example, v1 has stride = 2 in the first 1x1 convolution, whereas v1.5 has stride = 2 in the 3x3 convolution.

This difference makes ResNet50 v1.5 slightly more accurate (~0.5% top1) than v1,
but comes with a small performance drawback (~5% imgs/sec).

This model is trained with mixed precision using Tensor Cores on NVIDIA Volta and Turing GPUs. Therefore, researchers can get results 2x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.

The following performance optimizations were implemented in this model:
* JIT graph compilation with [XLA](https://www.tensorflow.org/xla)
* NVIDIA Data Loading ([DALI](https://github.com/NVIDIA/DALI)) support (experimental). 

### Default configuration

This model trains for 90 epochs, with default ResNet50 v1.5 setup:

* SGD with momentum (0.875)

* Learning rate = 0.256 for 256 batch size, for other batch sizes we lineary
scale the learning rate.

* Learning rate schedule - we use cosine LR schedule

* For bigger batch sizes (512 and up) we use linear warmup of the learning rate
during first 5 epochs according to [Training ImageNet in 1 hour](https://arxiv.org/abs/1706.02677).

* Weight decay: 3.0517578125e-05 (1/32768).

* We do not apply Weight Decay on Batch Norm trainable parameters (gamma/bias).

* Label Smoothing: 0.1

* We train for:

    * 50 Epochs -> configuration that reaches 75.9% top1 accuracy

    * 90 Epochs -> 90 epochs is a standard for ResNet50
    
    * 250 Epochs -> best possible accuracy. For 250 epoch training we also use [MixUp regularization](https://arxiv.org/pdf/1710.09412.pdf).

### Data Augmentation

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


### Other training recipes

This script does not target any specific benchmark.
There are changes that others have made which can speed up convergence and/or increase accuracy.

One of the more popular training recipes is provided by [fast.ai](https://github.com/fastai/imagenet-fast).

The fast.ai recipe introduces many changes to the training procedure, one of which is progressive resizing of the training images.

The first part of training uses 128px images, the middle part uses 224px images, and the last part uses 288px images.
The final validation is performed on 288px images.

The training script in this repository performs validation on 224px images, just like the original paper described.

These two approaches can't be directly compared, since the fast.ai recipe requires validation on 288px images,
and this recipe keeps the original assumption that validation is done on 224px images.

Using 288px images means that a lot more FLOPs are needed during inference to reach the same accuracy.

### Feature support matrix

The following features are supported by this model.

| Feature               | ResNet-50 v1.5 Tensorflow             |
|-----------------------|--------------------------
|Multi-GPU training with [Horovod](https://github.com/horovod/horovod)  |  Yes |
|[NVIDIA DALI](https://docs.nvidia.com/deeplearning/sdk/dali-release-notes/index.html)                |  Yes |


#### Features

Multi-GPU training with Horovod - Our model uses Horovod to implement efficient multi-GPU training with NCCL.
For details, see example sources in this repository or see the [TensorFlow tutorial](https://github.com/horovod/horovod/#usage).

NVIDIA DALI - DALI is a library accelerating data preparation pipeline. To accelerate your input pipeline, you only need to define your data loader
with the DALI library. For details, see example sources in this repository or see the [DALI documentation](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html)

### Mixed precision training
[Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format, while storing minimal information in single-precision to retain as much information as possible in critical parts of the network.  Since the introduction of [tensor cores](https://developer.nvidia.com/tensor-cores) in the Volta and Turing architectures, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures.  Using [mixed precision training](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html) previously required two steps:

1. Porting the model to use the FP16 data type where appropriate.
2. Manually adding loss scaling to preserve small gradient values. 

This can now be achieved using Automatic Mixed Precision (AMP) for TensorFlow to enable the full [mixed precision methodology](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#tensorflow) in your existing TensorFlow model code.  AMP enables mixed precision training on Volta and Turing GPUs automatically. The TensorFlow framework code makes all necessary model changes internally.

In TF-AMP, the computational graph is optimized to use as few casts as necessary and maximize the use of FP16, and the loss scaling is automatically applied inside of supported optimizers. AMP can be configured to work with the existing tf.contrib loss scaling manager by disabling the AMP scaling with a single environment variable to perform only the automatic mixed-precision optimization. It accomplishes this by automatically rewriting all computation graphs with the necessary operations to enable mixed precision training and automatic loss scaling.

For information about:
 * How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html) documentation.
 * How to access and enable AMP for TensorFlow, see [Using TF-AMP](https://docs.nvidia.com/deeplearning/dgx/tensorflow-user-guide/index.html#tfamp) from the TensorFlow User Guide.
 * Techniques used for mixed precision training, see the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
 
#### Enabling mixed precision

Mixed precision is enabled in TensorFlow by using the Automatic Mixed Precision (TF-AMP) extension which casts variables to half-precision upon retrieval, while storing variables in single-precision format.Furthermore, to preserve small gradient magnitudes in backpropagation, a [loss scaling](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#lossscaling) step must be included when applying gradients. In TensorFlow, loss scaling can be applied statically by using simple multiplication of loss by a constant value or automatically, by TF-AMP. Automatic mixed precision makes all the adjustments internally in TensorFlow, providing two benefits over manual operations. First, programmers need not modify network model code, reducing development and maintenance effort. Second, using AMP maintains forward and backward compatibility with all the APIs for defining and running TensorFlow models.

To enable mixed precision, you can simply the values of enviromental varialbes inside your training script:
- Enable TF-AMP graph rewrite:
  ```
  os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
  ```
  
- Enable Automated Mixed Precision:
  ```
  os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
  ```

## Setup

The following section list the requirements that you need to meet in order to use the ResNet50 v1.5 model.

### Requirements
This repository contains Dockerfile which extends the Tensorflow NGC container and encapsulates all dependencies.  Aside from these dependencies, ensure you have the following software:

* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [TensorFlow 19.08-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)
* [NVIDIA Volta based GPU](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)

For more information about how to get started with NGC containers, see the
following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
* [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html),
* [Accessing And Pulling From The NGC container registry](https://docs.nvidia.com/deeplearning/dgx/user-guide/index.html#accessing_registry)
* [Running Tensorflow](https://docs.nvidia.com/deeplearning/dgx/tensorflow-release-notes/running.html#running).

For those unable to use the [TensorFlow 19.08-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## Quick Start Guide
To train your model using mixed precision with tensor cores, perform the following steps using the default parameters of the ResNet-50 v1.5 model on the [ImageNet](http://www.image-net.org/) dataset. For the specifics concerning training and inference, see the [Advanced](#advanced) section.


1. Clone the repository.
```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/TensorFlow/Classification/RN50v1.5
```

2. Download and preprocess the dataset.
The ResNet50 v1.5 script operates on ImageNet 1k, a widely popular image classification dataset from ILSVRC challenge.

To download and preprocess the dataset, use the [Generate ImageNet for TensorFlow](https://github.com/tensorflow/models/blob/master/research/inception/inception/data/download_and_preprocess_imagenet.sh) script. The dataset will be downloaded to a directory specified as the first parameter of the script.


3. Build the ResNet-50 v1.5 TensorFlow NGC container.
```bash
bash scripts/docker/build.sh
```

4. Start an interactive session in the NGC container to run training/inference.
After you build the container image, you can start an interactive CLI session with
```bash
bash scripts/docker/interactive.sh
```
The interactive.sh script requires that the location on the dataset is specified.  For example, /data.

5. Start training.
To run training for a default configuration (as described in [Default configuration](#default-configuration), for example 1/4/8 GPUs, FP16/FP32), run one of the scripts in the ./scripts directory called `./scripts/RN50_{FP16, FP32}_{1, 4, 8}GPU.sh`. Each of the scripts require three parameters: 
* path to the root directory of the model as the first argument
* path to the dataset as a second argument 
* path to the results destination as a third argument

For example:
```bash
./scripts/RN50_FP16_8GPU.sh <path to model> <path to dataset> <path to results>
```

6. Start validation/evaluation.

Model evaluation on a checkpoint can be launched by running  one of the `./scripts/RN50_{FP16, FP32}_EVAL.sh` scripts in the `./scripts` directory. Each of the scripts requires three parameters: 
* path to the root directory of the model as the first argument
* path to the dataset as a second argument
* path to the results destination as a third argument

For example:
```bash
./scripts/RN50_FP16_EVAL.sh <path to model> <path to dataset> <path to results>
```

To run a non-default configuration, use:

`python ./main.py --mode=evaluate --use_tf_amp --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to checkpoint>`


## Advanced

The following sections provide greater details of the dataset,running training and inference, and the training results.

### Scripts and sample code

In the root directory, the most important files are:
 - `main.py`:               the script that controls the logic of training and validation of the ResNet-50 v1.5 model;
 - `Dockerfile`:            Instructions for docker to build a container with the basic set of dependencies to run ResNet-50 v1.5;
 - `requirements.txt`:      a set of extra Python requirements for running ResNet-50 v1.5;

The `model/` directory contains modules used to define ResNet-50 v1.5 model
 - `resnet_v1_5.py`: the definition of ResNet-50 v1.5 model
 - `blocks/conv2d_block.py`: the definition of ResNet-50 v1.5 2D convolution block
 - `blocks/resnet_bottleneck_block.py`: the definition of ResNet-50 v1.5 bottleneck block
 - `layers/*.py`: definitions of specific layers used in ResNet-50 v1.5 model
 
The `utils/` directory contains utility modules
 - `cmdline_helper.py`: helper module for command line processing
 - `data_utils.py`: module defining input data pipelines
 - `dali_utils.py`: helper module for DALI 
 - `hvd_utils.py`: helper module for Horovod
 - `image_processing.py`: image processing and data augmentation functions
 - `learning_rate.py`: definition of used learning rate schedule
 - `optimizers.py`: definition of used custom optimizers
 - `hooks/*.py`: defintions of specific hooks allowing loggin of training and inference process
 
The `runtime/` directory contains modules that define the mechanics of trainig process
 - `runner.py`: module encapsulating the training, inference and evaluation 
 
The `scripts/` directory contains scripts wrapping common scenarios.


### Parameters

#### The script `main.py`
The script for training end evaluating the ResNet-50 v1.5 model have a variety of parameters that control these processes.

##### Common parameters
`--mode`
: allow specification of mode in which the script will run: train, train_and_evaluate, evaluate, predict, training_benchmark or inference_benchmark

`--data_dir` `--data_idx_dir`
: allow specification of dataset location 

`--seed`
: allow specification of seed for RNGs.

`--batch_size`
: allow specification of the minibatch size.

`--num_iter` and `--iter_unit`
: allow specification of training/evaluation length 

`--use_tf_amp`
: flag enabling TF-AMP mixed precision computation.

`--use_xla`
: flag enabling XLA graph optimization.

`--use_dali`
: flag enabling DALI input pipeline. This parameter requires `--data_idx_dir` to be set.


##### Training related
`--use_auto_loss_scaling`
: flag enabling automatic loss scaling

`--lr_init`
: initial value of learning rate.

`--warmup_steps`
: allows you to specify the number of iterations considered as warmup and not taken into account for performance measurements.

`--momentum`
: momentum argument for SGD optimizer.

`--weight-decay`
: weight decay argument for SGD optimizer.

`--batch-size`
: a number of inputs processed at once for each iteration.

`--loss_scale`
: value of static loss scale. This parameter will have no effect if `--use_auto_loss_scaling` is set.

`--mixup`        
: value of alpha parameter for mixup (if 0 then mixup is not applied) (default: 0)

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

The ResNet-50 v1.5 model was trained on the ImageNet 1k, a widely popular image classification dataset from ILSVRC challenge.

To download and preprocess the dataset, use the [Generate ImageNet for TensorFlow](https://github.com/tensorflow/models/blob/master/research/inception/inception/data/download_and_preprocess_imagenet.sh) script. The dataset will be downloaded to a directory specified as the first parameter of the script.

#### Data Augmentation

This model uses the following data augmentation:

* During training, we perform the following augmentation techniques:
  * Normalization
  * Random resized crop to 224x224
    * Scale from 8% to 100%
    * Aspect ratio from 3/4 to 4/3
  * Random horizontal flip

* During inference, we perform the following augmentation techniques:
  * Normalization
  * Scale to 256x256
  * Center crop to 224x224

### Training process
To run a configuration that is not based on the default parameters, use:

* For 1 GPU
    * FP32
        `python ./main.py --batch_size=128 --data_dir=<path to imagenet> --results_dir=<path to results>`
    * FP16
        `python ./main.py --batch_size=256 --use_tf_amp --data_dir=<path to imagenet> --results_dir=<path to results>`

* For multiple GPUs
    * FP32
        `mpiexec --allow-run-as-root --bind-to socket -np <num_gpus> python ./main.py --batch_size=128 --data_dir=<path to imagenet> --results_dir=<path to results>`
    * FP16
        `mpiexec --allow-run-as-root --bind-to socket -np <num_gpus> python ./main.py --batch_size=256 --use_tf_amp --data_dir=<path to imagenet> --results_dir=<path to results>`

### Inference process
To run inference on single examples on checkpoint with model script, use: 

`python main.py --mode predict --model_dir <path to model> --to_predict <path to image> --results_dir <path to results>`


To run inference on a SavedModel with dedicated script, use:

`python scripts/inference/predict.py -m <path to model>  -f <path to image>`


## Performance

### Benchmarking

The following sections shows how to run benchmarks measuring the model performance in training and inference modes.

#### Training performance benchmark

To benchmark the training performance on a specific batch size, run:

* For 1 GPU
    * FP32
        `python ./main.py --mode=training_benchmark --warmup_steps 200 --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
    * FP16
        `python ./main.py --mode=training_benchmark  --use_tf_amp --warmup_steps 200 --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
* For multiple GPUs
    * FP32
        `mpiexec --allow-run-as-root --bind-to socket -np <num_gpus> python ./main.py --mode=training_benchmark --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
    * FP16
        `mpiexec --allow-run-as-root --bind-to socket -np <num_gpus> python ./main.py --mode=training_benchmark --use_tf_amp --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
        
Each of these scripts runs 200 warm-up iterations and measures the first epoch.

To control warmup and benchmark length, use `--warmup_steps`, `--num_iter` and `--iter_unit` flags.


#### Inference performance benchmark

To benchmark the inference performance on a specific batch size, run:

* FP32
`python ./main.py --mode=inference_benchmark --warmup_steps 20 --train_iter 100 --iter_unit batch --batch_size <batch size> --data_dir=<path to imagenet> --log_dir=<path to results directory>`

* FP16
`python ./main.py --mode=inference_benchmark --use_tf_amp --warmup_steps 20 --train_iter 100 --iter_unit batch --batch_size <batch size> --data_dir=<path to imagenet> --log_dir=<path to results directory>`

Each of these scripts, by default runs 20 warm-up iterations and measures the next 80 iterations.

To control warm-up and benchmark length, use `--warmup_steps`, `--num_iter` and `--iter_unit` flags.

### Results

The following sections provide details on how we achieved our results in training accuracy, performance and inference performance.

#### Training accuracy results

##### NVIDIA DGX-1 (8x V100 16G)

Our results were obtained by running 50 epochs training using the `./scripts/RN50_{FP16, FP32}_{1, 4, 8}GPU.sh` script in
the tensorflow-19.07-py3 Docker container on NVIDIA DGX-1 with 8 V100 16G GPUs.


| **number of GPUs** | **mixed precision top1** | **mixed precision training time** | **FP32 top1** | **FP32 training time** |
|:------------------:|:------------------------:|:---------------------------------:|:-------------:|:----------------------:|
| **1**                  | 76.21                    | 13.3h                             | 76.14         | 50.42h                  |
| **4**                  | 76.13                    | 3.58h                             | 76.11         | 12.65h                  |
| **8**                  | 76.08                    | 1.87h                              | 76.12        | 6.38h                  |

##### NVIDIA DGX-1 (8x V100 32G)

Our results were obtained by running 50 epochs training using the `./scripts/RN50_{FP16, FP32}_{1, 4, 8}GPU.sh` script in
the tensorflow-19.07-py3 Docker container on NVIDIA DGX-1 with 8 V100 32G GPUs.


| **number of GPUs** | **mixed precision top1** | **mixed precision training time** | **FP32 top1** | **FP32 training time** |
|:------------------:|:------------------------:|:---------------------------------:|:-------------:|:----------------------:|
| **1**                  | 76.14                    | 13.79h                             | 76.06         | 51.38h                  |
| **4**                  | 76.07                    | 3.72h                             | 76.17         | 12.7h                  |
| **8**                  | 76.04                   | 1.91h                              | 76.02         | 6.43h                  |


#### Training performance results

##### NVIDIA DGX-1 (8x V100 16G)
The results were obtained by running the `./scripts/benchmarking/DGX1V_trainbench_fp16.sh` and `./scripts/benchmarking/DGX1V_trainbench_fp32.sh` scripts in the tensorflow-19.07-py3 Docker container on NVIDIA DGX-1 with 8 V100 16G GPU.


| **number of GPUs** | **mixed precision img/s** | **FP32 img/s** | **mixed precision speedup** | **mixed precision weak scaling** | **FP32 weak scaling** |
|:------------------:|:-------------------------:|:--------------:|:---------------------------:|:--------------------------------:|:---------------------:|
| **1**                  | 825.2                     | 364.9          | 2.20                        | 1.00                             | 1.00                  |
| **4**                  | 3197.4                    | 1419.4         | 2.25                        | 3.96                             | 3.89                  |
| **8**                  | 6209.9                    | 2778.5         | 2.24                        | 7.74                             | 7.61                  |

##### XLA Enabled

| **number of GPUs** | **mixed precision img/s** | **mixed precision + XLA img/s** | **XLA speedup** | 
|:------------------:|:-------------------------:|:-------------------------------:|:---------------:|
| **1**              | 825.2                     | 1335.9                          | 1.61            |
| **4**              | 3197.4                    | 4964.9                          | 1.55            |
| **8**              | 6209.9                    | 9518.8                         | 1.53            |


##### NVIDIA DGX-2 (16x V100 32G)
The results were obtained by running the `./scripts/benchmarking/DGX2_trainbench_fp16.sh` and `./scripts/benchmarking/DGX2_trainbench_fp32.sh` scripts in the tensorflow-19.07-py3 Docker container on NVIDIA DGX-2 with 16 V100 32G GPU.

| **number of GPUs** | **mixed precision img/s** | **FP32 img/s** | **mixed precision speedup** | **mixed precision weak scaling** | **FP32 weak scaling** |
|:------------------:|:-------------------------:|:--------------:|:---------------------------:|:--------------------------------:|:---------------------:|
| **1**                  | 821.9                     | 377.5          | 2.18                        | 1.00                             | 1.00                  |
| **4**                  | 3239.8                    | 1478.5         | 2.19                        | 3.94                             | 3.92                  |
| **8**                  | 6439.4                    | 2888.6         | 2.23                        | 7.84                             | 7.65                  |
| **16**                  | 12467.5                    | 5660.8         | 2.20                        | 15.17                             | 15.00                  |

##### XLA Enabled

| **number of GPUs** | **mixed precision img/s** | **mixed precision + XLA img/s** | **XLA speedup** | 
|:------------------:|:-------------------------:|:-------------------------------:|:---------------:|
| **1**              | 821.9                     | 1248.5                          | 1.52            |
| **4**              | 3239.8                    | 4934.8                          | 1.52            |
| **8**              | 6439.4                    | 9295.5                          | 1.44            |
| **16**             | 12467.5                   | 15902.8                         | 1.27            |

#### Inference performance results

##### NVIDIA DGX-1 (8x V100 16G)
The results were obtained by running the `./scripts/benchmarking/DGX1V_inferbench_fp16.sh` and `./scripts/benchmarking/DGX1V_inferbench_fp32.sh` scripts in the tensorflow-19.07-py3 Docker container on a single GPU of NVIDIA DGX-1 with 8 V100 16G GPUs.

| **batch size** | **mixed precision img/s** | **FP32 img/s** | **mixed precision + XLA img/s** |
|:--------------:|:-------------------------:|:--------------:|:-------------------------------:|
|         **1** |   177.2 |   170.8 |   163.8 |      
|         **2** |   325.7 |   308.4 |   312.7 |
|         **4** |   587.0 |   499.4 |   581.5 |         
|         **8** |  1002.9 |   688.3 |   1077 |         
|        **16** |  1408.5 |   854.9 |   1848.2 |        
|        **32** |  1687.0 |   964.4 |   2486.3 |        
|        **64** |  1907.7 |  1045.1 |   2721.2 |
|       **128** |  2077.3 |  1100.1 |   3334.4 |       
|       **256** |  2129.3 |  N/A    |   3547.9 |


##### NVIDIA DGX-2 (16x V100 32G)
The results were obtained by running the `./scripts/benchmarking/DGX2_inferbench_fp16.sh` and `./scripts/benchmarking/DGX2_inferbench_fp32.sh` scripts in the tensorflow-19.07-py3 Docker container on a single GPU of NVIDIA DGX-2 with 16 V100 32G GPUs.

| **batch size** | **mixed precision img/s** | **FP32 img/s** | **mixed precision + XLA img/s** |
|:--------------:|:-------------------------:|:--------------:|:-------------------------------:|
|         **1** |   199.7 |   201.6 |   220.2 |       
|         **2** |   388.9 |   334.1 |   406.6 | 
|         **4** |   684.3 |   534.4 |   783.4 |          
|         **8** |  1095.9 |   739.8 |  1391.7 |          
|        **16** |  1484.1 |   906.5 |  1851.6 |         
|        **32** |  1789.9 |  1020.4 |  2746.5 |         
|        **64** |  2005.7 |  1111.9 |  3253.8 | 
|       **128** |  2126.5 |  1168.8 |  3469.6 |        
|       **256** |  2203.6 |  N/A    |  3713.2 | 

##### NVIDIA T4 (1x T4 16G)
The results were obtained by running the `./scripts/benchmarking/T4_inferbench_fp16.sh` and `./scripts/benchmarking/T4_inferbench_fp32.sh` scripts in the tensorflow-19.07-py3 Docker container on a single T4 GPU.

| **batch size** | **mixed precision img/s** | **FP32 img/s** | **mixed precision + XLA img/s** |
|:--------------:|:-------------------------:|:--------------:|:-------------------------------:|
|         **1** |    173.2 |   138.7 |    204.2 |       
|         **2** |   302.1 |   207.6 |  359.8 | 
|         **4** |   450.3 |   267.4 |   660.0 |          
|         **8** |  558.7 |   305.9 |  924.7 |          

## Release notes

### Changelog
1. March 1, 2019
  * Initial release
2. May 15, 2019
  * Added DALI support
  * Added scripts for DGX-2
  * Added benchmark results for DGX-2 and XLA-enabled DGX-1 and DGX-2.
3. July 15, 2019
  * Added Cosine learning rate schedule
3. August 15, 2019
  * Added mixup regularization
  * Added T4 benchmarks
  * Improved inference capabilities
  * Added SavedModel export 
  
### Known issues
There are no known issues with this model.

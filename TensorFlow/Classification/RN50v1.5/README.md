# ResNet-50 v1.5 for TensorFlow

## Table Of Contents
* [The model](#the-model)
    * [Default configuration](#default-configuration)
    * [Data Augmentation](#data-augmentation)
    * [Other training recipes](#other-training-recipes)
* [Setup](#setup)
    * [Requirements](#requirements)
* [Quick start guide](#quick-start-guide)
* [Details](#details)
    * [Command line options](#command-line-options)
    * [Training process](#training-process)
    * [Enabling mixed precision](#enabling-mixed-precision)
* [Benchmarking](#benchmarking)
    * [Training performance benchmark](#training-performance-benchmark)
    * [Inference performance benchmark](#inference-performance-benchmark)
* [Results](#results)
    * [Training accuracy results](#training-accuracy-results)
    * [Training performance results](#training-performance-results)
    * [Inference performance results](#inference-performance-results)
* [Changelog](#changelog)
* [Known issues](#known-issues)

# The model
The ResNet50 v1.5 model is a modified version of the [original ResNet50 v1 model](https://arxiv.org/abs/1512.03385).

The difference between v1 and v1.5 is in the bottleneck blocks which requires
downsampling, for example, v1 has stride = 2 in the first 1x1 convolution, whereas v1.5 has stride = 2 in the 3x3 convolution.

This difference makes ResNet50 v1.5 slightly more accurate (~0.5% top1) than v1,
but comes with a small performance drawback (~5% imgs/sec).

The following features were implemented in this model:
* Data-parallel multi-GPU training with Horovod
* Mixed precision support with TensorFlow Automatic Mixed Precision (TF-AMP), which enables mixed precision training without any changes to the code-base by performing automatic graph rewrites and loss scaling controlled by an environmental variable. Tensor Core operations to maximize throughput using NVIDIA Volta GPUs.
* Static loss scaling for Tensor Cores (mixed precision) training

The following performance optimizations were implemented in this model:
* XLA support (experimental)

## Default configuration

This model trains for 90 epochs, with default ResNet50 v1.5 setup:

* SGD with momentum (0.9)

* Learning rate = 0.1 for 256 batch size, for other batch sizes we linearly
scale the learning rate.

* Learning rate decay - multiply by 0.1 after 30, 60, and 80 epochs

* For bigger batch sizes (512 and up) we use linear warmup of the learning rate
during first 5 epochs according to [Training ImageNet in 1 hour](https://arxiv.org/abs/1706.02677).

* Weight decay: 1e-4


## Data Augmentation

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


## Other training recipes

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

# Setup

The following section list the requirements that you need to meet in order to use the ResNet50 v1.5 model.

## Requirements
This repository contains Dockerfile which extends the Tensorflow NGC container and encapsulates all dependencies.  Aside from these dependencies, ensure you have the following software:

* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [TensorFlow 19.03-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)
* [NVIDIA Volta based GPU](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)

For more information about how to get started with NGC containers, see the
following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
* [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html),
* [Accessing And Pulling From The NGC container registry](https://docs.nvidia.com/deeplearning/dgx/user-guide/index.html#accessing_registry)
* [Running Tensorflow](https://docs.nvidia.com/deeplearning/dgx/tensorflow-release-notes/running.html#running).

# Quick start guide
To train your model using mixed precision with tensor cores, perform the following steps using the default parameters of the ResNet-50 v1.5 model on the [ImageNet](http://www.image-net.org/) dataset.

## 1. Download and preprocess the dataset.
The ResNet50 v1.5 script operates on ImageNet 1k, a widely popular image classification dataset from ILSVRC challenge.

To download and preprocess the dataset, use the [Generate ImageNet for TensorFlow](https://github.com/tensorflow/models/blob/master/research/inception/inception/data/download_and_preprocess_imagenet.sh) script. The dataset will be downloaded to a directory specified as the first parameter of the script.


## 2. Build the ResNet-50 v1.5 TensorFlow NGC container.
```bash
bash scripts/docker/build.sh
```

## 3. Start an interactive session in the NGC container to run training/inference.
After you build the container image, you can start an interactive CLI session with
```bash
bash scripts/docker/interactive.sh
```
The interactive.sh script requires that the location on the dataset is specified.  For example, /data.

## 4. Start training.
To run training for a default configuration (as described in [Default configuration](#default-configuration), for example 1/4/8 GPUs, FP16/FP32), run one of the scripts in the ./scripts directory called `./scripts/RN50_{FP16, FP32}_{1, 4, 8}GPU.sh`. Each of the scripts require three parameters: 
* path to the root directory of the model as the first argument
* path to the dataset as a second argument 
* path to the results destination as a third argument

For example:
```bash
./scripts/RN50_FP16_8GPU.sh <path to model> <path to dataset> <path to results>
```

## 5. Start validation/evaluation.

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


# Details

## Command line options
To see the full list of available options and their descriptions, use the `-h` or `--help` command line option, for example:

```
python main.py --help
```

To summarize, the most important arguments are as follows:

```
  --mode {train,train_and_evaluate,evaluate,training_benchmark,inference_benchmark}
                        The execution mode of the script.
                        
  --data_dir DATA_DIR   Path to dataset in TFRecord format. Files should be
                        named 'train-*' and 'validation-*'.
                        
  --batch_size BATCH_SIZE
                        Size of each minibatch per GPU.
                                               
  --num_iter NUM_ITER   Number of iterations to run.
  
  --iter_unit {epoch,batch}
                        Unit of iterations.
                        
  --results_dir RESULTS_DIR
                        Directory in which to write training logs, summaries
                        and checkpoints.
                        
  --loss_scale LOSS_SCALE
                        Loss scale for mixed precision training.
                        
  --use_auto_loss_scaling
                        Use automatic loss scaling in fp32 AMP.
                        
  --use_xla             Enable XLA (Accelerated Linear Algebra) computation
                        for improved performance.
                        
  --use_tf_amp          Enable Automatic Mixed Precision to speedup fp32
                        computation using tensor cores.
                                           
```

## Training process
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


## Enabling mixed precision
[Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format, while storing minimal information in single-precision to retain as much information as possible in critical parts of the network.  Since the introduction of [tensor cores](https://developer.nvidia.com/tensor-cores) in the Volta and Turing architectures, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures.  Using [mixed precision training](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html) previously required two steps:

1. Porting the model to use the FP16 data type where appropriate.
2. Manually adding loss scaling to preserve small gradient values. 

This can now be achieved using Automatic Mixed Precision (AMP) for TensorFlow to enable the full [mixed precision methodology](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#tensorflow) in your existing TensorFlow model code.  AMP enables mixed precision training on Volta and Turing GPUs automatically. The TensorFlow framework code makes all necessary model changes internally.

In TF-AMP, the computational graph is optimized to use as few casts as necessary and maximize the use of FP16, and the loss scaling is automatically applied inside of supported optimizers. AMP can be configured to work with the existing tf.contrib loss scaling manager by disabling the AMP scaling with a single environment variable to perform only the automatic mixed-precision optimization. It accomplishes this by automatically rewriting all computation graphs with the necessary operations to enable mixed precision training and automatic loss scaling.

For information about:
 * How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html) documentation.
 * How to access and enable AMP for TensorFlow, see [Using TF-AMP](https://docs.nvidia.com/deeplearning/dgx/tensorflow-user-guide/index.html#tfamp) from the TensorFlow User Guide.
 * Techniques used for mixed precision training, see the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.


# Benchmarking

The following sections shows how to run benchmarks measuring the model performance in training and inference modes.

## Training performance benchmark

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


## Inference performance benchmark

To benchmark the inference performance on a specific batch size, run:

* FP32
`python ./main.py --mode=inference_benchmark --precision=fp32 --warmup_steps 20 --train_iter 100 --iter_unit batch --batch_size <batch size> --data_dir=<path to imagenet> --log_dir=<path to results directory>`

* FP16
`python ./main.py --mode=inference_benchmark --precision=fp16 --warmup_steps 20 --train_iter 100 --iter_unit batch --batch_size <batch size> --data_dir=<path to imagenet> --log_dir=<path to results directory>`

Each of these scripts, by default runs 20 warm-up iterations and measures the next 80 iterations.

To control warm-up and benchmark length, use `--warmup_steps`, `--num_iter` and `--iter_unit` flags.

# Results

The following sections provide details on how we achieved our results in training accuracy, performance and inference performance.

## Training accuracy results

Our results were obtained by running the `./scripts/RN50_{FP16, FP32}_{1, 4, 8}GPU.sh` script in
the tensorflow-19.02-py3 Docker container on NVIDIA DGX-1 with 8 V100 16G GPUs.


| **number of GPUs** | **mixed precision top1** | **mixed precision training time** | **FP32 top1** | **FP32 training time** |
|:------------------:|:------------------------:|:---------------------------------:|:-------------:|:----------------------:|
| **1**                  | 76.18                    | 41.3h                             | 76.38         | 89.4h                  |
| **4**                  | 76.30                    | 10.5h                             | 76.30         | 22.4h                  |
| **8**                  | 76.18                    | 5.6h                              | 76.26         | 11.5h                  |



## Training performance results

Our results were obtained by running the `./scripts/benchmarking/DGX1V_trainbench_fp16.sh` and `./scripts/benchmarking/DGX1V_trainbench_fp32.sh` scripts in the tensorflow-19.02-py3 Docker container on NVIDIA DGX-1 with 8 V100 16G GPUs.


| **number of GPUs** | **mixed precision img/s** | **FP32 img/s** | **mixed precision speedup** | **mixed precision weak scaling** | **FP32 weak scaling** |
|:------------------:|:-------------------------:|:--------------:|:---------------------------:|:--------------------------------:|:---------------------:|
| **1**                  | 818.3                     | 362.5          | 2.25                        | 1.00                             | 1.00                  |
| **4**                  | 3276.6                    | 1419.4         | 2.30                        | 4.00                             | 3.92                  |
| **8**                  | 6508.4                    | 2832.2         | 2.30                        | 7.95                             | 7.81                  |

Our results were obtained by running the `./scripts/benchmarking/DGX1V_inferbench_fp16.sh` and `./scripts/benchmarking/DGX1V_inferbench_fp32.sh` scripts in the tensorflow-19.02-py3 Docker container on NVIDIA DGX-1 with 8 V100 16G GPUs.

Those results can be improved when [XLA](https://www.tensorflow.org/xla) is used 
in conjunction with mixed precision, delivering up to 3.3x speedup over FP32 on a single GPU (~1179 img/s).
However XLA is still considered experimental.

## Inference performance results

| **batch size** | **mixed precision img/s** | **FP32 img/s** |
|:--------------:|:-------------------------:|:--------------:|
|         **1** |   177.2 |   170.8 |      
|         **2** |   325.7 |   308.4 |
|         **4** |   587.0 |   499.4 |         
|         **8** |  1002.9 |   688.3 |         
|        **16** |  1408.5 |   854.9 |        
|        **32** |  1687.0 |   964.4 |        
|        **64** |  1907.7 |  1045.1 |
|       **128** |  2077.3 |  1100.1 |       
|       **256** |  2129.3 |  N/A    |



# Changelog
1. March 1, 2019
  * Initial release

# Known issues
There are no known issues with this model.

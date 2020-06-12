# SE-ResNext101-32x4d for TensorFlow

This repository provides a script and recipe to train the SE-ResNext101-32x4d model to achieve state-of-the-art accuracy, and is tested and maintained by NVIDIA.

## Table Of Contents
* [Model overview](#model-overview)
    * [Model architecture](#model-architecture)
    * [Default configuration](#default-configuration)
        * [Optimizer](#optimizer)
        * [Data augmentation](#data-augmentation)
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
        * [The `main.py` script](#the-mainpy-script)
    * [Inference process](#inference-process)
* [Performance](#performance)
    * [Benchmarking](#benchmarking)
        * [Training performance benchmark](#training-performance-benchmark)
        * [Inference performance benchmark](#inference-performance-benchmark)
    * [Results](#results)
        * [Training accuracy results](#training-accuracy-results)
            * [Training accuracy: NVIDIA DGX-1 (8x V100 16G)](#training-accuracy-nvidia-dgx-1-8x-v100-16g)
        * [Training performance results](#training-performance-results)
            * [Training performance: NVIDIA DGX-1 (8x V100 16G)](#training-performance-nvidia-dgx-1-8x-v100-16g)
            * [Training performance: NVIDIA DGX-2 (16x V100 32G)](#training-performance-nvidia-dgx-2-16x-v100-32g)
        * [Training time for 90 Epochs](#training-time-for-90-epochs)
            * [Training time: NVIDIA DGX-1 (8x V100 16G)](#training-time-nvidia-dgx-1-8x-v100-16g)
            * [Training time: NVIDIA DGX-2 (16x V100 32G)](#training-time-nvidia-dgx-2-16x-v100-32g)
        * [Inference performance results](#inference-performance-results)
            * [Inference performance: NVIDIA DGX-1 (1x V100 16G)](#inference-performance-nvidia-dgx-1-1x-v100-16g)
            * [Inference performance: NVIDIA DGX-2 (1x V100 32G)](#inference-performance-nvidia-dgx-2-1x-v100-32g)
            * [Inference performance: NVIDIA T4 (1x T4 16G)](#inference-performance-nvidia-t4-1x-t4-16g)
* [Release notes](#release-notes)
    * [Changelog](#changelog)
    * [Known issues](#known-issues)

## Model overview
The SE-ResNeXt101-32x4d is a [ResNeXt101-32x4d](https://arxiv.org/pdf/1611.05431.pdf)
model with added Squeeze-and-Excitation module introduced in the [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf) paper.

The following performance optimizations were implemented in this model:
* JIT graph compilation with [XLA](https://www.tensorflow.org/xla)
* Multi-GPU training with [Horovod](https://github.com/horovod/horovod)
* Automated mixed precision [AMP](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)

This model is trained with mixed precision using Tensor Cores on NVIDIA Volta and Turing GPUs. Therefore, researchers can get results 2x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.

### Model architecture
Here is a diagram of the Squeeze and Excitation module architecture for ResNet-type models:

![SEArch](./imgs/SEArch.png)

_Image source: [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)_

This image shows the architecture of the SE block and where it is placed in the ResNet bottleneck block.

### Default configuration

The following sections highlight the default configuration for the SE-ResNext101-32x4d model.

#### Optimizer

This model uses the SGD optimizer with the following hyperparameters:

* Momentum (0.875).
* Learning rate (LR) = 0.256 for 256 batch size, for other batch sizes we linearly scale the learning
  rate.
* Learning rate schedule - we use cosine LR schedule.
* For bigger batch sizes (512 and up) we use linear warmup of the learning rate.
during the first 5 epochs according to [Training ImageNet in 1 hour](https://arxiv.org/abs/1706.02677).
* Weight decay: 6.103515625e-05 (1/16384).
* We do not apply Weight decay on batch norm trainable parameters (gamma/bias).
* Label Smoothing: 0.1.
* We train for:
    * 90 Epochs -> 90 epochs is a standard for ImageNet networks.
    * 250 Epochs -> best possible accuracy. 
* For 250 epoch training we also use [MixUp regularization](https://arxiv.org/pdf/1710.09412.pdf).

#### Data Augmentation

This model uses the following data augmentation:

* For training:
  * Normalization.
  * Random resized crop to 224x224.
    * Scale from 8% to 100%.
    * Aspect ratio from 3/4 to 4/3.
  * Random horizontal flip.

* For inference:
  * Normalization.
  * Scale to 256x256.
  * Center crop to 224x224.

### Feature support matrix

The following features are supported by this model.

| Feature               | SE-ResNext101-32x4d Tensorflow             |
|-----------------------|--------------------------
|Multi-GPU training with [Horovod](https://github.com/horovod/horovod)  |  Yes |
|[NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/release-notes/index.html)                |  Yes |
|Automatic mixed precision (AMP) | Yes |


#### Features

Multi-GPU training with Horovod - Our model uses Horovod to implement efficient multi-GPU training with NCCL.
For details, refer to the example sources in this repository or the [TensorFlow tutorial](https://github.com/horovod/horovod/#usage).

NVIDIA DALI - DALI is a library accelerating data preparation pipeline. To accelerate your input pipeline, you only need to define your data loader
with the DALI library. For details, refer to the example sources in this repository or the [DALI documentation](https://docs.nvidia.com/deeplearning/dali/index.html).

Automatic mixed precision (AMP) - Computation graph can be modified by TensorFlow on runtime to support mixed precision training. 
Detailed explanation of mixed precision can be found in the next section.

### Mixed precision training
Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format, while storing minimal information in single-precision to retain as much information as possible in critical parts of the network.  Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in the Volta and Turing architectures, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures.  Using [mixed precision training](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html) previously required two steps:

1. Porting the model to use the FP16 data type where appropriate.
2. Manually adding loss scaling to preserve small gradient values. 

This can now be achieved using Automatic Mixed Precision (AMP) for TensorFlow to enable the full [mixed precision methodology](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#tensorflow) in your existing TensorFlow model code.  AMP enables mixed precision training on Volta and Turing GPUs automatically. The TensorFlow framework code makes all necessary model changes internally.

In TF-AMP, the computational graph is optimized to use as few casts as necessary and maximize the use of FP16, and the loss scaling is automatically applied inside of supported optimizers. AMP can be configured to work with the existing tf.contrib loss scaling manager by disabling the AMP scaling with a single environment variable to perform only the automatic mixed-precision optimization. It accomplishes this by automatically rewriting all computation graphs with the necessary operations to enable mixed precision training and automatic loss scaling.

For information about:
 * How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) documentation.
 * How to access and enable AMP for TensorFlow, see [Using TF-AMP](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html#tfamp) from the TensorFlow User Guide.
 * Techniques used for mixed precision training, see the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
 
#### Enabling mixed precision

Mixed precision is enabled in TensorFlow by using the Automatic Mixed Precision (TF-AMP) extension which casts variables to half-precision upon retrieval, while storing variables in single-precision format. Furthermore, to preserve small gradient magnitudes in backpropagation, a [loss scaling](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#lossscaling) step must be included when applying gradients. In TensorFlow, loss scaling can be applied statically by using simple multiplication of loss by a constant value or automatically, by TF-AMP. Automatic mixed precision makes all the adjustments internally in TensorFlow, providing two benefits over manual operations. First, programmers need not modify network model code, reducing development and maintenance effort. Second, using AMP maintains forward and backward compatibility with all the APIs for defining and running TensorFlow models.

To enable mixed precision, you can simply add the values to the environmental variables inside your training script:
- Enable TF-AMP graph rewrite:
  ```
  os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
  ```
  
- Enable Automated Mixed Precision:
  ```
  os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
  ```

## Setup

The following section lists the requirements that you need to meet in order to use the SE-ResNext101-32x4d model.

### Requirements
This repository contains Dockerfile which extends the TensorFlow NGC container and encapsulates all dependencies.  Aside from these dependencies, ensure you have the following software:

* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [TensorFlow 20.03-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)
* [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) based GPU

For more information about how to get started with NGC containers, see the
following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
* [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html),
* [Accessing And Pulling From The NGC container registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry),
* [Running TensorFlow](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html#running).

For those unable to use the [TensorFlow 20.03-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## Quick Start Guide
To train your model using mixed precision with Tensor Cores or FP32, perform the following steps using the default parameters of the SE-ResNext101-32x4d model on the [ImageNet](http://www.image-net.org/) dataset. For the specifics concerning training and inference, see the [Advanced](#advanced) section.


1. Clone the repository.
```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/TensorFlow/Classification/RN50v1.5
```

2. Download and preprocess the dataset.
The SE-ResNext101-32x4d script operates on ImageNet 1k, a widely popular image classification dataset from the ILSVRC challenge.

To download and preprocess the dataset, use the [Generate ImageNet for TensorFlow](https://github.com/tensorflow/models/blob/master/research/inception/inception/data/download_and_preprocess_imagenet.sh) script. The dataset will be downloaded to a directory specified as the first parameter of the script.

3. Build the SE-ResNext101-32x4d TensorFlow NGC container.
```bash
docker build . -t nvidia_rn50
```

4. Start an interactive session in the NGC container to run training/inference.
After you build the container image, you can start an interactive CLI session with
```bash
nvidia-docker run --rm -it -v <path to imagenet>:/data/tfrecords --ipc=host nvidia_rn50
```

5. (Optional) Create index files to use DALI.
To allow proper sharding in a multi-GPU environment, DALI has to create index files for the dataset. To create index files, run inside the container:
```bash
bash ./utils/dali_index.sh /data/tfrecords <index file store location>
```
Index files can be created once and then reused. It is highly recommended to save them into a persistent location.

6. Start training.
To run training for a standard configuration (as described in [Default
configuration](#default-configuration), DGX1V, DGX2V, single GPU, FP16, FP32, 90, and 250 epochs), run
one of the scripts in the `se-resnext101-32x4d/training` directory. Ensure ImageNet is mounted in the
`/data/tfrecords` directory.

For example, to train on DGX-1 for 90 epochs using AMP, run:  

`bash ./se-resnext101-32x4d/training/AMP/DGX1_SE-RNxt101-32x4d_AMP_90E.sh`

Additionally, features like DALI data preprocessing or TensorFlow XLA can be enabled with
environmental variables when running those scripts:

`USE_XLA=1 USE_DALI=1 bash ./se-resnext101-32x4d/training/AMP/DGX1_SE-RNxt101-32x4d_AMP_90E.sh`

To store results in a specific location, add a location as a first argument:

`bash ./resnext101-32x4d/training/AMP/DGX1_RNxt101-32x4d_AMP_90E.sh <location to store>`

7. Start validation/evaluation.
To evaluate the validation dataset located in `/data/tfrecords`, run `main.py` with
`--mode=evaluate`. For example:

`python main.py --arch=se-resnext101-32x4d --mode=evaluate --data_dir=/data/tfrecords --batch_size <batch size> --model_dir
<model location> --result_dir <output location> [--use_xla] [--use_tf_amp]`

The optional `--use_xla` and `--use_tf_amp` flags control XLA and AMP during evaluation. 

## Advanced

The following sections provide greater details of the dataset, running training and inference, and the training results.

### Scripts and sample code

In the root directory, the most important files are:
 - `main.py`:               the script that controls the logic of training and validation of the ResNet-like models
 - `Dockerfile`:            Instructions for Docker to build a container with the basic set of dependencies to run ResNet like models for image classification
 - `requirements.txt`:      a set of extra Python requirements for running ResNet-like models

The `model/` directory contains the following modules used to define ResNet family models:
 - `resnet.py`: the definition of ResNet, ResNext, and SE-ResNext model
 - `blocks/conv2d_block.py`: the definition of 2D convolution block
 - `blocks/resnet_bottleneck_block.py`: the definition of ResNet-like bottleneck block
 - `layers/*.py`: definitions of specific layers used in the ResNet-like model
 
The `utils/` directory contains the following utility modules:
 - `cmdline_helper.py`: helper module for command line processing
 - `data_utils.py`: module defining input data pipelines
 - `dali_utils.py`: helper module for DALI 
 - `hvd_utils.py`: helper module for Horovod
 - `image_processing.py`: image processing and data augmentation functions
 - `learning_rate.py`: definition of used learning rate schedule
 - `optimizers.py`: definition of used custom optimizers
 - `hooks/*.py`: definitions of specific hooks allowing logging of training and inference process
 
The `runtime/` directory contains the following module that define the mechanics of the training process:
 - `runner.py`: module encapsulating the training, inference and evaluation 


### Parameters

#### The `main.py` script
The script for training and evaluating the ResNext101-32x4d model has a variety of parameters that control these processes.

```
usage: main.py [-h]
               [--arch {resnet50,resnext101-32x4d,se-resnext101-32x4d}]
               [--mode {train,train_and_evaluate,evaluate,predict,training_benchmark,inference_benchmark}]
               [--data_dir DATA_DIR] [--data_idx_dir DATA_IDX_DIR]
               [--export_dir EXPORT_DIR] [--to_predict TO_PREDICT]
               [--batch_size BATCH_SIZE] [--num_iter NUM_ITER]
               [--iter_unit {epoch,batch}] [--warmup_steps WARMUP_STEPS]
               [--model_dir MODEL_DIR] [--results_dir RESULTS_DIR]
               [--log_filename LOG_FILENAME] [--display_every DISPLAY_EVERY]
               [--lr_init LR_INIT] [--lr_warmup_epochs LR_WARMUP_EPOCHS]
               [--weight_decay WEIGHT_DECAY] [--weight_init {fan_in,fan_out}]
               [--momentum MOMENTUM] [--loss_scale LOSS_SCALE]
               [--label_smoothing LABEL_SMOOTHING] [--mixup MIXUP]
               [--use_static_loss_scaling | --nouse_static_loss_scaling]
               [--use_xla | --nouse_xla] [--use_dali | --nouse_dali]
               [--use_tf_amp | --nouse_tf_amp]
               [--use_cosine_lr | --nouse_cosine_lr] [--seed SEED]
               [--gpu_memory_fraction GPU_MEMORY_FRACTION] [--gpu_id GPU_ID]

JoC-RN50v1.5-TF

optional arguments:
  -h, --help            Show this help message and exit
  --arch {resnet50,resnext101-32x4d,se-resnext101-32x4d}
                        Architecture of model to run (to run se-resnext-32x4d set
                        --arch=se-rensext101-32x4d)
  --mode {train,train_and_evaluate,evaluate,predict,training_benchmark,inference_benchmark}
                        The execution mode of the script.
  --data_dir DATA_DIR   Path to dataset in TFRecord format. Files should be
                        named 'train-*' and 'validation-*'.
  --data_idx_dir DATA_IDX_DIR
                        Path to index files for DALI. Files should be named
                        'train-*' and 'validation-*'.
  --export_dir EXPORT_DIR
                        Directory in which to write exported SavedModel.
  --to_predict TO_PREDICT
                        Path to file or directory of files to run prediction
                        on.
  --batch_size BATCH_SIZE
                        Size of each minibatch per GPU.
  --num_iter NUM_ITER   Number of iterations to run.
  --iter_unit {epoch,batch}
                        Unit of iterations.
  --warmup_steps WARMUP_STEPS
                        Number of steps considered as warmup and not taken
                        into account for performance measurements.
  --model_dir MODEL_DIR
                        Directory in which to write the model. If undefined,
                        results directory will be used.
  --results_dir RESULTS_DIR
                        Directory in which to write training logs, summaries
                        and checkpoints.
  --log_filename LOG_FILENAME
                        Name of the JSON file to which write the training log
  --display_every DISPLAY_EVERY
                        How often (in batches) to print out running
                        information.
  --lr_init LR_INIT     Initial value for the learning rate.
  --lr_warmup_epochs LR_WARMUP_EPOCHS
                        Number of warmup epochs for the learning rate schedule.
  --weight_decay WEIGHT_DECAY
                        Weight Decay scale factor.
  --weight_init {fan_in,fan_out}
                        Model weight initialization method.
  --momentum MOMENTUM   SGD momentum value for the momentum optimizer.
  --loss_scale LOSS_SCALE
                        Loss scale for FP16 training and fast math FP32.
  --label_smoothing LABEL_SMOOTHING
                        The value of label smoothing.
  --mixup MIXUP         The alpha parameter for mixup (if 0 then mixup is not
                        applied).
  --use_static_loss_scaling
                        Use static loss scaling in FP16 or FP32 AMP.
  --nouse_static_loss_scaling
  --use_xla             Enable XLA (Accelerated Linear Algebra) computation
                        for improved performance.
  --nouse_xla
  --use_dali            Enable DALI data input.
  --nouse_dali
  --use_tf_amp          Enable AMP to speedup FP32
                        computation using Tensor Cores.
  --nouse_tf_amp
  --use_cosine_lr       Use cosine learning rate schedule.
  --nouse_cosine_lr
  --seed SEED           Random seed.
  --gpu_memory_fraction GPU_MEMORY_FRACTION
                        Limit memory fraction used by the training script for DALI
  --gpu_id GPU_ID       Specify the ID of the target GPU on a multi-device platform.
                        Effective only for single-GPU mode.
```

### Inference process
To run inference on a single example with a checkpoint and a model script, use: 

`python main.py --arch=se-resnext101-32x4d --mode predict --model_dir <path to model> --to_predict <path to image> --results_dir <path to results>`

The optional `--use_xla` and `--use_tf_amp` flags control XLA and AMP during inference.

## Performance

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in training and inference modes.

#### Training performance benchmark

To benchmark the training performance on a specific batch size, run:

* For 1 GPU
    * FP32
        `python ./main.py --arch=se-resnext101-32x4d --mode=training_benchmark --warmup_steps 200 --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
    * FP16
        `python ./main.py --arch=se-resnext101-32x4d --mode=training_benchmark  --use_tf_amp --warmup_steps 200 --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
* For multiple GPUs
    * FP32
        `mpiexec --allow-run-as-root --bind-to socket -np <num_gpus> python ./main.py --arch=se-resnext101-32x4d --mode=training_benchmark --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
    * FP16
        `mpiexec --allow-run-as-root --bind-to socket -np <num_gpus> python ./main.py --arch=se-resnext101-32x4d --mode=training_benchmark --use_tf_amp --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
        
Each of these scripts runs 200 warm-up iterations and measures the first epoch.

To control warmup and benchmark length, use the `--warmup_steps`, `--num_iter` and `--iter_unit` flags. Features like XLA or DALI can be controlled
with `--use_xla` and `--use_dali` flags. 
Suggested batch sizes for training are 96 for mixed precision training and 64 for single precision training per single V100 16 GB.


#### Inference performance benchmark

To benchmark the inference performance on a specific batch size, run:

* FP32
`python ./main.py --arch=se-resnext101-32x4d --mode=inference_benchmark --warmup_steps 20 --num_iter 100 --iter_unit batch --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`

* FP16
`python ./main.py --arch=se-resnext101-32x4d --mode=inference_benchmark --use_tf_amp --warmup_steps 20 --num_iter 100 --iter_unit batch --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`

By default, each of these scripts runs 20 warm-up iterations and measures the next 80 iterations.
To control warm-up and benchmark length, use the `--warmup_steps`, `--num_iter` and `--iter_unit` flags.

The benchmark can be automated with the `inference_benchmark.sh` script provided in `se-resnext101-32x4d`, by simply running:
`bash ./se-resnext101-32x4d/inference_benchmark.sh <data dir> <data idx dir>`

The `<data dir>` parameter refers to the input data directory (by default `/data/tfrecords` inside the container). 
By default, the benchmark tests the following configurations: **FP32**, **AMP**, **AMP + XLA** with different batch sizes.
When the optional directory with the DALI index files `<data idx dir>` is specified, the benchmark executes an additional **DALI + AMP + XLA** configuration.

### Results

The following sections provide details on how we achieved our performance and accuracy in training and inference. 

#### Training accuracy results

##### Training accuracy: NVIDIA DGX-1 (8x V100 16G)
Our results were obtained by running the `/resnext101-32x4d/training/{PRECISION}/DGX1_RNxt101-32x4d_{PRECISION}_{EPOCHS}E.sh` 
training script in the [TensorFlow 20.03-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) 
NGC container on NVIDIA DGX-1 with (8x V100 16G) GPUs.

| Epochs | Batch Size / GPU | Accuracy - FP32 | Accuracy - mixed precision | 
|--------|------------------|-----------------|----------------------------|
| 90   | 64 (FP32) / 96 (AMP) | 79.69              | 79.81   |
| 250  | 64 (FP32) / 96 (AMP) | 80.87              | 80.84   |

**Example training loss plot**

![TrainingLoss](./imgs/train_loss.png)

#### Training performance results

##### Training performance: NVIDIA DGX-1 (8x V100 16G)
Our results were obtained by running the steps from [Training performance benchmark](#training-performance-benchmark) in the 
[TensorFlow 20.03-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)  NGC container 
on NVIDIA DGX-1 with (8x V100 16G) GPUs. Performance numbers (in images per second) were averaged over an entire training epoch.


| GPUs | Batch Size / GPU | Throughput - FP32 | Throughput - mixed precision | Throughput speedup (FP32 - mixed precision) | Weak scaling - FP32 | Weak scaling - mixed precision |
|----|---------------|---------------|-----------------------|---------------|-----------|-------|
| 1  | 64 (FP32) / 96 (AMP) | 126.44 img/s | 285.99 img/s     | 2.26x         | 1.00x     | 1.00x             |
| 8  | 64 (FP32) / 96 (AMP) | 921.38 img/s | 2168.40 img/s    | 2.35x         | 7.28x     | 7.58x             |

**XLA Enabled**

| GPUs | Batch Size / GPU | Throughput - mixed precision | Throughput - mixed precision + XLA | Throughput speedup (mixed precision - XLA) |
|----|------------|---------------|---------------------|-----------|
| 1  | 128        | 285.99 img/s   |453.39 img/s         |1.58x      |
| 8  | 128        | 2168.40 img/s  |3297.39 img/s        |1.52x      |

##### Training performance: NVIDIA DGX-2 (16x V100 32G)
Our results were obtained by running the steps from [Training performance benchmark](#training-performance-benchmark) in the 
[TensorFlow 20.03-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)  NGC container 
on NVIDIA DGX-2 with (16x V100 32G) GPUs. Performance numbers (in images per second) were averaged over an entire training epoch.

| GPUs | Batch Size / GPU | Throughput - FP32 | Throughput - mixed precision | Throughput speedup (FP32 - mixed precision) | Weak scaling - FP32 | Weak scaling - mixed precision |
|----|---------------|---------------|-------------------------|-------|--------|--------|
| 1  | 64 (FP32) / 96 (AMP) | 128.08 img/s | 309.34 img/s    | 2.39x                 | 1.00x         | 1.00x  |
| 16 | 64 (FP32) / 96 (AMP) | 1759.44 img/s| 4210.85 img/s   | 2.51x                 | 13.73x        | 13.61x |

**XLA Enabled**

| GPUs | Batch Size / GPU | Throughput - mixed precision | Throughput - mixed precision + XLA | Throughput speedup (mixed precision - XLA) |
|----|-----|----------|---------------------|-----------|
| 1  | 96 | 309.34 img/s    |520.10 img/s         |1.68x      |
| 16 | 96 | 4210.85 img/s   |6835.66 img/s        |1.62x      |

#### Training Time for 90 Epochs

##### Training time: NVIDIA DGX-1 (8x V100 16G)

Our results were estimated based on the [training performance results](#training-performance-nvidia-dgx-1-8x-v100-16g) 
on NVIDIA DGX-1 with (8x V100 16G) GPUs.

| GPUs | Time to train - mixed precision + XLA | Time to train - mixed precision | Time to train - FP32 |
|---|--------|---------|---------|
| 1 | ~71h   |  ~112h   |  ~253h   |
| 8 | ~10h  |  ~15h    |  ~35h | 

##### Training time: NVIDIA DGX-2 (16x V100 32G)

Our results were estimated based on the [training performance results](#training-performance-nvidia-dgx-2-16x-v100-32g) 
on NVIDIA DGX-2 with (16x V100 32G) GPUs.

| GPUs | Time to train - mixed precision + XLA | Time to train - mixed precision | Time to train - FP32 |
|----|-------|--------|-------|
| 1  | ~61h  | ~103h  | ~247h |
| 16 | ~4.7h | ~7.5h  | ~19h   | 



#### Inference performance results

##### Inference performance: NVIDIA DGX-1 (1x V100 16G)

Our results were obtained by running the `inference_benchmark.sh` inferencing benchmarking script
in the [TensorFlow 20.03-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) NGC container 
on NVIDIA DGX-1 with (1x V100 16G) GPU.

**FP32 Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
|1	|54.71 img/s	|18.32 ms	|18.73 ms	|18.89 ms	|19.95 ms  |
|2	|98.36 img/s	|20.37 ms	|20.69 ms	|20.75 ms	|21.03 ms  |
|4	|150.60 img/s	|26.56 ms	|26.83 ms	|26.95 ms	|27.46 ms  |
|8	|235.17 img/s	|34.02 ms	|34.40 ms	|34.57 ms	|35.37 ms |
|16	|330.33 img/s	|48.43 ms	|48.91 ms	|49.22 ms	|49.79 ms |
|32	|393.96 img/s	|81.22 ms	|81.72 ms	|81.99 ms	|82.49 ms |
|64	|446.13 img/s	|143.54 ms	|144.37 ms	|144.74 ms	|145.93 ms |
|128	|490.61 img/s	|260.89 ms	|261.56 ms	|261.76 ms	|262.71 ms |

**Mixed Precision Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
|1	|92.56 img/s	|10.88 ms	|11.04 ms	|11.66 ms	|14.34 ms  |
|2	|180.61 img/s	|11.11 ms	|11.35 ms	|11.47 ms	|11.79 ms  |
|4	|354.41 img/s	|11.35 ms	|11.90 ms	|12.13 ms	|13.28 ms  |
|8	|547.79 img/s	|14.63 ms	|15.53 ms	|15.99 ms	|16.38 ms  |
|16	|772.41 img/s	|20.80 ms	|21.76 ms	|22.02 ms	|23.02 ms  |
|32	|965.89 img/s	|33.15 ms	|33.82 ms	|34.24 ms	|35.15 ms |
|64	|1086.99 img/s	|59.01 ms	|59.42 ms	|59.56 ms	|60.35 ms |
|128	|1162.59 img/s	|110.36 ms	|110.41 ms	|110.64 ms	|111.18 ms |

**Mixed Precision Inference Latency + XLA**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
|1	|84.98 img/s	|11.81 ms	|12.00 ms	|12.08 ms	|12.85 ms  |
|2	|150.35 img/s	|13.37 ms	|14.15 ms	|14.56 ms	|15.11 ms  |
|4	|288.89 img/s	|13.90 ms	|14.37 ms	|14.56 ms	|16.19 ms  |
|8	|526.94 img/s	|15.19 ms	|15.61 ms	|15.85 ms	|17.91 ms  |
|16	|818.86 img/s	|19.63 ms	|19.85 ms	|19.97 ms	|20.70 ms |
|32	|1134.72 img/s	|28.20 ms	|28.60 ms	|28.82 ms	|30.03 ms |
|64	|1359.55 img/s	|47.22 ms	|47.51 ms	|47.84 ms	|48.96 ms |
|128	|1515.12 img/s	|84.49 ms	|85.51 ms	|85.82 ms	|86.89 ms |

##### Inference performance: NVIDIA DGX-2 (1x V100 32G)

Our results were obtained by running the `inference_benchmark.sh` inferencing benchmarking script
in the [TensorFlow 20.03-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) NGC container 
on NVIDIA DGX-2 with (1x V100 32G) GPU.

**FP32 Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
|1	|59.41 img/s	|16.86 ms	|17.02 ms	|17.10 ms	|17.47 ms |
|2	|92.74 img/s	|21.59 ms	|21.88 ms	|22.01 ms	|22.65 ms |
|4	|141.53 img/s	|28.26 ms	|28.45 ms	|28.55 ms	|28.77 ms |
|8	|228.80 img/s	|34.96 ms	|35.18 ms	|35.38 ms	|35.72 ms |
|16	|324.11 img/s	|49.36 ms	|49.61 ms	|49.76 ms	|50.17 ms |
|32	|397.66 img/s	|80.47 ms	|80.69 ms	|80.82 ms	|82.15 ms |
|64	|468.28 img/s	|136.67 ms	|137.03 ms	|137.18 ms	|138.11 ms|
|128	|514.25 img/s	|248.91 ms	|250.42 ms	|251.89 ms	|253.55 ms|

**Mixed Precision Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
|1	|103.55 img/s	|9.70 ms	|9.94 ms	|10.03 ms	|10.26 ms |
|2	|227.06 img/s	|8.86 ms	|9.07 ms	|9.19 ms	|9.78 ms  |
|4	|380.22 img/s	|10.53 ms	|10.97 ms	|11.11 ms	|11.64 ms |
|8	|579.59 img/s	|13.82 ms	|14.07 ms	|14.29 ms	|14.99 ms |
|16	|833.08 img/s	|19.20 ms	|19.33 ms	|19.37 ms	|19.68 ms |
|32	|990.96 img/s	|32.30 ms	|32.53 ms	|32.70 ms	|33.45 ms |
|64	|1114.78 img/s	|57.41 ms	|57.61 ms	|57.86 ms	|58.78 ms |
|128	|1203.04 img/s	|106.40 ms	|106.54 ms	|106.62 ms	|107.93 ms|

**Mixed Precision Inference Latency + XLA**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
|1	|92.52 img/s	|10.85 ms	|10.95 ms	|11.12 ms	|11.99 ms |
|2	|177.54 img/s	|11.32 ms	|11.59 ms	|11.66 ms	|13.16 ms |
|4	|322.44 img/s	|12.44 ms	|12.63 ms	|12.71 ms	|12.98 ms |
|8	|548.33 img/s	|14.59 ms	|14.90 ms	|15.58 ms	|15.95 ms |
|16	|818.34 img/s	|19.55 ms	|19.75 ms	|19.91 ms	|20.30 ms |
|32	|1178.12 img/s	|27.16 ms	|27.41 ms	|27.53 ms	|28.24 ms|
|64	|1397.25 img/s	|45.96 ms	|46.03 ms	|46.19 ms	|47.01 ms|
|128	|1613.78 img/s	|79.32 ms	|80.02 ms	|80.46 ms	|81.71 ms|

##### Inference performance: NVIDIA T4 (1x T4 16G)

Our results were obtained by running the `inference_benchmark.sh` inferencing benchmarking script
in the [TensorFlow 20.03-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) NGC container 
on NVIDIA T4 with (1x T4 16G) GPU.

**FP32 Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
|1	|48.92 img/s	|20.47 ms	|20.75 ms	|20.88 ms	|21.23 ms    |
|2	|82.59 img/s	|24.24 ms	|24.42 ms	|24.50 ms	|24.64 ms    |
|4	|111.08 img/s	|36.03 ms	|36.37 ms	|36.45 ms	|36.67 ms   |
|8	|131.84 img/s	|60.68 ms	|61.36 ms	|61.62 ms	|62.21 ms   |
|16	|144.27 img/s	|110.90 ms	|112.04 ms	|112.29 ms	|112.80 ms   |
|32	|156.59 img/s	|204.35 ms	|206.12 ms	|206.78 ms	|208.08 ms   |
|64	|162.58 img/s	|393.66 ms	|396.45 ms	|396.94 ms	|397.52 ms  |
|128	|162.41 img/s	|788.13 ms	|790.86 ms	|791.47 ms	|792.43 ms  |

**Mixed Precision Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
|1	|79.57 img/s	|12.74 ms	|13.31 ms	|13.37 ms	|13.85 ms    |
|2	|190.25 img/s	|10.55 ms	|10.63 ms	|10.67 ms	|10.77 ms    |
|4	|263.72 img/s	|15.16 ms	|15.37 ms	|15.44 ms	|15.58 ms   |
|8	|312.07 img/s	|25.64 ms	|26.14 ms	|26.25 ms	|26.98 ms   |
|16	|347.43 img/s	|46.05 ms	|46.60 ms	|46.82 ms	|47.09 ms   |
|32	|360.20 img/s	|88.84 ms	|89.44 ms	|89.60 ms	|90.20 ms   |
|64	|367.23 img/s	|174.28 ms	|175.81 ms	|176.13 ms	|176.74 ms  |
|128	|362.43 img/s	|353.17 ms	|354.91 ms	|355.52 ms	|356.07 ms  |

**Mixed Precision Inference Latency + XLA**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
|1  |89.30 img/s	|11.24 ms	|11.31 ms	|11.38 ms	|11.75 ms  |
|2	|152.93 img/s	|13.11 ms	|13.20 ms	|13.25 ms	|13.58 ms  |
|4	|254.87 img/s	|15.69 ms	|15.84 ms	|15.87 ms	|15.95 ms  |
|8	|356.48 img/s	|22.44 ms	|22.79 ms	|22.86 ms	|23.08 ms |
|16	|442.24 img/s	|36.18 ms	|36.63 ms	|36.76 ms	|36.76 ms |
|32	|471.28 img/s	|67.90 ms	|68.62 ms	|68.80 ms	|69.14 ms |
|64	|483.18 img/s	|132.46 ms	|133.72 ms	|134.08 ms	|134.88 ms |
|128	|501.38 img/s	|255.31 ms	|258.46 ms	|259.19 ms	|260.17 ms|

## Release notes

### Changelog

April 2020
   - Initial release

### Known issues
There are no known issues with this model.

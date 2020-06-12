# ResNext101-32x4d for TensorFlow

This repository provides a script and recipe to train the ResNext101-32x4d model to achieve state-of-the-art accuracy, and is tested and maintained by NVIDIA.

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
            * [Inference performance: NVIDIA T4 (1x T4)](#inference-performance-nvidia-t4-1x-t4-16g)
* [Release notes](#release-notes)
    * [Changelog](#changelog)
    * [Known issues](#known-issues)

## Model overview
The ResNeXt101-32x4d is a model introduced in the [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf) paper.

It is based on a regular ResNet model, substituting 3x3 convolutions inside the bottleneck block for 3x3 grouped convolutions.

The following performance optimizations were implemented in this model:
* JIT graph compilation with [XLA](https://www.tensorflow.org/xla)
* Multi-GPU training with [Horovod](https://github.com/horovod/horovod)
* Automated mixed precision [AMP](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)

This model is trained with mixed precision using Tensor Cores on NVIDIA Volta and Turing GPUs. Therefore, researchers can get results 3x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.

### Model architecture

![ResNextArch](./imgs/ResNeXtArch.png)

_Image source: [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf)_

Image shows difference between ResNet bottleneck block and ResNeXt bottleneck block.
ResNeXt bottleneck block splits single convolution into multiple smaller, parallel convolutions.

ResNeXt101-32x4d model's cardinality equals 32 and bottleneck width equals 4. This means instead of single convolution with 64 filters 
32 parallel convolutions with only 4 filters are used.


### Default configuration

The following sections highlight the default configuration for the ResNext101-32x4d model.

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

| Feature               | ResNext101-32x4d Tensorflow             |
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

The following section lists the requirements that you need to meet in order to use the ResNext101-32x4d model.

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
To train your model using mixed precision with Tensor Cores or FP32, perform the following steps using the default parameters of the ResNext101-32x4d model on the [ImageNet](http://www.image-net.org/) dataset. For the specifics concerning training and inference, see the [Advanced](#advanced) section.


1. Clone the repository.
```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/TensorFlow/Classification/RN50v1.5
```

2. Download and preprocess the dataset.
The ResNext101-32x4d script operates on ImageNet 1k, a widely popular image classification dataset from the ILSVRC challenge.

To download and preprocess the dataset, use the [Generate ImageNet for TensorFlow](https://github.com/tensorflow/models/blob/master/research/inception/inception/data/download_and_preprocess_imagenet.sh) script. The dataset will be downloaded to a directory specified as the first parameter of the script.

3. Build the ResNext101-32x4d TensorFlow NGC container.
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
one of the scripts in the `resnext101-32x4d/training` directory. Ensure ImageNet is mounted in the
`/data/tfrecords` directory.

For example, to train on DGX-1 for 90 epochs using AMP, run: 

`bash ./resnext101-32x4d/training/AMP/DGX1_RNxt101-32x4d_AMP_90E.sh`

Additionally, features like DALI data preprocessing or TensorFlow XLA can be enabled with
environmental variables when running those scripts:

`USE_XLA=1 USE_DALI=1 bash ./resnext101-32x4d/training/AMP/DGX1_RNxt101-32x4d_AMP_90E.sh`

To store results in a specific location, add a location as a first argument:

`bash ./resnext101-32x4d/training/AMP/DGX1_RNxt101-32x4d_AMP_90E.sh <location to store>`

7. Start validation/evaluation.
To evaluate the validation dataset located in `/data/tfrecords`, run `main.py` with
`--mode=evaluate`. For example:

`python main.py --arch=resnext101-32x4d --mode=evaluate --data_dir=/data/tfrecords --batch_size <batch size> --model_dir
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
                        Architecture of model to run (to run Resnext-32x4d set
                        --arch=rensext101-32x4d)
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

`python main.py --arch=resnext101-32x4d --mode predict --model_dir <path to model> --to_predict <path to image> --results_dir <path to results>`

The optional `--use_xla` and `--use_tf_amp` flags control XLA and AMP during inference.

## Performance

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in training and inference modes.

#### Training performance benchmark

To benchmark the training performance on a specific batch size, run:

* For 1 GPU
    * FP32
        `python ./main.py --arch=resnext101-32x4d --mode=training_benchmark --warmup_steps 200 --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
    * FP16
        `python ./main.py --arch=resnext101-32x4d --mode=training_benchmark  --use_tf_amp --warmup_steps 200 --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
* For multiple GPUs
    * FP32
        `mpiexec --allow-run-as-root --bind-to socket -np <num_gpus> python ./main.py --arch=resnext101-32x4d --mode=training_benchmark --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
    * FP16
        `mpiexec --allow-run-as-root --bind-to socket -np <num_gpus> python ./main.py --arch=resnext101-32x4d --mode=training_benchmark --use_tf_amp --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
        
Each of these scripts runs 200 warm-up iterations and measures the first epoch.

To control warmup and benchmark length, use the `--warmup_steps`, `--num_iter` and `--iter_unit` flags. Features like XLA or DALI can be controlled
with `--use_xla` and `--use_dali` flags.
Suggested batch sizes for training are 128 for mixed precision training and 64 for single precision training per single V100 16 GB.


#### Inference performance benchmark

To benchmark the inference performance on a specific batch size, run:

* FP32
`python ./main.py --arch=resnext101-32x4d --mode=inference_benchmark --warmup_steps 20 --num_iter 100 --iter_unit batch --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`

* FP16
`python ./main.py --arch=resnext101-32x4d --mode=inference_benchmark --use_tf_amp --warmup_steps 20 --num_iter 100 --iter_unit batch --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`

By default, each of these scripts runs 20 warm-up iterations and measures the next 80 iterations.
To control warm-up and benchmark length, use the `--warmup_steps`, `--num_iter` and `--iter_unit` flags.

The benchmark can be automated with the `inference_benchmark.sh` script provided in `resnext101-32x4d`, by simply running:
`bash ./resnext101-32x4d/inference_benchmark.sh <data dir> <data idx dir>`

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
| 90   | 64 (FP32) / 128 (AMP) | 79.34              | 79.31   |
| 250  | 64 (FP32) / 128 (AMP) | 80.21              | 80.21   |

**Example training loss plot**

![TrainingLoss](./imgs/train_loss.png)

#### Training performance results

##### Training performance: NVIDIA DGX-1 (8x V100 16G)
Our results were obtained by running the steps from [Training performance benchmark](#training-performance-benchmark) in the 
[TensorFlow 20.03-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)  NGC container 
on NVIDIA DGX-1 with (8x V100 16G) GPUs. Performance numbers (in images per second) were averaged over an entire training epoch.


| GPUs | Batch Size / GPU | Throughput - FP32 | Throughput - mixed precision | Throughput speedup (FP32 - mixed precision) | Weak scaling - FP32 | Weak scaling - mixed precision |
|----|---------------|---------------|------------------------|-----------------|-----------|-------------------|
| 1  | 64 (FP32) / 128 (AMP) | 142.10 img/s  | 423.19 img/s   |  2.97x          | 1.00x     | 1.00x             |
| 8  | 64 (FP32) / 128 (AMP) | 1055.82 img/s | 3151.81 img/s  |  2.98x          | 7.43x     | 7.44x             |

**XLA Enabled**

| GPUs | Batch Size / GPU | Throughput - mixed precision | Throughput - mixed precision + XLA | Throughput speedup (mixed precision - XLA) |
|----|------------|---------------|---------------------|-----------|
| 1  | 128        | 423.19 img/s  | 588.49 img/s        | 1.39x    |
| 8  | 128        | 3151.81 img/s | 4231.42 img/s       | 1.34x    |

##### Training performance: NVIDIA DGX-2 (16x V100 32G)
Our results were obtained by running the steps from [Training performance benchmark](#training-performance-benchmark) in the 
[TensorFlow 20.03-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)  NGC container 
on NVIDIA DGX-2 with (16x V100 32G) GPUs. Performance numbers (in images per second) were averaged over an entire training epoch.

| GPUs | Batch Size / GPU | Throughput - FP32 | Throughput - mixed precision | Throughput speedup (FP32 - mixed precision) | Weak scaling - FP32 | Weak scaling - mixed precision |
|----|---------------|---------------|-------------------------|-------|--------|--------|
| 1  | 64 (FP32) / 128 (AMP) | 148.19 img/s  | 403.13 img/s    | 2.72x | 1.00x  | 1.00x  |
| 16 | 64 (FP32) / 128 (AMP) | 1961.31 img/s | 5601.13 img/s   | 2.86x | 13.23x | 13.89x |

**XLA Enabled**

| GPUs | Batch Size / GPU | Throughput - mixed precision | Throughput - mixed precision + XLA | Throughput speedup (mixed precision - XLA) |
|----|-----|----------|---------------------|-----------|
| 1  | 128 | 403.13 img/s   | 555.33 img/s  |1.13x      |
| 16 | 128 | 5601.13 img/s  | 7617.25 img/s |1.36x      |

#### Training Time for 90 Epochs

##### Training time: NVIDIA DGX-1 (8x V100 16G)

Our results were estimated based on the [training performance results](#training-performance-nvidia-dgx-1-8x-v100-16g) 
on NVIDIA DGX-1 with (8x V100 16G) GPUs.

| GPUs | Time to train - mixed precision + XLA | Time to train - mixed precision | Time to train - FP32 |
|---|--------|---------|---------|
| 1 | ~54h   |  ~75h   |  ~225h  |
| 8 | ~7.5h  |  ~10h   |  ~30h   | 

##### Training time: NVIDIA DGX-2 (16x V100 32G)

Our results were estimated based on the [training performance results](#training-performance-nvidia-dgx-2-16x-v100-32g) 
on NVIDIA DGX-2 with (16x V100 32G) GPUs.

| GPUs | Time to train - mixed precision + XLA | Time to train - mixed precision | Time to train - FP32 |
|----|-------|--------|-------|
| 1  | ~57h  | ~79h   | ~216h |
| 16 | ~4.2h | ~6h    | ~16h  | 



#### Inference performance results

##### Inference performance: NVIDIA DGX-1 (1x V100 16G)

Our results were obtained by running the `inference_benchmark.sh` inferencing benchmarking script
in the [TensorFlow 20.03-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) NGC container 
on NVIDIA DGX-1 with (1x V100 16G) GPU.

**FP32 Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
|1	|61.19 img/s	|16.36 ms	|16.66 ms	|16.87 ms	|17.31 ms |
|2	|120.52 img/s	|16.60 ms   |16.91 ms	|17.00 ms	|17.60 ms |
|4	|179.63 img/s	|22.26 ms	|22.44 ms	|22.50 ms	|22.73 ms |
|8	|287.94 img/s	|27.78 ms	|27.97 ms	|28.08 ms	|28.30 ms |
|16	|403.04 img/s	|39.72 ms	|39.93 ms	|40.01 ms	|40.29 ms |
|32	|463.61 img/s	|69.03 ms	|69.68 ms	|70.99 ms	|71.48 ms |
|64	|530.00 img/s	|120.75 ms	|121.12 ms	|121.38 ms	|123.17 ms |
|128	|570.60 img/s	|224.32 ms	|224.84 ms	|224.98 ms	|225.72 ms |

**Mixed Precision Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
|1	|164.92 img/s	|6.10 ms	|6.17 ms	|6.26 ms	|7.73 ms  |
|2	|326.59 img/s	|6.14 ms	|6.32 ms	|6.39 ms	|6.62 ms  |
|4	|607.20 img/s	|6.60 ms	|6.77 ms	|6.88 ms	|8.08 ms  |
|8	|892.31 img/s	|8.97 ms	|9.13 ms	|9.49 ms	|9.86 ms  |
|16	|1259.92 img/s	|12.82 ms	|13.31 ms	|13.44 ms	|13.58 ms |
|32	|1508.73 img/s	|31.30 ms	|21.70 ms	|21.86 ms	|22.02 ms |
|64	|1618.77 img/s	|39.55 ms	|40.71 ms	|41.33 ms	|41.94 ms |
|128	|1730.40 img/s	|73.98 ms	|74.27 ms	|76.01 ms	|76.74 ms |

**Mixed Precision Inference Latency + XLA**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
|1	|97.24 img/s	|10.31 ms	|10.48 ms	|10.57 ms	|10.81 ms  |
|2	|191.68 img/s	|10.44 ms	|10.74 ms	|10.84 ms	|11.42 ms  |
|4	|381.19 img/s	|10.50 ms	|10.85 ms	|10.98 ms	|11.74 ms  |
|8	|744.11 img/s	|10.77 ms	|11.42 ms	|11.85 ms	|12.44 ms  |
|16	|1174.29 img/s	|13.83 ms	|13.87 ms	|14.29 ms	|15.53 ms |
|32	|1439.07 img/s	|22.33 ms	|22.67 ms	|22.84 ms	|23.06 ms |
|64	|1712.76 img/s	|37.37 ms	|37.91 ms	|38.09 ms	|38.74 ms |
|128	|1883.71 img/s	|67.95 ms	|68.48 ms	|68.63 ms	|68.86 ms |

##### Inference performance: NVIDIA DGX-2 (1x V100 32G)

Our results were obtained by running the `inference_benchmark.sh` inferencing benchmarking script
in the [TensorFlow 20.03-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) NGC container 
on NVIDIA DGX-2 with (1x V100 32G) GPU.

**FP32 Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
|1	|62.02 img/s	|16.22 ms	|17.62 ms	|17.92 ms	|19.21 ms |
|2	|97.98 img/s	|20.54 ms	|20.59 ms	|20.72 ms	|23.21 ms |
|4	|168.16 img/s	|23.79 ms	|24.12 ms	|24.24 ms	|26.94 ms |
|8	|269.89 img/s	|29.66 ms	|30.01 ms	|30.35 ms	|34.05 ms|
|16	|379.81 img/s	|42.14 ms	|42.47 ms	|42.85 ms	|47.63 ms|
|32	|466.04 img/s	|68.67 ms	|68.99 ms	|69.26 ms	|74.87 ms|
|64	|547.64 img/s	|117.01 ms	|117.59 ms	|118.37 ms	|122.83 ms|
|128	|603.44 img/s	|212.21 ms	|212.92 ms	|214.09 ms	|217.06 ms|

**Mixed Precision Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
|1	|198.53 img/s	|5.14 ms	|5.23 ms	|5.41 ms	|5.54 ms |
|2	|343.00 img/s	|6.14 ms	|6.08 ms	|6.26 ms	|7.72 ms |
|4	|592.25 img/s	|6.77 ms	|7.06 ms	|7.18 ms	|8.70 ms |
|8	|918.45 img/s	|8.72 ms	|8.90 ms	|9.09 ms	|9.77 ms |
|16	|1306.53 img/s	|12.60 ms	|12.65 ms	|12.91 ms	|17.06 ms |
|32	|1483.83 img/s	|21.56 ms	|21.61 ms	|21.84 ms	|27.05 ms|
|64	|1668.63 img/s	|38.39 ms	|38.50 ms	|40.15 ms	|43.15 ms|
|128	|1748.25 img/s	|73.35 ms	|75.23 ms	|78.82 ms	|80.17 ms|

**Mixed Precision Inference Latency + XLA**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
|1	|143.65 img/s	|6.97 ms	|7.15 ms	|7.24 ms	|7.95 ms |
|2	|282.21 img/s	|7.09 ms	|7.32 ms	|7.56 ms	|7.97 ms |
|4	|511.55 img/s	|7.85 ms	|8.42 ms	|8.62 ms	|9.02 ms |
|8	|870.60 img/s	|9.23 ms	|9.46 ms	|9.54 ms	|9.88 ms |
|16	|1179.93 img/s	|13.62 ms	|14.04 ms	|14.19 ms	|14.51 ms|
|32	|1512.36 img/s	|21.19 ms	|21.70 ms	|21.80 ms	|22.04 ms|
|64	|1805.38 img/s	|35.56 ms	|36.33 ms	|36.48 ms	|36.94 ms|
|128	|1947.49 img/s	|65.88 ms	|66.50 ms	|66.72 ms	|67.17 ms|

##### Inference performance: NVIDIA T4 (1x T4 16G)

Our results were obtained by running the `inference_benchmark.sh` inferencing benchmarking script
in the [TensorFlow 20.03-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) NGC container 
on NVIDIA T4 with (1x T4 16G) GPU.

**FP32 Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
|1	|54.23 img/s	|18.48 ms	|19.62 ms	|19.78 ms	|20.13 ms   |
|2	|94.22 img/s	|21.24 ms	|21.58 ms	|21.71 ms	|21.97 ms   |
|4	|127.71 img/s	|31.33 ms	|31.90 ms	|32.10 ms	|32.50 ms   |
|8	|151.88 img/s	|52.67 ms	|53.45 ms	|53.80 ms	|54.12 ms   |
|16	|163.01 img/s	|98.16 ms	|99.52 ms	|99.94 ms	|100.49 ms  |
|32	|176.13 img/s	|181.71 ms	|183.91 ms	|184.54 ms	|185.60 ms  |
|64	|183.40 img/s	|349.00 ms	|352.65 ms	|353.55 ms	|355.03 ms  |
|128	|182.77 img/s	|700.35 ms	|707.89 ms	|708.80 ms	|710.28 ms  |


**Mixed Precision Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
|1	|157.87 img/s	|6.36 ms	|6.47 ms	|6.52 ms	|6.64 ms    |
|2	|274.10 img/s	|7.29 ms	|7.41 ms	|7.45 ms	|7.51 ms    |
|4	|395.41 img/s	|10.12 ms	|10.35 ms	|10.41 ms	|10.53 ms   |
|8	|479.83 img/s	|16.68 ms	|16.92 ms	|17.01 ms	|17.15 ms   |
|16	|525.83 img/s	|30.47 ms	|30.80 ms	|30.89 ms	|31.27 ms   |
|32	|536.31 img/s	|59.67 ms	|60.35 ms	|60.51 ms	|60.96 ms   |
|64	|541.26 img/s	|118.25 ms	|119.51 ms	|119.77 ms	|120.38 ms  |
|128	|538.20 img/s	|237.84 ms	|240.41 ms	|240.82 ms	|241.72 ms  |


**Mixed Precision Inference Latency + XLA**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
|1  |104.10 img/s	|9.63 ms	|9.75 ms	|9.78 ms	|9.86 ms  |
|2	|220.23 img/s	|9.08 ms	|9.22 ms	|9.26 ms	|9.35 ms  |
|4	|361.55 img/s	|11.06 ms	|11.19 ms	|11.29 ms	|11.68 ms |
|8	|452.95 img/s	|17.66 ms	|17.92 ms	|18.00 ms	|18.12 ms |
|16	|522.64 img/s	|30.65 ms	|30.92 ms	|31.04 ms	|31.36 ms |
|32	|542.06 img/s	|59.03 ms	|59.63 ms	|59.77 ms	|60.25 ms |
|64	|536.14 img/s	|119.37 ms	|120.31 ms	|120.68 ms	|121.39 ms |
|128	|548.43 img/s	|233.50 ms	|234.83 ms	|235.31 ms	|236.29 ms|

## Release notes

### Changelog

June 2020
   - Initial release

### Known issues
There are no known issues with this model.

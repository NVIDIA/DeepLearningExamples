# ResNext101-32x4d for TensorFlow

This repository provides a script and recipe to train the ResNext101-32x4d model to achieve state-of-the-art accuracy, and is tested and maintained by NVIDIA.
ResNext101-32x4d model for TensorFlow1 is no longer maintained and will soon become unavailable, please consider PyTorch or TensorFlow2 models as a substitute for your requirements.

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
        * [Enabling TF32](#enabling-tf32)
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
            * [Training accuracy: NVIDIA DGX A100 (8x A100 40GB)](#training-accuracy-nvidia-dgx-a100-8x-a100-40gb)
            * [Training accuracy: NVIDIA DGX-1 (8x V100 16G)](#training-accuracy-nvidia-dgx-1-8x-v100-16g)
        * [Training performance results](#training-performance-results)
            * [Training performance: NVIDIA DGX A100 (8x A100 40GB)](#training-performance-nvidia-dgx-a100-8x-a100-40gb) 
            * [Training performance: NVIDIA DGX-1 (8x V100 16G)](#training-performance-nvidia-dgx-1-8x-v100-16g)
            * [Training performance: NVIDIA DGX-2 (16x V100 32G)](#training-performance-nvidia-dgx-2-16x-v100-32g)
        * [Training time for 90 Epochs](#training-time-for-90-epochs)
            * [Training time: NVIDIA DGX A100 (8x A100 40G)](#training-time-nvidia-dgx-a100-8x-a100-40gb)
            * [Training time: NVIDIA DGX-1 (8x V100 16G)](#training-time-nvidia-dgx-1-8x-v100-16g)
            * [Training time: NVIDIA DGX-2 (16x V100 32G)](#training-time-nvidia-dgx-2-16x-v100-32g)
        * [Inference performance results](#inference-performance-results)
            * [Inference performance: NVIDIA DGX A100 (1x A100 40GB)](#inference-performance-nvidia-dgx-a100-1x-a100-40gb)
            * [Inference performance: NVIDIA DGX-1 (1x V100 16G)](#inference-performance-nvidia-dgx-1-1x-v100-16g)
            * [Inference performance: NVIDIA DGX-2 (1x V100 32G)](#inference-performance-nvidia-dgx-2-1x-v100-32g)
            * [Inference performance: NVIDIA T4 (1x T4 16G)](#inference-performance-nvidia-t4-1x-t4-16g)
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

This model is trained with mixed precision using Tensor Cores on Volta, Turing, and the NVIDIA Ampere GPU architectures. Therefore, researchers can get results 3x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.

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
    * 90 Epochs -> 90 epochs is a standard for ResNet family networks.
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
Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in Volta, and following with both the Turing and Ampere architectures, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using [mixed precision training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) previously required two steps:

1.  Porting the model to use the FP16 data type where appropriate.    
2.  Adding loss scaling to preserve small gradient values.

This can now be achieved using Automatic Mixed Precision (AMP) for TensorFlow to enable the full [mixed precision methodology](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#tensorflow) in your existing TensorFlow model code.  AMP enables mixed precision training on Volta and Turing GPUs automatically. The TensorFlow framework code makes all necessary model changes internally.

In TF-AMP, the computational graph is optimized to use as few casts as necessary and maximize the use of FP16, and the loss scaling is automatically applied inside of supported optimizers. AMP can be configured to work with the existing tf.contrib loss scaling manager by disabling the AMP scaling with a single environment variable to perform only the automatic mixed-precision optimization. It accomplishes this by automatically rewriting all computation graphs with the necessary operations to enable mixed precision training and automatic loss scaling.

For information about:
-   How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) documentation.
-   Techniques used for mixed precision training, see the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
-   How to access and enable AMP for TensorFlow, see [Using TF-AMP](https://docs.nvidia.com/deeplearning/dgx/tensorflow-user-guide/index.html#tfamp) from the TensorFlow User Guide.


#### Enabling mixed precision

Mixed precision is enabled in TensorFlow by using the Automatic Mixed Precision (TF-AMP) extension which casts variables to half-precision upon retrieval, while storing variables in single-precision format. Furthermore, to preserve small gradient magnitudes in backpropagation, a [loss scaling](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#lossscaling) step must be included when applying gradients. In TensorFlow, loss scaling can be applied statically by using simple multiplication of loss by a constant value or automatically, by TF-AMP. Automatic mixed precision makes all the adjustments internally in TensorFlow, providing two benefits over manual operations. First, programmers need not modify network model code, reducing development and maintenance effort. Second, using AMP maintains forward and backward compatibility with all the APIs for defining and running TensorFlow models.

To enable mixed precision, you can simply add the values to the environmental variables inside your training script:
- Enable TF-AMP graph rewrite:
  ```
  os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
  ```
  
- Enable Automated Mixed Precision:
  ```
  os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

  ```

#### Enabling TF32

TensorFloat-32 (TF32) is the new math mode in [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) GPUs for handling the matrix math also called tensor operations. TF32 running on Tensor Cores in A100 GPUs can provide up to 10x speedups compared to single-precision floating-point math (FP32) on Volta GPUs. 

TF32 Tensor Cores can speed up networks using FP32, typically with no loss of accuracy. It is more robust than FP16 for models which require high dynamic range for weights or activations.

For more information, refer to the [TensorFloat-32 in the A100 GPU Accelerates AI Training, HPC up to 20x](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) blog post.

TF32 is supported in the NVIDIA Ampere GPU architecture and is enabled by default.


## Setup

The following section lists the requirements that you need to meet in order to use the ResNext101-32x4d model.

### Requirements
This repository contains Dockerfile which extends the TensorFlow NGC container and encapsulates all dependencies.  Aside from these dependencies, ensure you have the following software:

- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [TensorFlow 20.06-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)
- GPU-based architecture:
  - [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
  - [NVIDIA Turing](https://www.nvidia.com/en-us/geforce/turing/)
  - [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)


For more information about how to get started with NGC containers, see the
following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
* [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html),
* [Accessing And Pulling From The NGC container registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry),
* [Running TensorFlow](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html#running).

For those unable to use the [TensorFlow NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## Quick Start Guide
To train your model using mixed precision or TF32 with Tensor Cores or FP32, perform the following steps using the default parameters of the ResNext101-32x4d model on the [ImageNet](http://www.image-net.org/) dataset. For the specifics concerning training and inference, see the [Advanced](#advanced) section.


1. Clone the repository.
```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/TensorFlow/Classification/ConvNets
```

2. Download and preprocess the dataset.
The ResNext101-32x4d script operates on ImageNet 1k, a widely popular image classification dataset from the ILSVRC challenge.

* [Download the images](http://image-net.org/download-images)
* Extract the training and validation data:
```bash
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
```
* Preprocess dataset to TFRecord form using [script](https://github.com/tensorflow/models/blob/archive/research/inception/inception/data/build_imagenet_data.py). Additional metadata from [autors repository](https://github.com/tensorflow/models/tree/archive/research/inception/inception/data) might be required.

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

`bash ./resnext101-32x4d/training/DGX1_RNxt101-32x4d_AMP_90E.sh /path/to/result /data`

Additionally, features like DALI data preprocessing or TensorFlow XLA can be enabled with
following arguments when running those scripts:

`bash ./resnext101-32x4d/training/DGX1_RNxt101-32x4d_AMP_90E.sh /path/to/result /data --xla --dali`

7. Start validation/evaluation.
To evaluate the validation dataset located in `/data/tfrecords`, run `main.py` with
`--mode=evaluate`. For example:

`python main.py --arch=resnext101-32x4d --mode=evaluate --data_dir=/data/tfrecords --batch_size <batch size> --model_dir
<model location> --results_dir <output location> [--xla] [--amp]`

The optional `--xla` and `--amp` flags control XLA and AMP during evaluation. 

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
usage: main.py [-h] [--arch {resnet50,resnext101-32x4d,se-resnext101-32x4d}]
               [--mode {train,train_and_evaluate,evaluate,predict,training_benchmark,inference_benchmark}]
               [--export_dir EXPORT_DIR] [--to_predict TO_PREDICT]       
               --batch_size BATCH_SIZE [--num_iter NUM_ITER]  
               [--run_iter RUN_ITER] [--iter_unit {epoch,batch}]              
               [--warmup_steps WARMUP_STEPS] [--model_dir MODEL_DIR]
               [--results_dir RESULTS_DIR] [--log_filename LOG_FILENAME]      
               [--display_every DISPLAY_EVERY] [--seed SEED]
               [--gpu_memory_fraction GPU_MEMORY_FRACTION] [--gpu_id GPU_ID]
               [--finetune_checkpoint FINETUNE_CHECKPOINT] [--use_final_conv]
               [--quant_delay QUANT_DELAY] [--quantize] [--use_qdq]        
               [--symmetric] [--data_dir DATA_DIR]         
               [--data_idx_dir DATA_IDX_DIR] [--dali]
               [--synthetic_data_size SYNTHETIC_DATA_SIZE] [--lr_init LR_INIT]
               [--lr_warmup_epochs LR_WARMUP_EPOCHS] 
               [--weight_decay WEIGHT_DECAY] [--weight_init {fan_in,fan_out}]
               [--momentum MOMENTUM] [--label_smoothing LABEL_SMOOTHING]
               [--mixup MIXUP] [--cosine_lr] [--xla]            
               [--data_format {NHWC,NCHW}] [--amp]
               [--static_loss_scale STATIC_LOSS_SCALE]
                                                            
JoC-RN50v1.5-TF                      
                                                                           
optional arguments:          
  -h, --help            show this help message and exit.
  --arch {resnet50,resnext101-32x4d,se-resnext101-32x4d}
                        Architecture of model to run.                           
  --mode {train,train_and_evaluate,evaluate,predict,training_benchmark,inference_benchmark}
                        The execution mode of the script.
  --export_dir EXPORT_DIR                                                                                                                                                                                                                                                  
                        Directory in which to write exported SavedModel.         
  --to_predict TO_PREDICT        
                        Path to file or directory of files to run prediction
                        on.
  --batch_size BATCH_SIZE      
                        Size of each minibatch per GPU.                    
  --num_iter NUM_ITER   Number of iterations to run.
  --run_iter RUN_ITER   Number of training iterations to run on single run.
  --iter_unit {epoch,batch}                                
                        Unit of iterations.                                  
  --warmup_steps WARMUP_STEPS                                    
                        Number of steps considered as warmup and not taken
                        into account for performance measurements.                                  
  --model_dir MODEL_DIR                
                        Directory in which to write model. If undefined,         
                        results dir will be used.                                                  
  --results_dir RESULTS_DIR
                        Directory in which to write training logs, summaries
                        and checkpoints.
  --log_filename LOG_FILENAME
                        Name of the JSON file to which write the training log.
  --display_every DISPLAY_EVERY
                        How often (in batches) to print out running
                        information.
  --seed SEED           Random seed.
  --gpu_memory_fraction GPU_MEMORY_FRACTION
                        Limit memory fraction used by training script for DALI.
  --gpu_id GPU_ID       Specify ID of the target GPU on multi-device platform.
                        Effective only for single-GPU mode.
  --finetune_checkpoint FINETUNE_CHECKPOINT
                        Path to pre-trained checkpoint which will be used for
                        fine-tuning.
  --use_final_conv      Use convolution operator instead of MLP as last layer.
  --quant_delay QUANT_DELAY
                        Number of steps to be run before quantization starts
                        to happen.
  --quantize            Quantize weights and activations during training.
                        (Defaults to Assymmetric quantization)
  --use_qdq             Use QDQV3 op instead of FakeQuantWithMinMaxVars op for
                        quantization. QDQv3 does only scaling.
  --symmetric           Quantize weights and activations during training using
                        symmetric quantization.

Dataset arguments:
  --data_dir DATA_DIR   Path to dataset in TFRecord format. Files should be
                        named 'train-*' and 'validation-*'.
  --data_idx_dir DATA_IDX_DIR
                        Path to index files for DALI. Files should be named
                        'train-*' and 'validation-*'.
  --dali                Enable DALI data input.
  --synthetic_data_size SYNTHETIC_DATA_SIZE
                        Dimension of image for synthetic dataset.

Training arguments:
  --lr_init LR_INIT     Initial value for the learning rate.
  --lr_warmup_epochs LR_WARMUP_EPOCHS
                        Number of warmup epochs for learning rate schedule.
  --weight_decay WEIGHT_DECAY
                        Weight Decay scale factor.
  --weight_init {fan_in,fan_out}
                        Model weight initialization method.
  --momentum MOMENTUM   SGD momentum value for the Momentum optimizer.
  --label_smoothing LABEL_SMOOTHING
                        The value of label smoothing.
  --mixup MIXUP         The alpha parameter for mixup (if 0 then mixup is not
                        applied).
  --cosine_lr           Use cosine learning rate schedule.

Generic optimization arguments:
  --xla                 Enable XLA (Accelerated Linear Algebra) computation
                        for improved performance.
  --data_format {NHWC,NCHW}
                        Data format used to do calculations.
  --amp                 Enable Automatic Mixed Precision to speedup
                        computation using tensor cores.

Automatic Mixed Precision arguments:
  --static_loss_scale STATIC_LOSS_SCALE
                        Use static loss scaling in FP32 AMP.

```

### Inference process
To run inference on a single example with a checkpoint and a model script, use: 

`python main.py --arch=resnext101-32x4d --mode predict --model_dir <path to model> --to_predict <path to image> --results_dir <path to results>`

The optional `--xla` and `--amp` flags control XLA and AMP during inference.

## Performance

The performance measurements in this document were conducted at the time of publication and may not reflect the performance achieved from NVIDIA’s latest software release. For the most up-to-date performance measurements, go to [NVIDIA Data Center Deep Learning Product Performance](https://developer.nvidia.com/deep-learning-performance-training-inference).

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in training and inference modes.

#### Training performance benchmark

To benchmark the training performance on a specific batch size, run:

* For 1 GPU
    * FP32 / TF32

        `python ./main.py --arch=resnext101-32x4d --mode=training_benchmark --warmup_steps 200 --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
    * AMP

        `python ./main.py --arch=resnext101-32x4d --mode=training_benchmark  --amp --warmup_steps 200 --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
* For multiple GPUs
    * FP32 / TF32

        `mpiexec --allow-run-as-root --bind-to socket -np <num_gpus> python ./main.py --arch=resnext101-32x4d --mode=training_benchmark --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
    * AMP

        `mpiexec --allow-run-as-root --bind-to socket -np <num_gpus> python ./main.py --arch=resnext101-32x4d --mode=training_benchmark --amp --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
        
Each of these scripts runs 200 warm-up iterations and measures the first epoch.

To control warmup and benchmark length, use the `--warmup_steps`, `--num_iter` and `--iter_unit` flags. Features like XLA or DALI can be controlled
with `--xla` and `--dali` flags. For proper throughput reporting the value of `--num_iter` must be greater than `--warmup_steps` value.
Suggested batch sizes for training are 128 for mixed precision training and 64 for single precision training per single V100 16 GB.

If no `--data_dir=<path to imagenet>` flag is specified then the benchmarks will use a synthetic dataset. The resolution of synthetic images used can be controlled with `--synthetic_data_size` flag.

#### Inference performance benchmark

To benchmark the inference performance on a specific batch size, run:

* FP32 / TF32

`python ./main.py --arch=resnext101-32x4d --mode=inference_benchmark --warmup_steps 20 --num_iter 100 --iter_unit batch --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`

* AMP

`python ./main.py --arch=resnext101-32x4d --mode=inference_benchmark --amp --warmup_steps 20 --num_iter 100 --iter_unit batch --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`

By default, each of these scripts runs 20 warm-up iterations and measures the next 80 iterations.
To control warm-up and benchmark length, use the `--warmup_steps`, `--num_iter` and `--iter_unit` flags.
If no `--data_dir=<path to imagenet>` flag is specified then the benchmarks will use a synthetic dataset.

The benchmark can be automated with the `inference_benchmark.sh` script provided in `resnext101-32x4d`, by simply running:
`bash ./resnext101-32x4d/inference_benchmark.sh <data dir> <data idx dir>`

The `<data dir>` parameter refers to the input data directory (by default `/data/tfrecords` inside the container). 
By default, the benchmark tests the following configurations: **FP32**, **AMP**, **AMP + XLA** with different batch sizes.
When the optional directory with the DALI index files `<data idx dir>` is specified, the benchmark executes an additional **DALI + AMP + XLA** configuration.
For proper throughput reporting the value of `--num_iter` must be greater than `--warmup_steps` value.

For performance benchamrk of raw model, synthetic dataset can be used. To use synthetic dataset, use `--synthetic_data_size` flag instead of `--data_dir` to specify input image size.

### Results

The following sections provide details on how we achieved our performance and accuracy in training and inference. 

#### Training accuracy results

##### Training accuracy: NVIDIA DGX A100 (8x A100 40GB)

Our results were obtained by running the `/resnet50v1.5/training/DGXA100_RN50_{PRECISION}_90E.sh` 
training script in the [TensorFlow 20.06-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) 
NGC container on NVIDIA DGX A100 (8x A100 40GB) GPUs.

| Epochs | Batch Size / GPU | Accuracy - TF32 (top1) | Accuracy - mixed precision (top1) | 
|--------|------------------|-----------------|----------------------------|
| 90     | 128 (TF32) / 256 (AMP) | 79.38           | 79.20                 |

##### Training accuracy: NVIDIA DGX-1 (8x V100 16G)
Our results were obtained by running the `/resnext101-32x4d/training/DGX1_RNxt101-32x4d_{PRECISION}_{EPOCHS}E.sh` 
training script in the [TensorFlow 20.06-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) 
NGC container on NVIDIA DGX-1 with (8x V100 16G) GPUs.

| Epochs | Batch Size / GPU | Accuracy - FP32 | Accuracy - mixed precision | 
|--------|------------------|-----------------|----------------------------|
| 90   | 64 (FP32) / 128 (AMP) | 79.35              | 79.30   |
| 250  | 64 (FP32) / 128 (AMP) | 80.21              | 80.21   |

**Example training loss plot**

![TrainingLoss](./imgs/train_loss.png)

#### Training performance results

##### Training performance: NVIDIA DGX A100 (8x A100 40GB)
Our results were obtained by running the `resnext101-32x4d/training/training_perf.sh` benchmark script in the 
[TensorFlow 20.06-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)  NGC container 
on NVIDIA DGX A100 (8x A100 40GB) GPUs. Performance numbers (in images per second) were averaged over an entire training epoch.


| GPUs | Batch Size / GPU | Throughput - TF32 + XLA | Throughput - mixed precision + XLA | Throughput speedup (TF32 - mixed precision) | Weak scaling - TF32 + XLA| Weak scaling - mixed precision + XLA |
|----|---------------|---------------|------------------------|-----------------|-----------|-------------------|
| 1  | 128 (TF) / 256 (AMP) | 371 img/s  | 1132 img/s    | 3.05x           | 1.00x     | 1.00x             |
| 8  | 128 (TF) / 256 (AMP) | 2854 img/s | 8500 img/s   | 2.98x           | 7.69x     | 7.51x             |


##### Training performance: NVIDIA DGX-1 (8x V100 16G)
Our results were obtained by running the `resnext101-32x4d/training/training_perf.sh` benchmark script in the 
[TensorFlow 20.06-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)  NGC container 
on NVIDIA DGX-1 with (8x V100 16G) GPUs. Performance numbers (in images per second) were averaged over an entire training epoch.


| GPUs | Batch Size / GPU | Throughput - FP32 | Throughput - mixed precision | Throughput speedup (FP32 - mixed precision) | Weak scaling - FP32 | Weak scaling - mixed precision |
|----|---------------|---------------|------------------------|-----------------|-----------|-------------------|
| 1  | 64 (FP32) / 128 (AMP) | 166 img/s  | 566 img/s   |  3.40x          | 1.00x     | 1.00x             |
| 8  | 64 (FP32) / 128 (AMP) | 1210 img/s  | 4160 img/s  |  3.44x          | 7.29x     | 7.35x             |

##### Training performance: NVIDIA DGX-2 (16x V100 32G)
Our results were obtained by running the `resnext101-32x4d/training/training_perf.sh` benchmark script in the 
[TensorFlow 20.06-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)  NGC container 
on NVIDIA DGX-2 with (16x V100 32G) GPUs. Performance numbers (in images per second) were averaged over an entire training epoch.

| GPUs | Batch Size / GPU | Throughput - FP32 | Throughput - mixed precision | Throughput speedup (FP32 - mixed precision) | Weak scaling - FP32 | Weak scaling - mixed precision |
|----|---------------|---------------|-------------------------|-------|--------|--------|
| 1  | 64 (FP32) / 128 (AMP) | 170 img/s  | 572 img/s    | 3.36x | 1.00x  | 1.00x  |
| 16 | 64 (FP32) / 128 (AMP) | 2500 img/s | 7750 img/s   | 3.10x | 14.70x  | 13.55x |

#### Training Time for 90 Epochs

##### Training time: NVIDIA DGX A100 (8x A100 40GB)

Our results were estimated based on the [training performance results](#training-performance-nvidia-dgx-a100-8x-a100-40g) 
on NVIDIA DGX A100 with (8x A100 40G) GPUs.

| GPUs | Time to train - mixed precision + XLA | Time to train - TF32 + XLA | 
|---|--------|---------|
| 1 | ~35h   | ~94h   |
| 8 | ~2h    | ~5h    |

##### Training time: NVIDIA DGX-1 (8x V100 16G)

Our results were estimated based on the [training performance results](#training-performance-nvidia-dgx-1-8x-v100-16g) 
on NVIDIA DGX-1 with (8x V100 16G) GPUs.

| GPUs | Time to train - mixed precision + XLA | Time to train - FP32 + XLA |
|---|--------|---------|
| 1 | ~56h   |  ~192h  |
| 8 | ~8h    |  ~27h   | 

##### Training time: NVIDIA DGX-2 (16x V100 32G)

Our results were estimated based on the [training performance results](#training-performance-nvidia-dgx-2-16x-v100-32g) 
on NVIDIA DGX-2 with (16x V100 32G) GPUs.

| GPUs | Time to train - mixed precision + XLA | Time to train - FP32 + XLA |
|----|-------|-------|
| 1  | ~55h  | ~188h |
| 16 | ~4h   | ~12h  | 


#### Inference performance results

##### Inference performance: NVIDIA DGX A100 (1x A100 40GB)

Our results were obtained by running the `inference_benchmark.sh` inferencing benchmarking script
in the [TensorFlow 20.06-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) NGC container 
on NVIDIA DGX A100 with (1x A100 40G) GPU.

**TF32 Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 111.07 img/s | 9.04 ms | 9.05 ms | 9.10 ms | 9.45 ms |
| 2 | 200.35 img/s | 10.01 ms | 10.05 ms | 10.08 ms | 10.24 ms |
| 4 | 283.11 img/s | 14.15 ms | 14.36 ms | 14.43 ms | 14.65 ms |
| 8 | 416.93 img/s | 19.19 ms | 19.64 ms | 19.90 ms | 20.14 ms |
| 16 | 629.64 img/s | 25.44 ms | 25.82 ms | 25.97 ms | 26.51 ms |
| 32 | 766.57 img/s | 41.83 ms | 42.30 ms | 42.65 ms | 43.45 ms |
| 64 | 836.72 img/s | 76.50 ms | 77.07 ms | 77.44 ms | 78.72 ms |
| 128 | 864.37 img/s | 148.27 ms | 148.54 ms | 148.93 ms | 149.62 ms |
| 256 | 902.67 img/s | 283.60 ms | 284.57 ms | 285.02 ms | 285.74 ms |

**TF32 Inference Latency + XLA**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 107.46 img/s | 9.34 ms | 9.36 ms | 9.40 ms | 9.95 ms |
| 2 | 192.54 img/s | 10.42 ms | 10.48 ms | 10.54 ms | 11.21 ms |
| 4 | 280.89 img/s | 14.26 ms | 14.41 ms | 14.53 ms | 14.94 ms |
| 8 | 387.41 img/s | 20.65 ms | 21.19 ms | 21.37 ms | 21.74 ms |
| 16 | 676.19 img/s | 23.67 ms | 24.34 ms | 24.55 ms | 25.61 ms |
| 32 | 902.44 img/s | 35.46 ms | 36.22 ms | 36.40 ms | 37.00 ms |
| 64 | 1028.06 img/s | 62.34 ms | 63.46 ms | 64.38 ms | 72.65 ms |
| 128 | 1096.39 img/s | 116.80 ms | 118.10 ms | 118.82 ms | 121.00 ms |
| 256 | 1153.50 img/s | 221.93 ms | 223.18 ms | 223.49 ms | 223.90 ms |


**Mixed Precision Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 127.96 img/s | 7.84 ms | 7.88 ms | 7.92 ms | 8.00 ms |
| 2 | 243.62 img/s | 8.24 ms | 8.28 ms | 8.31 ms | 8.58 ms |
| 4 | 491.02 img/s | 8.18 ms | 8.36 ms | 8.43 ms | 8.99 ms |
| 8 | 952.95 img/s | 8.40 ms | 8.80 ms | 8.94 ms | 9.31 ms |
| 16 | 1625.38 img/s | 9.85 ms | 10.19 ms | 10.45 ms | 10.86 ms |
| 32 | 1991.14 img/s | 16.22 ms | 16.46 ms | 16.78 ms | 17.59 ms |
| 64 | 2138.11 img/s | 30.08 ms | 31.02 ms | 31.34 ms | 32.27 ms |
| 128 | 2140.59 img/s | 59.81 ms | 61.37 ms | 61.77 ms | 62.53 ms |
| 256 | 2185.86 img/s | 117.12 ms | 118.35 ms | 118.72 ms | 119.84 ms |

**Mixed Precision Inference Latency + XLA**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 86.02 img/s | 11.66 ms | 11.78 ms | 11.82 ms | 12.18 ms |
| 2 | 166.91 img/s | 12.01 ms | 12.10 ms | 12.14 ms | 12.25 ms |
| 4 | 330.75 img/s | 12.10 ms | 12.45 ms | 12.87 ms | 13.27 ms |
| 8 | 675.53 img/s | 11.84 ms | 12.08 ms | 12.24 ms | 12.59 ms |
| 16 | 1234.52 img/s | 13.06 ms | 13.89 ms | 14.11 ms | 15.01 ms |
| 32 | 2501.78 img/s | 13.09 ms | 14.14 ms | 15.25 ms | 25.57 ms |
| 64 | 3049.35 img/s | 21.12 ms | 22.24 ms | 23.27 ms | 28.62 ms |
| 128 | 3324.24 img/s | 38.98 ms | 40.07 ms | 40.81 ms | 51.07 ms |
| 256 | 3166.28 img/s | 82.05 ms | 94.93 ms | 101.78 ms | 119.88 ms |

##### Inference performance: NVIDIA DGX-1 (1x V100 16G)

Our results were obtained by running the `inference_benchmark.sh` inferencing benchmarking script
in the [TensorFlow 20.06-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) NGC container 
on NVIDIA DGX-1 with (1x V100 16G) GPU.

**FP32 Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 98.34 img/s | 10.24 ms | 10.27 ms | 10.32 ms | 12.89 ms |
| 2 | 167.04 img/s | 11.98 ms | 12.17 ms | 12.24 ms | 12.59 ms |
| 4 | 214.18 img/s | 18.68 ms | 18.80 ms | 18.88 ms | 19.73 ms |
| 8 | 259.96 img/s | 30.78 ms | 31.04 ms | 31.08 ms | 31.44 ms |
| 16 | 350.71 img/s | 45.63 ms | 45.81 ms | 45.88 ms | 47.96 ms |
| 32 | 407.80 img/s | 78.74 ms | 78.66 ms | 79.04 ms | 110.32 ms |
| 64 | 461.88 img/s | 138.57 ms | 139.34 ms | 139.68 ms | 141.54 ms |
| 128 | 493.61 img/s | 259.57 ms | 260.38 ms | 260.84 ms | 262.40 ms |

**Mixed Precision Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 84.74 img/s | 11.85 ms | 11.95 ms | 12.02 ms | 12.17 ms |
| 2 | 183.64 img/s | 10.94 ms | 11.08 ms | 11.18 ms | 11.36 ms |
| 4 | 359.91 img/s | 11.17 ms | 11.35 ms | 11.46 ms | 11.80 ms |
| 8 | 736.61 img/s | 10.87 ms | 11.17 ms | 11.31 ms | 11.46 ms |
| 16 | 1058.59 img/s | 15.22 ms | 15.30 ms | 15.47 ms | 16.51 ms |
| 32 | 1152.14 img/s | 28.03 ms | 27.99 ms | 28.11 ms | 29.55 ms |
| 64 | 1275.35 img/s | 50.38 ms | 50.41 ms | 50.52 ms | 51.39 ms |
| 128 | 1347.11 img/s | 95.02 ms | 95.51 ms | 95.70 ms | 96.29 ms |


**Mixed Precision Inference Latency + XLA**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 59.84 img/s | 16.77 ms | 16.95 ms | 17.00 ms | 17.23 ms |
| 2 | 120.41 img/s | 16.66 ms | 16.90 ms | 16.97 ms | 17.21 ms |
| 4 | 242.75 img/s | 16.48 ms | 16.96 ms | 17.10 ms | 17.55 ms |
| 8 | 466.47 img/s | 17.15 ms | 17.50 ms | 17.65 ms | 17.94 ms |
| 16 | 861.72 img/s | 18.69 ms | 19.19 ms | 19.33 ms | 19.68 ms |
| 32 | 1472.21 img/s | 22.06 ms | 22.32 ms | 22.82 ms | 23.91 ms |
| 64 | 1728.76 img/s | 37.24 ms | 37.49 ms | 37.65 ms | 38.08 ms |
| 128 | 1892.97 img/s | 67.62 ms | 68.24 ms | 68.49 ms | 69.47 ms |
|

##### Inference performance: NVIDIA DGX-2 (1x V100 32G)

Our results were obtained by running the `inference_benchmark.sh` inferencing benchmarking script
in the [TensorFlow 20.06-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) NGC container 
on NVIDIA DGX-2 with (1x V100 32G) GPU.

**FP32 Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 96.91 img/s | 10.38 ms | 10.46 ms | 10.53 ms | 11.32 ms |
| 2 | 163.02 img/s | 12.33 ms | 12.54 ms | 12.77 ms | 13.45 ms |
| 4 | 206.76 img/s | 19.35 ms | 19.52 ms | 19.63 ms | 20.09 ms |
| 8 | 249.68 img/s | 32.05 ms | 32.24 ms | 32.31 ms | 33.26 ms |
| 16 | 330.36 img/s | 48.43 ms | 48.63 ms | 48.69 ms | 49.03 ms |
| 32 | 399.97 img/s | 80.00 ms | 80.44 ms | 80.62 ms | 81.28 ms |
| 64 | 481.88 img/s | 132.94 ms | 133.05 ms | 133.16 ms | 133.71 ms |
| 128 | 519.85 img/s | 246.22 ms | 247.09 ms | 247.71 ms | 250.49 ms |


**Mixed Precision Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 108.86 img/s | 9.24 ms | 9.36 ms | 9.42 ms | 9.57 ms |
| 2 | 215.01 img/s | 9.36 ms | 9.42 ms | 9.46 ms | 9.68 ms |
| 4 | 422.09 img/s | 9.48 ms | 9.70 ms | 9.80 ms | 10.10 ms |
| 8 | 791.52 img/s | 10.12 ms | 10.24 ms | 10.32 ms | 10.58 ms |
| 16 | 1064.30 img/s | 15.16 ms | 15.27 ms | 15.32 ms | 17.23 ms |
| 32 | 1190.90 img/s | 27.11 ms | 27.00 ms | 27.10 ms | 27.97 ms |
| 64 | 1319.63 img/s | 48.49 ms | 48.73 ms | 48.82 ms | 49.32 ms |
| 128 | 1397.36 img/s | 91.60 ms | 91.93 ms | 92.07 ms | 92.61 ms |


**Mixed Precision Inference Latency + XLA**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 76.34 img/s | 13.16 ms | 13.37 ms | 13.49 ms | 13.74 ms |
| 2 | 150.90 img/s | 13.31 ms | 13.54 ms | 13.61 ms | 13.87 ms |
| 4 | 284.88 img/s | 14.10 ms | 15.28 ms | 15.38 ms | 15.68 ms |
| 8 | 587.77 img/s | 13.61 ms | 13.87 ms | 13.94 ms | 14.06 ms |
| 16 | 1089.95 img/s | 14.80 ms | 14.91 ms | 15.04 ms | 15.46 ms |
| 32 | 1503.51 img/s | 21.55 ms | 21.33 ms | 21.38 ms | 21.91 ms |
| 64 | 1765.86 img/s | 36.47 ms | 36.39 ms | 36.51 ms | 37.15 ms |
| 128 | 2003.04 img/s | 63.91 ms | 64.95 ms | 65.07 ms | 65.47 ms |
|

##### Inference performance: NVIDIA T4 (1x T4 16G)

Our results were obtained by running the `inference_benchmark.sh` inferencing benchmarking script
in the [TensorFlow 20.06-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) NGC container 
on NVIDIA T4 with (1x T4 16G) GPU.

**FP32 Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 31.92 img/s | 31.42 ms | 31.58 ms | 31.78 ms | 37.56 ms |
| 2 | 45.62 img/s | 43.92 ms | 44.83 ms | 45.80 ms | 46.99 ms |
| 4 | 70.42 img/s | 56.80 ms | 57.14 ms | 57.47 ms | 59.30 ms |
| 8 | 85.68 img/s | 93.36 ms | 93.66 ms | 93.76 ms | 94.15 ms |
| 16 | 99.58 img/s | 160.65 ms | 160.91 ms | 161.39 ms | 162.34 ms |
| 32 | 105.04 img/s | 304.63 ms | 305.53 ms | 305.96 ms | 307.22 ms |
| 64 | 108.31 img/s | 590.85 ms | 591.31 ms | 591.70 ms | 593.23 ms |
| 128 | 110.05 img/s | 1163.04 ms | 1163.52 ms | 1163.75 ms | 1164.24 ms |


**Mixed Precision Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 80.61 img/s | 12.50 ms | 12.56 ms | 12.66 ms | 13.54 ms |
| 2 | 104.47 img/s | 19.23 ms | 19.73 ms | 19.92 ms | 20.68 ms |
| 4 | 143.68 img/s | 27.91 ms | 28.42 ms | 28.71 ms | 29.47 ms |
| 8 | 176.65 img/s | 45.29 ms | 45.93 ms | 46.15 ms | 46.75 ms |
| 16 | 203.55 img/s | 78.60 ms | 78.95 ms | 79.25 ms | 79.74 ms |
| 32 | 209.77 img/s | 152.54 ms | 153.41 ms | 153.75 ms | 154.82 ms |
| 64 | 222.97 img/s | 287.03 ms | 287.91 ms | 288.27 ms | 289.56 ms |
| 128 | 226.19 img/s | 565.89 ms | 566.21 ms | 566.38 ms | 567.52 ms |


**Mixed Precision Inference Latency + XLA**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 54.68 img/s | 18.40 ms | 19.17 ms | 19.34 ms | 19.53 ms |
| 2 | 102.20 img/s | 19.67 ms | 20.37 ms | 20.55 ms | 24.65 ms |
| 4 | 153.96 img/s | 26.05 ms | 26.31 ms | 27.01 ms | 28.96 ms |
| 8 | 177.98 img/s | 44.94 ms | 45.25 ms | 45.43 ms | 45.66 ms |
| 16 | 237.70 img/s | 67.31 ms | 68.35 ms | 68.87 ms | 69.63 ms |
| 32 | 241.79 img/s | 132.34 ms | 133.18 ms | 133.87 ms | 134.92 ms |
| 64 | 263.80 img/s | 242.60 ms | 244.25 ms | 245.27 ms | 246.56 ms |
| 128 | 272.17 img/s | 470.29 ms | 471.29 ms | 471.78 ms | 473.61 ms |

## Release notes

### Changelog

April 2023
  - Ceased maintenance of ConvNets in TensorFlow1
June 2020
   - Initial release
August 2020
   - Updated command line argument names
   - Added support for syntetic dataset with different image size
January 2022
   - Added barrier at the end of multiprocess run
### Known issues
Performance without XLA enabled is low due to BN + ReLU fusion bug.

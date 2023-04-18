# SE-ResNext101-32x4d for TensorFlow

This repository provides a script and recipe to train the SE-ResNext101-32x4d model to achieve state-of-the-art accuracy, and is tested and maintained by NVIDIA.
SE-ResNext101-32x4d model for TensorFlow1 is no longer maintained and will soon become unavailable, please consider PyTorch or TensorFlow2 models as a substitute for your requirements.

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
The SE-ResNeXt101-32x4d is a [ResNeXt101-32x4d](https://arxiv.org/pdf/1611.05431.pdf)
model with added Squeeze-and-Excitation module introduced in the [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf) paper.

The following performance optimizations were implemented in this model:
* JIT graph compilation with [XLA](https://www.tensorflow.org/xla)
* Multi-GPU training with [Horovod](https://github.com/horovod/horovod)
* Automated mixed precision [AMP](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)

This model is trained with mixed precision using Tensor Cores on Volta, Turing, and the NVIDIA Ampere GPU architectures. Therefore, researchers can get results 3x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.

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

The following section lists the requirements that you need to meet in order to use the SE-ResNext101-32x4d model.

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
To train your model using mixed precision or TF32 with Tensor Cores or FP32, perform the following steps using the default parameters of the SE-ResNext101-32x4d model on the [ImageNet](http://www.image-net.org/) dataset. For the specifics concerning training and inference, see the [Advanced](#advanced) section.


1. Clone the repository.
```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/TensorFlow/Classification/ConvNets
```

2. Download and preprocess the dataset.
The SE-ResNext101-32x4d script operates on ImageNet 1k, a widely popular image classification dataset from the ILSVRC challenge.

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

`bash ./se-resnext101-32x4d/training/DGX1_SE-RNxt101-32x4d_AMP_90E.sh /path/to/result /data`

Additionally, features like DALI data preprocessing or TensorFlow XLA can be enabled with
following arguments when running those scripts:

`bash ./se-resnext101-32x4d/training/DGX1_SE-RNxt101-32x4d_AMP_90E.sh /path/to/result /data/ --xla --dali`

7. Start validation/evaluation.
To evaluate the validation dataset located in `/data/tfrecords`, run `main.py` with
`--mode=evaluate`. For example:

`python main.py --arch=se-resnext101-32x4d --mode=evaluate --data_dir=/data/tfrecords --batch_size <batch size> --model_dir
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

`python main.py --arch=se-resnext101-32x4d --mode predict --model_dir <path to model> --to_predict <path to image> --results_dir <path to results>`

The optional `--xla` and `--amp` flags control XLA and AMP during inference.

## Performance

The performance measurements in this document were conducted at the time of publication and may not reflect the performance achieved from NVIDIA’s latest software release. For the most up-to-date performance measurements, go to [NVIDIA Data Center Deep Learning Product Performance](https://developer.nvidia.com/deep-learning-performance-training-inference).

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in training and inference modes.

#### Training performance benchmark

To benchmark the training performance on a specific batch size, run:

* For 1 GPU
    * FP32 / TF32
    
        `python ./main.py --arch=se-resnext101-32x4d --mode=training_benchmark --warmup_steps 200 --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
    * AMP

        `python ./main.py --arch=se-resnext101-32x4d --mode=training_benchmark  --amp --warmup_steps 200 --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
* For multiple GPUs
    * FP32 / TF32
    
        `mpiexec --allow-run-as-root --bind-to socket -np <num_gpus> python ./main.py --arch=se-resnext101-32x4d --mode=training_benchmark --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
    * AMP

        `mpiexec --allow-run-as-root --bind-to socket -np <num_gpus> python ./main.py --arch=se-resnext101-32x4d --mode=training_benchmark --amp --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`
        
        
Each of these scripts runs 200 warm-up iterations and measures the first epoch.

To control warmup and benchmark length, use the `--warmup_steps`, `--num_iter` and `--iter_unit` flags. Features like XLA or DALI can be controlled
with `--xla` and `--dali` flags. For proper throughput reporting the value of `--num_iter` must be greater than `--warmup_steps` value.
Suggested batch sizes for training are 96 for mixed precision training and 64 for single precision training per single V100 16 GB.

If no `--data_dir=<path to imagenet>` flag is specified then the benchmarks will use a synthetic dataset. The resolution of synthetic images used can be controlled with `--synthetic_data_size` flag.


#### Inference performance benchmark

To benchmark the inference performance on a specific batch size, run:

* FP32 / TF32

`python ./main.py --arch=se-resnext101-32x4d --mode=inference_benchmark --warmup_steps 20 --num_iter 100 --iter_unit batch --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`

* AMP

`python ./main.py --arch=se-resnext101-32x4d --mode=inference_benchmark --amp --warmup_steps 20 --num_iter 100 --iter_unit batch --batch_size <batch size> --data_dir=<path to imagenet> --results_dir=<path to results directory>`

By default, each of these scripts runs 20 warm-up iterations and measures the next 80 iterations.
To control warm-up and benchmark length, use the `--warmup_steps`, `--num_iter` and `--iter_unit` flags.
If no `--data_dir=<path to imagenet>` flag is specified then the benchmarks will use a synthetic dataset.

The benchmark can be automated with the `inference_benchmark.sh` script provided in `se-resnext101-32x4d`, by simply running:
`bash ./se-resnext101-32x4d/inference_benchmark.sh <data dir> <data idx dir>`

The `<data dir>` parameter refers to the input data directory (by default `/data/tfrecords` inside the container). 
By default, the benchmark tests the following configurations: **FP32**, **AMP**, **AMP + XLA** with different batch sizes.
When the optional directory with the DALI index files `<data idx dir>` is specified, the benchmark executes an additional **DALI + AMP + XLA** configuration.
For proper throughput reporting the value of `--num_iter` must be greater than `--warmup_steps` value.

For performance benchamrk of raw model, synthetic dataset can be used. To use synthetic dataset, use `--synthetic_data_size` flag instead of `--data_dir` to specify input image size.

### Results

The following sections provide details on how we achieved our performance and accuracy in training and inference. 

#### Training accuracy results

##### Training accuracy: NVIDIA DGX A100 (8x A100 40GB)

Our results were obtained by running the `/se-resnet50v1.5/training/DGXA100_RN50_{PRECISION}_90E.sh` 
training script in the [TensorFlow 20.06-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) 
NGC container on NVIDIA DGX A100 (8x A100 40GB) GPUs.

| Epochs | Batch Size / GPU | Accuracy - TF32 (top1) | Accuracy - mixed precision (top1) | 
|--------|------------------|-----------------|----------------------------|
| 90     | 128 (TF32) / 256 (AMP) | 79.73           | 79.60                 |

##### Training accuracy: NVIDIA DGX-1 (8x V100 16G)
Our results were obtained by running the `/se-resnext101-32x4d/training/{/DGX1_RNxt101-32x4d_{PRECISION}_{EPOCHS}E.sh` 
training script in the [TensorFlow 20.06-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) 
NGC container on NVIDIA DGX-1 with (8x V100 16G) GPUs.

| Epochs | Batch Size / GPU | Accuracy - FP32 | Accuracy - mixed precision | 
|--------|------------------|-----------------|----------------------------|
| 90   | 64 (FP32) / 96 (AMP) | 79.69              | 79.81   |
| 250  | 64 (FP32) / 96 (AMP) | 80.87              | 80.84   |

**Example training loss plot**

![TrainingLoss](./imgs/train_loss.png)

#### Training performance results

##### Training performance: NVIDIA DGX A100 (8x A100 40GB)
Our results were obtained by running the `se-resnext101-32x4d/training/training_perf.sh` benchmark script in the 
[TensorFlow 20.06-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)  NGC container 
on NVIDIA DGX A100 (8x A100 40GB) GPUs. Performance numbers (in images per second) were averaged over an entire training epoch.


| GPUs | Batch Size / GPU | Throughput - TF32 + XLA | Throughput - mixed precision + XLA | Throughput speedup (TF32 - mixed precision) | Weak scaling - TF32 + XLA | Weak scaling - mixed precision + XLA |
|----|---------------|---------------|------------------------|-----------------|-----------|-------------------|
| 1  | 128 (TF) / 256 (AMP) | 342 img/s  | 975 img/s    | 2.86x           | 1.00x     | 1.00x             |
| 8  | 128 (TF) / 256 (AMP) | 2610 img/s | 7230 img/s   | 2.77x           | 7.63x     | 7.41x             |

##### Training performance: NVIDIA DGX-1 (8x V100 16G)
Our results were obtained by running the `se-resnext101-32x4d/training/training_perf.sh` benchmark script in the 
[TensorFlow 20.06-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)  NGC container 
on NVIDIA DGX-1 with (8x V100 16G) GPUs. Performance numbers (in images per second) were averaged over an entire training epoch.


| GPUs | Batch Size / GPU | Throughput - FP32 + XLA | Throughput - mixed precision + XLA | Throughput speedup (FP32 - mixed precision) | Weak scaling - FP32 + XLA | Weak scaling - mixed precision + XLA |
|----|---------------|---------------|-----------------------|---------------|-----------|-------|
| 1  | 64 (FP32) / 96 (AMP) | 152 img/s | 475 img/s     | 3.12x         | 1.00x     | 1.00x      |
| 8  | 64 (FP32) / 96 (AMP) | 1120 img/s | 3360 img/s    | 3.00x         | 7.37x     | 7.07x      |

##### Training performance: NVIDIA DGX-2 (16x V100 32G)
Our results were obtained by running the `se-resnext101-32x4d/training/training_perf.sh` benchmark script in the 
[TensorFlow 20.06-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)  NGC container 
on NVIDIA DGX-2 with (16x V100 32G) GPUs. Performance numbers (in images per second) were averaged over an entire training epoch.

| GPUs | Batch Size / GPU | Throughput - FP32 + XLA | Throughput - mixed precision + XLA | Throughput speedup (FP32 - mixed precision) | Weak scaling - FP32 + XLA | Weak scaling - mixed precision + XLA |
|----|---------------|---------------|-------------------------|-------|--------|--------|
| 1  | 64 (FP32) / 96 (AMP) | 158 img/s | 472 img/s    | 2.98x                 | 1.00x        | 1.00x  |
| 16 | 64 (FP32) / 96 (AMP) | 2270 img/s| 6580 img/s   | 2.89x                 | 14.36x        | 13.94x |

#### Training Time for 90 Epochs

##### Training time: NVIDIA DGX A100 (8x A100 40GB)

Our results were estimated based on the [training performance results](#training-performance-nvidia-dgx-a100-8x-a100-40g) 
on NVIDIA DGX A100 with (8x A100 40G) GPUs.

| GPUs | Time to train - mixed precision + XLA |  Time to train - TF32 + XLA | 
|---|--------|---------|
| 1 | ~36h   |  ~102h  | 
| 8 | ~5h    | ~14h    | 

##### Training time: NVIDIA DGX-1 (8x V100 16G)

Our results were estimated based on the [training performance results](#training-performance-nvidia-dgx-1-8x-v100-16g) 
on NVIDIA DGX-1 with (8x V100 16G) GPUs.

| GPUs | Time to train - mixed precision + XLA | Time to train - FP32 + XLA |
|---|--------|---------|
| 1 | ~68h   | ~210h   |
| 8 | ~10h  |  ~29h | 

##### Training time: NVIDIA DGX-2 (16x V100 32G)

Our results were estimated based on the [training performance results](#training-performance-nvidia-dgx-2-16x-v100-32g) 
on NVIDIA DGX-2 with (16x V100 32G) GPUs.

| GPUs | Time to train - mixed precision + XLA |  Time to train - FP32 + XLA |
|----|-------|-------|
| 1  | ~68h  | ~202h |
| 16 | ~5h | ~14h   | 


#### Inference performance results

##### Inference performance: NVIDIA DGX A100 (1x A100 40GB)

Our results were obtained by running the `inference_benchmark.sh` inferencing benchmarking script
in the [TensorFlow 20.06-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) NGC container 
on NVIDIA DGX A100 with (1x A100 40G) GPU.

**TF32 Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 95.32 img/s | 10.52 ms | 10.52 ms | 10.55 ms | 11.10 ms |
| 2 | 169.59 img/s | 11.82 ms | 11.83 ms | 11.92 ms | 12.56 ms |
| 4 | 258.97 img/s | 15.45 ms | 15.70 ms | 15.78 ms | 16.22 ms |
| 8 | 355.09 img/s | 22.53 ms | 22.74 ms | 22.84 ms | 23.17 ms |
| 16 | 561.11 img/s | 28.52 ms | 28.85 ms | 29.09 ms | 29.50 ms |
| 32 | 698.94 img/s | 45.78 ms | 46.36 ms | 46.56 ms | 46.87 ms |
| 64 | 751.17 img/s | 85.21 ms | 86.74 ms | 87.27 ms | 87.95 ms |
| 128 | 802.64 img/s | 159.47 ms | 160.01 ms | 160.35 ms | 161.42 ms |
| 256 | 840.72 img/s | 304.50 ms | 305.87 ms | 306.11 ms | 306.57 ms |

**TF32 Inference Latency + XLA**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 92.46 img/s | 10.84 ms | 10.90 ms | 10.96 ms | 11.14 ms |
| 2 | 161.55 img/s | 12.40 ms | 12.44 ms | 12.51 ms | 12.62 ms |
| 4 | 237.41 img/s | 16.88 ms | 17.54 ms | 17.79 ms | 18.25 ms |
| 8 | 358.39 img/s | 22.35 ms | 23.56 ms | 24.29 ms | 25.53 ms |
| 16 | 577.33 img/s | 27.72 ms | 28.64 ms | 28.92 ms | 29.22 ms |
| 32 | 800.81 img/s | 39.97 ms | 40.93 ms | 41.42 ms | 41.87 ms |
| 64 | 921.00 img/s | 69.64 ms | 70.44 ms | 70.90 ms | 79.54 ms |
| 128 | 1024.70 img/s | 124.99 ms | 125.70 ms | 126.10 ms | 138.57 ms |
| 256 | 1089.80 img/s | 234.90 ms | 236.02 ms | 236.37 ms | 237.26 ms |

**Mixed Precision Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 84.06 img/s | 11.92 ms | 11.94 ms | 11.96 ms | 12.08 ms |
| 2 | 170.38 img/s | 11.76 ms | 11.82 ms | 11.87 ms | 11.94 ms |
| 4 | 336.09 img/s | 11.93 ms | 12.06 ms | 12.17 ms | 12.62 ms |
| 8 | 669.91 img/s | 11.94 ms | 12.33 ms | 12.47 ms | 12.88 ms |
| 16 | 1119.49 img/s | 14.36 ms | 14.86 ms | 15.11 ms | 16.11 ms |
| 32 | 1482.46 img/s | 21.66 ms | 22.04 ms | 22.38 ms | 23.72 ms |
| 64 | 1680.85 img/s | 38.09 ms | 39.02 ms | 39.34 ms | 41.02 ms |
| 128 | 1728.27 img/s | 74.30 ms | 74.92 ms | 75.22 ms | 75.60 ms |
| 256 | 1761.56 img/s | 145.33 ms | 146.54 ms | 146.83 ms | 147.34 ms |

**Mixed Precision Inference Latency + XLA**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 74.83 img/s | 13.39 ms | 13.45 ms | 13.49 ms | 13.57 ms |
| 2 | 135.28 img/s | 14.81 ms | 14.98 ms | 15.10 ms | 16.19 ms |
| 4 | 272.18 img/s | 14.70 ms | 15.07 ms | 15.30 ms | 15.80 ms |
| 8 | 517.69 img/s | 15.50 ms | 16.63 ms | 17.05 ms | 18.10 ms |
| 16 | 1050.03 img/s | 15.38 ms | 16.84 ms | 17.49 ms | 17.97 ms |
| 32 | 1781.06 img/s | 18.27 ms | 19.54 ms | 20.00 ms | 25.94 ms |
| 64 | 2551.55 img/s | 25.26 ms | 26.03 ms | 26.62 ms | 29.67 ms |
| 128 | 2834.59 img/s | 45.50 ms | 46.85 ms | 47.72 ms | 54.91 ms |
| 256 | 3367.18 img/s | 76.03 ms | 77.06 ms | 77.36 ms | 78.13 ms |


##### Inference performance: NVIDIA DGX-1 (1x V100 16G)

Our results were obtained by running the `inference_benchmark.sh` inferencing benchmarking script
in the [TensorFlow 20.06-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) NGC container 
on NVIDIA DGX-1 with (1x V100 16G) GPU.

**FP32 Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 75.72 img/s | 13.25 ms | 13.38 ms | 13.50 ms | 13.66 ms |
| 2 | 112.58 img/s | 17.90 ms | 20.74 ms | 20.91 ms | 21.87 ms |
| 4 | 191.09 img/s | 20.93 ms | 21.05 ms | 21.09 ms | 21.27 ms |
| 8 | 235.39 img/s | 33.98 ms | 34.14 ms | 34.19 ms | 34.28 ms |
| 16 | 315.24 img/s | 50.76 ms | 50.96 ms | 51.01 ms | 51.32 ms |
| 32 | 376.05 img/s | 85.09 ms | 85.56 ms | 85.71 ms | 86.40 ms |
| 64 | 427.39 img/s | 149.84 ms | 150.08 ms | 150.37 ms | 161.87 ms |
| 128 | 460.82 img/s | 277.76 ms | 278.97 ms | 279.48 ms | 280.95 ms |

**Mixed Precision Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 66.44 img/s | 15.10 ms | 15.17 ms | 15.25 ms | 16.01 ms |
| 2 | 132.33 img/s | 15.16 ms | 15.32 ms | 15.37 ms | 15.50 ms |
| 4 | 273.84 img/s | 14.63 ms | 15.14 ms | 15.83 ms | 17.38 ms |
| 8 | 509.35 img/s | 15.71 ms | 16.10 ms | 16.21 ms | 16.55 ms |
| 16 | 770.02 img/s | 20.78 ms | 20.96 ms | 21.03 ms | 21.24 ms |
| 32 | 926.46 img/s | 34.55 ms | 34.88 ms | 35.05 ms | 36.32 ms |
| 64 | 1039.74 img/s | 61.55 ms | 61.82 ms | 61.99 ms | 62.32 ms |
| 128 | 1102.00 img/s | 116.15 ms | 116.62 ms | 116.80 ms | 116.97 ms |

**Mixed Precision Inference Latency + XLA**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 58.55 img/s | 17.12 ms | 17.21 ms | 17.28 ms | 17.42 ms |
| 2 | 105.00 img/s | 19.10 ms | 19.29 ms | 19.36 ms | 19.67 ms |
| 4 | 207.60 img/s | 19.31 ms | 19.59 ms | 19.67 ms | 19.84 ms |
| 8 | 413.16 img/s | 19.37 ms | 19.77 ms | 19.87 ms | 20.24 ms |
| 16 | 739.12 img/s | 21.80 ms | 24.48 ms | 24.71 ms | 26.93 ms |
| 32 | 1196.83 img/s | 26.99 ms | 27.10 ms | 27.49 ms | 28.80 ms |
| 64 | 1470.31 img/s | 43.74 ms | 44.02 ms | 44.18 ms | 46.28 ms |
| 128 | 1683.63 img/s | 76.03 ms | 77.00 ms | 77.23 ms | 78.15 ms |


##### Inference performance: NVIDIA DGX-2 (1x V100 32G)

Our results were obtained by running the `inference_benchmark.sh` inferencing benchmarking script
in the [TensorFlow 20.06-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) NGC container 
on NVIDIA DGX-2 with (1x V100 32G) GPU.

**FP32 Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 71.44 img/s | 14.07 ms | 14.22 ms | 14.43 ms | 16.44 ms |
| 2 | 149.68 img/s | 13.43 ms | 13.79 ms | 13.94 ms | 16.63 ms |
| 4 | 183.01 img/s | 21.85 ms | 22.12 ms | 22.18 ms | 22.44 ms |
| 8 | 220.67 img/s | 36.25 ms | 36.84 ms | 37.17 ms | 37.43 ms |
| 16 | 310.27 img/s | 51.57 ms | 51.88 ms | 52.09 ms | 53.37 ms |
| 32 | 381.41 img/s | 83.89 ms | 84.30 ms | 84.66 ms | 85.04 ms |
| 64 | 440.37 img/s | 145.45 ms | 145.49 ms | 145.86 ms | 147.53 ms |
| 128 | 483.84 img/s | 264.54 ms | 265.04 ms | 265.46 ms | 266.43 ms |


**Mixed Precision Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 73.06 img/s | 13.74 ms | 14.07 ms | 14.20 ms | 14.35 ms |
| 2 | 155.23 img/s | 12.95 ms | 13.13 ms | 13.33 ms | 15.49 ms |
| 4 | 303.68 img/s | 13.23 ms | 13.38 ms | 13.46 ms | 14.34 ms |
| 8 | 583.43 img/s | 13.72 ms | 13.90 ms | 14.08 ms | 15.47 ms |
| 16 | 783.30 img/s | 20.43 ms | 20.66 ms | 21.31 ms | 21.97 ms |
| 32 | 932.10 img/s | 34.34 ms | 34.71 ms | 34.81 ms | 35.70 ms |
| 64 | 1058.07 img/s | 60.48 ms | 60.75 ms | 60.94 ms | 62.49 ms |
| 128 | 1129.65 img/s | 113.30 ms | 113.53 ms | 113.66 ms | 114.81 ms |


**Mixed Precision Inference Latency + XLA**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 66.43 img/s | 15.14 ms | 15.24 ms | 15.31 ms | 19.18 ms |
| 2 | 122.85 img/s | 16.39 ms | 18.28 ms | 18.45 ms | 20.33 ms |
| 4 | 247.80 img/s | 16.14 ms | 16.44 ms | 16.57 ms | 17.24 ms |
| 8 | 498.19 img/s | 16.07 ms | 16.26 ms | 16.66 ms | 17.70 ms |
| 16 | 831.20 img/s | 19.40 ms | 19.30 ms | 19.39 ms | 25.41 ms |
| 32 | 1223.75 img/s | 26.42 ms | 26.31 ms | 26.70 ms | 29.88 ms |
| 64 | 1520.64 img/s | 42.09 ms | 42.45 ms | 42.57 ms | 42.84 ms |
| 128 | 1739.61 img/s | 73.58 ms | 73.98 ms | 74.17 ms | 74.72 ms |

##### Inference performance: NVIDIA T4 (1x T4 16G)

Our results were obtained by running the `inference_benchmark.sh` inferencing benchmarking script
in the [TensorFlow 20.06-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) NGC container 
on NVIDIA T4 with (1x T4 16G) GPU.

**FP32 Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 27.39 img/s | 36.68 ms | 38.85 ms | 39.01 ms | 40.40 ms |
| 2 | 44.56 img/s | 44.96 ms | 46.25 ms | 46.92 ms | 48.92 ms |
| 4 | 65.11 img/s | 61.43 ms | 62.22 ms | 62.93 ms | 65.01 ms |
| 8 | 80.09 img/s | 99.88 ms | 100.34 ms | 100.85 ms | 101.79 ms |
| 16 | 93.98 img/s | 170.24 ms | 170.72 ms | 171.27 ms | 171.98 ms |
| 32 | 99.86 img/s | 320.42 ms | 320.99 ms | 321.37 ms | 322.28 ms |
| 64 | 103.31 img/s | 619.44 ms | 620.08 ms | 620.55 ms | 622.19 ms |
| 128 | 105.16 img/s | 1217.18 ms | 1218.09 ms | 1218.59 ms | 1221.16 ms |

**Mixed Precision Inference Latency**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 57.21 img/s | 17.57 ms | 18.06 ms | 18.15 ms | 20.74 ms |
| 2 | 80.34 img/s | 24.97 ms | 25.38 ms | 25.69 ms | 27.12 ms |
| 4 | 115.12 img/s | 34.77 ms | 35.61 ms | 36.74 ms | 37.61 ms |
| 8 | 147.51 img/s | 54.24 ms | 54.79 ms | 55.28 ms | 58.25 ms |
| 16 | 173.83 img/s | 92.04 ms | 92.50 ms | 93.26 ms | 94.72 ms |
| 32 | 182.19 img/s | 175.64 ms | 176.51 ms | 177.44 ms | 178.52 ms |
| 64 | 193.20 img/s | 331.25 ms | 332.56 ms | 333.34 ms | 334.58 ms |
| 128 | 195.17 img/s | 655.82 ms | 657.24 ms | 658.79 ms | 661.76 ms |


**Mixed Precision Inference Latency + XLA**

|**Batch Size**|**Avg throughput**|**Avg latency**|**90% Latency**|**95% Latency**|**99% Latency**|
|--------------|------------------|---------------|---------------|---------------|---------------|
| 1 | 46.19 img/s | 21.72 ms | 21.90 ms | 21.93 ms | 23.64 ms |
| 2 | 80.98 img/s | 24.77 ms | 24.99 ms | 25.15 ms | 25.63 ms |
| 4 | 129.49 img/s | 30.89 ms | 31.26 ms | 31.34 ms | 32.31 ms |
| 8 | 156.91 img/s | 51.00 ms | 52.17 ms | 52.51 ms | 53.32 ms |
| 16 | 204.45 img/s | 78.26 ms | 79.58 ms | 79.96 ms | 80.44 ms |
| 32 | 215.22 img/s | 148.68 ms | 149.63 ms | 150.41 ms | 151.62 ms |
| 64 | 235.36 img/s | 272.05 ms | 273.56 ms | 274.33 ms | 275.86 ms |
| 128 | 244.45 img/s | 523.62 ms | 525.12 ms | 525.89 ms | 528.42 ms |


## Release notes

### Changelog

April 2023
  - Ceased maintenance of ConvNets in TensorFlow1
April 2020
   - Initial release
August 2020
   - Updated command line argument names
   - Added support for syntetic dataset with different image size
January 2022
   - Added barrier at the end of multiprocess run

### Known issues
Performance without XLA enabled is low due to BN + ReLU fusion bug.

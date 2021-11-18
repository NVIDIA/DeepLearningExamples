# EfficientNet For TensorFlow 2.4

This repository provides a script and recipe to train the EfficientNet model to achieve state-of-the-art accuracy.
The content of the repository is tested and maintained by NVIDIA.

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

    * [Training process](#training-process)
        *[Multi-node](#multi-node)
    * [Inference process](#inference-process)
   - [Performance](#performance)
    * [Benchmarking](#benchmarking)
        * [Training performance benchmark](#training-performance-benchmark)
        * [Inference performance benchmark](#inference-performance-benchmark)
    * [Results](#results)
        * [Training accuracy results for EfficientNet-B0](#training-accuracy-results-for-efficientnet-b0)
            * [Training accuracy: NVIDIA DGX A100 (8x A100 80GB)](#training-accuracy-nvidia-dgx-a100-8x-a100-80gb)  
            * [Training accuracy: NVIDIA DGX-1 (8x V100 16GB)](#training-accuracy-nvidia-dgx-1-8x-v100-16gb)
        * [Training accuracy results for EfficientNet-B4](#training-accuracy-results-for-efficientnet-b4)
            * [Training accuracy: NVIDIA DGX A100 (8x A100 80GB)](#training-accuracy-nvidia-dgx-a100-8x-a100-80gb-1)  
            * [Training accuracy: NVIDIA DGX-1 (8x V100 32GB)](#training-accuracy-nvidia-dgx-1-8x-v100-32gb)
        * [Training performance results for EfficientNet-B0](#training-performance-results-for-efficientnet-b0)
            * [Training performance: NVIDIA DGX A100 (8x A100 80GB)](#training-performance-nvidia-dgx-a100-8x-a100-80gb) 
            * [Training performance: NVIDIA DGX-1 (8x V100 16GB)](#training-performance-nvidia-dgx-1-8x-v100-16gb)
        * [Training performance results for EfficientNet-B4](#training-performance-results-for-efficientnet-b4)
            * [Training performance: NVIDIA DGX A100 (8x A100 80GB)](#training-performance-nvidia-dgx-a100-8x-a100-80gb-1) 
            * [Training performance: NVIDIA DGX-1 (8x V100 32GB)](#training-performance-nvidia-dgx-1-8x-v100-32gb)
        * [Inference performance results for EfficientNet-B0](#inference-performance-results-for-efficientnet-b0)
            * [Inference performance: NVIDIA DGX A100 (1x A100 80GB)](#inference-performance-nvidia-dgx-a100-1x-a100-80gb)
            * [Inference performance: NVIDIA DGX-1 (1x V100 16GB)](#inference-performance-nvidia-dgx-1-1x-v100-16gb)
        * [Inference performance results for EfficientNet-B4](#inference-performance-results-for-efficientnet-b4)
            * [Inference performance: NVIDIA DGX A100 (1x A100 80GB)](#inference-performance-nvidia-dgx-a100-1x-a100-80gb-1)
            * [Inference performance: NVIDIA DGX-1 (1x V100 32GB)](#inference-performance-nvidia-dgx-1-1x-v100-32gb)
- [Release notes](#release-notes)
    * [Changelog](#changelog)
    * [Known issues](#known-issues)



## Model overview

EfficientNet  TensorFlow 2 is a family of image classification models, which achieve state-of-the-art accuracy, yet being an order-of-magnitude smaller and faster than previous models.
This model is based on [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946).
NVIDIA's implementation of EfficientNet TensorFlow 2 is an optimized version of [TensorFlow Model Garden](https://github.com/tensorflow/models/tree/master/official/vision/image_classification) implementation, 
leveraging mixed precision arithmetic on Volta, Turing, and the NVIDIA Ampere GPU architectures for faster training times while maintaining target accuracy.

The major differences between the original implementation of the paper and this version of EfficientNet are as follows:
- Automatic mixed precision (AMP) training support
- Cosine LR decay for better accuracy
- Weight initialization using `fan_out` for better accuracy
- Multi-node training support
- XLA enabled for better performance
- Lightweight logging using [dllogger](https://github.com/NVIDIA/dllogger)

Other publicly available implementations of EfficientNet include:

- [Tensorflow Model Garden](https://github.com/tensorflow/models/tree/master/official/vision/image_classification)
- [Pytorch version](https://github.com/rwightman/pytorch-image-models)
- [Google's implementation for TPU](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
 
This model is trained with mixed precision Tensor Cores on Volta, Turing, and the NVIDIA Ampere GPU architectures. It provides a push-button solution to pretraining on a corpus of choice. 
As a result, researchers can get results 1.5x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training.  This model is tested against each NGC monthly released container to ensure consistent accuracy and performance over time.


### Model architecture

EfficientNets are developed based on AutoML and Compound Scaling. In particular, 
a mobile-size baseline network called EfficientNet-B0 is developed from AutoML MNAS Mobile
framework, the building block is mobile inverted bottleneck MBConv with squeeze-and-excitation optimization. 
Then, through a compound scaling method, this baseline is scaled up to obtain EfficientNet-B1
to B7.

![Efficientnet_structure](https://1.bp.blogspot.com/-Cdtb97FtgdA/XO3BHsB7oEI/AAAAAAAAEKE/bmtkonwgs8cmWyI5esVo8wJPnhPLQ5bGQCLcBGAs/s1600/image4.png)

### Default configuration

Here is the Baseline EfficientNet-B0 structure. 
 ![Efficientnet-B0](https://miro.medium.com/max/1106/1*5oQHqmvS_q9Pq_lZ_Rv51A.png)

The following features were implemented in this model:
- General:
    -  XLA support
    -  Mixed precision support
    -  Multi-GPU support using Horovod
    -  Multi-node support using Horovod
    -  Cosine LR Decay
	
- Inference:
    -  Support for inference on single image is included
    -  Support for inference on batch of images is included
    
### Feature support matrix

The following features are supported by this model: 

| Feature               | EfficientNet                
|-----------------------|-------------------------- |                    
|Horovod Multi-GPU training (NCCL)              |       Yes      |           
|Multi-node training    |     Yes     |
|Automatic mixed precision (AMP)   |   Yes    |
|XLA     |    Yes    |
      
         
#### Features



**Multi-GPU training with Horovod**

Our model uses Horovod to implement efficient multi-GPU training with NCCL. For details, see example sources in this repository or see the [TensorFlow tutorial](https://github.com/horovod/horovod/#usage).


**Multi-node training with Horovod**

Our model also uses Horovod to implement efficient multi-node training. 



**Automatic Mixed Precision (AMP)**

Computation graphs can be modified by TensorFlow on runtime to support mixed precision training. Detailed explanation of mixed precision can be found in the next section.


### Mixed precision training

Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in Volta, and following with both the Turing and Ampere architectures, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using [mixed precision training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) previously required two steps:
1.  Porting the model to use the FP16 data type where appropriate.    
2.  Adding loss scaling to preserve small gradient values.

This can now be achieved using Automatic Mixed Precision (AMP) for TensorFlow to enable the full [mixed precision methodology](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#tensorflow) in your existing TensorFlow model code.  AMP enables mixed precision training on Volta, Turing, and NVIDIA Ampere GPU architectures automatically. The TensorFlow framework code makes all necessary model changes internally.

In TF-AMP, the computational graph is optimized to use as few casts as necessary and maximize the use of FP16, and the loss scaling is automatically applied inside of supported optimizers. AMP can be configured to work with the existing tf.contrib loss scaling manager by disabling the AMP scaling with a single environment variable to perform only the automatic mixed-precision optimization. It accomplishes this by automatically rewriting all computation graphs with the necessary operations to enable mixed precision training and automatic loss scaling.

For information about:
-   How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) documentation.
-   Techniques used for mixed precision training, see the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
-   How to access and enable AMP for TensorFlow, see [Using TF-AMP](https://docs.nvidia.com/deeplearning/dgx/tensorflow-user-guide/index.html#tfamp) from the TensorFlow User Guide.

#### Enabling mixed precision

Mixed precision is enabled in TensorFlow by using the Automatic Mixed Precision (TF-AMP) extension which casts variables to half-precision upon retrieval, while storing variables in single-precision format. Furthermore, to preserve small gradient magnitudes in backpropagation, a [loss scaling](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#lossscaling) step must be included when applying gradients. In TensorFlow, loss scaling can be applied statically by using simple multiplication of loss by a constant value or automatically, by TF-AMP. Automatic mixed precision makes all the adjustments internally in TensorFlow, providing two benefits over manual operations. First, programmers need not modify network model code, reducing development and maintenance effort. Second, using AMP maintains forward and backward compatibility with all the APIs for defining and running TensorFlow models.

To enable mixed precision,  you can simply add the `--use_amp` to the command-line used to run the model. This will enable the following code:

```
if params.use_amp:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16', loss_scale='dynamic')
    tf.keras.mixed_precision.experimental.set_policy(policy)
```


#### Enabling TF32

TensorFloat-32 (TF32) is the new math mode in [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) GPUs for handling the matrix math also called tensor operations. TF32 running on Tensor Cores in A100 GPUs can provide up to 10x speedups compared to single-precision floating-point math (FP32) on Volta GPUs. 

TF32 Tensor Cores can speed up networks using FP32, typically with no loss of accuracy. It is more robust than FP16 for models which require high dynamic range for weights or activations.

For more information, refer to the [TensorFloat-32 in the A100 GPU Accelerates AI Training, HPC up to 20x](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) blog post.

TF32 is supported in the NVIDIA Ampere GPU architecture and is enabled by default.



## Setup

The following section lists the requirements that you need to meet in order to start training the EfficientNet model.

### Requirements

This repository contains Dockerfile which extends the TensorFlow NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:
-   [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
-   [TensorFlow 20.08-py3] NGC container or later
-   Supported GPUs:
    - [NVIDIA Volta architecture](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
    - [NVIDIA Turing architecture](https://www.nvidia.com/en-us/geforce/turing/)
    - [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)

For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
-   [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
-   [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)
-   [Running TensorFlow](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html#running)
  
As an alternative  to the use of the Tensorflow2 NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

For multi-node, the sample provided in this repository requires [Enroot](https://github.com/NVIDIA/enroot) and [Pyxis](https://github.com/NVIDIA/pyxis) set up on a [SLURM](https://slurm.schedmd.com) cluster.

## Quick Start Guide

To train your model using mixed or TF32 precision with Tensor Cores or using FP32, perform the following steps using the default parameters of the EfficientNet model on the ImageNet dataset. For the specifics concerning training and inference, see the [Advanced](#advanced) section.

1. Clone the repository.

	```
    git clone https://github.com/NVIDIA/DeepLearningExamples.git
    
    cd DeepLearningExamples/TensorFlow2/Classification/ConvNets/efficientnet
	```

2. Download and prepare the dataset.
          `Runner.py` supports ImageNet with [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets/overview). Refer to  the [TFDS ImageNet readme](https://github.com/tensorflow/datasets/blob/master/docs/catalog/imagenet2012.md) for manual download instructions.

3. Build EfficientNet on top of the NGC container.
           `bash ./scripts/docker/build.sh`

4. Start an interactive session in the NGC container to run training/inference.
           `bash ./scripts/docker/launch.sh`

5. Start training.

    To run training for a standard configuration (DGX A100/DGX-1 V100, AMP/TF32/FP32, 500 Epochs, efficientnet-b0/efficientnet-b4), 
    run one of the scripts in the `./scripts/{B0, B4}/training` directory called `./scripts/{B0, B4}/training/{AMP, TF32, FP32}/convergence_8x{A100-80G, V100-16G, V100-32G}.sh`.
    Ensure ImageNet is mounted in the `/data` directory.
    For example:
    `bash ./scripts/B0/AMP/convergence_8xA100-80G.sh`

6. Start validation/evaluation.

   To run validation/evaluation for a standard configuration (DGX A100/DGX-1 V100, AMP/TF32/FP32, efficientnet-b0/efficientnet-b4), 
   run one of the scripts in the `./scripts/{B0, B4}/evaluation` directory called `./scripts/{B0, B4}/evaluation/evaluation_{AMP, FP32, TF32}_8x{A100-80G, V100-16G, V100-32G}.sh`.
    Ensure ImageNet is mounted in the `/data` directory.
    (Optional) Place the checkpoint in the `--model_dir` location to evaluate on a checkpoint.
    For example:
    `bash ./scripts/B0/evaluation/evaluation_AMP_8xA100-80G.sh`

7. Start inference/predictions.

   To run inference for a standard configuration (DGX A100/DGX-1 V100, AMP/TF32/FP32, efficientnet-b0/efficientnet-b4, batch size 8), 
   run one of the scripts in the `./scripts/{B0, B4}/inference` directory called `./scripts/{B0, B4}/inference/inference_{AMP, FP32, TF32}.sh`.
    Ensure your JPEG images to be ran inference on are mounted in the `/infer_data` directory with this folder structure :
    ```
    infer_data
    |   ├── images
    |   |   ├── image1.JPEG
    |   |   ├── image2.JPEG
    ```
    (Optional) Place the checkpoint in the `--model_dir` location to evaluate on a checkpoint.
    For example:
    `bash ./scripts/B0/inference/inference_{AMP, FP32}.sh`

Now that you have your model trained and evaluated, you can choose to compare your training results with our [Training accuracy results](#training-accuracy-results). You can also choose to benchmark yours performance to [Training performance benchmark](#training-performance-results), or [Inference performance benchmark](#inference-performance-results). Following the steps in these sections will ensure that you achieve the same accuracy and performance results as stated in the [Results](#results) section.

## Advanced

The following sections provide greater details of the dataset, running training and inference, and the training results.

### Scripts and sample code

The following lists the content for each folder:
- `scripts/` - shell scripts to build and launch EfficientNet container on top of NGC container,
and scripts to launch training, evaluation and inference
- `model/` - building blocks and EfficientNet model definitions 
- `runtime/` - detailed procedure for each running mode
- `utils/` - support util functions for `runner.py`

### Parameters

Important parameters for training are listed below with default values.

- `mode` (`train_and_eval`,`train`,`eval`,`prediction`) - the default is `train_and_eval`.
- `arch` - the default is `efficientnet-b0`
- `model_dir` - The folder where model checkpoints are saved (the default is `/workspace/output`)
- `data_dir` - The folder where data resides (the default is `/data/`)
- `augmenter_name` - Type of Augmentation (the default is `autoaugment`)
- `max_epochs` - The number of training epochs (the default is `300`)
- `warmup_epochs` - The number of epochs of warmup (the default is `5`)
- `train_batch_size` - The training batch size per GPU (the default is `32`)
- `eval_batch_size` - The evaluation batch size per GPU (the default is `32`)
- `lr_init` - The learning rate for a batch size of 128, effective learning rate will be automatically scaled according to the global training batch size (the default is `0.008`)

The main script `main.py` specific parameters are:
```
 --model_dir MODEL_DIR
                        The directory where the model and training/evaluation
                        summariesare stored.
  --save_checkpoint_freq SAVE_CHECKPOINT_FREQ
                        Number of epochs to save checkpoint.
  --data_dir DATA_DIR   The location of the input data. Files should be named
                        `train-*` and `validation-*`.
  --mode MODE           Mode to run: `train`, `eval`, `train_and_eval`, `predict` or
                        `export`.
  --arch ARCH           The type of the model, e.g. EfficientNet, etc.
  --dataset DATASET     The name of the dataset, e.g. ImageNet, etc.
  --log_steps LOG_STEPS
                        The interval of steps between logging of batch level
                        stats.
  --use_xla             Set to True to enable XLA
  --use_amp             Set to True to enable AMP
  --num_classes NUM_CLASSES
                        Number of classes to train on.
  --batch_norm BATCH_NORM
                        Type of Batch norm used.
  --activation ACTIVATION
                        Type of activation to be used.
  --optimizer OPTIMIZER
                        Optimizer to be used.
  --moving_average_decay MOVING_AVERAGE_DECAY
                        The value of moving average.
  --label_smoothing LABEL_SMOOTHING
                        The value of label smoothing.
  --max_epochs MAX_EPOCHS
                        Number of epochs to train.
  --num_epochs_between_eval NUM_EPOCHS_BETWEEN_EVAL
                        Eval after how many steps of training.
  --steps_per_epoch STEPS_PER_EPOCH
                        Number of steps of training.
  --warmup_epochs WARMUP_EPOCHS
                        Number of steps considered as warmup and not taken
                        into account for performance measurements.
  --lr_init LR_INIT     Initial value for the learning rate.
  --lr_decay LR_DECAY   Type of LR Decay.
  --lr_decay_rate LR_DECAY_RATE
                        LR Decay rate.
  --lr_decay_epochs LR_DECAY_EPOCHS
                        LR Decay epoch.
  --weight_decay WEIGHT_DECAY
                        Weight Decay scale factor.
  --weight_init {fan_in,fan_out}
                        Model weight initialization method.
  --train_batch_size TRAIN_BATCH_SIZE
                        Training batch size per GPU.
  --augmenter_name AUGMENTER_NAME
                        Type of Augmentation during preprocessing only during
                        training.
  --eval_batch_size EVAL_BATCH_SIZE
                        Evaluation batch size per GPU.
  --resume_checkpoint   Resume from a checkpoint in the model_dir.
  --use_dali            Use dali for data loading and preprocessing of train
                        dataset.
  --use_dali_eval       Use dali for data loading and preprocessing of eval
                        dataset.
  --dtype DTYPE         Only permitted
                        `float32`,`bfloat16`,`float16`,`fp32`,`bf16`
```

### Command-line options

To see the full list of available options and their descriptions, use the `-h` or `--help` command-line option, for example:
`python main.py --help`


### Getting the data

Refer to the [TFDS ImageNet readme](https://github.com/tensorflow/datasets/blob/master/docs/catalog/imagenet2012.md) for manual download instructions.
To train on ImageNet dataset, pass `$path_to_ImageNet_tfrecords` to `$data_dir` in the command-line.

Name the TFRecords in the following scheme:

- Training images - `/data/train-*`
- Validation images - `/data/validation-*`

### Training process

The training process can start from scratch, or resume from a checkpoint.

By default, bash script `scripts/{B0, B4}/training/{AMP, FP32, TF32}/convergence_8x{A100-80G, V100-16G, V100-32G}.sh` will start the training process from scratch with the following settings.
   - Use 8 GPUs by Horovod
   - Has XLA enabled
   - Saves checkpoints after every 5 epochs to `/workspace/output/` folder
   - AMP or FP32 or TF32 based on the folder `scripts/{B0, B4}/training/{AMP, FP32, TF32}`

To resume from a checkpoint, include `--resume_checkpoint` in the command-line and place the checkpoint into `--model_dir`.

#### Multi-node

Multi-node runs can be launched on a Pyxis/enroot Slurm cluster (see [Requirements](#requirements)) with the `run_{B0, B4}_multinode.sub` script with the following command for a 4-node NVIDIA DGX A100 example:

```
PARTITION=<partition_name> sbatch N 4 --ntasks-per-node=8 run_B0_multinode.sub
PARTITION=<partition_name> sbatch N 4 --ntasks-per-node=8 run_B4_multinode.sub
```
 
Checkpoint after `--save_checkpoint_freq` epochs will be saved in `checkpointdir`. The checkpoint will be automatically picked up to resume training in case it needs to be resumed. Cluster partition name has to be provided `<partition_name>`.
 
Note that the `run_{B0, B4}_multinode.sub` script is a starting point that has to be adapted depending on the environment. In particular, variables such as `--container-image` handle the container image to train using and `--datadir` handle the location of the ImageNet data.
 
Refer to the files contents to see the full list of variables to adjust for your system.

### Inference process

Validation is done every epoch and can be also run separately on a checkpointed model.

`bash ./scripts/{B0, B4}/evaluation/evaluation_{AMP, FP32, TF32}_8x{A100-80G, V100-16G, V100-32G}.sh`

Metrics gathered through this process are as follows:

```
- eval_loss
- eval_accuracy_top_1
- eval_accuracy_top_5
- avg_exp_per_second_eval
- avg_exp_per_second_eval_per_GPU
- avg_time_per_exp_eval : Average Latency
- latency_90pct : 90% Latency
- latency_95pct : 95% Latency
- latency_99pct : 99% Latency
```

To run inference on a JPEG image, you have to first store the checkpoint in the `--model_dir` and store the JPEG images in the following directory structure:

    ```
    infer_data
    |   ├── images
    |   |   ├── image1.JPEG
    |   |   ├── image2.JPEG
    ```

Run: 
`bash ./scripts/{B0, B4}/inference/inference_{AMP, FP32, TF32}.sh`

## Performance

The performance measurements in this document were conducted at the time of publication and may not reflect the performance achieved from NVIDIA’s latest software release. For the most up-to-date performance measurements, go to [NVIDIA Data Center Deep Learning Product Performance](https://developer.nvidia.com/deep-learning-performance-training-inference).

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in training and inference modes.

#### Training performance benchmark

Training benchmark for EfficientNet-B0 was run on NVIDIA DGX A100 80GB and NVIDIA DGX-1 V100 16GB.

To benchmark training performance with other parameters, run:

`bash ./scripts/B0/training/{AMP, FP32, TF32}/train_benchmark_8x{A100-80G, V100-16G}.sh`

Training benchmark for EfficientNet-B4 was run on NVIDIA DGX A100- 80GB and NVIDIA DGX-1 V100 32GB.

`bash ./scripts/B4/training/{AMP, FP32, TF32}/train_benchmark_8x{A100-80G, V100-16G}.sh`

#### Inference performance benchmark

Inference benchmark for EfficientNet-B0 was run on NVIDIA DGX A100- 80GB and NVIDIA DGX-1 V100 16GB.

Inference benchmark for EfficientNet-B4 was run on NVIDIA DGX A100- 80GB and NVIDIA DGX-1 V100 32GB.

### Results

The following sections provide details on how we achieved our performance and accuracy in training and inference.

#### Training accuracy results for EfficientNet-B0


##### Training accuracy: NVIDIA DGX A100 (8x A100 80GB)

Our results were obtained by running the training scripts in the tensorflow:21.02-tf2-py3 NGC container on NVIDIA DGX A100 (8x A100 80GB) GPUs.

| GPUs     | Accuracy - TF32  | Accuracy - mixed precision  |   Time to train - TF32  |  Time to train - mixed precision | Time to train speedup (TF32 to mixed precision)        |
|-------------------|-----------------------|-------------|-------|-------------------|---------------------------------------|
|     8              |             77.38          |           77.43  |     19   |  10.5     |     1.8           |
|     16              |           77.46          |         77.62  |        10 |  5.5   |  1.81  |


##### Training accuracy: NVIDIA DGX-1 (8x V100 16GB)

Our results were obtained by running the training scripts in the tensorflow:21.02-tf2-py3 NGC container on NVIDIA DGX-1 (8x V100 16GB) GPUs.

| GPUs     | Accuracy - FP32  | Accuracy - mixed precision  |   Time to train - FP32  |  Time to train - mixed precision | Time to train speedup (FP32 to mixed precision)        |
|-------------------|-----------------------|-------------|-------|-------------------|---------------------------------------|
|     8              |             77.54          |           77.51  |     48   |  44     |     1.09           |
|     32              |           77.38          |         77.62  |        11.48 |  11.44   |  1.003  |


#### Training accuracy results for EfficientNet-B4


##### Training accuracy: NVIDIA DGX A100 (8x A100 80GB)

Our results were obtained by running the training scripts in the tensorflow:21.02-tf2-py3 NGC container on multi-node NVIDIA DGX A100 (8x A100 80GB) GPUs.

| GPUs     | Accuracy - TF32  | Accuracy - mixed precision  |   Time to train - TF32  |  Time to train - mixed precision | Time to train speedup (TF32 to mixed precision)        |
|-------------------|-----------------------|-------------|-------|-------------------|---------------------------------------|
|     32              |             82.69          |           82.69  |     38   |  17.5     |     2.17           |
|     64              |           82.75          |         82.78  |        18 |  8.5   |  2.11  |


##### Training accuracy: NVIDIA DGX-1 (8x V100 32GB)

Our results were obtained by running the training scripts in the tensorflow:21.02-tf2-py3 NGC container on multi-node NVIDIA DGX-1 (8x V100 32GB) GPUs.

| GPUs     | Accuracy - FP32  | Accuracy - mixed precision  |   Time to train - FP32  |  Time to train - mixed precision | Time to train speedup (FP32 to mixed precision)        |
|-------------------|-----------------------|-------------|-------|-------------------|---------------------------------------|
|     32              |     82.78  |    82.78  |    95   |   39.5     |    2.40           |
|     64              |           82.74          |         82.74  |     53 |  19   |  2.78  |

#### Training performance results for EfficientNet-B0


##### Training performance: NVIDIA DGX A100 (8x A100 80GB)

Our results were obtained by running the training benchmark script in the tensorflow:21.02-tf2-py3 NGC container on NVIDIA DGX A100 (8x A100 80GB) GPUs. Performance numbers (in items/images per second) were averaged over 5 entire training epoch.

| GPUs  | Throughput - TF32    | Throughput - mixed precision    | Throughput speedup (TF32 - mixed precision)   | Weak scaling - TF32    | Weak scaling - mixed precision   |     
|-----|-----|-----|-----|------|-------|
|  1  | 1206 | 2549 | 2.11 | 1 | 1 |
|  8  | 9365 | 16336 | 1.74 | 7.76 | 6.41 |
|  16  | 18361 | 33000 | 1.79 | 15.223 | 12.95 |


To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).



##### Training performance: NVIDIA DGX-1 (8x V100 16GB)

Our results were obtained by running the training benchmark script in the tensorflow:21.02-tf2-py3 NGC container on NVIDIA DGX-1 (8x V100 16GB) GPUs. Performance numbers (in items/images per second) were averaged over an entire training epoch.


| GPUs  | Throughput - FP32    | Throughput - mixed precision    | Throughput speedup (FP32 - mixed precision)   | Weak scaling - FP32    | Weak scaling - mixed precision   |     
|-----|-----|-----|-----|------|-------|
|  1  | 629 | 712 | 1.13 | 1 | 1 |
|  8  | 4012 | 4065 | 1.01 | 6.38 | 5.71 |


To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).


#### Training performance results for EfficientNet-B4


##### Training performance: NVIDIA DGX A100 (8x A100 80GB)

Our results were obtained by running the training benchmark script in the tensorflow:21.02-tf2-py3 NGC container on NVIDIA DGX A100 (8x A100 80GB) GPUs. Performance numbers (in items/images per second) were averaged over 5 entire training epoch.

| GPUs  | Throughput - TF32    | Throughput - mixed precision    | Throughput speedup (TF32 - mixed precision)   | Weak scaling - TF32    | Weak scaling - mixed precision   |     
|-----|-----|-----|-----|------|-------|
|  1  | 167 | 394 | 2.34 | 1 | 1 |
|  8  | 1280 | 2984 | 2.33 | 7.66 | 7.57 |
|  32  | 5023 | 11034 | 2.19 | 30.07 | 28.01 |
|  64  | 9838 | 21844 | 2.22 | 58.91 | 55.44 |


To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).



##### Training performance: NVIDIA DGX-1 (8x V100 32GB)

Our results were obtained by running the training benchmark script in the tensorflow:21.02-tf2-py3 NGC container on NVIDIA DGX-1 (8x V100 16GB) GPUs. Performance numbers (in items/images per second) were averaged over an entire training epoch.


| GPUs  | Throughput - FP32    | Throughput - mixed precision    | Throughput speedup (FP32 - mixed precision)   | Weak scaling - FP32    | Weak scaling - mixed precision   |     
|-----|-----|-----|-----|------|-------|
|  1  | 89 | 193 | 2.16 | 1 | 1 |
|  8  | 643 | 1298 | 2.00 | 7.28 | 6.73 |
|  32  | 2095 | 4892 | 2.33 | 23.54 | 25.35 |
|  64  | 4109 | 9666 | 2.35 | 46.17 | 50.08 |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).




#### Inference performance results for EfficientNet-B0

##### Inference performance: NVIDIA DGX A100 (1x A100 80GB)

Our results were obtained by running the inferencing benchmarking script in the tensorflow:21.02-tf2-py3 NGC container on NVIDIA DGX A100 (1x A100 80GB) GPU.

FP16 Inference Latency

| Batch size | Resolution | Throughput Avg | Latency Avg (ms) | Latency 90% (ms) |Latency 95% (ms) |Latency 99% (ms) |
|------------|-----------------|-----|-----|-----|-----|-----|
|      1      |    224x224    | 111 |  8.97   | 8.88  | 8.92 | 8.96 |
|      2      |    224x224      | 233 | 8.56 | 8.44 | 8.5 | 8.54 |
|      4      |    224x224      | 432 | 9.24 | 9.12 | 9.16 | 9.2 |
|      8      |    224x224      | 771 | 10.32 | 10.16 | 10.24 | 10.24 |
|     1024       |    224x224     | 10269 |  102.4   |  102.4   |   102.4  | 102.4 |

TF32 Inference Latency

| Batch size | Resolution | Throughput Avg | Latency Avg (ms) | Latency 90% (ms) | Latency 95% (ms) | Latency 99% (ms) |
|------------|-----------------|-----|-----|-----|-----|-----|
|     1       |      224x224   |   101     |  9.87   | 9.78    |  9.82   | 9.86    |
|     2       |      224x224    | 204 |   9.78  |  9.66   |  9.7   |   9.76  |
|     4       |      224x224    | 381 |  10.48   |  10.36   |  10.4   |  10.44   |
|     8      |      224x224   | 584 |  13.68   |  13.52   |  13.6   | 13.68   |
|      512      |   224x224      | 5480 | 92.16 | 92.16 | 92.16 | 92.16 |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).



##### Inference performance: NVIDIA DGX-1 (1x V100 16GB)

Our results were obtained by running the `inference-script-name.sh` inferencing benchmarking script in the TensorFlow NGC container on NVIDIA DGX-1 (1x V100 16GB) GPU.

FP16 Inference Latency

| Batch size | Resolution | Throughput Avg | Latency Avg (ms) | Latency 90% (ms) |Latency 95% (ms) |Latency 99% (ms) |
|------------|-----------------|-----|-----|-----|-----|-----|
|     1      |    224x224     | 98.8 | 10.12 | 10.03 | 10.06 | 10.10 |
|     2      |    224x224      | 199.3 | 10.02 | 9.9 | 9.94 | 10.0 |
|     4      |    224x224      | 382.5 | 10.44 | 10.28 | 10.36 | 10.4 |
|     8      |    224x224      | 681.2 | 11.68 | 11.52 | 11.6 | 11.68 |
|      256      |   224x224      | 5271 | 48.64 | 46.08 | 46.08 | 48.64 |
FP32 Inference Latency

| Batch size | Resolution | Throughput Avg | Latency Avg (ms) | Latency 90% (ms) | Latency 95% (ms) | Latency 99% (ms) |
|------------|-----------------|-----|-----|-----|-----|-----|
|      1      |    224x224     | 68.39 | 14.62 | 14.45 | 14.51 | 14.56 |
|      2      |    224x224      | 125.62 | 15.92 | 15.78 | 15.82 | 15.82 |
|      4      |    224x224      | 216.41 | 18.48 | 18.24 | 18.4 | 18.44 |
|      8      |    224x224      | 401.60 | 19.92 | 19.6 | 19.76 | 19.84 |
|     128      |   224x224       | 2713 | 47.36 | 46.08 | 46.08 | 47.36 |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).


#### Inference performance results for EfficientNet-B4

##### Inference performance: NVIDIA DGX A100 (1x A100 80GB)

Our results were obtained by running the inferencing benchmarking script in the tensorflow:21.02-tf2-py3 NGC container on NVIDIA DGX A100 (1x A100 80GB) GPU.

FP16 Inference Latency

| Batch size | Resolution | Throughput Avg | Latency Avg (ms) | Latency 90% (ms) |Latency 95% (ms) |Latency 99% (ms) |
|------------|-----------------|-----|-----|-----|-----|-----|
|      1      |    380x380    | 57.54 |  17.37   | 17.24  | 17.30 | 17.35 |
|      2      |    380x380      | 112.06 | 17.84 | 17.7 | 17.76 | 17.82 |
|      4      |    380x380      | 219.71 | 18.2 | 18.08 | 18.12 | 18.16 |
|      8      |    380x380      | 383.39 | 20.8 | 20.64 | 20.72 | 20.8 |
|     128       |    380x380     | 1470 |  87.04   |  85.76   |  85.76  | 87.04 |

TF32 Inference Latency
| Batch size | Resolution | Throughput Avg | Latency Avg (ms) | Latency 90% (ms) | Latency 95% (ms) | Latency 99% (ms) |
|------------|-----------------|-----|-----|-----|-----|-----|
|     1       |      380x380   |   52.68     |  18.98   | 18.86    |  18.91   | 18.96    |
|     2       |      380x380    | 95.32 |   20.98  |  20.84   |  20.9   |  20.96  |
|     4       |      380x380    | 182.14 |  21.96  | 21.84   |  21.88   |  21.92   |
|     8      |      380x380   | 325.72 |  24.56   |  24.4   |  24.4   | 24.48  |
|      64      |   380x380      | 694 | 91.52 | 90.88 | 91.52 | 91.52 |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).



##### Inference performance: NVIDIA DGX-1 (1x V100 32GB)

Our results were obtained by running the `inference-script-name.sh` inferencing benchmarking script in the TensorFlow NGC container on NVIDIA DGX-1 (1x V100 16GB) GPU.

FP16 Inference Latency

| Batch size | Resolution | Throughput Avg | Latency Avg (ms) | Latency 90% (ms) | Latency 95% (ms) | Latency 99% (ms) |
|------------|-----------------|-----|-----|-----|-----|-----|
|     1      |    380x380     | 54.27 | 18.35 | 18.20 | 18.25 | 18.32 |
|     2      |    380x380      | 104.27 | 19.18 | 19.02 | 19.08 | 19.16 |
|     4      |    380x380      | 182.61 | 21.88 | 21.64 | 21.72 | 21.84 |
|     8      |    380x380      | 234.06 | 34.16 | 33.92 | 34.0 | 34.08 |
|      64      |   380x380      | 782.47 | 81.92 | 80.0 | 80.64 | 81.28 |


FP32 Inference Latency

| Batch size | Resolution | Throughput Avg | Latency Avg (ms) | Latency 90% (ms) |Latency 95% (ms) |Latency 99% (ms) |
|------------|-----------------|-----|-----|-----|-----|-----|
|      1      |    380x380     | 30.48 | 32.80 | 32.86 | 31.83 | 32.60 |
|      2      |    380x380      | 58.59 | 34.12 | 31.92 | 33.02 | 33.9 |
|      4      |    380x380      | 111.35 | 35.92 | 35.0 | 35.12 | 35.68 |
|      8      |    380x380      | 199.00 | 40.24 | 38.72 | 39.04 | 40.0 |
|     32      |   380x380       | 307.04  | 104.0 | 104.0 | 104.0 | 104.0 |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

## Release notes

### Changelog

March 2021
- Initial release

### Known issues

- EfficientNet-B0 does not improve training speed by using AMP as compared to FP32, because of the CPU bound Auto-augmentation.



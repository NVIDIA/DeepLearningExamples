# EfficientNet v2-S For TensorFlow 2.6

This repository provides scripts and recipes to train EfficientNet v2-S to achieve state-of-the-art accuracy.
The content of the repository is maintained by NVIDIA and is tested against each NGC monthly released container to ensure consistent accuracy and performance over time.

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
    * [Multi-node](#multi-node)
  * [Inference process](#inference-process)
- [Performance](#performance)
  
  * [Benchmarking](#benchmarking)
  * [Training performance benchmark](#training-performance-benchmark)
    * [Inference performance benchmark](#inference-performance-benchmark)
  * [Results](#results)
    * [Training results for EfficientNet v2-S](#training-results-for-efficientnet-v2-s)
      * [Training accuracy: NVIDIA DGX A100 (8x A100 80GB)](#training-accuracy-nvidia-dgx-a100-8x-a100-80gb)
      * [Training accuracy: NVIDIA DGX-1 (8x V100 32GB)](#training-accuracy-nvidia-dgx-1-8x-v100-32gb)
    * [Training performance results for EfficientNet v2-S](#training-performance-results-for-efficientnet-v2-s)
      * [Training performance: NVIDIA DGX A100 (8x A100 80GB)](#training-performance-nvidia-dgx-a100-8x-a100-80gb)
      * [Training performance: NVIDIA DGX-1 (8x V100 32GB)](#training-performance-nvidia-dgx-1-8x-v100-32gb)
    * [Training EfficientNet v2-S at scale](#training-efficientnet-v2-s-at-scale)
      * [10x NVIDIA DGX-1 V100 (8x V100 32GB)](#10x-nvidia-dgx-1-v100-8x-v100-32gb)
      * [10x NVIDIA DGX A100 (8x A100 80GB)](#10x-nvidia-dgx-a100-8x-a100-80gb)
    * [Inference performance results for EfficientNet v2-S](#inference-performance-results-for-efficientnet-v2-s)
      * [Inference performance: NVIDIA DGX A100 (1x A100 80GB)](#inference-performance-nvidia-dgx-a100-1x-a100-80gb)
      * [Inference performance: NVIDIA DGX-1 (1x V100 32GB)](#inference-performance-nvidia-dgx-1-1x-v100-32gb)
- [Release notes](#release-notes)
  * [Changelog](#changelog)
  * [Known issues](#known-issues)
## Model overview

EfficientNet  TensorFlow 2 is a family of image classification models, which achieve state-of-the-art accuracy, yet being an order-of-magnitude smaller and faster than previous models.
Specifically, this readme covers model v2-S as suggested in [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298).
NVIDIA's implementation of EfficientNet TensorFlow 2 is an optimized version of [TensorFlow Model Garden](https://github.com/tensorflow/models/tree/master/official/vision/image_classification) implementation, 
leveraging mixed precision arithmetic on NVIDIA Volta, NVIDIA Turing, and the NVIDIA Ampere GPU architectures for faster training times while maintaining target accuracy.

The major differences between the papers' original implementations and this version of EfficientNet are as follows:
- Automatic mixed precision (AMP) training support
- Cosine LR decay for better accuracy
- Weight initialization using `fan_out` for better accuracy
- Multi-node training support using Horovod
- XLA enabled for better performance
- Gradient accumulation support
- Lightweight logging using [dllogger](https://github.com/NVIDIA/dllogger)

Other publicly available implementations of EfficientNet include:

- [Tensorflow Model Garden](https://github.com/tensorflow/models/tree/master/official/vision/image_classification)
- [Pytorch version](https://github.com/rwightman/pytorch-image-models)
- [Google's implementation for TPU EfficientNet v1](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
- [Google's implementation for TPU EfficientNet v2](https://github.com/google/automl/tree/master/efficientnetv2)
 

This model is trained with mixed precision Tensor Cores on NVIDIA Volta, NVIDIA Turing, and the NVIDIA Ampere GPU architectures. It provides a push-button solution to pretraining on a corpus of choice. As a result, researchers can get results 1.5--2x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training.  This model is tested against each NGC monthly released container to ensure consistent accuracy and performance over time.

### Model architecture 
EfficientNet v2 is developed based on AutoML and compound scaling, but with a particular emphasis on faster training. For this purpose, the authors have proposed 3 major changes compared to v1: 1) the objective function of AutoML is revised so that the number of flops is now substituted by training time, because FLOPs is not an accurate surrogate of the actual training time; 2) a multi-stage training is proposed where the early stages of training use low resolution images and weak regularization, but the subsequent stages use larger images and stronger regularization; 3) an additional block called fused MBConv is used in AutoML, which replaces the 1x1 depth-wise convolution of MBConv with a regular 3x3 convolution. 

![Efficientnet v2-S](https://api.wandb.ai/files/wandb_fc/images/projects/273565/59864ee4.png)

EfficientNet v2 base model is scaled up using a non-uniform compounding scheme, through which the depth and width of blocks are scaled depending on where they are located in the base architecture. With this approach, the authors have identified the base "small" model, EfficientNet v2-S, and then scaled it up to obtain EfficientNet v2-M,L,XL. Below is the detailed overview of EfficientNet v2-S, which is reproduced in this repository.

### Default configuration
Here is the baseline EfficientNet v2-S structure.
![Efficientnet v2-S](https://api.wandb.ai/files/wandb_fc/images/projects/273565/1bb50059.png)



The following features are supported by this implementation: 
- General:
    -  XLA support
    -  Mixed precision support
    -  Multi-GPU support using Horovod
    -  Multi-node support using Horovod
    -  Cosine LR Decay
	
- Inference:
    -  Support for inference on a single image is included
    -  Support for inference on a batch of images is included

### Feature support matrix
| Feature               | EfficientNet                
|-----------------------|-------------------------- |                    
|Horovod Multi-GPU training (NCCL)              |       Yes      |        
|Multi-GPU training    |     Yes     |   
|Multi-node training    |     Yes     |
|Automatic mixed precision (AMP)   |   Yes    |
|XLA     |    Yes    |
|Gradient Accumulation| Yes |      
|Stage-wise Training| Yes |   

#### Features
**Multi-GPU training with Horovod**

Our model uses Horovod to implement efficient multi-GPU training with NCCL. For details, refer to example sources in this repository or refer to the [TensorFlow tutorial](https://github.com/horovod/horovod/#usage).


**Multi-node training with Horovod**

Our model also uses Horovod to implement efficient multi-node training. 



**Automatic Mixed Precision (AMP)**

Computation graphs can be modified by TensorFlow on runtime to support mixed precision training. A detailed explanation of mixed precision can be found in Appendix.

**Gradient Accumulation**

Gradient Accumulation is supported through a custom train_step function. This feature is enabled only when grad_accum_steps is greater than 1.

**Stage-wise Training**

Stage-wise training was proposed for EfficientNet v2  to further accelerate convergence. In this scheme, the early stages use low resolution images and weak regularization, but the subsequent stages use larger images and stronger regularization. This feature is activated when `--n_stages` is greater than 1. The current codebase allows the user to linearly schedule the following factors in the various stages of training:

| factor                | value in the first stage  | value in the last stage
|-----------------------|-------------------------- |-------------------------- |                          
| image resolution       |     --base_img_size      | --img_size                |
| strength of mixup      |      --base_mixup        | --mixup_alpha             |
| strength of cutmix     |      --base_cutmix       | --cutmix_alpha            |
| strength of random aug.|     --base_randaug_mag   | --raug_magnitude          |

Note that if `--n_stages` is set to 1, then the above hyper-parameters beginning with `base_`  will have no effect.

### Mixed precision training

Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in NVIDIA Volta, and following with both the NVIDIA Turing and NVIDIA Ampere Architectures, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using [mixed precision training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) previously required two steps:
1.  Porting the model to use the FP16 data type where appropriate.    
2.  Adding loss scaling to preserve small gradient values.

This can now be achieved using Automatic Mixed Precision (AMP) for TensorFlow to enable the full [mixed precision methodology](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#tensorflow) in your existing TensorFlow model code.  AMP enables mixed precision training on NVIDIA Volta, NVIDIA Turing, and NVIDIA Ampere GPU architectures automatically. The TensorFlow framework code makes all necessary model changes internally.

In TF-AMP, the computational graph is optimized to use as few casts as necessary and maximize the use of FP16, and the loss scaling is automatically applied inside of supported optimizers. AMP can be configured to work with the existing tf.contrib loss scaling manager by disabling the AMP scaling with a single environment variable to perform only the automatic mixed-precision optimization. It accomplishes this by automatically rewriting all computation graphs with the necessary operations to enable mixed precision training and automatic loss scaling.

For information about:
-   How to train using mixed precision, refer to the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) documentation.
-   Techniques used for mixed precision training, refer to the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
-   How to access and enable AMP for TensorFlow, refer to [Using TF-AMP](https://docs.nvidia.com/deeplearning/dgx/tensorflow-user-guide/index.html#tfamp) from the TensorFlow User Guide.

#### Enabling mixed precision

Mixed precision is enabled in TensorFlow by using the Automatic Mixed Precision (TF-AMP) extension which casts variables to half-precision upon retrieval, while storing variables in single-precision format. Furthermore, to preserve small gradient magnitudes in backpropagation, a [loss scaling](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#lossscaling) step must be included when applying gradients. In TensorFlow, loss scaling can be applied statically by using simple multiplication of loss by a constant value or automatically, by TF-AMP. Automatic mixed precision makes all the adjustments internally in TensorFlow, providing two benefits over manual operations. First, programmers need not modify network model code, reducing development and maintenance effort. Second, using AMP maintains forward and backward compatibility with all the APIs for defining and running TensorFlow models.

To enable mixed precision,  you can simply add the `--use_amp` to the command-line used to run the model. This will enable the following code:

```
if params.use_amp:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16', loss_scale='dynamic')
    tf.keras.mixed_precision.experimental.set_policy(policy)
```


#### Enabling TF32

TensorFloat-32 (TF32) is the new math mode in [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) GPUs for handling the matrix math also called tensor operations. TF32 running on Tensor Cores in A100 GPUs can provide up to 10x speedups compared to single-precision floating-point math (FP32) on NVIDIA Volta GPUs. 

TF32 Tensor Cores can speed up networks using FP32, typically with no loss of accuracy. It is more robust than FP16 for models which require a high dynamic range for weights or activations.

For more information, refer to the [TensorFloat-32 in the A100 GPU Accelerates AI Training, HPC up to 20x](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) blog post.

TF32 is supported in the NVIDIA Ampere GPU architecture and is enabled by default.


## Setup
The following section lists the requirements that you need to meet in order to start training the EfficientNet model.

### Requirements
This repository contains a Dockerfile that extends the TensorFlow NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:
-   [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
-   [TensorFlow 21.09-py3] NGC container or later
-   Supported GPUs:
    - [NVIDIA Volta architecture](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
    - [NVIDIA Turing architecture](https://www.nvidia.com/en-us/geforce/turing/)
    - [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)

For more information about how to get started with NGC containers, refer to the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
-   [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
-   [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)
-   [Running TensorFlow](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html#running)
  
As an alternative  to the use of the Tensorflow2 NGC container, to set up the required environment or create your own container, refer to the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

For multi-node, the sample provided in this repository requires [Enroot](https://github.com/NVIDIA/enroot) and [Pyxis](https://github.com/NVIDIA/pyxis) set up on a [SLURM](https://slurm.schedmd.com) cluster.

## Quick Start Guide

To train your model using mixed or TF32 precision with Tensor Cores or using FP32, perform the following steps using the default parameters of the EfficientNet model on the ImageNet dataset. For the specifics concerning training and inference, refer to the [Advanced](#advanced) section.

1. Clone the repository.

	```
    git clone https://github.com/NVIDIA/DeepLearningExamples.git
    
    cd DeepLearningExamples/TensorFlow2/Classification/ConvNets/efficientnet
	```

2. Download and prepare the dataset.
          `Runner.py` supports ImageNet with [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets/overview). Refer to  the [TFDS ImageNet readme](https://github.com/tensorflow/datasets/blob/master/docs/catalog/imagenet2012.md) for manual download instructions.

3. Build EfficientNet on top of the NGC container.
           `bash ./scripts/docker/build.sh YOUR_DESIRED_CONTAINER_NAME`

4. Start an interactive session in the NGC container to run training/inference. **Ensure that `launch.sh` has the correct path to ImageNet on your machine and that this path is mounted onto the `/data` directory, because this is where training and evaluation scripts search for data.** 
   
           `bash ./scripts/docker/launch.sh YOUR_DESIRED_CONTAINER_NAME`

5. Start training.

    To run training for a standard configuration, **under the container default entry point `/workspace`**, run one of the scripts in the `./efficinetnet_v2/S/training/{AMP,TF32,FP32}/convergence_8x{A100-80G, V100-16G, V100-32G}.sh`. For example:

    `bash ./efficinetnet_v2/S/training/AMP/convergence_8xA100-80G.sh`

6. Start validation/evaluation.

   To run validation/evaluation for a standard configuration, **under the container default entry point `/workspace`**, run one of the scripts in the `./efficinetnet_v2/S/evaluation/evaluation_{AMP,FP32,TF32}_8x{A100-80G,V100-16G,V100-32G}.sh`. The evaluation script is configured to  use the checkpoint specified in the `checkpoint` file for evaluation. The specified checkpoint will be read from the location passed by `--model_dir'.For example:
    
    `bash ./efficinetnet_v2/S/evaluation/evaluation_AMP_A100-80G.sh`

7. Start inference/predictions.

   To run inference for a standard configuration, **under the container default entry point `/workspace`**, run one of the scripts in the `./efficinetnet_v2/S/inference/inference_{AMP,FP32,TF32}.sh`.
    Ensure your JPEG images used to run inference on are mounted in the `/infer_data` directory with this folder structure :
    ```
    infer_data
    |   ├── images
    |   |   ├── image1.JPEG
    |   |   ├── image2.JPEG
    ```
    For example:
    `bash ./efficinetnet_v2/S/inference/inference_AMP.sh`

Now that you have your model trained and evaluated, you can choose to compare your training results with our [Training accuracy results](#training-accuracy-results). You can also choose to benchmark your performance to [Training performance benchmark](#training-performance-results), or [Inference performance benchmark](#inference-performance-results). Following the steps in these sections will ensure that you achieve the same accuracy and performance results as stated in the [Results](#results) section.

## Advanced
The following sections provide greater details of the dataset, running training and inference, and the training results.

### Scripts and sample code
The repository is structured as follows:
- `scripts/` - shell scripts to build and launch EfficientNet container on top of NGC container,
- `efficientnet_{v1,v2}` scripts to launch training, evaluation and inference
- `model/` - building blocks and EfficientNet model definitions 
- `runtime/` - detailed procedure for each running mode
- `utils/` - support util functions for learning rates, optimizers, etc.
- `dataloader/` provides data pipeline utils
- `config/` contains model definitions

### Parameters
The hyper parameters can be grouped into model-specific hyperparameters (e.g., #layers ) and general hyperparameters (e.g., #training epochs). 

The model-specific hyperparameters are to be defined in a python module, which must be passed in the command line via --cfg ( `python main.py --cfg config/efficientnet_v2/s_cfg.py`). To override model-specific hyperparameters, you can use a comma separated list of k=v pairs (e.g., `python main.py --cfg config/efficientnet_v2/s_cfg.py --mparams=bn_momentum=0.9,dropout=0.5`).

The general hyperparameters and their default values can be found in `utils/cmdline_helper.py`. The user can override these hyperparameters in the command line (e.g., `python main.py --cfg config/efficientnet_v2/s_cfg.py --data_dir xx --train_batch_size 128`).  Here is a list of important hyperparameters:

- `--mode` (`train_and_eval`,`train`,`eval`,`prediction`) - the default is `train_and_eval`.
- `--use_amp`    Set to True to enable AMP
- `--use_xla`    Set to True to enable XLA
- `--model_dir` The folder where model checkpoints are saved (the default is `/workspace/output`)
- `--data_dir`  The folder where data resides (the default is `/data/`)
- `--log_steps`  The interval of steps between logging of batch level stats.

  
- `--augmenter_name`  Type of data augmentation 
- `--raug_num_layers`  Number of layers used in the random data augmentation scheme
- `--raug_magnitude`  Strength of transformations applied in the random data augmentation scheme 
- `--cutmix_alpha`  Cutmix parameter used in the last stage of training.
- `--mixup_alpha`  Mixup parameter used in the last stage of training.
- `--defer_img_mixing`  Move image mixing ops from the data loader to the model/GPU (faster training)


- `--eval_img_size` Size of images used for evaluation
- `--eval_batch_size`  The evaluation batch size per GPU 

- `--n_stages`   Number of stages used for stage-wise training 
- `--train_batch_size`  The training batch size per GPU
- `--train_img_size`  Size of images used in the last stage of training
- `--base_img_size`  Size of images used in the first stage of training
- `--max_epochs`  The number of training epochs 
- `--warmup_epochs`  The number of epochs of warmup 
- `--moving_average_decay` The decay weight used for EMA 
- `--lr_init` The learning rate for a batch size of 128, effective learning rate will be automatically scaled according to the global training batch size: `lr=lr_init * global_BS/128 where global_BS=train_batch_size*n_GPUs`  
- `--lr_decay` Learning rate decay policy 
- `--weight_decay` Weight decay coefficient
- `--save_checkpoint_freq` Number of epochs to save checkpoints

**NOTE**: Avoid changing the default values of the general hyperparameters provided in `utils/cmdline_helper.py`. The reason is that some other models supported by this repository may rely on such default values. If you wish to change the values, override them via the command line.

### Command-line options
To display the full list of available options and their descriptions, use the `-h` or `--help` command-line option, for example:
`python main.py --help`


### Getting the data

Refer to the [TFDS ImageNet readme](https://github.com/tensorflow/datasets/blob/master/docs/catalog/imagenet2012.md) for manual download instructions.
To train on the ImageNet dataset, pass `$path_to_ImageNet_tfrecords` to `$data_dir` in the command-line.

Name the TFRecords in the following scheme:

- Training images - `/data/train-*`
- Validation images - `/data/validation-*`



### Training process
The training process can start from scratch, or resume from a checkpoint.

By default, bash script `scripts/S/training/{AMP,FP32,TF32}/convergence_8x{A100-80G,V100-16G,V100-32G}.sh` will start the training process with the following settings.
   - Use 8 GPUs by Horovod
   - Has XLA enabled
   - Saves checkpoints after every 10 epochs to `/workspace/output/` folder
   - AMP or FP32 or TF32 based on the folder `scripts/S/training/{AMP, FP32, TF32}`

The training starts from scratch if `--model_dir` has no checkpoints in it. To resume from a checkpoint, place the checkpoint into `--model_dir` and make sure the `checkpoint` file points to it.
#### Multi-node
Multi-node runs can be launched on a Pyxis/enroot Slurm cluster (refer to [Requirements](#requirements)) with the `run_S_multinode.sub` script with the following command for a 4-node NVIDIA DGX A100 example:

```
PARTITION=<partition_name> sbatch N 4 --ntasks-per-node=8 run_S_multinode.sub
```
 
Checkpoints will be saved after `--save_checkpoint_freq` epochs at `checkpointdir`. The latest checkpoint will be automatically picked up to resume training in case it needs to be resumed. Cluster partition name has to be provided `<partition_name>`.
 
Note that the `run_S_multinode.sub` script is a starting point that has to be adapted depending on the environment. In particular, pay attention to the variables such as `--container-image`, which handles the container image to train, and `--datadir`, which handles the location of the ImageNet data.
 
Refer to the scripts to find the full list of variables to adjust for your system.

### Inference process

Validation can be done either during training (when `--mode train_and_eval` is used) or in a post-training setting (`--mode eval`) on a checkpointed model. The evaluation script expects data in the tfrecord format.

`bash ./scripts/S/evaluation/evaluation_{AMP,FP32,TF32}_{A100-80G,V100-16G,V100-32G}.sh`

Metrics gathered through this process are listed below:

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
The scripts used for inference expect the inference data in the following directory structure:

    ```
    infer_data
    |   ├── images
    |   |   ├── image1.JPEG
    |   |   ├── image2.JPEG
    ```

Run: 
`bash ./scripts/S/inference/inference_{AMP,FP32,TF32}.sh`


## Performance
The performance measurements in this document were conducted at the time of publication and may not reflect the performance achieved from NVIDIA’s latest software release. For the most up-to-date performance measurements, go to [NVIDIA Data Center Deep Learning Product Performance](https://developer.nvidia.com/deep-learning-performance-training-inference).
### Benchmarking

The following section shows how to run benchmarks measuring the model performance in training and inference modes.

#### Training performance benchmark

Training benchmark for EfficientNet v2-S was run on NVIDIA DGX A100 80GB and NVIDIA DGX-1 V100 32GB.

`bash ./scripts/S/training/{AMP, FP32, TF32}/train_benchmark_8x{A100-80G, V100-32G}.sh`

#### Inference performance benchmark

Inference benchmark for EfficientNet v2-S was run on NVIDIA DGX A100 80GB and NVIDIA DGX-1 V100 32GB.

### Results
The following sections provide details on how we achieved our performance and accuracy in training and inference.
#### Training results for EfficientNet v2-S

##### Training accuracy: NVIDIA DGX A100 (8x A100 80GB)
Our results were obtained by running the training scripts in the tensorflow:21.09-tf2-py3 NGC container on multi-node NVIDIA DGX A100 (8x A100 80GB) GPUs. We evaluated the models using both the original and EMA weights and selected the higher accuracy to report.

<!---
|      8   |        83.87%    |            83.93%           |          33hrs          |            13.5hrs               |                   2.44                                 |
--->

| GPUs     | Accuracy - TF32  | Accuracy - mixed precision  |   Time to train - TF32  |  Time to train - mixed precision | Time to train speedup (TF32 to mixed precision)        |
|----------|------------------|-----------------------------|-------------------------|----------------------------------|--------------------------------------------------------|
|      8   |        83.87%    |            83.93%           |          32hrs          |            14hrs               |                      2.28                             |# PBR
|     16   |        83.89%    |            83.83%           |          16hrs          |             7hrs                 |                   2.28                                 |

##### Training accuracy: NVIDIA DGX-1 (8x V100 32GB)
Our results were obtained by running the training scripts in the tensorflow:21.09-tf2-py3 NGC container on multi-node NVIDIA DGX V100 (8x V100 32GB) GPUs. We evaluated the models using both the original and EMA weights and selected the higher accuracy to report.

<!---
[//]: |      8  |         83.86%   |            84.0%            |          126.5hrs       |            59hrs                 |                   2.14                                 | # RNO
--->

| GPUs     | Accuracy - FP32  | Accuracy - mixed precision  |   Time to train - FP32  |  Time to train - mixed precision | Time to train speedup (FP32 to mixed precision)        |
|----------|------------------|-----------------------------|-------------------------|----------------------------------|--------------------------------------------------------|
|     8    |         83.86%   |            84.0%            |          90.3hrs        |            55hrs                 |                   1.64                                 | # PBR
|     16   |        83.75%    |            83.87%           |          60.5hrs        |             28.5hrs              |                   2.12                                 | # RNO
|     32   |        83.81%    |            83.82%           |          30.2hrs        |             15.5hrs              |                   1.95                                 | # RNO



#### Training performance results for EfficientNet v2-S
##### Training performance: NVIDIA DGX A100 (8x A100 80GB)
EfficientNet v2-S uses images of increasing resolution during training. Since throughput changes depending on the image size, we have  measured throughput based on the image size used in the last stage of training (300x300).
<!---
# |  8  |         3100           |        7000                     |           2.25                                |          7.94          |             7.36                 | # without PBR
--->
| GPUs  | Throughput - TF32    | Throughput - mixed precision    | Throughput speedup (TF32 - mixed precision)   | Weak scaling - TF32    | Weak scaling - mixed precision   |     
|-----|------------------------|---------------------------------|-----------------------------------------------|------------------------|----------------------------------|
|  1  |         390            |        950                      |           2.43                                |          1             |             1                    |
|  8  |            2800        |         6600                    |            2.35                               |            7.17       |                6.94              | # PBR
|  16 |         5950           |        14517                    |           2.43                                |          15.25         |             15.28                |

##### Training performance: NVIDIA DGX-1 (8x V100 32GB)



EfficientNet v2-S uses images of increasing resolution during training. Since throughput changes depending on the image size, we have  measured throughput based on the image size used in the last stage of training (300x300).

| GPUs  | Throughput - FP32    | Throughput - mixed precision    | Throughput speedup (FP32 - mixed precision)   | Weak scaling - FP32    | Weak scaling - mixed precision   |     
|-----|------------------------|---------------------------------|-----------------------------------------------|------------------------|----------------------------------|
|  1  |         156            |        380                      |           2.43                                |          1             |             1                    | # DLCLUSTER
|  8  |         952            |        1774                     |           1.86                                |          6.10          |             4.66                 | # PBR
|  16 |         1668           |        3750                     |           2.25                                |          10.69         |             9.86                 | # RNO
|  32 |         3270           |        7250                     |           2.2                                 |          20.96         |             19.07                | # RNO



#### Training EfficientNet v2-S at scale
##### 10x NVIDIA DGX-1 V100 (8x V100 32GB)
We trained EfficientNet v2-S at scale using 10 DGX-1 machines each having 8x V100 32GB GPUs. We used the same set of hyperparameters and NGC container as before. Also, throughput numbers were measured in the last stage of training. The accuracy was selected as the better between that of the original weights and EMA weights.

| # Nodes | GPUs      | Optimizer |Accuracy - mixed precision   |Time to train - mixed precision | Time to train speedup        | Throughput - mixed precision    |  Throughput scaling    |
|----------|----------|-----------|-----------------------------|--------------------------------|------------------------------|---------------------------------|------------------------|
|  1       |      8   |    RMSPROP|             84.0%           |            55hrs               |                   1          |           1774                  |           1            |
| 10       |    80    |    RMSPROP|            83.76%           |        6.5hrs                  |                   8.46      |           16039                 |           9.04         |

<!---
| 20       |    160   |            83.74%           |        3.5hrs                  |                   15.71      |           31260                 |           17.62         |
--->
##### 10x NVIDIA DGX A100 (8x A100 80GB)
We trained EfficientNet v2-S at scale using 10 DGX A100 machines each having 8x A100 80GB GPUs. This training setting has an effective batch size of 36800 (460x8x10), which requires advanced optimizers particularly designed for large-batch training. For this purpose, we used the nvLAMB optimizer with the following hyper parameters: lr_warmup_epochs=10, beta_1=0.9, beta_2=0.999, epsilon=0.000001, grad_global_clip_norm=1, lr_init=0.00005, weight_decay=0.00001. As before, we used tensorflow:21.09-tf2-py3 NGC container and measured throughput numbers in the last stage of training. The accuracy was selected as the better between that of the original weights and EMA weights.

| # Nodes | GPUs      | Optimizer | Accuracy - mixed precision  |Time to train - mixed precision | Time to train speedup        | Throughput - mixed precision    |  Throughput scaling    | 
|----------|----------|-----------|-----------------------------|--------------------------------|------------------------------|---------------------------------|------------------------|
|  1       |      8   |    RMSPROP|       83.93%                |            14hrs               |                   1          |           6600                  |           1            | #PBR
| 10       |    80    |    nvLAMB |       82.84%                |          1.84hrs               |                   7.60       |           62130                 |           9.41         |




#### Inference performance results for EfficientNet v2-S

##### Inference performance: NVIDIA DGX A100 (1x A100 80GB)

Our results were obtained by running the inferencing benchmarking script in the tensorflow:21.09-tf2-py3 NGC container on the NVIDIA DGX A100 (1x A100 80GB) GPU.

FP16 Inference Latency

| Batch size | Resolution | Throughput Avg | Latency Avg (ms) | Latency 90% (ms) |Latency 95% (ms) |Latency 99% (ms) |
|------------|------------------|--------|--------|--------|--------|--------|
|      1      |    384x384      | 29     | 33.99  | 33.49  | 33.69  | 33.89  |
|      8      |    384x384      | 204    | 39.14  | 38.61  | 38.82  | 39.03  |
|      32     |    384x384      | 772    | 41.35  | 40.64  | 40.90  | 41.15  |
|     128     |    384x384      | 1674   | 76.45  | 74.20  | 74.70  | 75.80  |
|     256     |    384x384      | 1960   | 130.57 | 127.34 | 128.74 | 130.27 |
|     512     |    384x384      | 2062   | 248.18 | 226.80 | 232.86 | 248.18 |
|     1024    |    384x384      | 2032   | 503.73 | 461.78 | 481.50 | 503.73 |

TF32 Inference Latency

| Batch size | Resolution | Throughput Avg | Latency Avg (ms) | Latency 90% (ms) | Latency 95% (ms) | Latency 99% (ms) |
|-------------|-----------------|--------|--------|--------|--------|--------|
|      1      |    384x384      |    39  |  25.55 | 25.05  | 25.26  | 25.47  |
|      8      |    384x384      |   244  |  32.75 | 32.16  | 32.40  | 32.64  | 
|      32     |    384x384      |   777  |  41.13 | 40.69  | 40.84  | 41.00  | 
|     128     |    384x384      |   1000 |  127.94| 126.71 | 127.12 | 127.64 | 
|     256     |    384x384      |   1070 |  239.08| 235.45 | 236.79 | 238.39 | 
|     512     |    384x384      |  1130  | 452.71 | 444.64 | 448.18 | 452.71 | 


### Inference performance: NVIDIA DGX-1 (1x V100 32GB)

Our results were obtained by running the inferencing benchmarking script in the tensorflow:21.09-tf2-py3 NGC container on the NVIDIA DGX V100 (1x V100 32GB) GPU.

FP16 Inference Latency

| Batch size | Resolution | Throughput Avg | Latency Avg (ms) | Latency 90% (ms) |Latency 95% (ms) |Latency 99% (ms) |
|------------|------------------|--------|--------|--------|--------|--------|
|      1      |    384x384      | 29     | 33.99  | 33.49  | 33.69  | 33.89  |
|      8      |    384x384      | 184    | 43.37  | 42.80  | 43.01  | 43.26  |
|      32     |    384x384      | 592    | 52.96  | 53.20  | 53.45  | 53.72  |
|     128     |    384x384      | 933    | 136.98 | 134.44 | 134.79 | 136.05 |
|     256     |    384x384      | 988    | 258.94 | 251.56 | 252.86 | 257.92 |


FP32 Inference Latency

| Batch size | Resolution | Throughput Avg | Latency Avg (ms) | Latency 90% (ms) | Latency 95% (ms) | Latency 99% (ms) |
|-------------|-----------------|--------|--------|--------|--------|--------|
|      1      |    384x384      |    45  |  22.02 | 21.87  | 21.93  | 21.99  |
|      8      |    384x384      |   260  |  30.73 | 30.33  | 30.51  | 30.67  | 
|      32     |    384x384      |   416  |  76.89 | 76.57  | 76.65  | 76.74  | 
|     128     |    384x384      |   460  |  278.24| 276.56 | 276.93 | 277.74 | 




## Release notes

### Changelog

February 2022
- Initial release

### Known issues
- EfficientNet v2 might run into OOM or might exhibit a significant drop in throughput at the onset of the last stage of training. The fix is to resume training from the latest checkpoint. 




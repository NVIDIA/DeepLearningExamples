# UNet Medical Image Segmentation for TensorFlow 1.x
 
This repository provides a script and recipe to train UNet Medical to achieve state of the art accuracy, and is tested and maintained by NVIDIA.

## Table of Contents
 
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
     * [Multi-dataset](#multi-dataset)
   * [Training process](#training-process)
   * [Inference process](#inference-process)
- [Performance](#performance)   
   * [Benchmarking](#benchmarking)
     * [Training performance benchmark](#training-performance-benchmark)
     * [Inference performance benchmark](#inference-performance-benchmark)
   * [Results](#results)
     * [Training accuracy results](#training-accuracy-results)
       * [Training accuracy: NVIDIA DGX A100 (8x A100 40GB)](#training-accuracy-nvidia-dgx-a100-8x-a100-40gb)  
       * [Training accuracy: NVIDIA DGX-1 (8x V100 16GB)](#training-accuracy-nvidia-dgx-1-8x-v100-16gb)
     * [Training performance results](#training-performance-results)
       * [Training performance: NVIDIA DGX A100 (8x A100 40GB)](#training-performance-nvidia-dgx-a100-8x-a100-40gb) 
       * [Training performance: NVIDIA DGX-1 (8x V100 16GB)](#training-performance-nvidia-dgx-1-8x-v100-16gb)
     * [Inference performance results](#inference-performance-results)
        * [Inference performance: NVIDIA DGX A100 (1x A100 40GB)](#inference-performance-nvidia-dgx-a100-1x-a100-40gb)
        * [Inference performance: NVIDIA DGX-1 (1x V100 16GB)](#inference-performance-nvidia-dgx-1-1x-v100-16gb)
- [Release notes](#release-notes)
   * [Changelog](#changelog)
   * [Known issues](#known-issues)
 
 
## Model overview
 
The UNet model is a convolutional neural network for 2D image segmentation. This repository contains a UNet implementation as described in the original paper [UNet: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), without any alteration.
 
This model is trained with mixed precision using Tensor Cores on Volta, Turing, and the NVIDIA Ampere GPU architectures. Therefore, researchers can get results  2.2x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.

### Model architecture
 
UNet was first introduced by Olaf Ronneberger, Philip Fischer, and Thomas Brox in the paper: [UNet: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). UNet allows for seamless segmentation of 2D images, with high accuracy and performance, and can be adapted to solve many different segmentation problems.
 
The following figure shows the construction of the UNet model and its different components. UNet is composed of a contractive and an expanding path, that aims at building a bottleneck in its centermost part through a combination of convolution and pooling operations. After this bottleneck, the image is reconstructed through a combination of convolutions and upsampling. Skip connections are added with the goal of helping the backward flow of gradients in order to improve the training.
 
![UNet](images/unet.png)

Figure 1. UNet architecture
 
### Default configuration
 
UNet consists of a contractive (left-side) and expanding (right-side) path. It repeatedly applies unpadded convolutions followed by max pooling for downsampling. Every step in the expanding path consists of an upsampling of the feature maps and a concatenation with the correspondingly cropped feature map from the contractive path.
 
### Feature support matrix
 
The following features are supported by this model.
 
| **Feature** | **UNet Medical** |
|---------------------------------|-----|
| Automatic mixed precision (AMP) | Yes |
| Horovod Multi-GPU (NCCL)        | Yes |
| Accelerated Linear Algebra (XLA)| Yes |
 
#### Features
 
**Automatic Mixed Precision (AMP)**
 
This implementation of UNet uses AMP to implement mixed precision training. It allows us to use FP16 training with FP32 master weights by modifying just a few lines of code.
 
**Horovod**
 
Horovod is a distributed training framework for TensorFlow, Keras, PyTorch, and MXNet. The goal of Horovod is to make distributed deep learning fast and easy to use. For more information about how to get started with Horovod, see the [Horovod: Official repository](https://github.com/horovod/horovod).
 
**Multi-GPU training with Horovod**
 
Our model uses Horovod to implement efficient multi-GPU training with NCCL. For details, see example sources in this repository or see the [TensorFlow tutorial](https://github.com/horovod/horovod/#usage).
 
**XLA support (experimental)**
 
XLA is a domain-specific compiler for linear algebra that can accelerate TensorFlow models with potentially no source code changes. The results are improvements in speed and memory usage: most internal benchmarks run ~1.1-1.5x faster after XLA is enabled.
 
### Mixed precision training
 
Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in the Volta and Turing architecture, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using [mixed precision training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) previously required two steps:
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

TensorFloat-32 (TF32) is the new math mode in [NVIDIA A100](#https://www.nvidia.com/en-us/data-center/a100/) GPUs for handling the matrix math also called tensor operations. TF32 running on Tensor Cores in A100 GPUs can provide up to 10x speedups compared to single-precision floating-point math (FP32) on Volta GPUs. 

TF32 Tensor Cores can speed up networks using FP32, typically with no loss of accuracy. It is more robust than FP16 for models which require high dynamic range for weights or activations.

For more information, refer to the [TensorFloat-32 in the A100 GPU Accelerates AI Training, HPC up to 20x](#https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) blog post.

TF32 is supported in the NVIDIA Ampere GPU architecture and is enabled by default.

## Setup
 
The following section lists the requirements in order to start training the UNet Medical model.
 
### Requirements
 
This repository contains Dockerfile which extends the TensorFlow NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- TensorFlow 20.06-tf1-py3 [NGC container](https://ngc.nvidia.com/registry/nvidia-tensorflow)
-   GPU-based architecture:
    - [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
    - [NVIDIA Turing](https://www.nvidia.com/en-us/geforce/turing/)
    - [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)

 
For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
- [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
- [Accessing And Pulling From The NGC container registry](https://docs.nvidia.com/deeplearning/dgx/user-guide/index.html#accessing_registry)
- [Running TensorFlow](https://docs.nvidia.com/deeplearning/dgx/tensorflow-release-notes/running.html#running)
 
For those unable to use the TensorFlow NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## Quick Start Guide
 
To train your model using mixed precision with Tensor Cores or using FP32, perform the following steps using the default parameters of the UNet model on the [EM segmentation challenge dataset](http://brainiac2.mit.edu/isbi_challenge/home). These steps enable you to build the UNet TensorFlow NGC container, train and evaluate your model, and generate predictions on the test data. Furthermore, you can then choose to:
* compare your evaluation accuracy with our [Training accuracy results](#training-accuracy-results),
* compare your training performance with our [Training performance benchmark](#training-performance-benchmark),
* compare your inference performance with our [Inference performance benchmark](#inference-performance-benchmark).
 
For the specifics concerning training and inference, see the [Advanced](#advanced) section.
 
1. Clone the repository.
 
   Executing this command will create your local repository with all the code to run UNet.
  
   ```bash
   git clone https://github.com/NVIDIA/DeepLearningExamples
   cd DeepLearningExamples/TensorFlow/Segmentation/UNet_Medical_TF
 
2. Build the UNet TensorFlow NGC container.
 
   This command will use the `Dockerfile` to create a Docker image named `unet_tf`, downloading all the required components automatically.
  
   ```
   docker build -t unet_tf .
   ```
  
   The NGC container contains all the components optimized for usage on NVIDIA hardware.
 
3. Start an interactive session in the NGC container to run preprocessing/training/inference.
 
   The following command will launch the container and mount the `./data` directory as a volume to the `/data` directory inside the container, and `./results` directory to the `/results` directory in the container.
  
   ```bash
   mkdir data
   mkdir results
   docker run --runtime=nvidia -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --rm --ipc=host -v ${PWD}/data:/data -v ${PWD}/results:/results unet_tf:latest /bin/bash
   ```
  
   Any datasets and experiment results (logs, checkpoints, etc.) saved to `/data` or `/results` will be accessible
   in the `./data` or `./results` directory on the host, respectively.
 
4. Download and preprocess the data.
  
   The UNet script `main.py` operates on data from the [ISBI Challenge](http://brainiac2.mit.edu/isbi_challenge/home), the dataset originally employed in the [UNet paper](https://arxiv.org/abs/1505.04597). The data is available to download upon registration on the website.
    
   Training and test data are composed of 3 multi-page `TIF` files, each containing 30 2D-images (around 30 Mb total). Once downloaded, the data can be used to run the training and benchmark scripts described below, by pointing `main.py` to its location using the `--data_dir` flag.
  
   **Note:** Masks are only provided for training data.
 
5. Start training.
  
   After the Docker container is launched, the training with the [default hyperparameters](#default-parameters) (for example 1/8 GPUs FP32/TF-AMP) can be started with:
  
   ```bash
   bash examples/unet{_TF-AMP}_{1,8}GPU.sh <path/to/dataset> <path/to/checkpoint>
   ```
  
   For example, to run with full precision (FP32) on 1 GPU from the project’s folder, simply use:
  
   ```bash
   bash examples/unet_1GPU.sh /data /results
   ```
  
   This script will launch a training on a single fold and store the model’s checkpoint in <path/to/checkpoint> directory. 
  
   The script can be run directly by modifying flags if necessary, especially the number of GPUs, which is defined after the `-np` flag. Since the test volume does not have labels, 20% of the training data is used for validation in 5-fold cross-validation manner. The number of fold can be changed using `--crossvalidation_idx` with an integer in range 0-4. For example, to run with 4 GPUs using fold 1 use:
  
   ```bash
   horovodrun -np 4 python main.py --data_dir /data --model_dir /results --batch_size 1 --exec_mode train --crossvalidation_idx 1 --xla --amp
   ```
  
   Training will result in a checkpoint file being written to `./results` on the host machine.
 
6. Start validation/evaluation.
  
   The trained model can be evaluated by passing the `--exec_mode evaluate` flag. Since evaluation is carried out on a validation dataset, the `--crossvalidation_idx` parameter should be filled. For example:
  
   ```bash
   python main.py --data_dir /data --model_dir /results --batch_size 1 --exec_mode evaluate --crossvalidation_idx 0 --xla --amp
   ```
  
   Evaluation can also be triggered jointly after training by passing the `--exec_mode train_and_evaluate` flag.
 
7. Start inference/predictions.
   To run inference on a checkpointed model, run:
   ```bash
   bash examples/unet_INFER{_TF-AMP}.sh <path/to/dataset> <path/to/checkpoint>
   ```
   For example:
   ```bash
   bash examples/unet_INFER_FP32.sh /data /results
   ```
  
   Now that you have your model trained and evaluated, you can choose to compare your training results with our [Training accuracy results](#training-accuracy-results). You can also choose to benchmark the performance of your training [Training performance benchmark](#training-performance-benchmark), or [Inference performance benchmark](#inference-performance-benchmark). Following the steps in these sections will ensure that you achieve the same accuracy and performance results as stated in the [Results](#results) section.
 
## Advanced
 
The following sections provide greater details of the dataset, running training and inference, and the training results.
 
### Scripts and sample code
 
In the root directory, the most important files are:
* `main.py`: Serves as the entry point to the application.
* `Dockerfile`: Container with the basic set of dependencies to run UNet.
* `requirements.txt`: Set of extra requirements for running UNet.
 
The `utils/` folder encapsulates the necessary tools to train and perform inference using UNet. Its main components are:
* `cmd_util.py`: Implements the command-line arguments parsing.
* `data_loader.py`: Implements the data loading and augmentation.
* `model_fn.py`: Implements the logic for training and inference.
* `hooks/training_hook.py`: Collects different metrics during training.
* `hooks/profiling_hook.py`: Collects different metrics to be used for benchmarking and testing.
* `parse_results.py`: Implements the intermediate results parsing.
* `setup.py`: Implements helper setup functions.
 
The `model/` folder contains information about the building blocks of UNet and the way they are assembled. Its contents are:
* `layers.py`: Defines the different blocks that are used to assemble UNet
* `unet.py`: Defines the model architecture using the blocks from the `layers.py` script
 
Other folders included in the root directory are:
* `examples/`: Provides examples for training and benchmarking UNet
* `images/`: Contains a model diagram
 
### Parameters
 
The complete list of the available parameters for the main.py script contains:
* `--exec_mode`: Select the execution mode to run the model (default: `train`). Modes available:
  * `train` - trains model from scratch.
  * `evaluate` - loads checkpoint (if available) and performs evaluation on validation subset (requires `--crossvalidation_idx` other than `None`).
  * `train_and_evaluate` - trains model from scratch and performs validation at the end (requires `--crossvalidation_idx` other than `None`).
  * `predict` - loads checkpoint (if available) and runs inference on the test set. Stores the results in `--model_dir` directory.
  * `train_and_predict` - trains model from scratch and performs inference.
* `--model_dir`: Set the output directory for information related to the model (default: `/results`).
* `--log_dir`: Set the output directory for logs (default: None).
* `--data_dir`: Set the input directory containing the dataset (default: `None`).
* `--batch_size`: Size of each minibatch per GPU (default: `1`).
* `--crossvalidation_idx`: Selected fold for cross-validation (default: `None`).
* `--max_steps`: Maximum number of steps (batches) for training (default: `1000`).
* `--seed`: Set random seed for reproducibility (default: `0`).
* `--weight_decay`: Weight decay coefficient (default: `0.0005`).
* `--log_every`: Log performance every n steps (default: `100`).
* `--learning_rate`: Model’s learning rate (default: `0.0001`).
* `--augment`: Enable data augmentation (default: `False`).
* `--benchmark`: Enable performance benchmarking (default: `False`). If the flag is set, the script runs in a benchmark mode - each iteration is timed and the performance result (in images per second) is printed at the end. Works for both `train` and `predict` execution modes.
* `--warmup_steps`: Used during benchmarking - the number of steps to skip (default: `200`). First iterations are usually much slower since the graph is being constructed. Skipping the initial iterations is required for a fair performance assessment.
* `--xla`: Enable accelerated linear algebra optimization (default: `False`).
* `--amp`: Enable automatic mixed precision (default: `False`).
 
### Command line options
 
To see the full list of available options and their descriptions, use the `-h` or `--help` command-line option, for example:
```bash
python main.py --help
```
 
The following example output is printed when running the model:
```python main.py --help
usage: main.py [-h]
              [--exec_mode {train,train_and_predict,predict,evaluate,train_and_evaluate}]
              [--model_dir MODEL_DIR] --data_dir DATA_DIR [--log_dir LOG_DIR]
              [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]
              [--crossvalidation_idx CROSSVALIDATION_IDX]
              [--max_steps MAX_STEPS] [--weight_decay WEIGHT_DECAY]
              [--log_every LOG_EVERY] [--warmup_steps WARMUP_STEPS]
              [--seed SEED] [--augment] [--benchmark]
              [--amp] [--xla]
 
UNet-medical
 
optional arguments:
 -h, --help            show this help message and exit
 --exec_mode {train,train_and_predict,predict,evaluate,train_and_evaluate}
                       Execution mode of running the model
 --model_dir MODEL_DIR
                       Output directory for information related to the model
 --data_dir DATA_DIR   Input directory containing the dataset for training
                       the model
 --log_dir LOG_DIR     Output directory for training logs
 --batch_size BATCH_SIZE
                       Size of each minibatch per GPU
 --learning_rate LEARNING_RATE
                       Learning rate coefficient for AdamOptimizer
 --crossvalidation_idx CROSSVALIDATION_IDX
                       Chosen fold for cross-validation. Use None to disable
                       cross-validation
 --max_steps MAX_STEPS
                       Maximum number of steps (batches) used for training
 --weight_decay WEIGHT_DECAY
                       Weight decay coefficient
 --log_every LOG_EVERY
                       Log performance every n steps
 --warmup_steps WARMUP_STEPS
                       Number of warmup steps
 --seed SEED           Random seed
 --augment             Perform data augmentation during training
 --benchmark           Collect performance metrics during training
 --amp                 Train using TF-AMP
 --xla                 Train using XLA
```
 
## Getting the data

The UNet model uses the [EM segmentation challenge dataset](http://brainiac2.mit.edu/isbi_challenge/home). Test images provided by the organization were used to produce the resulting masks for submission. The challenge's data is made available upon registration.

Training and test data are comprised of three 512x512x30 `TIF` volumes (`test-volume.tif`, `train-volume.tif` and `train-labels.tif`). Files `test-volume.tif` and `train-volume.tif` contain grayscale 2D slices to be segmented. Additionally, training masks are provided in `train-labels.tif` as a 512x512x30 `TIF` volume, where each pixel has one of two classes:
* 0 indicating the presence of cellular membrane,
* 1 corresponding to background.
 
The objective is to produce a set of masks that segment the data as accurately as possible. The results are expected to be submitted as a 32-bit `TIF` 3D image, with values between `0` (100% membrane certainty) and `1` (100% non-membrane certainty).

#### Dataset guidelines
 
The training and test datasets are given as stacks of 30 2D-images provided as a multi-page `TIF` that can be read using the Pillow library and NumPy (both Python packages are installed by the `Dockerfile`).
 
Initially, data is loaded from a multi-page `TIF` file and converted to 512x512x30 NumPy arrays with the use of Pillow. The process of loading, normalizing and augmenting the data contained in the dataset can be found in the `data_loader.py` script.
 
These NumPy arrays are fed to the model through `tf.data.Dataset.from_tensor_slices()`, in order to achieve high performance.
 
The voxel intensities then normalized to an interval `[-1, 1]`, whereas labels are one-hot encoded for their later use in dice or pixel-wise cross-entropy loss, becoming 512x512x30x2 tensors.
 
If augmentation is enabled, the following set of augmentation techniques are applied:
* Random horizontal flipping
* Random vertical flipping
* Crop to a random dimension and resize to input dimension
* Random brightness shifting
 
In the end, images are reshaped to 388x388 and padded to 572x572 to fit the input of the network. Masks are only reshaped to 388x388 to fit the output of the network. Moreover, pixel intensities are clipped to the `[-1, 1]` interval.
 
#### Multi-dataset
 
This implementation is tuned for the EM segmentation challenge dataset. Using other datasets is possible, but might require changes to the code (data loader) and tuning some hyperparameters (e.g. learning rate, number of iterations).
 
In the current implementation, the data loader works with NumPy arrays by loading them at the initialization, and passing them for training in slices by `tf.data.Dataset.from_tensor_slices()`. If you’re able to fit your dataset into the memory, then convert the data into three NumPy arrays - training images, training labels, and testing images (optional). If your dataset is large, you will have to adapt the optimizer for the lazy-loading of data. For a walk-through, check the [TensorFlow tf.data API guide](https://www.tensorflow.org/guide/data_performance)
 
The performance of the model depends on the dataset size.
Generally, the model should scale better for datasets containing more data. For a smaller dataset, you might experience lower performance.
 
### Training process
 
The model trains for a total 6,400 batches (6,400 / number of GPUs), with the default UNet setup:
* Adam optimizer with learning rate of 0.0001.
 
This default parametrization is applied when running scripts from the `./examples` directory and when running `main.py` without explicitly overriding these parameters. By default, the training is in full precision. To enable AMP, pass the `--amp` flag. AMP can be enabled for every mode of execution.
 
The default configuration minimizes a function _L = 1 - DICE + cross entropy_ during training.
 
The training can be run directly without using the predefined scripts. The name of the training script is `main.py`. Because of the multi-GPU support, training should always be run with the Horovod distributed launcher like this:
```bash
horovodrun -np <number/of/gpus> python main.py --data_dir /data [other parameters]
```
 
*Note:* When calling the `main.py` script manually, data augmentation is disabled. In order to enable data augmentation, use the `--augment` flag in your invocation.
 
The main result of the training are checkpoints stored by default in `./results/` on the host machine, and in the `/results` in the container. This location can be controlled
by the `--model_dir` command-line argument, if a different location was mounted while starting the container. In the case when the training is run in `train_and_predict` mode, the inference will take place after the training is finished, and inference results will be stored to the `/results` directory.
 
If the `--exec_mode train_and_evaluate` parameter was used, and if `--crossvalidation_idx` parameter is set to an integer value of {0, 1, 2, 3, 4}, the evaluation of the validation set takes place after the training is completed. The results of the evaluation will be printed to the console.
### Inference process
 
Inference can be launched with the same script used for training by passing the `--exec_mode predict` flag:
```bash
python main.py --exec_mode predict --data_dir /data --model_dir <path/to/checkpoint> [other parameters]
```
 
The script will then:
* Load the checkpoint from the directory specified by the `<path/to/checkpoint>` directory (`/results`),
* Run inference on the test dataset,
* Save the resulting binary masks in a `TIF` format.
 
## Performance
 
### Benchmarking
 
The following section shows how to run benchmarks measuring the model performance in training and inference modes.
 
#### Training performance benchmark
 
To benchmark training, run one of the `TRAIN_BENCHMARK` scripts in `./examples/`:
```bash
bash examples/unet_TRAIN_BENCHMARK{_TF-AMP}_{1, 8}GPU.sh <path/to/dataset> <path/to/checkpoint> <batch/size>
```
For example, to benchmark training using mixed-precision on 8 GPUs use:
```bash
bash examples/unet_TRAIN_BENCHMARK_TF-AMP_8GPU.sh <path/to/dataset> <path/to/checkpoint> <batch/size>
```
 
Each of these scripts will by default run 200 warm-up iterations and benchmark the performance during training in the next 800 iterations.
 
To have more control, you can run the script by directly providing all relevant run parameters. For example:
```bash
horovodrun -np <num/of/gpus> python main.py --exec_mode train --benchmark --augment --data_dir <path/to/dataset> --model_dir <optional, path/to/checkpoint> --batch_size <batch/size> --warmup_steps <warm-up/steps> --max_steps <max/steps>
```
 
At the end of the script, a line reporting the best train throughput will be printed.
 
#### Inference performance benchmark
 
To benchmark inference, run one of the scripts in `./examples/`:
```bash
bash examples/unet_INFER_BENCHMARK{_TF-AMP}.sh <path/to/dataset> <path/to/checkpoint> <batch/size>
```
 
For example, to benchmark inference using mixed-precision:
```bash
bash examples/unet_INFER_BENCHMARK_TF-AMP.sh <path/to/dataset> <path/to/checkpoint> <batch/size>
```
 
Each of these scripts will by default run 200 warm-up iterations and benchmark the performance during inference in the next 400 iterations.
 
To have more control, you can run the script by directly providing all relevant run parameters. For example:
```bash
python main.py --exec_mode predict --benchmark --data_dir <path/to/dataset> --model_dir <optional, path/to/checkpoint> --batch_size <batch/size> --warmup_steps <warm-up/steps> --max_steps <max/steps>
```
 
At the end of the script, a line reporting the best inference throughput will be printed.
 
### Results
 
The following sections provide details on how we achieved our performance and accuracy in training and inference.
 
#### Training accuracy results

##### Training accuracy: NVIDIA DGX A100 (8x A100 40GB)
 
The following table lists the average DICE score across 5-fold cross-validation. Our results were obtained by running the `examples/unet_TRAIN{_TF-AMP}_{1, 8}GPU.sh` training script in the `tensorflow:20.06-tf1-py3` NGC container on NVIDIA DGX A100 (8x A100 40GB) GPUs.
 
| GPUs | Batch size / GPU | Accuracy - TF32  | Accuracy - mixed precision  |   Time to train - TF32 [min]  |  Time to train - mixed precision [min] | Time to train speedup (TF32 to mixed precision) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 8 | 0.8908 | 0.8910 | 22 | 10 | 2.2 |
| 8 | 8 | 0.8938 | 0.8942 | 2.6 | 2.5 | 1.04 |


##### Training accuracy: NVIDIA DGX-1 (8x V100 16GB)
 
The following table lists the average DICE score across 5-fold cross-validation. Our results were obtained by running the `examples/unet_TRAIN_{FP32, TF-AMP}_{1, 8}GPU.sh` training script in the `tensorflow:20.06-tf1-py3` NGC container on NVIDIA DGX-1 with (8x V100 16GB) GPUs.
 
| GPUs | Batch size / GPU | Accuracy - FP32 | Accuracy - mixed precision | Time to train - FP32 [min] | Time to train - mixed precision [min] | Time to train speedup (FP32 to mixed precision) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 8 | 0.8910 | 0.8903 | 48 | 19 | 2.53 |
| 8 | 8 | 0.8942 | 0.8940 | 7 | 7.5 | 0.93 |
 
To reproduce this result, start the Docker container interactively and run one of the TRAIN scripts:
```bash
bash examples/unet_TRAIN{_TF-AMP}_{1, 8}GPU.sh <path/to/dataset> <path/to/checkpoint> <batch/size>
```
 for example
```bash
bash examples/unet_TRAIN_TF-AMP_8GPU.sh /data /results 8
```

This command will launch a script which will run 5-fold cross-validation training for 40,000 iterations and print the validation DICE score and cross-entropy loss. The time reported is for one fold, which means that the training for 5 folds will take 5 times longer. The default batch size is 8, however if you have less than 16 Gb memory card and you encounter GPU memory issue you should decrease the batch size. The logs of the runs can be found in `/results` directory once the script is finished.
 
**Learning curves**

The following image show the training loss as a function of iteration for training using DGX A100 (TF32 and TF-AMP) and DGX-1 V100 (FP32 and TF-AMP).
![LearningCurves](images/U-NetMed_TF1_conv.png)

#### Training performance results

##### Training performance: NVIDIA DGX A100 (8x A100 40GB)

Our results were obtained by running the `examples/unet_TRAIN_BENCHMARK{_TF-AMP}_{1, 8}GPU.sh` training script in the `examples/unet_TRAIN_BENCHMARK_{TF-AMP, FP32}_{1, 8}GPU.sh` NGC container on NVIDIA DGX A100 (8x A100 40GB) GPUs. Performance numbers (images per second) were averaged over 1000 iterations, excluding the first 200 warm-up steps.

| GPUs | Batch size / GPU | Throughput - TF32 [img/s] | Throughput - mixed precision [img/s] | Throughput speedup (TF32 - mixed precision) | Weak scaling - TF32 | Weak scaling - mixed precision |
|:----:|:----------------:|:-------------------------:|:------------------------------------:|:-------------------------------------------:|:-------------------:|:------------------------------:|
|  1   |        1         |           29.81           |                64.22                 |                    2.15                     |          -          |               -                |
|  1   |        8         |           46.53           |                120.08                |                    2.58                     |          -          |               -                |
|  8   |        1         |          169.62           |                293.31                |                    1.73                     |        5.69         |              4.57              |
|  8   |        8         |          304.64           |                738.64                |                    2.42                     |        6.55         |              6.15              |

##### Training performance: NVIDIA DGX-1 (8x V100 16GB)
 
Our results were obtained by running the `examples/unet_TRAIN_BENCHMARK{_TF-AMP}_{1, 8}GPU.sh` training script in the `tensorflow:20.06-tf1-py3` NGC container on NVIDIA DGX-1 with (8x V100 16GB) GPUs. Performance numbers (in images per second) were averaged over 1000 iterations, excluding the first 200 warm-up steps.

| GPUs | Batch size / GPU | Throughput - FP32 [img/s] | Throughput - mixed precision [img/s] | Throughput speedup (FP32 - mixed precision) | Weak scaling - FP32 | Weak scaling - mixed precision |
|:----:|:----------------:|:-------------------------:|:------------------------------------:|:-------------------------------------------:|:-------------------:|:------------------------------:|
|  1   |        1         |           15.70           |                39.62                 |                    2.52                     |          -          |               -                |
|  1   |        8         |           18.85           |                60.28                 |                    3.20                     |          -          |               -                |
|  8   |        1         |          102.52           |                212.51                |                    2.07                     |        6.53         |              5.36              |
|  8   |        8         |          141.75           |                403.88                |                    2.85                     |        7.52         |              6.70              |

 
To achieve these same results, follow the steps in the [Training performance benchmark](#training-performance-benchmark) section.
 
Throughput is reported in images per second. Latency is reported in milliseconds per image.
 
#### Inference performance results

##### Inference performance: NVIDIA DGX A100 (1x A100 40GB)

Our results were obtained by running the `examples/unet_INFER_BENCHMARK{_TF-AMP}.sh` inferencing benchmarking script in the `tensorflow:20.06-tf1-py3` NGC container on NVIDIA DGX A100 (1x A100 40GB) GPU.

FP16

| Batch size | Resolution | Throughput Avg [img/s] | Latency Avg [ms] | Latency 90% [ms] | Latency 95% [ms] | Latency 99% [ms] |
|:----------:|:----------:|:----------------------:|:----------------:|:----------------:|:----------------:|:----------------:|
|     1      | 572x572x1  |         251.11         |      3.983       |      3.990       |      3.991       |      3.993       |
|     2      | 572x572x1  |         179.70         |      11.130      |      11.138      |      11.139      |      11.142      |
|     4      | 572x572x1  |         197.53         |      20.250      |      20.260      |      20.262      |      20.266      |
|     8      | 572x572x1  |         382.48         |      24.050      |      29.356      |      30.372      |      32.359      |
|     16     | 572x572x1  |         400.58         |      45.759      |      55.615      |      57.502      |      61.192      |

TF32

| Batch size | Resolution | Throughput Avg [img/s] | Latency Avg [ms] | Latency 90% [ms] | Latency 95% [ms] | Latency 99% [ms] |
|:----------:|:----------:|:----------------------:|:----------------:|:----------------:|:----------------:|:----------------:|
|     1      | 572x572x1  |         88.80          |      11.261      |      11.264      |      11.264      |      11.265      |
|     2      | 572x572x1  |         104.62         |      19.120      |      19.149      |      19.155      |      19.166      |
|     4      | 572x572x1  |         117.02         |      34.184      |      34.217      |      34.223      |      34.235      |
|     8      | 572x572x1  |         131.54         |      65.094      |      72.577      |      74.009      |      76.811      |
|     16     | 572x572x1  |         137.41         |     121.552      |     130.795      |     132.565      |     136.027      |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

##### Inference performance: NVIDIA DGX-1 (1x V100 16GB)
 
Our results were obtained by running the `examples/unet_INFER_BENCHMARK{_TF-AMP}.sh` inferencing benchmarking script in the `tensorflow:20.06-tf1-py3` NGC container on NVIDIA DGX-1 with (1x V100 16GB) GPU.
 
FP16
 
| Batch size | Resolution | Throughput Avg [img/s] | Latency Avg [ms] | Latency 90% [ms] | Latency 95% [ms] | Latency 99% [ms] |
|:----------:|:----------:|:----------------------:|:----------------:|:----------------:|:----------------:|:----------------:|
|     1      | 572x572x1  |         127.11         |      7.868       |      7.875       |      7.876       |      7.879       |
|     2      | 572x572x1  |         140.32         |      14.256      |      14.278      |      14.283      |      14.291      |
|     4      | 572x572x1  |         148.28         |      26.978      |      27.005      |      27.010      |      27.020      |
|     8      | 572x572x1  |         178.28         |      48.432      |      54.613      |      55.797      |      58.111      |
|     16     | 572x572x1  |         181.94         |      94.812      |     106.743      |     109.028      |     113.496      |
 
FP32
 
| Batch size | Resolution | Throughput Avg [img/s] | Latency Avg [ms] | Latency 90% [ms] | Latency 95% [ms] | Latency 99% [ms] |
|:----------:|:----------:|:----------------------:|:----------------:|:----------------:|:----------------:|:----------------:|
|     1      | 572x572x1  |         47.32          |      21.133      |      21.155      |      21.159      |      21.167      |
|     2      | 572x572x1  |         51.43          |      38.888      |      38.921      |      38.927      |      38.940      |
|     4      | 572x572x1  |         53.56          |      74.692      |      74.763      |      74.777      |      74.804      |
|     8      | 572x572x1  |         54.41          |     152.733      |     163.148      |     165.142      |     169.042      |
|     16     | 572x572x1  |         67.11          |     245.775      |     259.548      |     262.186      |     267.343      |
 
To achieve these same results, follow the steps in the [Inference performance benchmark](#inference-performance-benchmark) section.
 
Throughput is reported in images per second. Latency is reported in milliseconds per batch.
 
## Release notes
 
### Changelog

June 2020
* Updated training and inference accuracy with A100 results
* Updated training and inference performance with A100 results

February 2020
* Updated README template
* Added cross-validation for accuracy measurements
* Changed optimizer to Adam and updated accuracy table
* Updated performance values
 
July 2019
* Added inference benchmark for T4
* Added inference example scripts
* Added inference benchmark measuring latency
* Added TRT/TF-TRT support
* Updated Pre-trained model on NGC registry
 
June 2019
* Updated README template
 
April 2019
* Initial release
 
 
### Known issues
 
There are no known issues in this release.
 
 
 


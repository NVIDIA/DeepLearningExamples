# Wide & Deep Recommender Model Training For TensorFlow

This repository provides a script and recipe to train the Wide and Deep Recommender model to achieve state-of-the-art accuracy and is tested and maintained by NVIDIA.

## Table Of Contents

- [Model overview](#model-overview)
    * [Model architecture](#model-architecture)
    * [Applications and dataset](#applications-and-dataset)
    * [Default configuration](#default-configuration)
    * [Feature support matrix](#feature-support-matrix)
	    * [Features](#features)
    * [Mixed precision](#mixed-precision)
	    * [Enabling mixed precision](#enabling-mixed-precision)
            * [Impact of mixed precision on training accuracy](#impact-of-mixed-precision-on-training-accuracy)
            * [Impact of mixed precision on inference accuracy](#impact-of-mixed-precision-on-inference-accuracy)
    * [Glossary](#glossary)
- [Setup](#setup)
    * [Requirements](#requirements)
- [Quick Start Guide](#quick-start-guide)
- [Advanced](#advanced)
    * [Scripts and sample code](#scripts-and-sample-code)
    * [Parameters](#parameters)
    * [Command-line options](#command-line-options)
    * [Getting the data](#getting-the-data)
        * [Dataset guidelines](#dataset-guidelines)
    * [Training process](#training-process)
- [Performance](#performance)
    * [Benchmarking](#benchmarking)
        * [Training performance benchmark](#training-performance-benchmark)
    * [Results](#results)
        * [Training accuracy results](#training-accuracy-results)
            * [Training accuracy: NVIDIA DGX-1 (8x V100 16G)](#training-accuracy-nvidia-dgx-1-8x-v100-16g)
            * [Training accuracy plots](#training-accuracy-plots)
            * [Training stability test](#training-stability-test)
        * [Training performance results](#training-performance-results)
            * [Training performance: NVIDIA DGX-1 (8x V100 16G)](#training-performance-nvidia-dgx-1-8x-v100-16g)
- [Release notes](#release-notes)
    * [Changelog](#changelog)
    * [Known issues](#known-issues)


## Model overview

Recommendation systems drive engagement on many of the most popular online platforms. As the volume of data available to power these systems grows exponentially, data scientists are increasingly turning from more traditional machine learning methods to highly expressive deep learning models to improve the quality of their recommendations. Google's [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) has emerged as a popular model for these problems both for its robustness to signal sparsity as well as its user-friendly implementation in [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedClassifier).

The differences between this Wide & Deep Recommender Model and the model from the paper is the size of the Deep part of the model. Originally, in Google's paper, the fully connected part was three layers of 1024, 512, and 256 neurons. Our model consists of 5 layers each of 1024 neurons. 

The model enables you to train a recommender model that combines the memorization of the Wide part and generalization of the Deep part of the network.

This model is trained with mixed precision using Tensor Cores on NVIDIA Volta and Turing GPUs. Therefore, researchers can get results 1.44 times faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.

### Model architecture

Wide & Deep refers to a class of networks that use the output of two parts working in parallel - wide model and deep model - to make predictions of recommenders. The wide model is a generalized linear model of features together with their transforms. The deep model is a series of 5 hidden MLP layers of 1024 neurons each beginning with a dense embedding of features. The architecture is presented in Figure 1.

<p align="center">
  <img width="70%" src="https://developer.download.nvidia.com/w-and-d-recommender/model.svg" />
  <br>
Figure 1. The architecture of the Wide & Deep model.</a>
</p>

### Applications and dataset

As a reference dataset, we used a subset of [the features engineered](https://github.com/gabrielspmoreira/kaggle_outbrain_click_prediction_google_cloud_ml_engine) by the 19th place finisher in the [Kaggle Outbrain Click Prediction Challenge](https://www.kaggle.com/c/outbrain-click-prediction/). This competition challenged competitors to predict the likelihood with which a particular ad on a website's display would be clicked on. Competitors were given information about the user, display, document, and ad in order to train their models. More information can be found [here](https://www.kaggle.com/c/outbrain-click-prediction/data).


### Default configuration

For reference, and to give context to the acceleration numbers described below, some important properties of our features and model are as follows:

- Features
    - Request Level
        - 16 scalar numeric features `(shape=(1,)`)
        - 12 one-hot categorical features (all `int` dtype)
            - 5 indicator embeddings
	        - sizes 2, 2, 3, 3, 6
            - 7 trainable embeddings
                - all except two have an embedding size of 64 (remaining two have 128), though it's important to note for *all* categorical features that we *do not* leverage that information to short-circuit the lookups by treating them as a single multi-hot lookup. Our API is fully general to any combination of embedding sizes.
                - all use hash bucketing with `num_buckets=` 300k, 100k, 4k, 2.5k, 2k, 1k, and 300 respectively
        - 3 multi-hot categorical features (all `int` dtype)
            - all trainable embeddings
            - all with embedding size 64
            - all use hash bucketing with `num_buckets=` 10k, 350, and 100 respectively
    - Item Level
        - 16 scalar numeric features
        - 4 one hot categorical features (all `int` dtype)
            - embedding sizes of 128, 64, 64, 64 respectively
            - hash bucketing with `num_buckets=` 250k, 4k, 2.5k, and 1k respectively
        - 3 multi-hot categorical features (all `int` dtype)
            - all with embedding size 64
            - hash bucketing with `num_buckets=` 10k, 350, and 100 respectively
    - All features are used in both wide *and* deep branches of the network

- Model
    - Total embedding dimension is 1328
    - 5 hidden layers each with size 1024
    - Output dimension is 1 (probability of click)

### Feature support matrix

The following features are supported by this model: 

| Feature               | Wide & Deep                
|-----------------------|--------------------------
|Horovod Multi-GPU      | Yes        
|Automatic mixed precision (AMP)   | Yes          
         
#### Features

Horovod

Horovod is a distributed training framework for TensorFlow, Keras, PyTorch and MXNet. The goal of Horovod is to make distributed deep learning fast and easy to use. For more information about how to get started with Horovod, see the [Horovod: Official repository](https://github.com/horovod/horovod).

Multi-GPU training with Horovod

Our model uses Horovod to implement efficient multi-GPU training with NCCL. For details, see example sources in this repository or see the [TensorFlow tutorial](https://github.com/horovod/horovod/#usage).


### Mixed precision

Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in the Volta and Turing architecture, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using mixed precision training requires two steps:
1.  Porting the model to use the FP16 data type where appropriate.    
2.  Adding loss scaling to preserve small gradient values.

The ability to train deep learning networks with lower precision was introduced in the Pascal architecture and first supported in [CUDA 8](https://devblogs.nvidia.com/parallelforall/tag/fp16/) in the NVIDIA Deep Learning SDK.

For information about:
-   How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html) documentation.
-   Techniques used for mixed precision training, see the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
-   How to access and enable AMP for TensorFlow, see [Using TF-AMP](https://docs.nvidia.com/deeplearning/dgx/tensorflow-user-guide/index.html#tfamp) from the TensorFlow User Guide.

#### Enabling mixed precision

To enable Wide & Deep training to use mixed precision you don't need to perform input quantization, only an additional flag `--amp` to the training script is needed (see [Quick Start Guide](#quick-start-guide)).

##### Impact of mixed precision on training accuracy
The accuracy of training, measured with MAP@12 metric was not impacted by enabling mixed precision. The obtained results were statistically similar (i.e. similar run-to-run variance was observed, with standard deviation of the level of `0.002`).

##### Impact of mixed precision on inference accuracy
For our reference model, the average absolute error on the probability of interaction induced by reduced precision inference is `0.0002`, producing a near-perfect fit between predictions produced by full and mixed precision models. Moreover, this error is uncorrelated with the magnitude of the predicted value, which means for most predictions of interest (i.e. greater than `0.01` or `0.1` likelihood of interaction), the relative magnitude of the error is approaching the noise floor of the problem.



### Glossary

Request level features: Features that describe the person or object _to which_ we wish to make recommendations.

Item level features: Features that describe those objects which we are considering recommending.

## Setup

The following section lists the requirements that you need to meet in order to start training the Wide & Deep model.

### Requirements

This repository contains Dockerfile which extends the TensorFlow NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:
-   [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
-   [20.02-tf1-py3](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) NGC container
-   [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) based GPU

For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
-   [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
-   [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)
-   [Running TensorFlow](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html#running)

For those unable to use the TensorFlow NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## Quick Start Guide

To train your model using mixed precision with Tensor Cores or using FP32, perform the following steps using the default parameters of the Wide & Deep model on the Outbrain dataset. For the specifics concerning training and inference, see the [Advanced](#advanced) section.

1. Clone the repository.

```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/TensorFlow/Recommendation/WideAndDeep
```

2.  Download the Outbrain dataset.

The Outbrain dataset can be downloaded from [Kaggle](https://www.kaggle.com/c/outbrain-click-prediction/data) (requires Kaggle account).
Unzip the downloaded archive e.g. to `/raid/outbrain/orig` and set the `HOST_OUTBRAIN_PATH` variable to the parent directory:

```bash
HOST_OUTBRAIN_PATH=/raid/outbrain
```

3.  Build the Wide & Deep Tensorflow NGC container.

```bash
docker build . -t wide_deep
```

4.  Start an interactive session in the NGC container to run preprocessing/training/inference.

```bash
docker run --runtime=nvidia --rm -ti -v ${HOST_OUTBRAIN_PATH}:/outbrain wide_deep /bin/bash
```
5. Start preprocessing.

```bash
bash scripts/preproc.sh 4096
```
The result of preprocessing scripts are prebatched TFRecords. The argument to the script is the prebatch
size (4096 is the default).

6. Start training.

Single GPU:
```bash
python -m trainer.task --gpu --amp --global_batch_size 131072 --num_epochs 120
```
8 GPU:
```bash
mpiexec --allow-run-as-root --bind-to socket -np 8 python -m trainer.task --gpu --amp --hvd --global_batch_size 131072 --num_epochs 120
```

If you want to run validation or inference, you can either use the checkpoint obtained from the training 
commands above, or download the pretrained checkpoint from NGC. 

In order to download the checkpoint from NGC, visit [ngc.nvidia.com](https://ngc.nvidia.com) website and
browse the available models.
Download the checkpoint files and unzip them to some path, e.g. to `/raid/outbrain/checkpoints/`
(which is the default path for storing the checkpoints during training).


7. Start validation/evaluation.

In order to validate the checkpoint on the evaluation set, run the `task.py` script with `--evaluate` flag:

```bash
python -m trainer.task --gpu --amp --evaluate --model_dir /outbrain/checkpoints 
```

8. Start inference/predictions.

In order to run inference and predict the results, run the `task.py`
script with `--predict` flag:

```bash
python -m trainer.task --gpu --amp --predict --model_dir /outbrain/checkpoints
```


## Advanced

The following sections provide greater details of the dataset, running training, and the training results.

### Scripts and sample code

These are the important scripts in this repository:
*  `trainer/task.py` - Python script for training the Wide & Deep recommender model
*  `trainer/features.py` - Python file describing the request and item level features

### Parameters

These are the important parameters in the `trainer/task.py` script:

```
--model_dir: Path to model checkpoint directory
--deep_hidden_units: [DEEP_LAYER1 DEEP_LAYER2 ...] hidden units per layer, separated by spaces
--prebatch_size: Number of samples in each pre-batch in tfrecords
--batch_size: Training batch size (must be a multiplicity of prebatch_size)
--eval_batch_size: Evaluation batch size (must be a multiplicity of prebatch_size)
--num_epochs: Number of epochs to train
--linear_learning_rate: Learning rate for the wide part of the model
--linear_l1_regularization: L1 regularization for the wide part of the model
--linear_l2_regularization: L2 regularization for the wide part of the model
--deep_learning_rate: Learning rate for the deep part of the model
--deep_l1_regularization: L1 regularization for the deep part of the model
--deep_l2_regularization: L2 regularization for the deep part of the model
--deep_dropout: Dropout probability for deep model
--predict: Perform only the prediction on the validation set, do not train
--evaluate: Perform only the evaluation on the validation set, do not train
--gpu: Run computations on GPU
--amp: Enable Automatic Mixed Precision
--xla: Enable XLA
--hvd: Use Horovod for multi-GPU training
```

### Command-line options

To see the full list of available options and their descriptions, use the `-h` or `--help` command-line option:
```bash
python -m trainer.task --help
```


### Getting the data

The Outbrain dataset can be downloaded from [Kaggle](https://www.kaggle.com/c/outbrain-click-prediction/data) (requires Kaggle account).


#### Dataset guidelines

The dataset contains a sample of users’ page views and clicks, as observed on multiple publisher sites. Viewed pages and clicked recommendations have further semantic attributes of the documents.

The dataset contains sets of content recommendations served to a specific user in a specific context. Each context (i.e. a set of recommendations) is given a display_id. In each such set, the user has clicked on at least one recommendation. The page view logs originally has more than 2 billion rows (around 100 GB uncompressed). 

The data within the preprocessing stage are transferred into tabular data of 54 features, for training having 55 million rows.


### Training process

The training can be started by running the `trainer/task.py` script. By default the script is in train mode. Other training related 
configs are also present in the `trainer/task.py` and can be seen using the command `python -m trainer.task --help`. Training happens for `--num_epochs` epochs with custom estimator for the model. The model has a wide linear part and a deep feed forward network, and the networks are built according to the default configuration.

Two separate optimizers are used to optimize the wide and the deep part of the network:
    
-  FTLR (Follow the Regularized Leader) optimizer is used to optimize the wide part of the network.
-  Proximal Adagrad optimizer is used to optimize the deep part of the network.

The training log will contain information about:

-  Loss value after every 100 steps.
-  Training throughput if `--benchmark` option is selected.
-  Evaluation metrics after every evaluation cycle at the end of every epoch.

Checkpoints are stored at the end of every `--save_checkpoints_steps` at the `--model_dir` location.

## Performance

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in training mode.

#### Training performance benchmark

We provide 6 scripts to benchmark the performance of training:
```bash
bash scripts/benchmark_training_fp32_1gpu.sh
bash scripts/benchmark_training_fp16_1gpu.sh
bash scripts/benchmark_training_fp32_4gpu.sh
bash scripts/benchmark_training_fp16_4gpu.sh
bash scripts/benchmark_training_fp32_8gpu.sh
bash scripts/benchmark_training_fp16_8gpu.sh
```

### Results

The following sections provide details on how we achieved our performance and
accuracy in training.

#### Training accuracy results

##### Training accuracy: NVIDIA DGX-1 (8x V100 16G)

Our results were obtained by running the benchmark scripts from the `scripts` directory in the TensorFlow NGC container on NVIDIA DGX-1 with (8x V100 16G) GPUs.

|**GPUs**|**Batch Size / GPU**|**Accuracy - FP32 (MAP@12)**|**Accuracy - Mixed precision (MAP@12)**|**Time to Train - FP32 (minutes)**|**Time to Train - Mixed precision (minutes)**|**Time to Train Speedup (FP32 to Mixed precision)**|
|-------:|-------------------:|----------------------------:|---------------------------------------:|-----------------------------------------------:|----------------------:|---------------------------------:|
| 1 | 131,072 |  0.67647 | 0.67634  | 654 | 454 | 1.44 |
| 4 | 32,768 | 0.67599 | 0.67652  | 226 | 183 | 1.23 |
| 8 | 16,384 | 0.67688 | 0.67690  | 167 | 153 | 1.09 |

To achieve the same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

##### Training accuracy plots

![MAP12](img/map12_WnD.png)

##### Training stability test

The Wide and Deep model was trained for 54,713 training steps, starting
from 50 different initial random seeds. The training was performed in the 20.02-tf1-py3-stage NGC container on
NVIDIA DGX-1 with 8x V100 16G GPUs with mixed precision enabled.
After training, the models were evaluated on the test dataset. The following
table summarizes the final MAP@12 score on the test set.

|**Average MAP@12**|**Standard deviation**|**Minimum**|**Maximum**|
|---------------------:|---------------------:|----------:|----------:|
| 0.67690 | 0.00081 | 0.67432 | 0.67821 | 


#### Training performance results


##### Training performance: NVIDIA DGX-1 (8x V100 16G)

Our results were obtained by running the `trainer/task.py` training script in the TensorFlow NGC container on NVIDIA DGX-1 with (8x V100 16G) GPUs. Performance numbers (in samples per second) were averaged over 50 training iterations. Improving model scaling for multi-GPU is planned, see [known issues](#known-issues).

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

|**GPUs**|**Batch Size / GPU**|**Throughput - FP32 (samples/s)**|**Throughput - Mixed precision (samples/s)**|**Throughput speedup (FP32 to Mixed precision)**|**Weak Scaling - FP32**|**Weak Scaling - Mixed precision**|
|-------:|-------------------:|----------------------------:|---------------------------------------:|-----------------------------------------------:|----------------------:|---------------------------------:|
| 1 | 131,072 | 168,181 | 242,332 | 1.44 | 1.00 | 1.00 |
| 4 | 131,072 | 487,719 | 602,027 | 1.23 | 2.47 | 2.89 |
| 8 | 131,072 | 659,533 | 718,820 | 1.09 | 3.11 | 3.91 |



## Release notes

### Changelog

This section needs to include the date of the release and the most important changes after the initial release.

March 2020
- Initial release

### Known issues

- Limited tf.feature_column support
- Limited scaling for multi-GPU because of inefficient handling of embedding operations (multiple memory transfers between CPU and GPU), work in progress to cover all the operations on GPU.

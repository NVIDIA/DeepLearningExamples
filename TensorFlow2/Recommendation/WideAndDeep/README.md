# Wide & Deep Recommender Model Training For TensorFlow 2

This repository provides a script and recipe to train the Wide and Deep Recommender model to achieve state-of-the-art accuracy.
The content of the repository is tested and maintained by NVIDIA.


- [Model overview](#model-overview)
  * [Model architecture](#model-architecture)
  * [Applications and dataset](#applications-and-dataset)
  * [Default configuration](#default-configuration)
  * [Model accuracy metric](#model-accuracy-metric)
  * [Feature support matrix](#feature-support-matrix)
    + [Features](#features)
  * [Mixed precision training](#mixed-precision-training)
    + [Enabling mixed precision](#enabling-mixed-precision)
    + [Enabling TF32](#enabling-tf32)
  * [Glossary](#glossary)
- [Setup](#setup)
  * [Requirements](#requirements)
- [Quick Start Guide](#quick-start-guide)
- [Advanced](#advanced)
  * [Scripts and sample code](#scripts-and-sample-code)
  * [Parameters](#parameters)
  * [Command-line options](#command-line-options)
  * [Getting the data](#getting-the-data)
    + [Dataset guidelines](#dataset-guidelines)
    + [Dataset preprocessing](#dataset-preprocessing)
      - [Spark CPU Dataset preprocessing](#spark-cpu-dataset-preprocessing)
      - [NVTabular GPU preprocessing](#nvtabular-gpu-preprocessing)
  * [Training process](#training-process)
  * [Evaluation process](#evaluation-process)
- [Performance](#performance)
  * [Benchmarking](#benchmarking)
    + [NVTabular and Spark CPU Preprocessing comparison](#nvtabular-and-spark-cpu-preprocessing-comparison)
    + [Training and inference performance benchmark](#training-and-inference-performance-benchmark)
  * [Results](#results)
    + [Training accuracy results](#training-accuracy-results)
      - [Training accuracy: NVIDIA DGX A100 (8x A100 80GB)](#training-accuracy-nvidia-dgx-a100-8x-a100-80gb)
      - [Training accuracy: NVIDIA DGX-1 (8x V100 16GB)](#training-accuracy-nvidia-dgx-1-8x-v100-16gb)
      - [Training accuracy plots](#training-accuracy-plots)
      - [Training stability test](#training-stability-test)
      - [Impact of mixed precision on training accuracy](#impact-of-mixed-precision-on-training-accuracy)
    + [Training performance results](#training-performance-results)
      - [Training performance: NVIDIA DGX A100 (8x A100 80GB)](#training-performance-nvidia-dgx-a100-8x-a100-80gb)
      - [Training performance: NVIDIA DGX-1 (8x V100 16GB)](#training-performance-nvidia-dgx-1-8x-v100-16gb)
    + [Inference performance results](#inference-performance-results)
      - [Inference performance: NVIDIA DGX A100 (8x A100 80GB)](#inference-performance-nvidia-dgx-a100-8x-a100-80gb)
      - [Inference performance: NVIDIA DGX-1 (8x V100 16GB)](#inference-performance-nvidia-dgx-1-8x-v100-16gb)
- [Release notes](#release-notes)
  * [Changelog](#changelog)
  * [Known issues](#known-issues)



## Model overview

Recommendation systems drive engagement on many of the most popular online platforms. As the volume of data available to power these systems grows exponentially, Data Scientists are increasingly turning from more traditional machine learning methods to highly expressive deep learning models to improve the quality of their recommendations.

Google's [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) has emerged as a popular model for Click Through Rate (CTR) prediction tasks thanks to its power of generalization (deep part) and memorization (wide part).
The differences between this Wide & Deep Recommender Model and the model from the paper is the size of the deep part of the model. Originally, in Google's paper, the fully connected part was three layers of 1024, 512, and 256 neurons. Our model consists of 5 layers each of 1024 neurons.

This model is trained with mixed precision using Tensor Cores on NVIDIA Volta and NVIDIA Ampere GPU architectures. Therefore, researchers can get results 4.5 times faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.

### Model architecture

Wide & Deep refers to a class of networks that use the output of two parts working in parallel - wide model and deep model - to make a binary prediction of CTR. The wide model is a linear model of features together with their transforms. The deep model is a series of 5 hidden MLP layers of 1024 neurons. The model can handle both numerical continuous features as well as categorical features represented as dense embeddings. The architecture of the model is presented in Figure 1.

<p align="center">
  <img width="100%" src="./img/model.svg">
  <br>
Figure 1. The architecture of the Wide & Deep model.</a>
</p>





### Applications and dataset

As a reference dataset, we used a subset of [the features engineered](https://github.com/gabrielspmoreira/kaggle_outbrain_click_prediction_google_cloud_ml_engine) by the 19th place finisher in the [Kaggle Outbrain Click Prediction Challenge](https://www.kaggle.com/c/outbrain-click-prediction/). This competition challenged competitors to predict the likelihood with which a particular ad on a website's display would be clicked on. Competitors were given information about the user, display, document, and ad in order to train their models. More information can be found [here](https://www.kaggle.com/c/outbrain-click-prediction/data).

### Default configuration

The Outbrain Dataset is preprocessed in order to get features input to the model. To give context to the acceleration numbers described below, some important properties of our features and model are as follows.

Features:
- Request Level:
    * 5 scalar numeric features `dtype=float32`
    * 8 categorical features (all INT32 `dtype`)
    * 8 trainable embeddings of (dimension, cardinality of categorical variable): (128,300000), (16,4), (128,100000), (64 ,4000), (64,1000), (64,2500), (64,300), (64,2000)
    * 8  trainable embeddings for wide part of size 1 (serving as an embedding from the categorical to scalar space for input to the wide portion of the model)

- Item Level:
    * 8 scalar numeric features `dtype=float32`
    * 5 categorical features (all INT32 `dtype`)
    * 5 trainable embeddings of dimensions (cardinality of categorical variable): 128 (250000), 64 (2500), 64 (4000), 64 (1000),64 (5000)
    * 5 trainable embeddings for wide part of size 1 (working as trainable one-hot embeddings)

Features describe both the user (Request Level features) and Item (Item Level Features).

- Model:
    * Input dimension is 26 (13 categorical and 13 numerical features)
    * Total embedding dimension is 976
    * 5 hidden layers each with size 1024
    * Total number of model parameter is ~90M
    * Output dimension is 1 (`y` is the probability of click given Request-level and Item-level features)
    * Loss function: Binary Crossentropy

For more information about feature preprocessing, go to [Dataset preprocessing](#dataset-preprocessing).

### Model accuracy metric

Model accuracy is defined with the [MAP@12](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision) metric. This metric follows the way of assessing model accuracy in the original [Kaggle Outbrain Click Prediction Challenge](https://www.kaggle.com/c/outbrain-click-prediction/). In this repository, the leaked clicked ads are not taken into account since in industrial setup Data Scientists do not have access to leaked information when training the model. For more information about data leak in Kaggle Outbrain Click Prediction challenge, visit this  [blogpost](https://medium.com/unstructured/how-feature-engineering-can-help-you-do-well-in-a-kaggle-competition-part-ii-3645d92282b8) by the 19th place finisher in that competition.

Training and inference script also reports AUC ROC, binary accuracy, and Loss (BCE) values.

### Feature support matrix

The following features are supported by this model:

| Feature                          | Wide & Deep |
| -------------------------------- | ----------- |
| Horovod Multi-GPU (NCCL)         | Yes         |
| Accelerated Linear Algebra (XLA) | Yes         |
| Automatic mixed precision (AMP)  | Yes         |

#### Features

**Horovod** is a distributed training framework for TensorFlow, Keras, PyTorch and MXNet. The goal of Horovod is to make distributed deep learning fast and easy to use. For more information about how to get started with Horovod, see: [Horovod: Official repository](https://github.com/horovod/horovod).

**Multi-GPU training with Horovod**
Our model uses Horovod to implement efficient multi-GPU training with NCCL. For details, see example sources in this repository or see: [TensorFlow tutorial](https://github.com/horovod/horovod/#usage).

**XLA** is a domain-specific compiler for linear algebra that can accelerate TensorFlow models with potentially no source code changes. Enabling XLA results in improvements to speed and memory usage: most internal benchmarks run ~1.1-1.5x faster after XLA is enabled. For more information on XLA visit [official XLA page](https://www.tensorflow.org/xla).

### Mixed precision training

Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in Volta, and following with both the Turing and Ampere architectures, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using [mixed precision training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) previously required two steps:
1.  Porting the model to use the FP16 data type where appropriate.    
2.  Adding loss scaling to preserve small gradient values.

The ability to train deep learning networks with lower precision was introduced in the Pascal architecture and first supported in [CUDA 8](https://devblogs.nvidia.com/parallelforall/tag/fp16/) in the NVIDIA Deep Learning SDK.

For more information:
* How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) documentation.
* Techniques used for mixed precision training, see the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
* How to access and enable AMP for TensorFlow, see [Using TF-AMP](https://docs.nvidia.com/deeplearning/dgx/tensorflow-user-guide/index.html#tfamp) from the TensorFlow User Guide.

For information on the influence of mixed precision training on model accuracy in train and inference, go to [Training accuracy results](Training-accuracy-results).

#### Enabling mixed precision

To enable Wide & Deep training to use mixed precision, add the additional flag `--amp` to the training script. Refer to the [Quick Start Guide](#quick-start-guide) for more information.

#### Enabling TF32

TensorFloat-32 (TF32) is the new math mode in NVIDIA A100 GPUs for handling tensor operations. TF32 running on Tensor Cores in A100 GPUs can provide up to 10x speedups compared to single-precision floating-point math (FP32) on Volta GPUs.

TF32 Tensor Cores can speed up networks using FP32, typically with no loss of accuracy. It is more robust than FP16 for models which require high dynamic range for weights or activations.

For more information, refer to the [TensorFloat-32 in the A100 GPU Accelerates AI Training, HPC up to 20x](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) blog post.

TF32 is supported in the NVIDIA Ampere GPU architecture and is enabled by default.

### Glossary

**Request level features**
Features that describe the person and context to which we wish to make recommendations.

**Item level features**
Features that describe those objects which we are considering recommending.

## Setup
The following section lists the requirements that you need to meet in order to start training the Wide & Deep model.

### Requirements

This repository contains Dockerfile which extends the TensorFlow2 NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [20.12-tf2-py3](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) NGC container

Supported GPUs:
- [NVIDIA Volta architecture](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
- [NVIDIA Turing architecture](https://www.nvidia.com/en-us/design-visualization/technologies/turing-architecture/)
- [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)


For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
* [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
* [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)
* [Running TensorFlow](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html#running)

For those unable to use the TensorFlow2 NGC container, to set up the required environment or create their own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## Quick Start Guide

To train your model using the default parameters of the Wide & Deep model on the Outbrain dataset in TF32 or FP32 precision, perform the following steps. For the specifics concerning training and inference with custom settings, see the [Advanced section](#advanced).

1. Clone the repository.
```
git clone https://github.com/NVIDIA/DeepLearningExamples
```

2. Go to `WideAndDeep` TensorFlow2 directory within the `DeepLearningExamples` repository:
```
cd DeepLearningExamples/TensorFlow2/Recommendation/WideAndDeep
```

3. Download the Outbrain dataset.

The Outbrain dataset can be downloaded from Kaggle (requires Kaggle account). Unzip the downloaded archive (for example, to `/raid/outbrain/orig`) and set the `HOST_OUTBRAIN_PATH` variable to the parent directory:
```
HOST_OUTBRAIN_PATH=/raid/outbrain
```

4. Preprocess the Outbrain dataset.

4.1. Build the Wide & Deep Preprocessing Container.
```
cd DeepLearningExamples/TensorFlow2/Recommendation/WideAndDeep
docker build -f Dockerfile-preproc . -t wd2-prep
```

4.2. Start an interactive session in the Wide&Deep Preprocessing Container. Run preprocessing against the original Outbrain dataset to `tf_records`. You can run preprocessing using Spark (CPU) or NVTabular preprocessing (GPU).
```
nvidia-docker run --rm -it --ipc=host -v ${HOST_OUTBRAIN_PATH}:/outbrain wd2-prep bash
```

4.3. Start preprocessing.
You can preprocess the data using either Spark on CPU or NVTabular on GPU. For more information, go to the [Dataset preprocessing](#dataset-preprocessing) section.

4.3.1. CPU Preprocessing (Spark).
```
cd /wd && bash scripts/preproc.sh spark 40
```

4.3.2. GPU Preprocessing (NVTabular).
```
cd /wd && bash scripts/preproc.sh nvtabular 40
```

The result of preprocessing scripts are prebatched TFRecords. The argument to the script is the number of TFRecords files that will be generated by the script (here 40). TFRecord files are generated in `${HOST_OUTBRAIN_PATH}/tfrecords`.

4.4. Training of the model
4.4.1. Build the Wide&Deep Training Container
```
cd DeepLearningExamples/TensorFlow2/Recommendation/WideAndDeep
docker build -f Dockerfile-train . -t wd2-train
```

4.4.2. Start an interactive session in the Wide&Deep Training Container
```
nvidia-docker run --rm -it --privileged --ipc=host -v ${HOST_OUTBRAIN_PATH}:/outbrain wd2-train bash
```

4.4.3. Run training
For 1 GPU:

```
python main.py
```

For 1 GPU with Mixed Precision training with XLA:

```
python main.py --xla --amp
```


For complete usage, run:
```
python main.py -h
```

For 8 GPUs:
```
mpiexec --allow-run-as-root --bind-to socket -np 8 python main.py
```

For 8 GPU with Mixed Precision training with XLA:
```
mpiexec --allow-run-as-root --bind-to socket -np 8 python main.py --xla --amp
```



5. Run validation or evaluation.
If you want to run validation or evaluation, you can either:
* use the checkpoint obtained from the training commands above, or
* download the pretrained checkpoint from NGC.

In order to download the checkpoint from NGC, visit [ngc.nvidia.com](https://ngc.nvidia.com) website and browse the available models. Download the checkpoint files and unzip them to some path, for example, to `$HOST_OUTBRAIN_PATH/checkpoints/` (which is the default path for storing the checkpoints during training). The checkpoint requires around 700MB disk space.

6. Start validation/evaluation.
In order to validate the checkpoint on the evaluation set, run the `main.py` script with the `--evaluate` and `--use_checkpoint` flags.

For 1 GPU:
```
python main.py --evaluate --use_checkpoint
```

For 8 GPUs:
```
mpiexec --allow-run-as-root --bind-to socket -np 8 python main.py --evaluate --use_checkpoint
```

Now that you have your model trained and evaluated, you can choose to compare your training results with our [Training accuracy results](#training-accuracy-results). You can also choose to benchmark yours performance to [Training and inference performance benchmark](#training-and-inference-performance-benchmark). Following the steps in these sections will ensure that you achieve the same accuracy and performance results as stated in the [Results](#results) section.

## Advanced

The following sections provide greater details of the dataset, running training, and the training results.

### Scripts and sample code

These are the important scripts in this repository:
* `main.py` - Python script for training the Wide & Deep recommender model. This script is run inside the training container (named `wd-train` in the [Quick Start Guide](#quick-start-guide)).
* `scripts/preproc.sh` - Bash script for Outbrain dataset preparation for training, preprocessing and saving into TFRecords format. This script is run inside a preprocessing container (named `wd-prep` in the [Quick Start Guide](#quick-start-guide)).
* `data/outbrain/dataloader.py` - Python file containing data loaders for training and evaluation set.
* `data/outbrain/features.py` - Python file describing the request and item level features as well as embedding dimensions and hash buckets’ sizes.
* `trainer/model/widedeep.py` - Python file with model definition.
* `trainer/utils/run.py` - Python file with training loop.

### Parameters

These are the important parameters in the `main.py` script:

| Scope| parameter| Comment| Default Value |
| -------------------- | ----------------------------------------------------- | ------------------------------------------------------------ | ------------- |
| location of datasets | --transformed_metadata_path TRANSFORMED_METADATA_PATH | Path to transformed_metadata for feature specification reconstruction |               |
|location of datasets| --use_checkpoint|Use checkpoint stored in model_dir path |False
|location of datasets|--model_dir MODEL_DIR|Destination where model checkpoint will be saved |/outbrain/checkpoints
|location of datasets|--results_dir RESULTS_DIR|Directory to store training results | /results
|location of datasets|--log_filename LOG_FILENAME|Name of the file to store dlloger output |log.json|
|training parameters|--training_set_size TRAINING_SET_SIZE|Number of samples in the training set | 59761827
|training parameters|--global_batch_size GLOBAL_BATCH_SIZE|Total size of training batch | 131072
|training parameters|--eval_batch_size EVAL_BATCH_SIZE|Total size of evaluation batch | 131072
|training parameters|--num_epochs NUM_EPOCHS|Number of training epochs | 20
|training parameters|--cpu|Run computations on the CPU | False
|training parameters|--amp|Enable automatic mixed precision conversion | False
|training parameters|--xla|Enable XLA conversion | False
|training parameters|--linear_learning_rate LINEAR_LEARNING_RATE|Learning rate for linear model | 0.02
|training parameters|--deep_learning_rate DEEP_LEARNING_RATE|Learning rate for deep model | 0.00012
|training parameters|--deep_warmup_epochs DEEP_WARMUP_EPOCHS|Number of learning rate warmup epochs for deep model | 6
|model construction|--deep_hidden_units DEEP_HIDDEN_UNITS [DEEP_HIDDEN_UNITS ...]|Hidden units per layer for deep model, separated by spaces|[1024, 1024, 1024, 1024, 1024]
|model construction|--deep_dropout DEEP_DROPOUT|Dropout regularization for deep model|0.1
|run mode parameters|--evaluate|Only perform an evaluation on the validation dataset, don't train | False
|run mode parameters|--benchmark|Run training or evaluation benchmark to collect performance metrics | False
|run mode parameters|--benchmark_warmup_steps BENCHMARK_WARMUP_STEPS|Number of warmup steps before start of the benchmark | 500
|run mode parameters|--benchmark_steps BENCHMARK_STEPS|Number of steps for performance benchmark | 1000
|run mode parameters|--affinity{socket,single,single_unique,<br>socket_unique_interleaved,<br>socket_unique_continuous,disabled}|Type of CPU affinity | socket_unique_interleaved




### Command-line options
To see the full list of available options and their descriptions, use the `-h` or `--help` command-line option:
```
python main.py -h
```

### Getting the data

The Outbrain dataset can be downloaded from [Kaggle](https://www.kaggle.com/c/outbrain-click-prediction/data) (requires Kaggle account).

#### Dataset guidelines

The dataset contains a sample of users’ page views and clicks, as observed on multiple publisher sites. Viewed pages and clicked recommendations have additional semantic attributes of the documents. The dataset contains sets of content recommendations served to a specific user in a specific context. Each context (i.e. a set of recommended ads) is given a `display_id`. In each such recommendation set, the user has clicked on exactly one of the ads.

The original data is stored in several separate files:
* `page_views.csv` - log of users visiting documents (2B rows, ~100GB uncompressed)
* `clicks_train.csv` - data showing which ad was clicked in each recommendation set (87M rows)
* `clicks_test.csv` - used only for the submission in the original Kaggle contest
* `events.csv` - metadata about the context of each recommendation set (23M rows)
* `promoted_content.csv` - metadata about the ads
* `document_meta.csv`, `document_topics.csv`, `document_entities.csv`, `document_categories.csv` - metadata about the documents

During the preprocessing stage, the data is transformed into 87M rows tabular data of 26 features. The dataset is split into training and evaluation parts that have approx 60M and approx 27M rows, respectively. Splitting into train and eval is done in this way so that random 80% of daily events for the first 10 days of the dataset form a training set and remaining part (20% of events daily for the first 10 days and all events in the last two days) form an evaluation set. Eventually the dataset is saved in pre-batched TFRecord format.

#### Dataset preprocessing

Dataset preprocessing aims in creating in total 26 features: 13 categorical and 13 numerical. These features are obtained from the original Outbrain dataset in preprocessing. There are 2 types of preprocessing available for the model:
Spark CPU preprocessing
[NVTabular](https://nvidia.github.io/NVTabular/v0.3.0/index.html) GPU preprocessing

Both split the dataset into train and evaluation sets and produce the same feature set, therefore, the training is agnostic to the preprocessing step.

For comparison of Spark CPU and NVTabular preprocessing go to [NVTabular and Spark CPU Preprocessing comparison](#nvtabular-and-spark-cpu-preprocessing-comparison)

##### Spark CPU Dataset preprocessing

The original dataset is preprocessed using the scripts provided in `data/outbrain/spark`. Preprocessing is split into 3 preprocessing steps: `preproc1.py`, `preproc2.py`, and `preproc3.py` that form a complete workflow. The workflow consists of the following operations:
* separating out the validation set for cross-validation
* filling missing data with mode, median, or imputed values
* joining click data, ad metadata,  and document category, topic and entity tables to create an enriched table
* computing  7 click-through rates (CTR) for ads grouped by 7 features
* computing attribute cosine similarity between the landing page and ad to be featured on the page
* math transformations of the numeric features (logarithmic, scaling, binning)
* categorifying data using hash-bucketing
* storing the resulting set of features in pre-batched TFRecord format

The `preproc1-3.py` preprocessing scripts use PySpark. In the Docker image, we have installed Spark 2.3.1 as a standalone cluster of Spark. The `preproc1.py` script splits the data into a training set and a validation set. The `preproc2.py` script computes the click-through rates (CTR) and cosine similarities between the features. The `preproc3.py` script performs the math transformations and generates the final TFRecord files. The data in the output files is pre-batched (with the default batch size of 4096) to avoid the overhead of the TFRecord format, which otherwise is not suitable for the tabular data.
The preprocessing includes some very resource-exhaustive operations including joining tables having over 2 billions of rows. Such operations may not fit into the RAM memory, and therefore we use Spark which is well suited for handling tabular operations on large data with limited RAM. Note that the Spark job requires about 500 GB disk space and 300 GB RAM to perform the preprocessing.

For more information about Spark, refer to the [Spark documentation](https://spark.apache.org/docs/2.3.1/).

##### NVTabular GPU preprocessing

The NVTabular dataset is preprocessed using the script provided in `data/outbrain/nvtabular`. The workflow consists of most of the same operations in the Spark pipeline:
* separating out the validation set for cross-validation
* filling missing data with themode, median, or imputed values most frequent value
* joining click data, ad metadata, and document category, topic and entity tables to create an enriched table.joining the  tables for the ad clicks data
* computing  7 click-through rates (CTR) for ads grouped by 7 features different contexts
* computing attribute cosine similarity between the landing page and ad to be featured on the page features of the clicked ads and the viewed ads
* math transformations of the numeric features (logarithmic, normalization)
* categorifying data using hash-bucketing
* storing the result in a Parquet format

**Transforming the result into the pre-batched TFRecord format**

Most of the code describing operations in this workflow are in `data/outbrain/nvtabular/utils/workflow.py` and leverage NVTabular v0.3. As stated in its repository, [NVTabular](https://github.com/NVIDIA/NVTabular), a component of [NVIDIA Merlin Open Beta](https://developer.nvidia.com/nvidia-merlin), is a feature engineering and preprocessing library for tabular data that is designed to quickly and easily manipulate terabyte scale datasets and train deep learning based recommender systems. It provides a high-level abstraction to simplify code and accelerates computation on the GPU using the [RAPIDS Dask-cuDF](https://github.com/rapidsai/cudf/tree/main/python/dask_cudf) library. The code to transform the NVTabular Parquet output into TFRecords is in `data/outbrain/nvtabular/utils/converter.py`.
The NVTabular version of preprocessing is not subject to the same memory and storage constraints as its Spark counterpart, since NVTabular is able to manipulate tables on GPU and work with tables much larger than even physical RAM memory. The NVTabular Outbrain workflow has been successfully tested on DGX-1 V100 and DGX A100 for single and multigpu preprocessing.

For more information about NVTabular, refer to the [NVTabular documentation](https://github.com/NVIDIA/NVTabular).

### Training process

The training can be started by running the `main.py` script. By default, the script is in train mode. Other training related configs are also present in the `trainer/utils/arguments.py` and can be seen using the command `python main.py -h`. Training happens on a TFRecords training dataset files that match `--train_data_pattern`.  Training is run for `--num_epochs` epochs with a global batch size of `--global_batch_size` in strong scaling mode (i.e. the effective batch size per GPU equals `global_batch_size/gpu_count`).

The model:
`tf.keras.experimental.WideDeepModel` consists of a wide part and deep part with a sigmoid activation in the output layer (see [Figure 1](#model-architecture)) for reference and `trainer/model/widedeep.py` for model definition).

During training (default configuration):
Two separate optimizers are used to optimize the wide and the deep part of the network:
* FTLR (Follow the Regularized Leader) optimizer is used to optimize the wide part of the network.
* RMSProp optimizer is used to optimize the deep part of the network.

Checkpoint of the model:
* Can be loaded at the beginning of training when `--use_checkpoint` is set
* is saved into `--model_dir` after each training epoch. Only the last checkpoint is kept.
* Contains information about number of training epochs

The model is evaluated on an evaluation dataset after every training epoch training log is displayed in the console and stored in  `--log_filename`.

Every 100 batches with training metrics:
loss, binary accuracy, AUC ROC, MAP@12 value

Every epoch after training, evaluation metrics are logged:
loss, binary accuracy, AUC ROC, MAP@12 value

### Evaluation process

The evaluation can be started by running the `main.py --evaluation` script. Evaluation is done for TFRecords dataset stored in `--eval_data_pattern`. Other evaluation related configs are also present in the `trainer/utils/arguments.py` and can be seen using the command `python main.py -h`.

During evaluation (`--evaluation flag`):
* Model is restored from checkpoint in `--model_dir` if `--use_checkpoint` is set
* Evaluation log is displayed in console and stored in  `--log_filename`
* Every 100 batches evaluation metrics are logged - loss, binary accuracy, AUC ROC, MAP@12 value

After the whole evaluation, the total evaluation metrics are logged,  loss, binary accuracy, AUC ROC, MAP@12 value.

## Performance

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in training mode.

#### NVTabular and Spark CPU Preprocessing comparison

Two types of dataset preprocessing are presented in Spark-CPU and NVTabular on GPU repositories. Both of these preprocess return prebatched TFRecords files with the same structure. The following table shows the comparison of both preprocessing in terms of code complication (Lines of code), top RAM consumption, and preprocessing time.

| |CPU preprocessing     | CPU Preprocessing        | GPU preprocessing        | GPU Preprocessing        | GPU preprocessing        | GPU Preprocessing        |
| -------------------------- | ----- | --------------------- | ------------------------ | ------------------------ | ------------------------ | ------------------------|
|| Spark on NVIDIA DGX-1 | Spark on NVIDIA DGX A100 | NVTabular on DGX-1 1 GPU | NVTabular on DGX-1 8 GPU | NVTabular DGX A100 1 GPU | NVTabular DGX A100 8 GPU |      |
| Lines of code*             | ~1500 | ~1500| ~500| ~500| ~500| ~500|
| Top RAM consumption \[GB\] | 167.0 | 223.4| 34.3| 48.7| 37.7 | 50.6|
| Top VRAM consumption per GPU \[GB\] | 0 | 0 | 16 | 13 | 45 | 67|
| Preprocessing time \[min\] |45.6|38.5|4.4|3.9|2.6| 2.3|


To achieve the same results for Top RAM consumption and preprocessing time, run a preprocessing container (`${HOST_OUTBRAIN_PATH}` is the path with Outbrain dataset).
```
nvidia-docker run --rm -it --ipc=host -v ${HOST_OUTBRAIN_PATH}:/outbrain wd2-prep bash
```

In the preprocessing container, run the preprocessing benchmark.

For Spark CPU preprocessing:
```
cd /wd && bash scripts/preproc_benchmark.sh -m spark
```

For GPU NVTabular preprocessing:
```
cd /wd && bash scripts/preproc_benchmark.sh -m nvtabular
```


#### Training and inference performance benchmark

Benchmark script is prepared to measure performance of the model during training (default configuration) and evaluation (`--evaluation`). Benchmark runs training or evaluation for `--benchmark_steps` batches, however measurement of performance starts after `--benchmark_warmup_steps`. Benchmark can be run for single and 8 GPUs and with a combination of XLA (`--xla`), AMP (`--amp`), batch sizes (`--global_batch_size` , `--eval_batch_size`) and affinity (`--affinity`).

In order to run benchmark follow these steps:
Run training container (`${HOST_OUTBRAIN_PATH}` is the path with Outbrain dataset):
```
nvidia-docker run --rm -it --ipc=host --privileged -v ${HOST_OUTBRAIN_PATH}:/outbrain wd2-train bash
```

Run the benchmark script:
For 1 GPU:
```
python main.py --benchmark
```

The benchmark will be run for training with default training parameters.

For 8GPUs:
```
mpiexec --allow-run-as-root --bind-to socket -np 8 python main.py --benchmark
```

### Results

The following sections provide details on how we achieved our performance and accuracy in training.

#### Training accuracy results

##### Training accuracy: NVIDIA DGX A100 (8x A100 80GB)

Our results were obtained by running the `main.py` training script in the TensorFlow2 NGC container on NVIDIA DGX A100 with (8x A100 80GB) GPUs.

| GPUs | Batch size / GPU | XLA | Accuracy - TF32 (MAP@12), Spark dataset | Accuracy - mixed precision (MAP@12),Spark Dataset | Accuracy - TF32 (MAP@12), NVTabular dataset | Accuracy - mixed precision (MAP@12), NVTabular Dataset | Time to train - TF32 (minutes) | Time to train - mixed precision (minutes) | Time to train speedup (TF32 to mixed precision) |
| ---- | ---------------- | --- | --------------|---|------- | ----------------------------------- | ------------------------------ | ----------------------------------------- | ----------------------------------------------- |
1|131072|Yes|0.65536|0.65537|0.65537|0.65646|16.40|13.71|1.20
1|131072|No|0.65538|0.65533|0.65533|0.65643|19.58|18.49|1.06
8|16384|Yes|0.65527|0.65525|0.65525|0.65638|7.77|9.71|0.80
8|16384|No|0.65517|0.65525|0.65525|0.65638|7.84|9.48|0.83


To achieve the same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

##### Training accuracy: NVIDIA DGX-1 (8x V100 16GB)

Our results were obtained by running the main.py training script in the TensorFlow2 NGC container on NVIDIA DGX-1 with (8x V100 16GB) GPUs.


| GPUs | Batch size / GPU | XLA | Accuracy - TF32 (MAP@12), Spark dataset | Accuracy - mixed precision (MAP@12),Spark Dataset | Accuracy - TF32 (MAP@12), NVTabular dataset | Accuracy - mixed precision (MAP@12), NVTabular Dataset | Time to train - TF32 (minutes) | Time to train - mixed precision (minutes) | Time to train speedup (TF32 to mixed precision) |
| ---- | ---------------- | --- | --------------|---|------- | ----------------------------------- | ------------------------------ | ----------------------------------------- | ----------------------------------------------- |
1|131072|Yes|0.65531|0.65529|0.65529|0.65651|66.01|23.66|2.79
1|131072|No|0.65542|0.65534|0.65534|0.65641|72.68|29.18|2.49|
8|16384|Yes|0.65544|0.65547|0.65547|0.65642|16.28|13.90|1.17|
8|16384|No|0.65548|0.65540|0.65540|0.65633|16.34|12.65|1.29|                                  |


To achieve the same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

##### Training accuracy plots

Models trained with FP32, TF32 and Automatic Mixed Precision (AMP), with and without XLA enabled achieve similar accuracy.

The plot represents MAP@12 in a function of steps (step is single batch) during training for default precision (FP32 for Volta architecture (DGX-1) and TF32 for Ampere GPU architecture (DGX-A100)) and AMP  for XLA and without it for both datasets. All other parameters of training are default.

<p align="center">
  <img width="100%" src="./img/leraning_curve_spark.svg" />
  <br>
  Figure 2. Learning curves for Spark dataset for different configurations.</a>
</p>

<p align="center">
  <img width="100%" src="./img/learning_curve_nvt.svg" />
  <br>
  Figure 3. Learning curves for NVTabular dataset for different configurations.</a>
</p>






##### Training stability test

Training of the model is stable for multiple configurations achieving the standard deviation of 10e-4. The model achieves similar MAP@12 scores for A100 and V100, training precisions, XLA usage and single/multi GPU. The Wide and Deep model was trained for 9100 training steps (20 epochs, 455 batches in each epoch, every batch containing 131072), starting from 20 different initial random seeds for each setup. The training was performed in the 20.12-tf1-py3 NGC container on NVIDIA DGX A100 80GB and DGX-1 16GB machines with and without mixed precision enabled, with and without XLA enabled for Spark- and NVTabular generated datasets. The provided charts and numbers consider single and 8 GPU training. After training, the models were evaluated on the validation set. The following plots compare distributions of MAP@12 on the evaluation set. In columns there is single vs 8 GPU training, in rows DGX A100 and DGX-1 V100.

<p align="center">
  <img width="100%" src="./img/training_stability_spark.svg" />
  <br>
  Figure 4. Training stability for Spark dataset: distribution of MAP@12 across different configurations. 'All configurations' refer to the distribution of MAP@12 for cartesian product of architecture, training precision, XLA usage, single/multi GPU. </a>
</p>

<p align="center">
  <img width="100%" src="./img/training_stability_nvtabular.svg" />
  <br>
  Figure 5. Training stability for NVtabular dataset: distribution of MAP@12 across different configurations. 'All configurations' refer to the distribution of MAP@12 for cartesian product of architecture, training precision, XLA usage, single/multi GPU.</a>
</p>



Training stability was also compared in terms of point statistics for MAP@12 distribution for multiple configurations. Refer to the expandable table below.

<details>
<summary>Full tabular data for training stability tests</summary>

||GPUs|Precicision|Dataset|XLA|mean|std|Min|Max
|--------|-|---------|-----------|---|----|---|---|---
DGX A100|1|TF32|Spark preprocessed|Yes|0.65536|0.00016|0.65510|0.65560|
DGX A100|1|TF32|Spark preprocessed|No|0.65538|0.00013|0.65510|0.65570|
DGX A100|1|TF32|NVTabular preprocessed|Yes|0.65641|0.00038|0.65530|0.65680|
DGX A100|1|TF32|NVTabular preprocessed|No|0.65648|0.00024|0.65580|0.65690|
DGX A100|1|AMP|Spark preprocessed|Yes|0.65537|0.00013|0.65510|0.65550|
DGX A100|1|AMP|Spark preprocessed|No|0.65533|0.00016|0.65500|0.65550|
DGX A100|1|AMP|NVTabular preprocessed|Yes|0.65646|0.00036|0.65530|0.65690|
DGX A100|1|AMP|NVTabular preprocessed|No|0.65643|0.00027|0.65590|0.65690|
DGX A100|8|TF32|Spark preprocessed|Yes|0.65527|0.00013|0.65500|0.65560|
DGX A100|8|TF32|Spark preprocessed|No|0.65517|0.00025|0.65460|0.65560|
DGX A100|8|TF32|NVTabular preprocessed|Yes|0.65631|0.00038|0.65550|0.65690|
DGX A100|8|TF32|NVTabular preprocessed|No|0.65642|0.00022|0.65570|0.65680|
DGX A100|8|AMP|Spark preprocessed|Yes|0.65525|0.00018|0.65490|0.65550|
DGX A100|8|AMP|Spark preprocessed|No|0.65525|0.00016|0.65490|0.65550|
DGX A100|8|AMP|NVTabular preprocessed|Yes|0.65638|0.00026|0.65580|0.65680|
DGX A100|8|AMP|NVTabular preprocessed|No|0.65638|0.00031|0.65560|0.65700|
DGX-1 V100|1|FP32|Spark preprocessed|Yes|0.65531|0.00017|0.65490|0.65560|
DGX-1 V100|1|FP32|Spark preprocessed|No|0.65542|0.00012|0.65520|0.65560|
DGX-1 V100|1|FP32|NVTabular preprocessed|Yes|0.65651|0.00019|0.65610|0.65680|
DGX-1 V100|1|FP32|NVTabular preprocessed|No|0.65638|0.00035|0.65560|0.65680|
DGX-1 V100|1|AMP|Spark preprocessed|Yes|0.65529|0.00015|0.65500|0.65570|
DGX-1 V100|1|AMP|Spark preprocessed|No|0.65534|0.00015|0.65500|0.65560|
DGX-1 V100|1|AMP|NVTabular preprocessed|Yes|0.65651|0.00028|0.65560|0.65690|
DGX-1 V100|1|AMP|NVTabular preprocessed|No|0.65641|0.00032|0.65570|0.65680|
DGX-1 V100|8|FP32|Spark preprocessed|Yes|0.65544|0.00019|0.65500|0.65580|
DGX-1 V100|8|FP32|Spark preprocessed|No|0.65548|0.00013|0.65510|0.65560|
DGX-1 V100|8|FP32|NVTabular preprocessed|Yes|0.65645|0.00012|0.65630|0.65670|
DGX-1 V100|8|FP32|NVTabular preprocessed|No|0.65638|0.00015|0.65610|0.65670|
DGX-1 V100|8|AMP|Spark preprocessed|Yes|0.65547|0.00015|0.65520|0.65580|
DGX-1 V100|8|AMP|Spark preprocessed|No|0.65540|0.00019|0.65500|0.65580|
DGX-1 V100|8|AMP|NVTabular preprocessed|Yes|0.65642|0.00028|0.65580|0.65690|
DGX-1 V100|8|AMP|NVTabular preprocessed|No|0.65633|0.00037|0.65510|0.65680|
</details>




##### Impact of mixed precision on training accuracy

The accuracy of training, measured with [MAP@12](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision) on the evaluation set after the final epoch metric was not impacted by enabling mixed precision. The obtained results were statistically similar. The similarity was measured according to the following procedure:

The model was trained 20 times for default settings (FP32 or TF32 for Volta and Ampere architecture respectively) and 20 times for AMP. After the last epoch, the accuracy score MAP@12 was calculated on the evaluation set.

Distributions for four configurations: architecture (A100, V100) and single/multi gpu for 2 datasets are presented below.

<p align="center">
  <img width="100%" src="./img/amp_influence_spark.svg" />
  <br>
  Figure 6. Influence of AMP on MAP@12 distribution for DGX A100 and DGX-1 V100 for single and multi gpu training on Spark dataset. </a>
</p>

<p align="center">
  <img width="100%" src="./img/amp_influence_nvtabular.svg" />
  <br>
  Figure 7. Influence of AMP on MAP@12 distribution for DGX A100 and DGX-1 V100 for single and multi gpu training on NVTabular dataset.
</p>



Distribution scores for full precision training and AMP training were compared in terms of mean, variance and [Kolmogorov–Smirnov test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) to state statistical difference between full precision and AMP results. Refer to the expandable table below.

<details>
<summary>Full tabular data for AMP influence on MAP@12</summary>

|            |GPUs                   | Dataset | XLA    | Mean MAP@12 for Full precision (TF32 for A100, FP32 for V100) | Std MAP@12 for Full precision (TF32 for A100, FP32 for V100) | Mean MAP@12 for AMP | Std MAP@12 for AMP | KS test value: statistics, p-value |
| ------------ | ---------------------- | ------- | ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------- | ------------------ | ---------------------------------- |
| DGX A100   | 1    | NVTabular preprocessed | No      | 0.6565 | 0.0002                                                       | 0.6564                                                       | 0.0003              | 0.2000 (0.8320)    |                                    |
| DGX A100   | 8    | NVTabular preprocessed | No      | 0.6564 | 0.0002                                                       | 0.6564                                                       | 0.0003              | 0.1500 (0.9831)    |                                    |
| DGX A100   | 1    | Spark preprocessed     | No      | 0.6554 | 0.0001                                                       | 0.6553                                                       | 0.0002              | 0.2500 (0.5713)    |                                    |
| DGX A100   | 8    | Spark preprocessed     | No      | 0.6552 | 0.0002                                                       | 0.6552                                                       | 0.0002              | 0.3000 (0.3356)    |                                    |
| DGX A100   | 1    | NVTabular preprocessed | No      | 0.6564 | 0.0004                                                       | 0.6565                                                       | 0.0004              | 0.1500 (0.9831)    |                                    |
| DGX A100   | 8    | NVTabular preprocessed | No      | 0.6563 | 0.0004                                                       | 0.6564                                                       | 0.0003              | 0.2500 (0.5713)    |                                    |
| DGX A100   | 1    | Spark preprocessed     | No      | 0.6554 | 0.0002                                                       | 0.6554                                                       | 0.0001              | 0.1500 (0.9831)    |                                    |
| DGX A100   | 8    | Spark preprocessed     | No      | 0.6553 | 0.0001                                                       | 0.6552                                                       | 0.0002              | 0.1500 (0.9831))   |                                    |
| DGX-1 V100 | 1    | NVTabular preprocessed | No      | 0.6564 | 0.0004                                                       | 0.6564                                                       | 0.0003              | 0.1000 (1.0000)    |                                    |
| DGX-1 V100 | 8    | NVTabular preprocessed | No      | 0.6564 | 0.0001                                                       | 0.6563                                                       | 0.0004              | 0.2500 (0.5713)    |                                    |
| DGX-1 V100 | 1    | Spark preprocessed     | No      | 0.6554 | 0.0001                                                       | 0.6553                                                       | 0.0001              | 0.2000 (0.8320)    |                                    |
| DGX-1 V100 | 8    | Spark preprocessed     | No      | 0.6555 | 0.0001                                                       | 0.6554                                                       | 0.0002              | 0.3500 (0.1745)    |                                    |
| DGX-1 V100 | 1    | NVTabular preprocessed | No      | 0.6565 | 0.0002                                                       | 0.6565                                                       | 0.0003              | 0.1500 (0.9831)    |                                    |
| DGX-1 V100 | 8    | NVTabular preprocessed | No      | 0.6564 | 0.0001                                                       | 0.6564                                                       | 0.0003              | 0.2000 (0.8320)    |                                    |
| DGX-1 V100 | 1    | Spark preprocessed     | No      | 0.6553 | 0.0002                                                       | 0.6553                                                       | 0.0002              | 0.2000 (0.8320)    |                                    |
| DGX-1 V100 | 8    | Spark preprocessed     | No      | 0.6554 | 0.0002                                                       | 0.6555                                                       | 0.0002              | 0.1500 (0.9831)    |                                    |

</details>

#### Training performance results

##### Training performance: NVIDIA DGX A100 (8x A100 80GB)

Our results were obtained by running the benchmark script (`main.py --benchmark`) in the TensorFlow2 NGC container on NVIDIA DGX A100 with (8x A100 80GB) GPUs. 

|GPUs | Batch size / GPU | XLA | Throughput - TF32 (samples/s)|Throughput - mixed precision (samples/s)|Throughput speedup (TF32 - mixed precision)| Strong scaling - TF32|Strong scaling - mixed precision
| ---- | ---------------- | --- | ----------------------------- | ---------------------------------------- | ------------------------------------------- | --------------------- | -------------------------------- |
|1|131,072|Yes|1642892|1997414|1.22|1.00|1.00|
|1|131,072|No|1269638|1355523|1.07|1.00|1.00|
|8|16,384|Yes|3376438|2508278|0.74|2.06|1.26|
|8|16,384|No|3351118|2643009|0.79|2.64|1.07|


##### Training performance: NVIDIA DGX-1 (8x V100 16GB)

Our results were obtained by running the benchmark script (`main.py --benchmark`) in the TensorFlow2 NGC container on NVIDIA DGX-1 with (8x V100 16GB) GPUs.

|GPUs | Batch size / GPU | XLA | Throughput - FP32 (samples/s)|Throughput - mixed precision (samples/s)|Throughput speedup (FP32 - mixed precision)| Strong scaling - FP32|Strong scaling - mixed precision
| ---- | ---------------- | --- | ----------------------------- | ---------------------------------------- | ------------------------------------------- | --------------------- | -------------------------------- |
|1|131,072|Yes|361202|1091584|3.02|1.00|1.00
|1|131,072|No|321816|847229|2.63|1.00|1.00
|8|16,384|Yes|1512691|1731391|1.14|4.19|1.59
|8|16,384|No|1490044|1837962|1.23|4.63|2.17


#### Inference performance results

##### Inference performance: NVIDIA DGX A100 (8x A100 80GB)

Our results were obtained by running the benchmark script (`main.py --evaluate --benchmark`) in the TensorFlow2 NGC container on NVIDIA DGX A100 with 8x A100 80GB GPUs. 


|GPUs|Batch size / GPU|XLA|Throughput \[samples/s\] TF32|Throughput \[samples/s\]AMP|Throughput speedup AMP to TF32
|----|----------------|---|------------------------------|-----------------------------|-------------------------------
|1|4096|NO|648058|614053|0.95|
|1|8192|NO|1063986|1063203|1.00|
|1|16384|NO|1506679|1573248|1.04|
|1|32768|NO|1983238|2088212|1.05|
|1|65536|NO|2280630|2523812|1.11|
|1|131072|NO|2568911|2915340|1.13|
|8|4096|NO|4516588|4374181|0.97|
|8|8192|NO|7715609|7718173|1.00|
|8|16384|NO|11296845|11624159|1.03|
|8|32768|NO|14957242|15904745|1.06|
|8|65536|NO|17671055|19332987|1.09|
|8|131072|NO|19779711|21761656|1.10|

For more results go to the expandable table below.

<details>
<summary>Full tabular data for inference performance results for DGX A100</summary>

|GPUs|Batch size / GPU|XLA|Throughput \[samples/s\] TF32|Throughput \[samples/s\]AMP|Throughput speedup AMP to TF32
|----|----------------|---|------------------------------|-----------------------------|-------------------------------
|1|4096|YES|621024|648441|1.04|
|1|4096|NO|648058|614053|0.95|
|1|8192|YES|1068943|1045790|0.98|
|1|8192|NO|1063986|1063203|1.00|
|1|16384|YES|1554101|1710186|1.10|
|1|16384|NO|1506679|1573248|1.04|
|1|32768|YES|2014216|2363490|1.17|
|1|32768|NO|1983238|2088212|1.05|
|1|65536|YES|2010050|2450872|1.22|
|1|65536|NO|2280630|2523812|1.11|
|1|131072|YES|2321543|2885393|1.24|
|1|131072|NO|2568911|2915340|1.13|
|8|4096|YES|4328154|4445315|1.03|
|8|4096|NO|4516588|4374181|0.97|
|8|8192|YES|7410554|7640191|1.03|
|8|8192|NO|7715609|7718173|1.00|
|8|16384|YES|11412928|12422567|1.09|
|8|16384|NO|11296845|11624159|1.03|
|8|32768|YES|11428369|12525670|1.10|
|8|32768|NO|14957242|15904745|1.06|
|8|65536|YES|13453756|15308455|1.14|
|8|65536|NO|17671055|19332987|1.09|
|8|131072|YES|17047482|20930042|1.23|
|8|131072|NO|19779711|21761656|1.10|
 </details>


##### Inference performance: NVIDIA DGX-1 (8x V100 16GB)

Our results were obtained by running the benchmark script (`main.py --evaluate --benchmark`) in the TensorFlow2 NGC container on NVIDIA DGX-1 with (8x V100 16GB) GPUs.

|GPUs|Batch size / GPU|XLA|Throughput \[samples/s\] TF32|Throughput \[samples/s\]AMP|Throughput speedup AMP to TF32
|----|----------------|---|------------------------------|-----------------------------|-------------------------------
|1|4096|NO|375928|439395|1.17|
|1|8192|NO|526780|754517|1.43|
|1|16384|NO|673971|1133696|1.68|
|1|32768|NO|791637|1470221|1.86|
|1|65536|NO|842831|1753500|2.08|
|1|131072|NO|892941|1990898|2.23|
|8|4096|NO|2893390|3278473|1.13|
|8|8192|NO|3881996|5337866|1.38|
|8|16384|NO|5003135|8086178|1.62|
|8|32768|NO|6124648|11087247|1.81|
|8|65536|NO|6631887|13233484|2.00|
|8|131072|NO|7030438|15081861|2.15|



For more results go to the expandable table below.

<details>
<summary>Full tabular data for inference performance for DGX-1 V100 results</summary>

|GPUs|Batch size / GPU|XLA|Throughput \[samples/s\] TF32|Throughput \[samples/s\]AMP|Throughput speedup AMP to TF32
|----|----------------|---|------------------------------|-----------------------------|-------------------------------
|1|4096|YES|356963|459481|1.29|
|1|4096|NO|375928|439395|1.17|
|1|8192|YES|517016|734515|1.42|
|1|8192|NO|526780|754517|1.43|
|1|16384|YES|660772|1150292|1.74|
|1|16384|NO|673971|1133696|1.68|
|1|32768|YES|776357|1541699|1.99|
|1|32768|NO|791637|1470221|1.86|
|1|65536|YES|863311|1962275|2.27|
|1|65536|NO|842831|1753500|2.08|
|1|131072|YES|928290|2235968|2.41|
|1|131072|NO|892941|1990898|2.23|
|8|4096|YES|2680961|3182591|1.19|
|8|4096|NO|2893390|3278473|1.13|
|8|8192|YES|3738172|5185972|1.39|
|8|8192|NO|3881996|5337866|1.38|
|8|16384|YES|4961435|8170489|1.65|
|8|16384|NO|5003135|8086178|1.62|
|8|32768|YES|6218767|11658218|1.87|
|8|32768|NO|6124648|11087247|1.81|
|8|65536|YES|6808677|14921211|2.19|
|8|65536|NO|6631887|13233484|2.00|
|8|131072|YES|7205370|16923294|2.35|
|8|131072|NO|7030438|15081861|2.15|
 </details>

## Release notes

### Changelog

February 2021
Initial release

### Known issues
* In this model the TF32 precision can in some cases be as fast as the FP16 precision on Ampere GPUs. This is because TF32 also uses Tensor Cores and doesn't need any additional logic such as maintaining FP32 master weights and casts. However, please note that W&D is, by modern recommender standards, a very small model. Larger models should still see significant benefits of using FP16 math.

# BERT For TensorFlow

This repository provides a script and recipe to train BERT to achieve state of the art accuracy and is tested and maintained by NVIDIA.


## Table Of Contents:
* [Model overview](#model-overview)
  * [Model architecture](#model-architecture)
  * [Default configuration](#default-configuration)
  * [Feature support matrix](#feature-support-matrix)
    * [Features](#features)
  * [Mixed Precision training](#mixed-precision-training)
    * [Enabling Mixed Precision](#enabling-mixed-precision)
  * [Glossary](#glossary)
* [Setup](#setup)
  * [Requirements](#requirements)
* [Quick Start Guide](#quick-start-guide)
* [Advanced](#advanced)
  * [Scripts and sample code](#scripts-and-sample-code)
  * [Parameters](#parameters)
  * [Command-line options](#command-line-options)
  * [Getting the data](#getting-the-data)
    * [Dataset guidelines](#dataset-guidelines)
    * [Multi-dataset](#multi-dataset)
  * [Training process](#training-process)
    * [Pre-training](#pre-training)
    * [Fine tuning](#fine-tuning)
  * [Inference process](#inference-process)
  * [Deploying the BERT model using TensorRT Inference Server](#deploying-the-bert-model-using-tensorrt-inference-server)
    * [Performance analysis for TensorRT Inference Server](#performance-analysis-for-tensorrt-inference-server)
      * [Advanced Details](#advanced-details)
    * [Running the TensorRT Inference Server and client](#running-the-tensorrt-inference-server-and-client)
* [Performance](#performance)
  * [Benchmarking](#benchmarking)
    * [Training performance benchmark](#training-performance-benchmark)
    * [Inference performance benchmark](#inference-performance-benchmark)
  * [Results](#results)
    * [Training accuracy results](#training-accuracy-results)
      * [Training accuracy: NVIDIA DGX-1 (8x V100 32G)](#training-accuracy-nvidia-dgx-1-(8x-v100-32G))
      * [Training stability test](#training-stability-test)
    * [Training performance results](#training-performance-results)
      * [Training performance: NVIDIA DGX-1 (8x V100 16G)](#training-performance-nvidia-dgx-1-8x-v100-16g)
      * [Training performance: NVIDIA DGX-1 (8x V100 32G)](#training-performance-nvidia-dgx-1-8x-v100-32g)
      * [Training performance: NVIDIA DGX-2 (16x V100 32G)](#training-performance-nvidia-dgx-2-16x-v100-32g)
    * [Inference performance results](#inference-performance-results)
      * [Inference performance: NVIDIA DGX-1 (1x V100 16G)](#inference-performance-nvidia-dgx-1-1x-v100-16g)
      * [Inference performance: NVIDIA DGX-1 (1x V100 32G)](#inference-performance-nvidia-dgx-1-1x-v100-32g)
      * [Inference performance: NVIDIA DGX-2 (1x V100 32G)](#inference-performance-nvidia-dgx-2-1x-v100-32g)
* [Release notes](#release-notes)
  * [Changelog](#changelog)
  * [Known issues](#known-issues)

## Model overview

BERT, or Bidirectional Encoder Representations from Transformers, is a new method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks. This model is based on the [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) paper. NVIDIA's BERT is an optimized version of [Google's official implementation](https://github.com/google-research/bert), leveraging mixed precision arithmetic and Tensor Cores on V100 GPUs for faster training times while maintaining target accuracy.


Other publicly available implementations of BERT include:
1. [Hugging Face](https://github.com/huggingface/pytorch-pretrained-BERT)
2. [codertimo](https://github.com/codertimo/BERT-pytorch)
3. [gluon-nlp](https://github.com/dmlc/gluon-nlp/tree/master/scripts/bert)


This model is trained with mixed precision using Tensor Cores on NVIDIA Volta and Turing GPUs. Therefore, researchers can get results faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.

### Model architecture

BERT's model architecture is a multi-layer bidirectional Transformer encoder. Based on the model size, we have the following two default configurations of BERT.

| **Model** | **Hidden layers** | **Hidden unit size** | **Attention heads** | **Feedforward filter size** | **Max sequence length** | **Parameters** |
|:---------:|:----------:|:----:|:---:|:--------:|:---:|:----:|
|BERTBASE |12 encoder| 768| 12|4 x  768|512|110M|
|BERTLARGE|24 encoder|1024| 16|4 x 1024|512|330M|

BERT training consists of two steps, pre-training the language model in an unsupervised fashion on vast amounts of unannotated datasets, and then using this pre-trained model for fine-tuning for various NLP tasks, such as question and answer, sentence classification, or sentiment analysis. Fine-tuning typically adds an extra layer or two for the specific task and further trains the model using a task-specific annotated dataset, starting from the pre-trained backbone weights. The end-to-end process can be summarized using Figure 1 and the results are covered in the following sections.

![](data/images/bert_pipeline.png?raw=true)

Figure 1: BERT Pipeline

### Default configuration

This repository contains scripts to interactively launch data download, training, benchmarking and inference routines in a Docker container for both pre-training and fine tuning for Question Answering. The major differences between the official implementation of the paper and our version of BERT are as follows:
- Mixed precision support with TensorFlow Automatic Mixed Precision (TF-AMP), which enables mixed precision training without any changes to the code-base by performing automatic graph rewrites and loss scaling controlled by an environmental variable.
- Scripts to download dataset for:
    - Pre-training - [Wikipedia](https://dumps.wikimedia.org/),  [Books Corpus](http://yknzhu.wixsite.com/mbweb)
    - Fine Tuning - [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (Stanford Question Answering Dataset), Pretrained Weights from Google
- Custom fused CUDA kernels for faster computations
- Multi-GPU/Multi-Node support using Horovod


The following performance optimizations were implemented in this model:
- [XLA](https://www.tensorflow.org/xla) support (experimental).


These techniques and optimizations improve model performance and reduce training time, allowing you to perform various NLP tasks with no additional effort.


### Feature support matrix

The following features are supported by this model.

| **Feature**               | **BERT** |
|:-----------------------:|:--------------------------:|
| Horovod Multi-GPU      | Yes |                         

#### Features

Horovod - Horovod is a distributed training framework for TensorFlow, Keras, PyTorch and MXNet. The goal of Horovod is to make distributed deep learning fast and easy to use. For more information about how to get started with Horovod, see the [Horovod: Official repository](https://github.com/horovod/horovod).

### Mixed precision training

Mixed precision is the combined use of different numerical precision in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format, while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in the Volta and Turing architecture, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using mixed precision training requires two steps:
1.  Porting the model to use the FP16 data type where appropriate.    
2.  Adding loss scaling to preserve small gradient values.

The ability to train deep learning networks with lower precision was introduced in the Pascal architecture and first supported in [CUDA 8](https://devblogs.nvidia.com/parallelforall/tag/fp16/) in the NVIDIA Deep Learning SDK.

For information about:
-   How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/sdk/Mixed-Precision-training/index.html) documentation.
-   Techniques used for mixed precision training, see the [Mixed Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
-   How to access and enable AMP for TensorFlow, see [Using TF-AMP](https://docs.nvidia.com/deeplearning/dgx/tensorflow-user-guide/index.html#tfamp) from the TensorFlow User Guide.
-   APEX tools for mixed precision training, see the [NVIDIA Apex: Tools for Easy Mixed Precision Training in PyTorch](https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/).

#### Enabling mixed precision
Automatic Mixed Precision (AMP) for TensorFlow to enables the full [mixed precision methodology](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#tensorflow) in your existing TensorFlow model code.  AMP enables mixed precision training on Volta and Turing GPUs automatically. The TensorFlow framework code makes all necessary model changes internally.

In TF-AMP, the computational graph is optimized to use as few casts as necessary and maximize the use of FP16, and the loss scaling is automatically applied inside of supported optimizers. AMP can be configured to work with the existing `tf.contrib` loss scaling manager by disabling the AMP scaling with a single environment variable to perform only the automatic mixed precision optimization. It accomplishes this by automatically rewriting all computation graphs with the necessary operations to enable mixed precision training and automatic loss scaling.


### Glossary

Fine-tuning
Training an already pretrained model further using a task specific dataset for subject-specific refinements, by adding task-specific layers on top if required.

Language Model
Assigns a probability distribution over a sequence of words. Given a sequence of words, it assigns a probability to the whole sequence.
Pre-training
Training a model on vast amounts of data on the same (or different) task to build general understandings.

Transformer
The paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) introduces a novel architecture called Transformer that uses an attention mechanism and transforms one sequence into another.


## Setup

The following section lists the requirements in order to start training the BERT model.


### Requirements

This repository contains `Dockerfile` which extends the TensorFlow NGC container and encapsulates some dependencies.  Aside from these dependencies, ensure you have the following components:
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [TensorFlow 19.06-py3+](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) NGC container
- [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) based GPU


For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
- [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
- [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)
- [Running TensorFlow](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html#running)

For those unable to use the TensorFlow NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## Quick Start Guide

To pretrain or fine tune your model for Question Answering using mixed precision with Tensor Cores or using FP32, perform the following steps using the default parameters of the BERT model.

1. Clone the repository.

```bash
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/TensorFlow/LanguageModeling/BERT
```

2. Build the BERT TensorFlow NGC container.

```bash
bash scripts/docker/build.sh
```

3. Download and preprocess the dataset.

This repository provides scripts to download, verify and extract the SQuaD dataset and pretrained weights for fine tuning as well as Wikipedia and BookCorpus dataset for pre-training.

To download, verify, and extract the required datasets, issue:

```bash
bash scripts/data_download.sh  
```

The script launches a Docker container with the current directory mounted and downloads the datasets to a `data/` folder on the host.

Note : The dataset is 170GB+ and takes 15+ hours to download. Expired dataset links are ignored during data download.


4. Download pretrained models from NGC.

We have uploaded checkpoints for both fine tuning and pre-training for various configurations on the NGC Model Registry. You can download them directly from the [NGC model catalog](https://ngc.nvidia.com/catalog/models). Download them to the BERT directory to easily access them in your scripts.



5. Start an interactive session in the NGC container to run training/inference.

After you build the container image and download the data, you can start an interactive CLI session as follows:  

```bash
bash scripts/docker/launch.sh
```

The `launch.sh` script assumes that the datasets are in the following locations by default after downloading the data.
- SQuAD v1.1 - `data/squad/v1.1`
- SQuaD v2.0 - `data/squad/v2.0`
- BERT Large - `data/pretrained_models_google/uncased_L-24_H-1024_A-16`
- BERT Base - `data/pretrained_models_google/uncased_L-12_H-768_A-12`
- BERT - `data/pretrained_models_google/uncased_L-24_H-1024_A-16`
- Wikipedia - `data/wikipedia_corpus/final_tfrecords_sharded`
- Books Corpus -  `data/bookcorpus/final_tfrecords_sharded`

6. Start pre-training.

BERT is designed to pre-train deep bidirectional representations for language representations. The following scripts are to replicate pre-training on Wikipedia and Books Corpus from the [paper](https://arxiv.org/pdf/1810.04805.pdf). These scripts are general and can be used for pre-training language representations on any corpus of choice.

From within the container, you can use the following script to run pre-training.
```bash
bash scripts/run_pretraining.sh <train_batch_size_per_gpu> <eval_batch_size> <learning_rate_per_gpu> <precision> <use_xla> <num_gpus> <warmup_steps> <train_steps> <save_checkpoint_steps> <create_logfile>
```

For FP16 training with XLA using a DGX-1 V100 32G, run:
```bash
bash scripts/run_pretraining.sh 14 8 5e-5 fp16 true 8 5000 2285000 5000 true
```

For FP32 training without XLA using a DGX-1 V100 32G, run:
```bash
bash scripts/run_pretraining.sh 6 6 2e-5 fp32 false 8 2000 5333333 5000 true
```

7. Start fine tuning.

The above pretrained BERT representations can be fine tuned with just one additional output layer for a state-of-the-art Question Answering system. From within the container, you can use the following script to run fine-training for SQuaD.

```bash
bash scripts/run_squad.sh <batch_size_per_gpu> <learning_rate_per_gpu> <precision> <use_xla> <num_gpus> <seq_length> <doc_stride> <bert_model> <squad_version> <checkpoint> <epochs>
```

For SQuAD 1.1 FP16 training with XLA using a DGX-1 V100 32G, run:
```bash
bash scripts/run_squad.sh 10 5e-6 fp16 true 8 384 128 large 1.1  
data/pretrained_models_google/uncased_L-24_H-1024_A-16/bert_model
.ckpt 1.1
```

For SQuAD 2.0 FP32 training without XLA using a DGX-1 V100 32G, run:
```bash
bash scripts/run_squad.sh 5 5e-6 fp32 false 8 384 128 large 1.1 data/pretrained_models_google/uncased_L-24_H-1024_A-16/bert_model.ckpt 2.0
```

8. Start validation/evaluation.

The `run_squad_inference.sh` script runs inference on a checkpoint fine tuned for SQuaD and evaluates the validity of predictions on the basis of exact match and F1 score.

```bash
bash scripts/run_squad_inference.sh <init_checkpoint> <batch_size> <precision> <use_xla> <seq_length> <doc_stride> <bert_model> <squad_version>
```

For SQuAD 2.0 FP16 inference with XLA using a DGX-1 V100 32G, run:
```bash
bash scripts/run_squad_inference.sh /results/model.ckpt 8 fp16 true 384 128 large 2.0
```

For SQuAD 1.1 FP32 inference without XLA using a DGX-1 V100 32G, run:
```bash
bash scripts/run_squad_inference.sh /results/model.ckpt 8 fp32 false 384 128 large 1.1
```

## Advanced
The following sections provide greater details of the dataset, running training and inference, and the training results.

### Scripts and sample code

In the root directory, the most important files are:
* `run_pretraining.py` - Serves as entry point for pre-training
* `run_squad.py` - Serves as entry point for SQuAD training
* Dockerfile - Container with the basic set of dependencies to run BERT

The `scripts/` folder encapsulates all the one-click scripts required for running various functionalities supported such as:
* `run_squad.sh` - Runs SQuAD training and inference using `run_squad.py` file
* `run_pretraining.sh` - Runs pre-training using `run_pretraining.py` file
* `data_download.sh` - Downloads datasets using files in `data/` folder
* `finetune_train_benchmark.sh` - Captures performance metrics of training for multiple configurations
* `finetune_inference_benchmark.sh` - Captures performance metrics of inference for multiple configurations

Other folders included in the root directory are:
* `data/` - Necessary folders and scripts to download datasets required for fine tuning and pre-training BERT
* `utils/` - Necessary files for preprocessing data before feeding into BERT and hooks for obtaining performance metrics from BERT.

### Parameters

Aside from the options to set hyperparameters, the relevant options to control the behaviour of the `run_pretraining.py` script are:
```bash
  --[no]use_fp16: Whether to enable AMP ops.(default: 'false')
  --bert_config_file: The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.
  --[no]do_eval: Whether to run evaluation on the dev set.(default: 'false')
  --[no]do_train: Whether to run training.(evaluation: 'false')
  --eval_batch_size: Total batch size for eval.(default: '8')(an integer)
  --[no]horovod: Whether to use Horovod for multi-gpu runs(default: 'false')
  --init_checkpoint: Initial checkpoint (usually from a pre-trained BERT model).
  --input_file: Input TF example files (can be a glob or comma separated).
  --iterations_per_loop: How many steps to make in each estimator call.(default: '1000')
  --output_dir: The output directory where the model checkpoints will be    written.
```

Aside from the options to set hyperparameters, some relevant options to control the behaviour of the `run_squad.py` script are:
```bash
  --bert_config_file: The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.
--output_dir: The output directory where the model checkpoints will be written.
  --[no]do_predict: Whether to run evaluation on the dev set. (default: 'false')
  --[no]do_train: Whether to run training. (default: 'false')
  --learning_rate: The initial learning rate for Adam.(default: '5e-06')(a number)
  --max_answer_length: The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.(default: '30')(an integer)
  --max_query_length: The maximum number of tokens for the question. Questions longer than this will be truncated to this length.(default: '64')(an integer)
  --max_seq_length: The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.(default: '384')(an integer)
  --predict_batch_size: Total batch size for predictions.(default: '8')(an integer)
  --train_batch_size: Total batch size for training.(default: '8')(an integer)
  --[no]use_fp16: Whether to enable AMP ops.(default: 'false')
  --[no]use_xla: Whether to enable XLA JIT compilation.(default: 'false')
  --[no]verbose_logging: If true, all of the warnings related to data processing will be printed. A number of warnings are expected for a normal SQuAD evaluation.(default: 'false')
  --[no]version_2_with_negative: If true, the SQuAD examples contain some that do not have an answer.(default: 'false')
```

### Command-line options

To see the full list of available options and their descriptions, use the `-h` or `--help` command-line option with the python file, for example:

```bash
python run_pretraining.py --help
python run_squad.py --help
```

### Getting the data

For pre-training BERT, we use the concatenation of Wikipedia (2500M words) as well as Books Corpus (800M words). For Wikipedia, we extract only the text passages from [here](ftp://ftpmirror.your.org/pub/wikimedia/dumps/enwiki/20190301/enwiki-20190301-pages-articles-multistream.xml.bz2) and ignore headers list and tables. It is structured as a document level corpus rather than a shuffled sentence level corpus because it is critical to extract long contiguous sentences. The next step is to run `create_pretraining_data.py` with the document level corpus as input, which generates input data and labels for the masked language modeling and next sentence prediction tasks. Pre-training can also be performed on any corpus of your choice. The collection of data generation scripts are intended to be modular to allow modifications for additional preprocessing steps or to use additional data. They can hence, easily be modified for an arbitrary corpus.

The preparation of an individual pre-training dataset is described in the `run_preprocessing.sh` script found in the `data/bookcorpus` and `data/wikipedia_corpus` folders. The component steps to prepare the datasets are as follows:

1.  Data download and extract - the dataset is downloaded and extracted

2.  Clean and format - document tags, etc. are removed from the dataset. The end result of this step is a `{dataset_name}.txt` file that contains the entire corpus. Each line in the text file contains an entire document from the corpus.

3.  Sentence segmentation - the corpus text file is processed into separate sentences. The result of this step is a `{dataset_name}.segmented.nltk.txt` file in a `final_text_file_single` directory that contains the entire corpus, with each sentence having its own line. Documents are separated by a new line between documents.

4.  Sharding - the sentence segmented corpus file is split into a number of smaller text documents. The sharding is configured so that a document will not be split between two shards.

5.  TFRecord file creation - each text file shard is processed by the `create_pretraining_data.py` script to produce a corresponding TFRecord file. The script generates input data and labels for masked language modeling and sentence prediction tasks for the input text shard.


For fine tuning BERT for the task of Question Answering. We use SQuaD for this task. SQuaD v1.1 has 100,000+ question-answer pairs on 500+ articles. SQuaD v2.0 combines v1.1 with an additional 50,000 new unanswerable questions and must not only answer questions but also determine when that is not possible.

#### Dataset guidelines

The procedure to prepare a text corpus for pre-training is described in the previous section. This section provides additional insight into how exactly raw text is processed so that it is ready for pre-training.

First, raw text is tokenized using [WordPiece tokenization](https://arxiv.org/pdf/1609.08144.pdf). A [CLS] token is inserted at the start of every sequence, and the two sentences in the sequence are separated with a [SEP] token.

Note: BERT pre-training looks at pairs of sentences at a time. A sentence embedding token [A] is added to the first sentence and token [B] to the next.

BERT pre-training optimizes for two unsupervised classification tasks. The first is Masked Language Modelling (Masked LM). One training instance of Masked LM is a single modified sentence. Each token in the sentence has a 15% chance of being replaced by a [MASK] token. The chosen token is replaced with [MASK] 80% of the time, 10% with another random token and the remaining 10% with the same token. The task is then to predict the original token.

The second task is next sentence prediction. One training instance of BERT pre-training is two sentences (a sentence pair). A sentence pair may be constructed by simply taking two adjacent sentences from a single document, or by pairing up two random sentences with equal probability. The goal of this task is to predict whether or not the second sentence followed the first in the original document.

The `create_pretraining_data.py` script takes in raw text and creates training instances for both pre-training tasks.

#### Multi-dataset

We are able to combine multiple datasets into a single dataset for pre-training on a diverse text corpus. Once TFRecords have been created for each component dataset, then one can simply create a combined dataset by adding the directory to `SOURCES` in `run_pretraining.sh`. This will feed all matching files to the input pipeline in `run_pretraining.py`. However, note that in the training process only one TFRecord file is consumed at a time, therefore, the training instances of any given training batch will all belong to the same source dataset.

### Training process

The training process consists of two steps: pre-training and fine tuning.

#### Pre-training

Pre-training is performed using the `run_pretraining.py` script along with parameters defined in the `scripts/run_pretraining.sh`.


The `run_pretraining.sh` script runs a job on a single node that trains the BERT-large model from scratch using the Wikipedia and Book corpus datasets as training data. By default, the training script:
- Runs on 8 GPUs with training batch size of 14 and evaluation batch size of 8 per GPU.
- Has FP16 precision enabled.
- Is XLA enabled.
- Runs for 1144000 steps with 10000 warm-up steps.
- Saves a checkpoint every 5000 iterations (keeps only the latest checkpoint) and at the end of training. All checkpoints, evaluation results and training logs are saved to the `/results` directory (in the container which can be mounted to a local directory).
- Creates a log file containing all the output.
- Evaluates the model at the end of training. To skip evaluation, modify `--do_eval` to `False`.

These parameters will train Wikipedia and Books Corpus to reasonable accuracy on a DGX1 with 32GB V100 cards. If you want to match Google’s best results from the BERT paper, you should either train for twice as many steps (2,288,000 steps) on a DGX-1, or train on 16 GPUs on a DGX-2. The DGX-2 having 16 GPUs will be able to fit a batch size twice as large as a DGX-1 (224 vs 112), hence the DGX-2 can finish in half as many steps.


For example:
```bash
run_pretraining.sh <training_batch_size> <eval_batch_size> <learning-rate> <precision> <num_gpus> <warmup_steps> <training_steps> <save_checkpoint_steps> <create_logfile>
```

Where:
- <training_batch_size> is per-GPU batch size used for training. Batch size varies with precision, larger batch sizes run more efficiently, but require more memory.

- <eval_batch_size> is per-GPU batch size used for evaluation after training.

- <learning_rate> is the default rate of 1e-4 is good for global batch size 256.

- <precision> is the type of math in your model, can be either `fp32` or `amp`. Specifically:

    - `fp32` is 32-bit IEEE single precision floats.
    - `amp` is Automatic rewrite of TensorFlow compute graph to take advantage of 16-bit arithmetic whenever it is safe.

- <num_gpus> is the number of GPUs to use for training. Must be equal to or smaller than the number of GPUs attached to your node.

- <warmup_steps> is the number of warm-up steps at the start of training.

- <training_steps> is the total number of training steps.

- <save_checkpoint_steps> controls how often checkpoints are saved. Default is 5000 steps.

- <create_logfile> is a flag that indicates whether output should be written to a logfile or not (acceptable values are ‘true’ or ‘false’, ‘true’ indicates output should be saved to a logfile.)

The following sample code, trains BERT-large from scratch on a single DGX-2 using FP16 arithmetic. This will take around 156 hours / 6.5 days. Checkpoints are written out every 5000 steps and all printouts are saved to a logfile.

```bash
bert_tf/scripts/run_pretraining.sh 14 8 1e-4 fp16_xla 16 10000 1144000 5000 true
```

#### Fine tuning

Fine tuning is performed using the `run_squad.py` script along with parameters defined in `scripts/run_squad.sh`.

The `run_squad.sh` script trains a model and performs evaluation on the SQuaD dataset. By default, the training script:
- Trains for SQuAD v1.1 dataset
- Trains on BERT Large Model
- Uses 8 GPUs and batch size of 10 on each GPU.
- Has FP16 precision enabled.
- Is XLA enabled.
- Runs for 2 epochs.
- Saves a checkpoint every 1000 iterations (keeps only the latest checkpoint) and at the end of training. All checkpoints, evaluation results and training logs are saved to the `/results` directory (in the container which can be mounted to a local directory).
- Evaluation is done at the end of training. To skip evaluation, modify `--do_predict` to `False`.

This script outputs checkpoints to the `/results` directory, by default, inside the container. Mount point of `/results` can be changed in the `scripts/docker/launch.sh` file. The training log contains information about:
- Loss for the final step
- Training and evaluation performance
- F1 and exact match score on the Dev Set of SQuaD after evaluation.

The summary after training is printed in the following format:
```bash
I0312 23:10:45.137036 140287431493376 run_squad.py:1332] 0 Total Training Time = 3007.00 Training Time W/O start up overhead = 2855.92 Sentences processed = 175176
I0312 23:10:45.137243 140287431493376 run_squad.py:1333] 0 Training Performance = 61.3378 sentences/sec
I0312 23:14:00.550846 140287431493376 run_squad.py:1396] 0 Total Inference Time = 145.46 Inference Time W/O start up overhead = 131.86 Sentences processed = 10840
I0312 23:14:00.550973 140287431493376 run_squad.py:1397] 0 Inference Performance = 82.2095 sentences/sec
{"exact_match": 83.69914853358561, "f1": 90.8477003317459}
```

Multi-GPU training is enabled with the Horovod TensorFlow module. The following example runs training on 8 GPUs:
```bash
BERT_DIR=data/pretrained_models_google/uncased_L-24_H-1024_A-16

mpi_command="mpirun -np 8 -H localhost:8 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH -mca pml ob1 -mca btl ^openib" \
     python run_squad.py --horovod --vocab_file=$BERT_DIR/vocab.txt \
     --bert_config_file=$BERT_DIR/bert_config.json \
     --output_dir=/results
```

### Inference process

Inference on a fine tuned Question Answering system is performed using the `run_squad.py` script along with parameters defined in `scripts/run_squad_inference.sh`. Inference is supported on a single GPU.

The `run_squad_inference.sh` script trains a model and performs evaluation on the SQuaD dataset. By default, the inferencing script:
- Uses SQuAD v1.1 dataset
- Has FP16 precision enabled
- Is XLA enabled
- Evaulates the latest checkpoint present in `/results` with a batch size of 8

This script outputs predictions file to `/results/predictions.json` and computes F1 score and exact match score using SQuaD's evaluate file. Mount point of `/results` can be changed in the `scripts/docker/launch.sh` file.

The output log contains information about:
Inference performance
Inference Accuracy (F1 and exact match scores) on the Dev Set of SQuaD after evaluation.

The summary after inference is printed in the following format:
```bash
I0312 23:14:00.550846 140287431493376 run_squad.py:1396] 0 Total Inference Time = 145.46 Inference Time W/O start up overhead = 131.86 Sentences processed = 10840
I0312 23:14:00.550973 140287431493376 run_squad.py:1397] 0 Inference Performance = 82.2095 sentences/sec
{"exact_match": 83.69914853358561, "f1": 90.8477003317459}
```

### Deploying the BERT model using TensorRT Inference Server

The [NVIDIA TensorRT Inference Server](https://github.com/NVIDIA/tensorrt-inference-server) provides a datacenter and cloud inferencing solution optimized for NVIDIA GPUs. The server provides an inference service via an HTTP or gRPC endpoint, allowing remote clients to request inferencing for any number of GPU or CPU models being managed by the server.

A typical TensorRT Inference Server pipeline can be broken down into the following 8 steps:
Client serializes the inference request into a message and sends it to the server (Client Send)
Message travels over the network from the client to the server (Network)
Message arrives at server, and is deserialized (Server Receive)
Request is placed on the queue (Server Queue)
Request is removed from the queue and computed (Server Compute)
Completed request is serialized in a message and sent back to the client (Server Send)
Completed message travels over network from the server to the client (Network)
Completed message is deserialized by the client and processed as a completed inference request (Client Receive)

Generally, for local clients, steps 1-4 and 6-8 will only occupy a small fraction of time, compared to steps 5-6. As backend deep learning systems like BERT are rarely exposed directly to end users, but instead only interfacing with local front-end servers, for the sake of BERT, we can consider that all clients are local.
In this section, we will go over how to launch TensorRT Inference Server and client and get the best performant solution that fits your specific application needs.

Note: The following instructions are run from outside the container and call `docker run` commands as required.

#### Performance analysis for TensorRT Inference Server

Based on the figures 2 and 3 below, we recommend using the Dynamic Batcher with `max_batch_size = 8`, `max_queue_delay_microseconds` as large as possible to fit within your latency window (The values used below are extremely large to exaggerate their effect), and only 1 instance of the engine. The largest improvements to both throughput and latency come from increasing the batch size due to efficiency gains in the GPU with larger batches. The Dynamic Batcher combines the best of both worlds by efficiently batching together a large number of simultaneous requests, while also keeping latency down for infrequent requests. We recommend only 1 instance of the engine due to the negligible improvement to throughput at the cost of significant increases in latency. Many models can benefit from multiple engine instances but as the figures below show, that is not the case for this model.


![](data/images/trtis_base_summary.png?raw=true)

Figure 2: Latency vs Throughput for BERT Base, fp16, Sequence Length = 128 using Various configurations available in TRTIS

![](data/images/trtis_large_summary.png?raw=true)

Figure 3: Latency vs Throughput for BERT Large, fp16, Sequence Length = 384 using Various configurations available in TRTIS

##### Advanced Details

This section digs deeper into the performance numbers and configurations corresponding to running TensorRT Inference Server for BERT fine tuning for Question Answering. It explains the tradeoffs in selecting maximum batch sizes, batching techniques and number of inference engines on the same GPU to understand how we arrived at the optimal configuration specified previously.

Results can be reproduced by running `generate_figures.sh`. It exports the Tensorflow BERT model as a `tensorflow_savedmodel` that TensorRT Inference Server accepts, builds a matching [TensorRT Inference Server model config](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/model_configuration.html#), starts the server on localhost in a detached state and runs [perf_client](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/client.html#performance-example-application) for various configurations.

```bash
bash scripts/trtis/generate_figures.sh <bert_model> <seq_length> <precision> <init_checkpoint>
```

All results below are obtained on 1 DGX-1 V100 32 GB GPU for BERT Base, Sequence Length = 128 and FP16 precision running on a local server. Latencies are indicated by bar plots using the left axis. Throughput is indicated by the blue line plot using the right axis. X-axis indicates the concurrency - the maximum number of inference requests that can be in the pipeline at any given time. For example, when the concurrency is set to 1, the client waits for an inference request to be completed (Step 8) before it sends another to the server (Step 1).  A high number of concurrent requests can reduce the impact of network latency on overall throughput

1. Maximum batch size

As we can see in figure 4 below, the throughput at BS=1, Client Concurrent Requests = 64 is 119 and in figure 5 below, the throughput at BS=8, Client Concurrent Requests = 8 is 517, respectively giving a speedup of ~4.3x (Note: We compare BS=1, Client Concurrent Requests = 64 to BS=8, Client Concurrent Requests = 8 to keep the Total Number of Outstanding Requests equal between the two different modes. Where Total Number of Outstanding Requests = Batch Size * Client Concurrent Requests. This is also why there are 8 times as many bars on the BS=1 chart than the BS=8 chart). Increasing the batch size from 1 to 8 results in an increase in compute time by 1.8x (8.38ms to 15.46ms) showing that computation is more efficient at higher batch sizes. Hence, an optimal batch size would be the maximum batch size that can both fit in memory and is within the preferred latency threshold.


![](data/images/trtis_bs_1.png?raw=true)

Figure 4: Latency & Throughput vs Concurrency at Batch size = 1

![](data/images/trtis_bs_8.png?raw=true)

Figure 5: Latency & Throughput vs Concurrency at Batch size = 8

2.  Batching techniques

Static batching is a feature of the inference server that allows inference requests to be served as they are received. It is preferred in scenarios where low latency is desired at the cost of throughput when the GPU is under utilized.

Dynamic batching is a feature of the inference server that allows inference requests to be combined by the server, so that a batch is created dynamically, resulting in an increased throughput. It is preferred in scenarios where we would like to maximize throughput and GPU utilization at the cost of higher latencies. User can set the [Dynamic Batcher parameters](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-master-branch-guide/docs/model_configuration.html#dynamic-batcher) `max_queue_delay_microseconds` to indicate the maximum amount of time they are willing to wait and ‘preferred_batchsize’ to indicate their optimal batch sizes in the TensorRT Inference Server model config.

The figures 6 & 7 below emphasize the increase in overall throughput with dynamic batching. At low numbers of concurrent requests, the increased throughput comes at the cost of increasing latency as the requests are queued up to `max_queue_delay_microseconds`. The effect of `preferred_batchsize` for dynamic batching is visually depicted by the dip in Server Queue time at integer multiples of the preferred batch sizes. At higher numbers of concurrent requests, observe that the throughput approach a maximum limit as we saturate the GPU utilization.

![](data/images/trtis_static.png?raw=true)

Figure 6: Latency & Throughput vs Concurrency using Static Batching at `Batch size` = 1

![](data/images/trtis_dynamic.png?raw=true)

Figure 7: Latency & Throughput vs Concurrency using Dynamic Batching at `Batch size` = 1, `preferred_batchsize` = [4, 8] and `max_queue_delay_microseconds` = 5000

3. Model execution instance count

TensorRT Inference Server enables us to launch multiple engines in separate CUDA streams by setting the `instance_group_count` parameter to improve both latency and throughput. Multiple engines are useful when the model doesn’t saturate the GPU allowing the GPU to run multiple instances of the model in parallel.

From the figures 8 & 9 below, we can see a drop in queue time as more models are available to serve an inference request. However, this is countered by an increase in compute time as multiple models compete for resources. Since BERT is a large model which utilizes the majority of the GPU, the benefit to running multiple engines is not seen.


![](data/images/trtis_ec_1.png?raw=true)

Figure 8: Latency & Throughput vs Concurrency at Batch size = 1, Engine Count = 1
(One copy of the model loaded in GPU memory)

![](data/images/trtis_ec_4.png?raw=true)

Figure 9: Latency & Throughput vs Concurrency at Batch size = 1, Engine count = 4
(Four copies the model loaded in GPU memory)

#### Run the TensorRT Inference Server and client

The `run_trtis.sh` script exports the Tensorflow BERT model as a `tensorflow_savedmodel` that TensorRT Inference Server accepts, builds a matching [TensorRT Inference Server model config](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/model_configuration.html#), starts the server on local host in a detached state, runs client and then evaluates the validity of predictions on the basis of exact match and F1 score all in one step.

```bash
bash scripts/trtis/run_trtis.sh <init_checkpoint> <batch_size> <precision> <use_xla> <seq_length> <doc_stride> <bert_model> <squad_version> <trtis_version_name> <trtis_model_name> <trtis_export_model> <trtis_dyn_batching_delay> <trtis_engine_count> <trtis_model_overwrite>
```

## Performance

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in training and inference modes.

Both of these benchmarking scripts enable you to run a number of epochs, extract performance numbers, and run the BERT model for fine tuning.

#### Training performance benchmark

Training benchmarking can be performed by running the script:
``` bash
scripts/finetune_train_benchmark.sh <bert_model> <use_xla> <num_gpu> squad
```

This script runs 2 epochs by default on the SQuAD v1.1 dataset and extracts performance numbers for various batch sizes and sequence lengths in both FP16 and FP32. These numbers are saved at `/results/squad_inference_benchmark_bert_<bert_model>_gpu_<num_gpu>.log`.

#### Inference performance benchmark

Inference benchmarking can be performed by running the script:

``` bash
scripts/finetune_inference_benchmark.sh <bert_model> <use_xla> squad
```

This script runs 1024 eval iterations by default on the SQuAD v1.1 dataset and extracts performance and latency numbers for various batch sizes and sequence lengths in both FP16 and FP32. These numbers are saved at `/results/squad_train_benchmark_bert_<bert_model>.log`.

### Results

The following sections provide details on how we achieved our performance and accuracy in training and inference for Question Answering fine tuning. All results are on BERT-large model for a sequence length of 384 on SQuAD v1.1 unless otherwise mentioned.

#### Training accuracy results


##### NVIDIA DGX-1 (8x V100 16G)

Our results were obtained by running the `run_squad.py` training script in the TensorFlow 19.06-py3 NGC container on NVIDIA DGX-1 with 8x V100 16G GPUs.


| **GPUs** | **Batch size / GPU** | **Accuracy - FP32** | **Accuracy - mixed precision** | **Time to Train - FP32 (Hrs)** | **Time to Train - mixed precision (Hrs)** |
|:---:|:----:|:----:|:---:|:----:|:----:|
| 8 | 4 |90.84|90.86|0.97|0.64|


##### Training stability test

The following tables compare `F1` scores across 5 different training runs with different seeds, for both FP16 and FP32 respectively.  The runs showcase consistent convergence on all 5 seeds with very little deviation.

| **FP16, 8x GPUs** | **seed 1** | **seed 2** | **seed 3** | **seed 4** | **seed 5** | **mean** | **std** |
|:-----------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|F1         |90.75|90.82|90.89|91.05|90.79|90.86|0.12|
|Exact match|83.85|83.93|83.95|84.25|83.59|83.91|0.24|

| **FP32, 8x GPUs** | **seed 1** | **seed 2** | **seed 3** | **seed 4** | **seed 5** | **mean** | **std** |
|:-----------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|F1         |90.70|90.80|90.89|91.08|90.73|90.84|0.15 |
|Exact match|83.82|83.77|84.23|84.19|83.63|83.93|0.27 |


#### Training performance results

Our results were obtained by running batch sizes up to 3x GPUs on a 16GB V100 and up to 10x GPUs on a 32G V100 with mixed precision.


##### Training performance: NVIDIA DGX-1 (8x V100 16G)

Our results were obtained by running the `scripts/run_squad.sh` training script in the TensorFlow 19.06-py3 NGC container on NVIDIA DGX-1 with 8x V100 16G GPUs. Performance (in sentences per second) is the mean throughput from 2 epochs.


| **GPUs** | **Batch size / GPU** | **Throughput - FP32** | **Throughput - mixed precision** | **Throughput speedup (FP32 to mixed precision)** | **Weak scaling - FP32** | **Weak scaling - mixed precision** |
|:---:|:---:|:------:|:-----:|:----:|:----:|:----:|
| 1 | 2 | 7.19 |14.37|2.0 |1.0  |1.0 |
| 4 | 2 |25.61 |40.44|1.58 |3.56 |2.81|
| 8 | 2 |49.79 |74.61|1.5 |6.92 |5.19|


| **GPUs** | **Batch size / GPU** | **Throughput - FP32** | **Throughput - mixed precision** | **Throughput speedup (FP32 to mixed precision)** | **Weak scaling - FP32** | **Weak scaling - mixed precision** |
|:---:|:---:|:-----:|:-----:|:---:|:---:|:----:|
| 1 | 3 |  -  |17.2| - | - |1.0 |
| 4 | 3 |  -  |50.71| - | - |2.95 |
| 8 | 3 |  -  |91.88| - | - |5.34|


Note: The respective values for FP32 runs that use a batch size of 3 are not available due to out of memory errors that arise. Batch size of 3 is only available on using FP16.

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.


##### Training performance: NVIDIA DGX-1 (8x V100 32G)

Our results were obtained by running the `scripts/run_squad.sh` training script in the TensorFlow 19.06-py3 NGC container on NVIDIA DGX-1 with 8x V100 32G GPUs. Performance (in sentences per second) is the mean throughput from 2 epochs.


| **GPUs** | **Batch size / GPU** | **Throughput - FP32** | **Throughput - mixed precision** | **Throughput speedup (FP32 to mixed precision)** | **Weak scaling - FP32** | **Weak scaling - mixed precision** |
|---|---|-----|------|----|----|----|
| 1 | 4 | 8.74|20.55 |2.35|1.0 |1.0 |
| 4 | 4 |32.22|57.58 |1.79|3.69|2.81|
| 8 | 4 |62.69|100.22|1.60|7.17|4.88|


| **GPUs** | **Batch size / GPU** | **Throughput - FP32** | **Throughput - mixed precision** | **Throughput speedup (FP32 to mixed precision)** | **Weak scaling - FP32** | **Weak scaling - mixed precision** |
|---|---|-----|-------|---|---|----|
| 1 | 10|  -  | 31.33 | - | - |1.0 |
| 4 | 10|  -  | 94.19| - | - |3.0|
| 8 | 10|  -  | 155.53| - | - |4.96|


Note: The respective values for FP32 runs that use a batch size of 10 are not available due to out of memory errors that arise. Batch size of 10 is only available on using FP16.

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

##### Training performance: NVIDIA DGX-2 (16x V100 32G)

Our results were obtained by running the `scripts/run_squad.sh` training script in the TensorFlow 19.06-py3 NGC container on NVIDIA DGX-2 with 16x V100 32G GPUs. Performance (in sentences per second) is the mean throughput from 2 epochs.

| **GPUs** | **Batch size / GPU** | **Throughput - FP32** | **Throughput - mixed precision** | **Throughput speedup (FP32 to mixed precision)** | **Weak scaling - FP32** | **Weak scaling - mixed precision** |
|---|---|------|------|----|-----|-----|
|  1| 4 | 9.39 | 20.69 |2.20| 1.0 | 1.0 |
|  4| 4 | 34.63| 62.79|1.81| 3.69| 3.03|
|  8| 4 | 66.95|111.47|1.66| 7.13 | 5.39|
| 16| 4 |126.09|179.09|1.42| 13.43 |8.66|



| **GPUs** | **Batch size / GPU** | **Throughput - FP32** | **Throughput - mixed precision** | **Throughput speedup (FP32 to mixed precision)** | **Weak scaling - FP32** | **Weak scaling - mixed precision** |
|---|---|---|------|---|---|-----|
|  1| 10| - | 32.72| - | - | 1.0 |
|  4| 10| - |100.73| - | - | 3.07 |
|  8| 10| - |168.92| - | - | 5.16 |
| 16| 10| - |249.54| - | - | 7.63 |


Note: The respective values for FP32 runs that use a batch size of 10 are not available due to out of memory errors that arise. Batch size of 10 is only available on using FP16.


To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.


#### Inference performance results

##### Inference performance: NVIDIA DGX-1 (1x V100 16G)

Our results were obtained by running the `scripts/finetune_inference_benchmark.sh` script in the TensorFlow 19.06-py3 NGC container on NVIDIA DGX-1 with 1x V100 16G GPUs. Performance numbers (throughput in sentences per second and latency in milliseconds) were averaged from 1024 iterations. Latency is computed as the time taken for a batch to process as they are fed in one after another in the model ie no pipelining.

BERT LARGE Fp16

| Sequence Length | Batch Size | Throughput-Average(sent/sec) | Latency-Average(ms) | Latency-90%(ms) | Latency-95%(ms) | Latency-99%(ms) |
|-----------------|------------|------------------------------|---------------------|-----------------|-----------------|-----------------|
| 128             | 1          | 89.4                         | 11.19               | 11.29           | 11.44           | 11.71           |
| 128             | 2          | 162.29                       | 12.32               | 12.5            | 12.57           | 12.74           |
| 128             | 4          | 263.44                       | 15.18               | 15.32           | 15.54           | 17              |
| 128             | 8          | 374.33                       | 21.37               | 21.56           | 21.72           | 23.23           |
| 384             | 1          | 64.57                        | 15.49               | 15.61           | 15.73           | 16.18           |
| 384             | 2          | 94.04                        | 21.27               | 21.34           | 21.4            | 21.9            |
| 384             | 4          | 118.81                       | 33.67               | 33.89           | 34.37           | 36.18           |
| 384             | 8          | 137.65                       | 58.12               | 58.53           | 59.34           | 61.32           |

BERT LARGE FP32

| Sequence Length | Batch Size | Throughput-Average(sent/sec) | Latency-Average(ms) | Latency-90%(ms) | Latency-95%(ms) | Latency-99%(ms) |
|-----------------|------------|------------------------------|---------------------|-----------------|-----------------|-----------------|
| 128             | 1          | 75.28                        | 13.28               | 13.4            | 13.49           | 13.66           |
| 128             | 2          | 104.16                       | 19.2                | 19.51           | 19.69           | 20.83           |
| 128             | 4          | 117.4                        | 34.07               | 34.4            | 34.76           | 36.99           |
| 128             | 8          | 125.63                       | 63.68               | 64.58           | 65.1            | 67.54           |
| 384             | 1          | 34.53                        | 28.96               | 29.32           | 29.61           | 31.08           |
| 384             | 2          | 38.03                        | 52.59               | 53.16           | 53.75           | 55.5            |
| 384             | 4          | 40.16                        | 99.6                | 100.76          | 101.62          | 103.4           |
| 384             | 8          | 42.2                         | 189.57              | 190.82          | 191.47          | 193.27          |

BERT BASE Fp16

| Sequence Length | Batch Size | Throughput-Average(sent/sec) | Latency-Average(ms) | Latency-90%(ms) | Latency-95%(ms) | Latency-99%(ms) |
|-----------------|------------|------------------------------|---------------------|-----------------|-----------------|-----------------|
| 128             | 1          | 196.58                       | 5.09                | 5.18            | 5.23            | 5.42            |
| 128             | 2          | 361.92                       | 5.53                | 5.62            | 5.67            | 5.85            |
| 128             | 4          | 605.43                       | 6.61                | 6.71            | 6.8             | 7.04            |
| 128             | 8          | 916                          | 8.73                | 8.83            | 8.95            | 9.19            |
| 384             | 1          | 154.05                       | 6.49                | 6.6             | 6.72            | 7.05            |
| 384             | 2          | 238.89                       | 8.37                | 8.42            | 8.47            | 9.1             |
| 384             | 4          | 327.18                       | 12.23               | 12.3            | 12.36           | 13.08           |
| 384             | 8          | 390.95                       | 20.46               | 20.5            | 20.8            | 21.89           |


BERT BASE FP32

| Sequence Length | Batch Size | Throughput-Average(sent/sec) | Latency-Average(ms) | Latency-90%(ms) | Latency-95%(ms) | Latency-99%(ms) |
|-----------------|------------|------------------------------|---------------------|-----------------|-----------------|-----------------|
| 128             | 1          | 165.51                       | 6.04                | 6.19            | 6.3             | 6.62            |
| 128             | 2          | 257.54                       | 7.77                | 7.86            | 7.92            | 8.28            |
| 128             | 4          | 338.52                       | 11.82               | 11.98           | 12.05           | 12.27           |
| 128             | 8          | 419.94                       | 19.05               | 19.25           | 19.35           | 20.12           |
| 384             | 1          | 97.4                         | 10.27               | 10.39           | 10.44           | 10.56           |
| 384             | 2          | 119.84                       | 16.69               | 16.78           | 16.85           | 17.66           |
| 384             | 4          | 132.5                        | 30.19               | 30.41           | 30.5            | 31.13           |
| 384             | 8          | 138.63                       | 57.71               | 58.15           | 58.37           | 59.33           |


To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

##### Inference performance: NVIDIA DGX-1 (1x V100 32G)

Our results were obtained by running the `scripts/finetune_inference_benchmark.sh` training script in the TensorFlow 19.06-py3 NGC container on NVIDIA DGX-1 with 1x V100 32G GPUs. Performance numbers (throughput in sentences per second and latency in milliseconds) were averaged from 1024 iterations. Latency is computed as the time taken for a batch to process as they are fed in one after another in the model ie no pipelining.

BERT LARGE FP16

| Sequence Length | Batch Size | Throughput-Average(sent/sec) | Latency-Average(ms) | Latency-90%(ms) | Latency-95%(ms) | Latency-99%(ms) |
|-----------------|------------|------------------------------|---------------------|-----------------|-----------------|-----------------|
| 128             | 1          | 86.4                         | 11.57               | 11.74           | 11.86           | 12.04           |
| 128             | 2          | 155.32                       | 12.88               | 12.98           | 13.05           | 13.31           |
| 128             | 4          | 252.18                       | 15.86               | 15.78           | 15.89           | 17.01           |
| 128             | 8          | 359.19                       | 22.27               | 22.44           | 22.58           | 23.94           |
| 384             | 1          | 62.45                        | 16.01               | 16.16           | 16.23           | 16.42           |
| 384             | 2          | 89.34                        | 22.39               | 22.45           | 22.53           | 23.13           |
| 384             | 4          | 113.77                       | 35.16               | 35.24           | 35.33           | 35.9            |
| 384             | 8          | 131.9                        | 60.65               | 61              | 61.49           | 65.3            |

BERT LARGE FP32

| Sequence Length | Batch Size | Throughput-Average(sent/sec) | Latency-Average(ms) | Latency-90%(ms) | Latency-95%(ms) | Latency-99%(ms) |
|-----------------|------------|------------------------------|---------------------|-----------------|-----------------|-----------------|
| 128             | 1          | 73.42                        | 13.62               | 13.78           | 13.85           | 14.13           |
| 128             | 2          | 102.47                       | 19.52               | 19.66           | 19.73           | 19.98           |
| 128             | 4          | 115.76                       | 34.55               | 34.86           | 35.34           | 37.87           |
| 128             | 8          | 124.84                       | 64.08               | 64.78           | 65.78           | 69.55           |
| 384             | 1          | 33.93                        | 29.47               | 29.7            | 29.8            | 29.98           |
| 384             | 2          | 37.62                        | 53.16               | 53.52           | 53.73           | 55.03           |
| 384             | 4          | 39.99                        | 100.02              | 100.91          | 101.69          | 106.63          |
| 384             | 8          | 42.09                        | 190.08              | 191.35          | 192.29          | 196.47          |

BERT BASE FP16

| Sequence Length | Batch Size | Throughput-Average(sent/sec) | Latency-Average(ms) | Latency-90%(ms) | Latency-95%(ms) | Latency-99%(ms) |
|-----------------|------------|------------------------------|---------------------|-----------------|-----------------|-----------------|
| 128             | 1          | 192.89                       | 5.18                | 5.29            | 5.35            | 5.55            |
| 128             | 2          | 348.23                       | 5.74                | 5.91            | 6.02            | 6.26            |
| 128             | 4          | 592.54                       | 6.75                | 6.96            | 7.08            | 7.34            |
| 128             | 8          | 888.58                       | 9                   | 9.11            | 9.22            | 9.5             |
| 384             | 1          | 148.64                       | 6.73                | 6.82            | 6.87            | 7.06            |
| 384             | 2          | 230.74                       | 8.67                | 8.75            | 8.87            | 9.44            |
| 384             | 4          | 318.45                       | 12.56               | 12.65           | 12.76           | 13.36           |
| 384             | 8          | 380.14                       | 21.05               | 21.1            | 21.25           | 21.83           |


BERT BASE FP32

| Sequence Length | Batch Size | Throughput-Average(sent/sec) | Latency-Average(ms) | Latency-90%(ms) | Latency-95%(ms) | Latency-99%(ms) |
|-----------------|------------|------------------------------|---------------------|-----------------|-----------------|-----------------|
| 128             | 1          | 161.69                       | 6.18                | 6.26            | 6.31            | 6.51            |
| 128             | 2          | 254.84                       | 7.85                | 8               | 8.09            | 8.29            |
| 128             | 4          | 331.72                       | 12.06               | 12.17           | 12.26           | 12.51           |
| 128             | 8          | 412.85                       | 19.38               | 19.6            | 19.72           | 20.13           |
| 384             | 1          | 94.42                        | 10.59               | 10.71           | 10.8            | 11.36           |
| 384             | 2          | 117.64                       | 17                  | 17.07           | 17.1            | 17.83           |
| 384             | 4          | 131.72                       | 30.37               | 30.64           | 30.77           | 31.26           |
| 384             | 8          | 139.75                       | 57.25               | 57.74           | 58.08           | 59.53           |



To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

##### Inference performance: NVIDIA DGX-2 (1x V100 32G)

Our results were obtained by running the `scripts/finetune_inference_benchmark.sh` training script in the TensorFlow 19.06-py3 NGC container on NVIDIA DGX-2 with 1x V100 32G GPUs. Performance numbers (throughput in sentences per second and latency in milliseconds) were averaged from 1024 iterations. Latency is computed as the time taken for a batch to process as they are fed in one after another in the model ie no pipelining.



BERT LARGE FP16

| Sequence Length | Batch Size | Throughput-Average(sent/sec) | Latency-Average(ms) | Latency-90%(ms) | Latency-95%(ms) | Latency-99%(ms) |
|-----------------|------------|------------------------------|---------------------|-----------------|-----------------|-----------------|
| 128             | 1          | 79                           | 12.66               | 13.13           | 13.36           | 14.49           |
| 128             | 2          | 151.28                       | 13.22               | 13.66           | 13.89           | 14.84           |
| 128             | 4          | 250.41                       | 15.97               | 16.13           | 16.3            | 17.81           |
| 128             | 8          | 369.76                       | 21.64               | 21.88           | 22.08           | 26.35           |
| 384             | 1          | 61.66                        | 16.22               | 16.46           | 16.62           | 17.26           |
| 384             | 2          | 91.54                        | 21.85               | 22.11           | 22.3            | 23.44           |
| 384             | 4          | 121.04                       | 33.05               | 33.08           | 33.31           | 34.97           |
| 384             | 8          | 142.03                       | 56.33               | 56.46           | 57.49           | 59.85           |


BERT LARGE FP32

| Sequence Length | Batch Size | Throughput-Average(sent/sec) | Latency-Average(ms) | Latency-90%(ms) | Latency-95%(ms) | Latency-99%(ms) |
|-----------------|------------|------------------------------|---------------------|-----------------|-----------------|-----------------|
| 128             | 1          | 70.1                         | 14.27               | 14.6            | 14.84           | 15.38           |
| 128             | 2          | 101.3                        | 19.74               | 20.09           | 20.27           | 20.77           |
| 128             | 4          | 122.19                       | 32.74               | 32.99           | 33.39           | 36.76           |
| 128             | 8          | 134.09                       | 59.66               | 60.36           | 61.79           | 69.33           |
| 384             | 1          | 34.52                        | 28.97               | 29.28           | 29.46           | 31.78           |
| 384             | 2          | 39.84                        | 50.21               | 50.61           | 51.53           | 54              |
| 384             | 4          | 42.79                        | 93.48               | 94.73           | 96.52           | 104.37          |
| 384             | 8          | 45.91                        | 174.24              | 175.34          | 176.59          | 183.76          |


BERT BASE FP16

| Sequence Length | Batch Size | Throughput-Average(sent/sec) | Latency-Average(ms) | Latency-90%(ms) | Latency-95%(ms) | Latency-99%(ms) |
|-----------------|------------|------------------------------|---------------------|-----------------|-----------------|-----------------|
| 128             | 1          | 192.89                       | 5.18                | 5.29            | 5.35            | 5.55            |
| 128             | 2          | 348.23                       | 5.74                | 5.91            | 6.02            | 6.26            |
| 128             | 4          | 592.54                       | 6.75                | 6.96            | 7.08            | 7.34            |
| 128             | 8          | 888.58                       | 9                   | 9.11            | 9.22            | 9.5             |
| 384             | 1          | 148.64                       | 6.73                | 6.82            | 6.87            | 7.06            |
| 384             | 2          | 230.74                       | 8.67                | 8.75            | 8.87            | 9.44            |
| 384             | 4          | 318.45                       | 12.56               | 12.65           | 12.76           | 13.36           |
| 384             | 8          | 380.14                       | 21.05               | 21.1            | 21.25           | 21.83           |

BERT BASE FP32

| Sequence Length | Batch Size | Throughput-Average(sent/sec) | Latency-Average(ms) | Latency-90%(ms) | Latency-95%(ms) | Latency-99%(ms) |
|-----------------|------------|------------------------------|---------------------|-----------------|-----------------|-----------------|
| 128             | 1          | 161.69                       | 6.18                | 6.26            | 6.31            | 6.51            |
| 128             | 2          | 254.84                       | 7.85                | 8               | 8.09            | 8.29            |
| 128             | 4          | 331.72                       | 12.06               | 12.17           | 12.26           | 12.51           |
| 128             | 8          | 412.85                       | 19.38               | 19.6            | 19.72           | 20.13           |
| 384             | 1          | 94.42                        | 10.59               | 10.71           | 10.8            | 11.36           |
| 384             | 2          | 117.64                       | 17                  | 17.07           | 17.1            | 17.83           |
| 384             | 4          | 131.72                       | 30.37               | 30.64           | 30.77           | 31.26           |
| 384             | 8          | 139.75                       | 57.25               | 57.74           | 58.08           | 59.53           |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

## Release notes
### Changelog

March 2019
- Initial release

July 2019
- Results obtained using 19.06
- Inference Studies using TensorRT Inference Server

### Known issues

There are no known issues with this model.
# BERT For PyTorch

This repository provides a script and recipe to train the BERT model to achieve state of the art accuracy, and is tested and maintained by NVIDIA.

**Table Of Contents**

- [Model overview](#model-overview)
    * [Model architecture](#model-architecture)
    * [Default configuration](#default-configuration)
    * [Feature support matrix](#feature-support-matrix)
        * [Features](#features)
    * [Mixed precision training](#mixed-precision-training)
        * [Enabling mixed precision](#enabling-mixed-precision)
- [Setup](#setup)
    * [Requirements](#requirements)
- [Quick Start Guide](#quick-start-guide)
- [Advanced](#advanced)
    * [Scripts and sample code](#scripts-and-sample-code)
    * [Parameters](#parameters)
        * [Pre-training parameters](#pre-training-parameters)
        * [Fine-tuning parameters](#fine-tuning-parameters)     
    * [Command-line options](#command-line-options)
    * [Getting the data](#getting-the-data)
        * [Dataset guidelines](#dataset-guidelines)
        * [Multi-dataset](#multi-dataset)
            * [Relocating hdf5 files](#relocating-hdf5-files)
            * [Inter sequence-pair mixing](#inter-sequence-pair-mixing)
            * [Retaining document-level granularity](#retaining-document-level-granularity)
    * [Training process](#training-process)
        * [Pre-training](#pre-training)
        * [Fine-tuning](#fine-tuning)   
    * [Inference process](#inference-process)
        * [Pre-training inference](#pre-training-inference)
        * [Fine-tuning inference](#fine-tuning-inference)
- [Performance](#performance)
    * [Benchmarking](#benchmarking)
        * [Training performance benchmark](#training-performance-benchmark)
        * [Inference performance benchmark](#inference-performance-benchmark)
    * [Results](#results)
        * [Training accuracy results](#training-accuracy-results)
            * [Pre-training loss results](#pre-training-loss-results)
            * [Fine-tuning accuracy results](#fine-tuning-accuracy-results) 
            * [Training stability test](#training-stability-test)
                * [Pre-training stability test](#pre-training-stability-test)
                * [Fine-tuning stability test](#fine-tuning-stability-test) 
            * [Training performance results](#training-performance-results)
                * [Training performance: NVIDIA DGX-1 (8x V100 16G)](#training-performance-nvidia-dgx-1-8x-v100-16g)
                    * [Pre-training NVIDIA DGX-1 With 16G](#pre-training-nvidia-dgx-1-with-16g)
                    * [Fine-tuning NVIDIA DGX-1 With 16G](#fine-tuning-nvidia-dgx-1-with-16g)   
                * [Training performance: NVIDIA DGX-1 (8x V100 32G)](#training-performance-nvidia-dgx-1-8x-v100-32g)
                    * [Pre-training NVIDIA DGX-1 With 32G](#pre-training-nvidia-dgx-1-with-32g)
                    * [Fine-tuning NVIDIA DGX-1 With 32G](#fine-tuning-nvidia-dgx-1-with-32g)   
                * [Training performance: NVIDIA DGX-2 (16x V100 32G)](#training-performance-nvidia-dgx-2-16x-v100-32g)
                    * [Pre-training NVIDIA DGX-2 With 32G](#pre-training-nvidia-dgx-2-with-32g)
                    * [Fine-tuning NVIDIA DGX-2 With 32G](#fine-tuning-nvidia-dgx-2-with-32g)   
            * [Inference performance results](#inference-performance-results)
                * [Inference performance: NVIDIA DGX-1 (1x V100 16G)](#inference-performance-nvidia-dgx-1-1x-v100-16g)
                    * [Pre-training inference on NVIDIA DGX-1 with 16G](#pre-training-inference-on-nvidia-dgx-1-with-16g)
                    * [Fine-tuning inference on NVIDIA DGX-1 with 16G](#fine-tuning-inference-on-nvidia-dgx-1-with-16g)
                * [Inference performance: NVIDIA DGX-1 (1x V100 32G)](#inference-performance-nvidia-dgx-1-1x-v100-32g)
                    * [Pre-training inference on NVIDIA DGX-1 with 32G](#pre-training-inference-on-nvidia-dgx-1-with-32g)
                    * [Fine-tuning inference on NVIDIA DGX-1 with 32G](#fine-tuning-inference-on-nvidia-dgx-1-with-32g)
                * [Inference performance: NVIDIA DGX-2 (1x V100 32G)](#inference-performance-nvidia-dgx-2-1x-v100-32g)
                    * [Pre-training inference on NVIDIA DGX-2 with 32G](#pre-training-inference-on-nvidia-dgx-2-with-32g)
                    * [Fine-tuning inference on NVIDIA DGX-2 with 32G](#fine-tuning-inference-on-nvidia-dgx-2-with-32g)
- [Release notes](#release-notes)
    * [Changelog](#changelog)
    * [Known issues](#known-issues)


## Model overview
 
BERT, or Bidirectional Encoder Representations from Transformers, is a new method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks. This model is based on the [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) paper. NVIDIA's implementation of BERT is an optimized version of the [Hugging Face implementation](https://github.com/huggingface/pytorch-pretrained-BERT), leveraging mixed precision arithmetic and Tensor Cores on V100 GPUs for faster training times while maintaining target accuracy.

The repository also contains scripts to interactively launch data download, training, benchmarking and inference routines in a Docker container for both pre-training and fine-tuning for tasks such as question answering. The major differences between the original implementation of the paper and this version of BERT are as follows:

-   Scripts to download Wikipedia and BookCorpus datasets
-   Scripts to preprocess downloaded data or a custom corpus into inputs and targets for pre-training in a modular fashion.
-   Fused [LAMB](https://arxiv.org/pdf/1904.00962.pdf) optimizer to support training with larger batches
-   Fused CUDA kernels for better performance LayerNorm
-   Automatic Mixed precision training support

Other publicly available implementations of BERT include:

1.  [Google's official implementation](https://github.com/google-research/bert)
2.  [codertimo](https://github.com/codertimo/BERT-pytorch)
    
This model trains with mixed precision Tensor Cores on Volta and provides a push-button solution to pretraining on a corpus of choice. As a result, researchers can get results 4x faster than training without Tensor Cores. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.

### Model architecture

The BERT architecture uses the same architecture as the encoder half of the Transformer. Input sequences are projected into an embedding space before being fed into the encoder structure. Additionally, a positional and segment encodings are added to the embeddings to preserve positional information. The encoder structure is simply a stack of Transformer blocks, which consist of a multi-head attention layer followed by successive stages of feed-forward networks and layer normalization. The multi-head attention layer accomplishes self-attention on multiple input representations.

An illustration of the architecture taken from the [Transformer paper](https://arxiv.org/pdf/1706.03762.pdf) is shown below.

 ![BERT](images/model.png)

### Default configuration

The architecture of the BERT model is almost identical to the Transformer model that was first introduced in the [Attention Is All You Need paper](https://arxiv.org/pdf/1706.03762.pdf). The main innovation of BERT lies in the pre-training step, where the model is trained on two unsupervised prediction tasks using a large text corpus. Training on these unsupervised tasks produces a generic language model, which can then be quickly fine-tuned to achieve state-of-the-art performance on language processing tasks such as question answering.

The BERT paper reports results two configurations of BERT, each corresponding to a unique model size. This implementation provides the same configurations by default, which are described in the table below.  

| **Model** | **Hidden layers** | **Hidden unit size** | **Attention heads** | **Feedforward filter size** | **Max sequence length** | **Parameters** |
|:---------:|:----------:|:----:|:---:|:--------:|:---:|:----:|
|BERTBASE |12 encoder| 768| 12|4 x  768|512|110M|
|BERTLARGE|24 encoder|1024| 16|4 x 1024|512|330M|

Additionally, this implementation supports training on multiple GPUs. Mixed precision training and inference with dynamic loss scaling is also supported.

### Feature support matrix

The following features are supported by this model.  

| **Feature** | **BERT** |
|:---------:|:----------:|
|APEX AMP|Yes|
|APEX DDP|Yes|
|LAMB|Yes|

#### Features
 
[APEX](https://github.com/NVIDIA/apex) is a Pytorch extension with NVIDIA-maintained utilities to streamline mixed precision and distributed training. 
 
[DDP](https://nvidia.github.io/apex/parallel.html) stands for DistributedDataParallel and is used for multi-GPU training, where as [AMP](https://nvidia.github.io/apex/amp.html) is an abbreviation used for automatic mixed precision training.

[LAMB](https://arxiv.org/pdf/1904.00962.pdf) stands for Layerwise Adaptive Moments based optimizer, is a large batch optimization technique that helps accelerate training of deep neural networks using large minibatches. It allows using a global batch size of 65536 and 32768 on sequence lengths 128 and 512 respectively, compared to a batch size of 256 for Adam. The optimized implementation accumulates 1024 gradients batches in phase 1 and 4096 steps in phase 2 before updating weights once. This results in 15% training speedup. On multi-node systems, LAMB allows scaling up to 1024 GPUs resulting in training speedups of up to 72x in comparison to [Adam](https://arxiv.org/pdf/1412.6980.pdf). Adam has limitations on the learning rate that can be used since it is applied globally on all parameters whereas LAMB follows a layerwise learning rate strategy.

### Mixed precision training

Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format, while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in the Volta and Turing architecture, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using mixed precision training requires two steps:

1.  Porting the model to use the FP16 data type where appropriate.
2.  Adding loss scaling to preserve small gradient values.

For information about:

-   How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html) documentation.
-   Techniques used for mixed precision training, see the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
-   APEX tools for mixed precision training, see the [NVIDIA Apex: Tools for Easy Mixed-Precision Training in PyTorch](https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/).
 
#### Enabling mixed precision

In this repository, mixed precision training is enabled by NVIDIA’s APEX library. The APEX library has an automatic mixed precision module that allows mixed precision to be enabled with minimal code changes.

Automatic mixed precision can be enabled with the following code changes:

```
from apex import amp
if fp16:
	# Wrap optimizer and model
	model, optimizer = amp.initialize(model, optimizer, opt_level=<opt_level>, loss_scale=”dynamic”)
 
if fp16:
	with amp.scale_loss(loss, optimizer) as scaled_loss:
    	scaled_loss.backward()
   ```

Where `<opt_level>` is the optimization level. In the pretraining, “O2” is set as the optimization level. Mixed precision training can be turned on by passing the `fp16` argument to the pre-training and fine-tuning Python scripts. Shell scripts all have a positional argument available to enable mixed precision training.
 
## Setup

The following section lists the requirements in order to start training the BERT model.

### Requirements

This repository contains Dockerfile which extends the PyTorch NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:

-   [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
-   [PyTorch 19.07-py3 NGC container or later](https://ngc.nvidia.com/registry/nvidia-pytorch)
-   [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) based GPU

For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
-   [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
-   [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/dgx/user-guide/index.html#accessing_registry)
-   [Running PyTorch](https://docs.nvidia.com/deeplearning/dgx/pytorch-release-notes/running.html#running)

For those unable to use the PyTorch NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/dgx/support-matrix/index.html).

## Quick Start Guide

To train your model using mixed precision with Tensor Cores or using FP32, perform the following steps using the default parameters of the BERT model. The default parameters for pretraining have been set to run on 8 x V100 32G cards. For the specifics concerning training and inference, see [Advanced](#advanced).
 

1. Clone the repository.

`git clone https://github.com/NVIDIA/DeepLearningExamples.git`

`cd DeepLearningExamples/PyTorch/LanguageModeling/BERT`


2. Download NVIDIA pretrained checkpoint.

If you want to use a pretrained checkpoint, visit [NGC](https://ngc.nvidia.com/catalog/models) and browse the available models. This downloaded checkpoint is used to fine-tune on SQuAD.

3. Build the BERT 19.07 NGC container.

`bash scripts/docker/build.sh`

4. Start an interactive session in the NGC container to run training/inference.

`bash scripts/docker/launch.sh <DATA_DIR> <VOCAB_DIR> <CHECKPOINT_DIR> <RESULTS_DIR>`

`<DATA_DIR>` - Path to `data` folder in the cloned repository. This directory contains scripts needed to download datasets and where the data will be downloaded.

`<VOCAB_DIR>` - Path to `vocab` folder in the cloned repository. This is the vocabulary with which BERT checkpoint is pretrained.

`<CHECKPOINT_DIR>` - Path to folder containing the downloaded pretrained checkpoint from step 2 for fine-tuning.

`<RESULTS_DIR>` - Path to folder where logs and checkpoints will be saved.

The above paths present on the local machine get mounted to predefined locations in the container.

`data` and `vocab` are a part of `.dockerignore` in order to provide the user the ability to mount datasets of choice and not necessarily the ones downloaded by the script below. In this case, `<DATA_DIR>` points to users corpus. Refer to the [Getting the data](#getting-the-data) section for more details on how to process a custom corpus as required for BERT pretraining.

5. Download and preprocess the dataset.

This repository provides scripts to download, verify and extract the following datasets:

-   [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (fine-tuning for question answering)
-   Wikipedia (pre-training)
-   BookCorpus (pre-training)

To download, verify, extract the datasets, and create the shards in hdf5 format, run:
`/workspace/bert/data/create_datasets_from_start.sh`

6. Start pre-training.

BERT is designed to pre-train deep bidirectional representations for language representations. The following scripts are to replicate pretraining on Wikipedia+Book Corpus from this [paper](https://arxiv.org/pdf/1810.04805.pdf). These scripts are general and can be used for pre-training language representations on any corpus of choice.

From within the container, you can use the following script to run pre-training.
`bash scripts/run_pretraining.sh 14 0.875e-4 fp16 16 0.01 1142857 2000 false true`

More details can be found in Details/Training Process
 
7. Start fine-tuning with the SQUAD dataset.

The above pretrained BERT representations can be fine tuned with just one additional output layer for a state-of-the-art question answering system. Running the following script launches fine-tuning for question answering with the SQuaD dataset.

`bash scripts/run_squad.sh /workspace/checkpoints/<downloaded_checkpoint>`

Default arguments are listed below in order,

-   Initial checkpoint - The default is `/workspace/checkpoints/bert_uncased.pt`.
-   Number of training Epochs - The default is `2`.
-   Batch size - The default is `3`.
-   Learning rate - The default is `3e-5`.
-   Precision (either `fp16` or `fp32`) - The default is `fp16`.
-   Number of GPUs - The default is `8`.
-   Seed - The default is `1`.
-   SQuAD directory -  The default is `/workspace/bert/data/v1.1`.
-   Vocabulary file (token to ID mapping) - The default is `/workspace/bert/vocab/vocab`.
-   Output directory for result - The default is `/results/SQuAD`.
-   Mode (“train”, “eval”, “train eval”, "predict") - The default is `train`.
-   Config file for the bert model (It should be the same as the pretrained model) - The default is `/workspace/bert/bert_config.json`.

The script will save the final checkpoint to the `/results/SQuAD/pytorch_model.bin` file.

9. Start validation/evaluation.

Validation can be performed with the same script as above, setting `Mode` to "prediction".

10. Start inference/predictions.

Inference can be performed with the same script as above, setting `Mode` to `eval`. Inference predictions get saved to `<OUTPUT_DIRECTORY>/predictions.json`.

## Advanced

The following sections provide greater details of the dataset, running training and inference, and the training results.

### Scripts and sample code

Descriptions of the key scripts and folders are provided below.

-   `data/` - Contains scripts for downloading and preparing individual datasets, and will contain downloaded and processed datasets.
-   `scripts/` - Contains shell scripts to launch data download, pre-training, and fine-tuning.
-   `data_download.sh` - Launches download and processing of required datasets.
-   `run_squad.sh`  - Interface for launching question answering fine-tuning with `run_squad.py`.
-   `run_pretraining.sh`  - Interface for launching BERT pre-training with `run_pretraining.py`.
-   `create_pretraining_data.py` - Creates `.hdf5` files from shared text files in the final step of dataset creation.
-   `model.py` - Implements the BERT pre-training and fine-tuning model architectures with PyTorch.
-   `optimization.py` - Implements the LAMB optimizer with PyTorch.
-   `run_squad.py` - Implements fine tuning training and evaluation for question answering on the [SQuaD](https://rajpurkar.github.io/SQuAD-explorer/) dataset.
-   `run_pretraining.py` - Implements BERT pre-training.
-   `run_pretraining_inference.py` - Implements evaluation of a BERT pre-trained model.

### Parameters

#### Pre-training parameters

The complete list of the available parameters for the `run_pretraining.py` script are:

```
  --input_dir INPUT_DIR   	- The input data directory.
                            	Should contain .hdf5 files for the task.

  --config_file CONFIG_FILE   - Path to a json file describing the BERT model
                            	configuration. This file configures the model
                            	architecture, such as the number of transformer
                            	blocks, number of attention heads, etc.

  --bert_model BERT_MODEL 	- Specifies the type of BERT model to use;
                            	should be one of the following:
    	bert-base-uncased
    	bert-large-uncased
    	bert-base-cased
    	bert-base-multilingual
    	bert-base-chinese

  --output_dir OUTPUT_DIR 	- Path to the output directory where the model
                            	checkpoints will be written.

  --max_seq_length MAX_SEQ_LENGTH
                          	- The maximum total input sequence length after
                            	WordPiece tokenization. Sequences longer than
                            	this will be truncated, and sequences shorter
                            	than this will be padded.

  --max_predictions_per_seq MAX_PREDICTIONS_PER_SEQ
                          	- The maximum total of masked tokens per input
                            	sequence for Masked LM.

  --train_batch_size TRAIN_BATCH_SIZE
                          	- Batch size per GPU for training.

  --learning_rate LEARNING_RATE
                          	- The initial learning rate for LAMB optimizer.

  --max_steps MAX_STEPS   	- Total number of training steps to perform.

  --warmup_proportion WARMUP_PROPORTION
                          	- Proportion of training to perform linear learning
                            	rate warmup for. For example, 0.1 = 10% of training.

  --seed SEED             	- Sets the seed to use for random number generation.

  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                          	- Number of update steps to accumulate before
                            	performing a backward/update pass.

  --fp16                  	- If set, will perform computations using
                            	automatic mixed precision.

  --loss_scale LOSS_SCALE 	- Sets the loss scaling value to use when
                            	mixed precision is used. The default value (0)
                            	tells the script to use dynamic loss scaling
                            	instead of fixed loss scaling.

  --log_freq LOG_FREQ     	- If set, the script will output the training
                            	loss every LOG_FREQ steps.
 
  --resume_from_checkpoint	- If set, training will resume from a checkpoint
                            	that currently exists in OUTPUT_DIR.

  --num_steps_per_checkpoint NUM_STEPS_PER_CHECKPOINT
                          	- Number of update steps until a model checkpoint
                            	is saved to disk.`

```
 
#### Fine-tuning parameters

The run_squad.py script contains many of the same arguments as `run_pretraining.py`.
The main script specific parameters are:

```
 --bert_model BERT_MODEL  	- Specifies the type of BERT model to use;
                            	should be one of the following:
    	bert-base-uncased
    	bert-large-uncased
    	bert-base-cased
    	bert-base-multilingual
    	bert-base-chinese

 --train_file TRAIN_FILE  	- Path to the SQuAD json for training.
                            	For example, train-v1.1.json.

 --predict_file PREDICT_FILE  - Path to the SQuAD json for predictions.
                            	For example, dev-v1.1.json or test-v1.1.json.

 --max_seq_length MAX_SEQ_LENGTH
                          	- The maximum total input sequence length
                            	after WordPiece tokenization.
                            	Sequences longer than this will be truncated,
                            	and sequences shorter than this will be padded.

 --doc_stride DOC_STRIDE  	- When splitting up a long document into chunks
                            	this parameters sets how much stride to take
                            	between chunks of tokens.

 --max_query_length MAX_QUERY_LENGTH
                          	- The maximum number of tokens for the question.
                            	Questions longer than <max_query_length>
                            	will be truncated to the value specified.

 --n_best_size N_BEST_SIZE	- The total number of n-best predictions to
                            	generate in the nbest_predictions.json
                            	output file.

 --max_answer_length MAX_ANSWER_LENGTH
                          	- The maximum length of an answer that can be
                            	generated. This is needed because the start and
                            	end predictions are not conditioned on one another.

 --verbose_logging        	- If true, all the warnings related to data
                            	processing will be printed. A number of warnings
                            	are expected for a normal SQuAD evaluation.

 --do_lower_case          	- Whether to lower case the input text. Set to
                            	true for uncased models and false for cased models.

 --version_2_with_negative	- If true, the SQuAD examples contain questions
                            	that do not have an answer.

 --null_score_diff_threshold NULL_SCORE_DIFF_THRES HOLD
                          	- A null answer will be predicted if null_score if
                            	best_non_null is greater than NULL_SCORE_DIFF_THRESHOLD.
```

### Command-line options

To see the full list of available options and their descriptions, use the -h or --help command line option, for example:

`python run_pretraining.py --help`

`python run_squad.py --help`

Detailed descriptions of command line options can be found in the Parameters section above.

### Getting the data

For pre-training BERT, we use the concatenation of Wikipedia (2500M words) as well as Book Corpus (800M words). For Wikipedia, we extract only the text passages and ignore headers, lists, and tables. BERT requires that datasets are structured as a document level corpus rather than a shuffled sentence level corpus because it is critical to extract long contiguous sentences.

The preparation of pre-training dataset is described in the `bertPrep.py` script found in the `data/` folder. The component steps in the automated scripts to prepare the datasets are as follows:

1.  Data download and extract - the dataset is downloaded and extracted.

2.  Clean and format - document tags, etc. are removed from the dataset.

3.  Sentence segmentation - the corpus text file is processed into separate sentences.

4.  Sharding - the sentence segmented corpus file is split into a number of uniformly distributed smaller text documents.

5.  hdf5 file creation - each text file shard is processed by the `create_pretraining_data.py` script to produce a corresponding hdf5 file. The script generates input data and labels for masked language modeling and sentence prediction tasks for the input text shard.

The tools used for preparing the Bookcorpus and Wikipedia datasets can be applied to prepare an arbitrary corpus. The `create_datasets_from_start.sh` script in the `data/` directory applies sentence segmentation, sharding, and hdf5 file creation given an arbitrary text file containing a document-separated text corpus.


For fine-tuning a pre-trained BERT model for specific tasks, by default this repository prepares the following dataset:

-   [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/): for question answering 

#### Dataset guidelines

The procedure to prepare a text corpus for pre-training is described in the above section. This section will provide additional insight into how exactly raw text is processed so that it is ready for pre-training.

First, raw text is tokenized using [WordPiece tokenization](https://arxiv.org/pdf/1609.08144.pdf). A [CLS] token is inserted at the start of every sequence, and the two sentences in the sequence are separated by a [SEP] token.

Note: BERT pre-training looks at pairs of sentences at a time. A sentence embedding token [A] is added to the first sentence and token [B] to the next.

BERT pre-training optimizes for two unsupervised classification tasks. The first is Masked Language Modelling (Masked LM). One training instance of Masked LM is a single modified sentence. Each token in the sentence has a 15% chance of being replaced by a [MASK] token. The chosen token is replaced with [MASK] 80% of the time, 10% with a random token and the remaining 10% the token is retained. The task is then to predict the original token.

The second task is next sentence prediction. One training instance of BERT pre-training is two sentences (a sentence pair). A sentence pair may be constructed by simply taking two adjacent sentences from a single document, or by pairing up two random sentences with equal probability. The goal of this task is to predict whether or not the second sentence followed the first in the original document.

The `create_pretraining_data.py` script takes in raw text and creates training instances for both pre-training tasks.

#### Multi-dataset

This repository provides functionality to combine multiple datasets into a single dataset for pre-training on a diverse text corpus at the shard level in `. /workspace/bert/data/create_datasets_from_start.sh`.

### Training process

The training process consists of two steps: pre-training and fine-tuning.

##### Pre-training

Pre-training is performed using the `run_pretraining.py` script along with parameters defined in the `scripts/run_pretraining.sh`.

The `run_pretraining.sh` script runs a job on a single node that trains the BERT-large model from scratch using the Wikipedia and BookCorpus datasets as training data using LAMB optimizer. By default, the training script runs two phases of training:

Phase 1: (Maximum sequence length of 128)
-   Runs on 8 GPUs with training batch size of 64 per GPU
-   Uses a learning rate of 6e-3
-   Has FP16 precision enabled
-   Runs for 7038 steps, where the first 28.43% (2000) are warm-up steps
-   Saves a checkpoint every 200 iterations (keeps only the latest 3 checkpoints) and at the end of training. All checkpoints, and training logs are saved to the `/results` directory (in the container which can be mounted to a local directory).
-   Creates a log file containing all the output

Phase 2: (Maximum sequence length of 512)
-   Runs on 8 GPUs with training batch size of 8 per GPU
-   Uses a learning rate of 4e-3
-   Has FP16 precision enabled
-   Runs for 1563 steps, where the first 12.8% are warm-up steps
-   Saves a checkpoint every 200 iterations (keeps only the latest 3 checkpoints) and at the end of training. All checkpoints, and training logs are saved to the `/results` directory (in the container which can be mounted to a local directory).
-   Creates a log file containing all the output

These parameters will train on Wikipedia and BooksCorpus to SoTA accuracy on a DGX-1 with 32GB V100 cards.

`bash run_pretraining.sh <training_batch_size> <learning-rate> <precision> <num_gpus> <warmup_proportion> <training_steps> <save_checkpoint_steps> <resume_training> <create_logfile> <accumulate_gradients> <gradient_accumulation_steps> <seed> <job_name> <allreduce_post_accumulation> <allreduce_post_accumulation_fp16> <accumulate_into_fp16> <train_bath_size_phase2> <learning_rate_phase2> <warmup_proportion_phase2> <train_steps_phase2> <gradient_accumulation_steps_phase2> `

Where:

-   `<training_batch_size>` is per-GPU batch size used for training. Larger batch sizes run more efficiently, but require more memory.
-   `<learning_rate>` is the base learning rate for training
-   `<precision>` is the type of math in your model, can be either `fp32` or `fp16`. The options mean:
	-   FP32: 32-bit IEEE single precision floats.
	-   FP16: Mixed precision 16 and 32 bit floats.
-   `<num_gpus>` is the number of GPUs to use for training. Must be equal to or smaller than the number of GPUs attached to your node.
-   `<warmup_proportion>` is the percentage of training steps used for warm-up at the start of training.
-   `<training_steps>` is the total number of training steps.
-   `<save_checkpoint_steps>` controls how often checkpoints are saved.
-   `<resume_training>` if set to true, training should resume from latest model in /results/checkpoints. Default is false.
-   `<create_logfile>` a flag indicating if output should be written to a log file or not (acceptable values are true or false. true indicates output should be saved to a log file.)
-   `<accumulate_gradient>` a flag indicating whether a larger batch should be simulated with gradient accumulation.
-   `<gradient_accumulation_steps>` an integer indicating the number of steps to accumulate gradients over. Effective batch size = `training_batch_size` / `gradient_accumulation_steps`.
-   `<seed>` random seed for the run.
- `<allreduce_post_accumulation>` - If set to `true`, performs allreduce only after the defined number of gradient accumulation steps.
- `<allreduce_post_accumulation_fp16>` -  If set to `true`, performs allreduce after gradient accumulation steps in FP16.
- `<accumulate_into_fp16>` - If set to `true`, accumulates/sums the gradients in FP16.
-  `<training_batch_size_phase2>` is per-GPU batch size used for training in phase 2. Larger batch sizes run more efficiently, but require more memory.
-   `<learning_rate_phase2>` is the base learning rate for training phase 2.
-   `<warmup_proportion_phase2>` is the percentage of training steps used for warm-up at the start of training.
-   `<training_steps_phase2>` is the total number of training steps for phase 2, to be continued in addition to phase 1.
-   `<gradient_accumulation_steps_phase2>` an integer indicating the number of steps to accumulate gradients over in phase 2. Effective batch size = `training_batch_size_phase2` / `gradient_accumulation_steps_phase2`.

For example:

`bash scripts/run_pretraining.sh`

Trains BERT-large from scratch on a single DGX-1 32G using FP16 arithmetic. 90% of the training steps are done with sequence length 128 (phase1 of training) and 10% of the training steps are done with sequence length 512 (phase2 of training).

##### Fine-tuning

Fine-tuning is provided for a variety of tasks. The following tasks are included with this repository through the following scripts:

-   Question Answering (`scripts/run_squad.sh`)

By default, each Python script implements fine-tuning a pre-trained BERT model for a specified number of training epochs as well as evaluation of the fine-tuned model. Each shell script invokes the associated python script with the following default parameters:

-   Uses 8 GPUs
-   Has FP16 precision enabled
-   Saves a checkpoint at the end of training to the `/results/<dataset_name>` folder

Fine-tuning Python scripts implement support for mixed precision and multi-GPU training through NVIDIA’s [Apex](https://github.com/NVIDIA/apex) library. For a full list of parameters and associated explanations, consult the [Parameters](#parameters) section.

All fine-tuning shell scripts have the same positional arguments, outlined below:

`bash scripts/run_squad.sh <checkpoint_to_load> <epochs> <batch_size per GPU> <learning rate> <precision (either `fp16` or `fp32`)> <number of GPUs to use> <seed> <SQUAD_DATA_DIR> <VOCAB_FILE> <OUTPUT_DIR> <mode (either `train`, `eval` or `train eval`)> <CONFIG_FILE>`

By default, the mode positional argument is set to train eval. See the [Quick Start Guide](#quick-start-guide) for explanations of each positional argument.

Note: The first positional argument (the path to the checkpoint to load) is required.

Each fine-tuning script assumes that the corresponding dataset files exist in the `data/` directory or separate path can be a command line input to `run_squad.sh`.

### Inference process

##### Pre-training inference

Inference on a pretrained model is performed using the `run_pretraining_inference.py` script along with parameters defined in `scripts/run_pretraining_inference.sh`. Inference is supported both for single and multi-GPU. By setting either the `--eval` or `--prediction` flag, you can choose between running evaluation on a given dataset or doing prediction (on both masked language model and next sentence prediction).

Prediction mode can be used to measure the inference turnaround time.

The `run_pretraining_inference.sh` script takes a model and a dataset and performs inference/evaluation on it. By default, the inferencing script:

-   Has FP16 precision enabled
-   Runs on 8 GPUs
-   Evaluates the latest checkpoint present in `/results/checkpoints` with a batch size of 14
-   Runs inference on the entire Wikipedia dataset

This script outputs prediction file to `/results/pyt_bert_pretraining_inference_<precision>_<global_batchsize>.<datestamp>.log`. The output log contains information about:

-   Inference performance
-   Loss (masked language model loss and next sentence prediction loss) of the specified dataset if ground truths exist with the `--eval` flag.

For example:

`bash scripts/run_pretraining_inference.sh <evaluation_batch_size> <precision> <num_gpus> <inference_mode><model_checkpoint><inference_steps><create_logfile>`

Where:

-   `<evaluation_batch_size>` is per-GPU batch size used for inference. Larger batch sizes run more efficiently, but require more memory.
-   `<precision>` is the type of math in your model, can be either `fp32` or `fp16`. The options mean:
	-   `fp32`: 32-bit IEEE single precision floats
	-   `fp16`: 16-bit floats for 3.2x faster inference
-   `<num_gpus>` is the number of GPUs to use for inference. Must be equal to or smaller than the number of GPUs attached to your node.
-   `<inference_mode>` is either `--eval` for evaluation or `--prediction` for inference
-   `<model_checkpoint>` is the model checkpoint to run inference on. Default is `-1`, which takes the most recent model checkpoint from the checkpoints folder.
-   `<inference_steps>` is the total number of inference steps per process. Default is `-1`, which iterates over the entire dataset.
-   `<create_logfile>` a flag indicating if output should be written to a logfile or not (acceptable values are true or false. true indicates output should be saved to a logfile.)

For example:

`bash scripts/run_pretraining_inference.sh 14 fp16 8 eval -1 -1 true`

#### Fine-tuning inference

Evaluation fine-tuning is enabled by the same scripts as training:

-   Question Answering (`scripts/run_squad.sh`)

The mode positional argument of the shell script is used to run in evaluation mode. The fine-tuned BERT model will be run on the evaluation dataset, and the evaluation loss and accuracy will be displayed.

Each inference shell script expects dataset files to exist in the same locations as the corresponding training scripts. The inference scripts can be run with default settings. By setting `mode` variable in the script to either `eval` or `prediction` flag, you can choose between running evaluation on a given dataset or doing prediction.

`bash scripts/run_squad.sh <path to fine-tuned model checkpoint>`

Note: Fine-tuning evaluation is only supported on single GPU.

## Performance

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in training and inference modes.

#### Training performance benchmark

Training performance benchmarks for both pretraining and fine-tuning can be obtained by running `scripts/run_pretraining.sh` and `scripts/run_squad.sh` respectively. The required parameters can be passed through the command line as described in [Training process](#training-process).

To benchmark the training performance on a specific batch size, run:

`bash scripts/run_squad.sh <pretrained model path> <epochs> <batch size> <learning rate> <fp16|fp32> <num_gpus> <seed> <path to squad dataset> <path to vocab set> <results directory> train <BERT config path] <max steps>`

An example call used to generate throughput numbers:

`bash scripts/run_squad.sh /workspace/bert/bert_large_uncased_wiki+books.pt.model 2.0 4 3e-5 fp16 8 42 /workspace/bert/squad_data /workspace/bert/scripts/vocab/vocab /results/SQuAD train /workspace/bert/bert_config.json -1`



#### Inference performance benchmark

Inference performance benchmarks for both pretraining and fine-tuning can be obtained by running `scripts/run_pretraining_inference.sh` and `scripts/run_squad.sh` respectively. The required parameters can be passed through the command line as described in [Inference process](#inference-process).

To benchmark the inference performance on a specific batch size, run:

`bash scripts/run_squad.sh <pretrained model path> <epochs> <batch size> <learning rate> <fp16|fp32> <num_gpus> <seed> <path to squad dataset> <path to vocab set> <results directory> eval <BERT config path> <max steps>`

An example call used to generate throughput numbers:

`bash scripts/run_squad.sh /workspace/bert/bert_large_uncased_wiki+books.pt.model 2.0 4 3e-5 fp16 8 42 /workspace/bert/squad_data /workspace/bert/scripts/vocab/vocab /results/SQuAD eval /workspace/bert/bert_config.json -1`



### Results
 

#### Training accuracy results



Our results were obtained by running `scripts/run_squad.sh` and `scripts/run_pretraining.sh` training scripts in the pytorch:19.07-py3 NGC container on NVIDIA DGX-2 with (16x V100 32G) GPUs for pretraining and NVIDIA DGX-1 with (8x V100 16G) GPUs for fine-tuning.

Note: Pretraining results were obtained with a dataset that was created using an earlier version of the data preprocessing scripts than are currently in this repository, and with an an earlier snapshot of wikidumps. The results in the table will be updated soon with results using the latest data prep scripts. Early data show the results are quite similar.


##### Pre-training loss results

| DGX System | GPUs | Accumulated Batch size / GPU (Phase 1 and Phase 2) | Accumulation steps (Phase 1 and Phase 2) | Final Loss - FP32 | Final Loss - mixed precision | Time to train(days) - FP32 | Time to train(days) - mixed precision | Time to train speedup (FP32 to mixed precision)
|-|-|-|-|-|-|-|-|-
| NVIDIA DGX-1 With 16G|8|8192 and 4196 |512 and 1024|-|1.53|-|6.84|- 
| NVIDIA DGX-2 With 32G|16|4096 and 2048 |64 and 256|-|1.52|-|2.71|- 

##### Fine-tuning accuracy results

| GPUs | Batch size / GPU | Accuracy - FP32(% F1) | Accuracy - mixed precision(% F1) | Time to train(hours) - FP32 | Time to train(hours) - mixed precision | Time to train speedup (FP32 to mixed precision)
|-|-|-|-|-|-|-
| 8|4 | 91.18|91.24|.77|.21| 3.66

##### Training stability test

###### Pre-training stability test

| Accuracy Metric | Seed 1
|-|-
| Final Loss | 1.52

###### Fine-tuning stability test

Training stability with 8 GPUs, FP16 computations, batch size of 4:

| Accuracy Metric | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 | Mean | Standard Deviation
|-|-|-|-|-|-|-|-
|Exact Match %| 84.50 | 84.07 | 84.52 | 84.23 | 84.17 | 84.30 | .200
| f1 % | 91.29 | 91.01 | 91.14 |  91.10 | 90.85 | 91.08 | 0.162

#### Training performance results

##### Training performance: NVIDIA DGX-1 (8x V100 16G)

Our results were obtained by running `scripts/run_pretraining.sh` and `scripts/run_squad.shtraining scripts in the pytorch:19.07-py3 NGC container on NVIDIA DGX-1 with (8x V100 16G) GPUs. Performance numbers (in sequences per second) were averaged over an entire training epoch.

###### Pre-training NVIDIA DGX-1 With 16G

| GPUs | Batch size / GPU | Sequence length | Throughput - FP32(sequences/sec) | Throughput - mixed precision(sequences/sec) | Throughput speedup (FP32 - mixed precision) | Weak scaling - FP32 | Weak scaling - mixed precision
|------------------|----------------------|-------------------|-----------------------------------------------|------------------------------------|---------------------------------|----------------------|----------------------------------------------
|1 | 4| 512| 7.56 |26.64 |3.52 |3.50 | 1.00
|4 | 4| 512| 28 |98.24 | 3.50| 3.70| 3.69
| 8| 4| 512| 56.16 |194.56 | 3.46| 7.43| 7.30
|1 | 16| 128| N/A |151.36 |N/A |N/A | 1.00
|4 | 16| 128| N/A |602.88 | N/A| N/A| 3.98
| 8| 16| 128| N/A |1192.96 | N/A| N/A| 7.88


###### Fine-tuning NVIDIA DGX-1 With 16G


| GPUs | Batch size / GPU | Throughput - FP32(sequences/sec) | Throughput - mixed precision(sequences/sec) | Throughput speedup (FP32 - mixed precision) | Weak scaling - FP32 | Weak scaling - mixed precision
|------------------|----------------------|-----------------------------------------------|------------------------------------|---------------------------------|----------------------|----------------------------------------------
|1 | 4|8.96 |35.88 | 3.99| 1.00| 1.00
|4 | 4|31.04 |120.00 | 3.86| 3.46| 3.34
| 8| 4|64.64 |227.84 | 3.52| 7.20| 6.35
|1 | 10|N/A |45.2| N/A| N/A| 1.0
|4 | 10|N/A |163.6 | N/A| N/A| 3.62
| 8| 10|N/A |327.2| N/A| N/A| 7.24


##### Training performance: NVIDIA DGX-1 (8x V100 32G)

Our results were obtained by running `scripts/run_pretraining.sh` and `scripts/run_squad.sh` training scripts in the pytorch:19.07-py3 NGC container on NVIDIA DGX-1 with (8x V100 32G) GPUs. Performance numbers (in sequences per second) were averaged over an entire training epoch.

###### Pre-training NVIDIA DGX-1 With 32G

| GPUs | Batch size / GPU | Sequence length | Throughput - FP32(sequences/sec) | Throughput - mixed precision(sequences/sec) | Throughput speedup (FP32 - mixed precision) | Weak scaling - FP32 | Weak scaling - mixed precision
|------------------|----------------------|-------------------|-----------------------------------------------|------------------------------------|---------------------------------|----------------------|----------------------------------------------
|1 | 8| 512|8.36 |30.08 | 3.68| 1.00| 1.00
|4 | 8| 512|31.52 |116.80 | 3.70| 3.84| 3.82
| 8| 8| 512|62.72 |231.68 | 3.69| 7.68| 7.61
|1 | 10| 512|N/A |46.00| N/A| N/A| 1.0
|4 | 10| 512|N/A |164.00 | N/A| N/A| 3.57
| 8| 10| 512|N/A |325.60| N/A| N/A| 7.08
|1 | 64| 128|N/A |186.32| N/A| N/A| 1.0
|4 | 64| 128|N/A |739.84 | N/A| N/A| 3.97
| 8| 64| 128|N/A |1479.68| N/A| N/A| 7.94

###### Fine-tuning NVIDIA DGX-1 With 32G

| GPUs | Batch size / GPU | Throughput - FP32(sequences/sec) | Throughput - mixed precision(sequences/sec) | Throughput speedup (FP32 - mixed precision) | Weak scaling - FP32 | Weak scaling - mixed precision
|------------------|----------------------|-----------------------------------------------|------------------------------------|---------------------------------|----------------------|----------------------------------------------
|1 | 8|8.64 |36.04 | 4.171| 1.00| 1.00
|4 | 8|31.52 |116.80 | 3.71| 3.64| 3.24
| 8| 8|64.32 |231.04 | 3.59| 7.44| 6.41
|1 | 10|N/A |46.00| N/A| N/A| 1.0
|4 | 10|N/A |164.00 | N/A| N/A| 3.57
| 8| 10|N/A |325.60| N/A| N/A| 7.08


##### Training performance: NVIDIA DGX-2 (16x V100 32G)

Our results were obtained by running `scripts/run_pretraining.sh` and `scripts/run_squad.sh` training scripts in the pytorch:19.07-py3 NGC container on NVIDIA DGX-2 with (16x V100 32G) GPUs. Performance numbers (in sequences per second) were averaged over an entire training epoch.

###### Pre-training NVIDIA DGX-2 With 32G

| GPUs | Batch size / GPU | Sequence length | Throughput - FP32(sequences/sec) | Throughput - mixed precision(sequences/sec) | Throughput speedup (FP32 - mixed precision) | Weak scaling - FP32 | Weak scaling - mixed precision
|------------------|----------------------|-------------------|-----------------------------------------------|------------------------------------|---------------------------------|----------------------|----------------------------------------------
|1 |8 | 512|9.0| 32.32| 3.59| 1.00| 1.00
|4 |8 | 512| 34.4| 124.16| 3.60| 3.82| 3.84
|8 | 8| 512| 68.1671.36| 247.04| 3.62| 7.57| 7.64
|16 | 8| 512| 135.68| 488.96| 3.60| 15.08| 15.13
|1 |10 | 512|N/A | 47.40| N/A| N/A| 1.00
|4 |10 | 512| N/A| 165.60| N/A| N/A| 3.49
|8 | 10| 512| N/A| 325.60| N/A| N/A| 6.87
|16 | 10| 512| N/A| 648.00| N/A| N/A| 13.67
|1 |64 | 128|N/A | 199.05| N/A| N/A| 1.00
|4 |64 | 128| N/A| 796.16| N/A| N/A| 3.99
|8 | 64| 128| N/A| 1592.32| N/A| N/A| 7.99
|16 | 64| 128| N/A| 3174.4| N/A| N/A| 15.94

###### Fine-tuning NVIDIA DGX-2 With 32G

| GPUs | Batch size / GPU | Throughput - FP32(sequences/sec) | Throughput - mixed precision(sequences/sec) | Throughput speedup (FP32 - mixed precision) | Weak scaling - FP32 | Weak scaling - mixed precision
|------------------|----------------------|-----------------------------------------------|------------------------------------|---------------------------------|----------------------|----------------------------------------------
|1 |4 |9.92| 38.16| 3.84| 1.00| 1.00
|4 |4 | 35.52| 122.08| 3.43| 3.58| 3.20
|8 | 4| 71.36| 241.28| 3.38| 7.19| 6.32
|16 | 4| 141.40| 462.08| 3.27| 14.25| 12.11
|1 |10 |N/A | 47.40| N/A| N/A| 1.00
|4 |10 | N/A| 165.60| N/A| N/A| 3.49
|8 | 10| N/A| 325.60| N/A| N/A| 6.87
|16 | 10| N/A| 648.00| N/A| N/A| 13.67

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

#### Inference performance results

##### Inference performance: NVIDIA DGX-1 (1x V100 16G)

Our results were obtained by running `scripts/run_pretraining_inference.sh` on data of sequence length 512 and `scripts/run_squad.sh` scripts in the pytorch:19.07-py3 NGC container on NVIDIA DGX-1 with (1x V100 16G) GPUs.

###### Pre-training inference on NVIDIA DGX-1 with 16G

|GPUs | Throughput - FP32(sequences/sec)|Throughput - Mixed Precision(sequences/sec)
|---------- |---------|---------------
| 1| 28.32| 94.36

###### Fine-tuning inference on NVIDIA DGX-1 with 16G

|GPUs | Throughput - FP32(sequences/sec)|Throughput - Mixed Precision(sequences/sec)
|---------- |---------|---------------
| 1| 37.64| 119.76

##### Inference performance: NVIDIA DGX-1 (1x V100 32G)

Our results were obtained by running `scripts/run_pretraining_inference.sh` and  `scripts/run_squad.sh` scripts in the pytorch:19.07-py3 NGC container on NVIDIA DGX-1 with (1x V100 32G) GPUs.

###### Pre-training inference on NVIDIA DGX-1 with 32G

|GPUs | Throughput(sequences/sec) - FP32|Throughput - Mixed Precision(sequences/sec)
|---------- |---------|---------------
| 1| 27.58| 90.16

###### Fine-tuning inference on NVIDIA DGX-1 with 32G

|GPUs | Throughput(sequences/sec) - FP32|Throughput - Mixed Precision(sequences/sec)
|---------- |---------|---------------
| 1| 37.64| 119.76

##### Inference performance: NVIDIA DGX-2 (1x V100 32G)

Our results were obtained by running `scripts/run_pretraining_inference.sh` and  `scripts/run_squad.sh` scripts in the pytorch:19.07-py3 NGC container on NVIDIA DGX-2 with (1x V100 32G) GPUs.

###### Pre-training inference on NVIDIA DGX-2 with 32G

|GPUs | Throughput - FP32(sequences/sec)|Throughput - Mixed Precision(sequences/sec)
|---------- |---------|---------------
| 1| 30.24 97.72

###### Fine-tuning inference on NVIDIA DGX-2 with 32G

|GPUs | Throughput - FP32(sequences/sec)|Throughput - Mixed Precision(sequences/sec)
|---------- |---------|---------------
| 1| 35.76| 112.60
 
To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

The inference performance metrics used were items/second.

## Release notes

### Changelog

August 2019

- Pretraining support with LAMB optimizer

- Updated Data download and Preprocessing

July 2019

- Initial release

### Known issues

There are no known issues with this model.

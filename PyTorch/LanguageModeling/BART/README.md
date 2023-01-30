# BART For PyTorch

This repository provides a script and recipe to train the BART model to achieve state-of-the-art accuracy and is tested and maintained by NVIDIA.

## Table Of Contents

- [Model overview](#model-overview)
    * [Model architecture](#model-architecture)
    * [Default configuration](#default-configuration)
    * [Feature support matrix](#feature-support-matrix)
	    * [Features](#features)
    * [Mixed precision training](#mixed-precision-training)
	    * [Enabling mixed precision](#enabling-mixed-precision)
          * [TF32](#tf32)
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
    * [Inference process](#inference-process)
- [Performance](#performance)
    * [Benchmarking](#benchmarking)
        * [Training performance benchmark](#training-performance-benchmark)
        * [Inference performance benchmark](#inference-performance-benchmark)
    * [Results](#results)
        * [Training accuracy results](#training-accuracy-results)
            * [Pre-training accuracy: NVIDIA DGX A100 (320x A100 80GB)](#pre-training-accuracy-nvidia-dgx-a100-320x-a100-80gb)
            * [Fine-tuning accuracy: NVIDIA DGX A100 (8x A100 80GB)](#fine-tuning-accuracy-nvidia-dgx-a100-8x-a100-80gb)
            * [Training stability test](#training-stability-test)
        * [Training performance results](#training-performance-results)
            * [Pre-training performance: Single-node on NVIDIA DGX A100 (8x A100 80GB)](#pre-training-performance-single-node-on-nvidia-dgx-a100-8x-a100-80gb)
            * [Pre-training performance: Multi-node on NVIDIA DGX A100 (8x A100 80GB)](#pre-training-performance-multi-node-on-nvidia-dgx-a100-8x-a100-80gb)
            * [Fine-tuning performance: NVIDIA DGX A100 (8x A100 80GB)](#fine-tuning-performance-nvidia-dgx-a100-8x-a100-80gb)
        * [Inference performance results](#inference-performance-results)
            * [Inference performance: NVIDIA DGX A100 (1x A100 80GB)](#inference-performance-nvidia-dgx-a100-1x-a100-80gb)
- [Release notes](#release-notes)
    * [Changelog](#changelog)
    * [Known issues](#known-issues)



## Model overview

BART is a denoising autoencoder for pretraining sequence-to-sequence models. According to the [paper](https://arxiv.org/abs/1910.13461), the model uses a standard seq2seq/machine translation architecture with a bidirectional encoder (like BERT) and a left-to-right decoder (like GPT).

BART is particularly effective when fine tuned for text generation but also works well for comprehension tasks. It matches the performance of RoBERTa with comparable training resources on GLUE and SQuAD, achieves new state-of-the-art results on a range of abstractive dialogue, question answering, and summarization tasks, with gains of up to 6 ROUGE.

Other publicly available implementations of BART include:
1. [Hugging Face](https://huggingface.co/transformers/model_doc/bart.html)
2. [Fairseq](https://github.com/pytorch/fairseq/tree/master/examples/bart)

This model is trained with mixed precision using Tensor Cores on Volta, Turing, and the NVIDIA Ampere GPU architectures. Therefore, researchers can get results 1.4 to 2.1x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.

### Model architecture

BART uses a standard sequence-to-sequence Transformer architecture with GeLU activations. The base model consists of 6 layers in encoder and decoder, whereas large consists of 12. The architecture has roughly 10% more parameters than BERT.


BART is trained by corrupting documents and then optimizing the reconstruction loss. The pretraining task involves randomly shuffling the order of the original sentences and a novel in-filling scheme, where spans of text are replaced with a single mask token.

### Default configuration

BART model is similar to BERT with the following differences:
Decoder layers additionally perform cross-attention over final hidden encoder layer
BART removes the additional feed-forward network before word prediction that BERT uses

Inference is done by default with beam search 4 for CNN-DM dataset and 6 for XSum Dataset.
### Feature support matrix

The following features are supported by this model:

| **Feature** | **BART** |
|:---------:|:----------:|
|  PyTorch AMP   |   Yes    |
|  PyTorch DDP   |   Yes    |
|      LAMB      |   Yes    |
|   Multi-node   |   Yes    |
|      LDDL      |   Yes    |
|     Pre-LN     |   Yes    |

#### Features

[APEX](https://github.com/NVIDIA/apex) is a PyTorch extension with NVIDIA-maintained utilities to streamline mixed precision and distributed training, whereas [AMP](https://nvidia.github.io/apex/amp.html) is an abbreviation used for automatic mixed precision training.

[DDP](https://nvidia.github.io/apex/parallel.html) stands for DistributedDataParallel and is used for multi-GPU training.

[LAMB](https://arxiv.org/pdf/1904.00962.pdf) stands for Layerwise Adaptive Moments based optimizer, is a large batch optimization technique that helps accelerate training of deep neural networks using large minibatches. It allows using a global batch size of 65536 and 32768 on sequence lengths 128 and 512 respectively, compared to a batch size of 256 for [Adam](https://arxiv.org/pdf/1412.6980.pdf). The optimized implementation accumulates 1024 gradient batches in phase 1 and 4096 steps in phase 2 before updating weights once. This results in a 15% training speedup. On multi-node systems, LAMB allows scaling up to 1024 GPUs resulting in training speedups of up to 72x in comparison to Adam. Adam has limitations on the learning rate that can be used since it is applied globally on all parameters whereas LAMB follows a layerwise learning rate strategy.

NVLAMB adds the necessary tweaks to [LAMB version 1](https://arxiv.org/abs/1904.00962v1), to ensure correct convergence. The algorithm is as follows:

  ![NVLAMB](images/nvlamb.png)

In this PyTorch BART example, we used global batch size of 64000 and 30720 on sequence lengths 128 and 512 respectively, compared to a batch size of 8000 and sequence lengths 512 on [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf) which Facebook used for BART. We only trained with 44% total number of tokens compared to Facebook's BART. It can get 2.7x training speedup and achieve similar accuracy.

[LDDL](../lddl) is a library that enables scalable data preprocessing and loading. LDDL is used by this PyTorch BART example.

[Pre-LN](https://arxiv.org/pdf/2002.04745.pdf) is an transformer architecture, which layer normalization is put inside the residual blocks. In our experiments, For Pre-LN transformer, the loss decays faster and it makes training more stable without gradient exploding or vanishing .

### Mixed precision training

Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in Volta, and following with both the Turing and Ampere architectures, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using [mixed precision training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) previously required two steps:
1.  Porting the model to use the FP16 data type where appropriate.
2.  Adding loss scaling to preserve small gradient values.

For information about:
-   How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) documentation.
-   Techniques used for mixed precision training, see the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
-   APEX tools for mixed precision training, see the [NVIDIA Apex: Tools for Easy Mixed-Precision Training in PyTorch](https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/).

#### Enabling mixed precision

In this repository, mixed precision is enabled in PyTorch by using the Automatic Mixed Precision (AMP)
autocast [torch.cuda.amp.autocast](https://pytorch.org/docs/stable/amp.html#autocasting) which casts variables
to half-precision upon retrieval, while storing variables in single-precision format.
Furthermore, to preserve small gradient magnitudes in backpropagation,
a [gradient scaling](https://pytorch.org/docs/stable/amp.html#gradient-scaling)
step must be included.

For an in-depth walk through on AMP, check out sample usage
[here](https://pytorch.org/docs/stable/amp.html).

#### TF32

TensorFloat-32 (TF32) is the new math mode in [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) GPUs for handling the matrix math also called tensor operations. TF32 running on Tensor Cores in A100 GPUs can provide up to 10x speedups compared to single-precision floating-point math (FP32) on Volta GPUs.

TF32 Tensor Cores can speed up networks using FP32, typically with no loss of accuracy. It is more robust than FP16 for models which require high dynamic range for weights or activations.

For more information, refer to the [TensorFloat-32 in the A100 GPU Accelerates AI Training, HPC up to 20x](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) blog post.

TF32 is supported in the NVIDIA Ampere GPU architecture and is enabled by default.



### Glossary

**Fine-tuning**
Training an already pretrained model further using a task specific dataset for subject-specific refinements, by adding task-specific layers on top if required.

**Language Model**
Assigns a probability distribution over a sequence of words. Given a sequence of words, it assigns a probability to the whole sequence.

**Pre-training**
Training a model on vast amounts of data on the same (or different) task to build general understandings.

**Transformer**
The paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) introduces a novel architecture called Transformer that uses an attention mechanism and transforms one sequence into another.

## Setup

The following section lists the requirements that you need to meet in order to start training the BART model.

### Requirements

This repository contains Dockerfile which extends the PyTorch
 NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:
-   [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
-   [PyTorch 22.08-py3+](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) NGC container
-   Supported GPUs:
- [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)

For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
-   [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
-   [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)
-   [Running PyTorch](https://docs.nvidia.com/deeplearning/dgx/pytorch-release-notes/running.html#running)

For those unable to use the PyTorch NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/dgx/support-matrix/index.html).

## Quick Start Guide
To train your model using mixed or TF32 precision with Tensor Cores or using FP32, perform the following steps using the default parameters of the BART model on the CNN-DM/XSum datasets. For the specifics concerning training and inference, see the [Advanced](#advanced) section.

1. Clone the repository.
```bash
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/PyTorch/LanguageModeling/BART
```

2. Build the BART PyTorch container.

```bash
bash scripts/docker/build.sh
```

3. Start an interactive session in the container to run training/inference.

After you build the container image, you can start an interactive CLI session as follows:

```bash
bash scripts/docker/launch.sh
```

The `launch.sh` script, by default, mounts the current directory to `/workspace/bart`.


4. Download and preprocess the dataset.

Use the following script to download and preprocess CNN DM data as well as XSum dataset.

```bash
bash scripts/get_data.sh <path to data folder>
```

Use the script to download Wikipedia, Common Crawl, and OpenWebTextCorpus for pre-training dataset
```bash
bash scripts/get_pretraining_data.sh <path to data folder>
```
The pretraining dataset is 200GB+ and takes 24+ hours to download.

For downloading less dataset, you can change the date period of Common Crawl archive to take less time. For example:
```bash
download_common_crawl \
    --outdir $data_folder/common_crawl \
    --warc-files-start-date 2016-09-01 \
    --warc-files-end-date 2016-10-31 \
    --start-date 2016-09-01 \
    --end-date 2016-10-31
```

Use the script to preprocess the pre-training dataset into LDDL Parquet shards
```bash
bash scripts/preprocess_pretrain_data.sh <path to Wikipedia> <path to Common Crawl> <path to OpenWebTextCorpus> <path to data folder>
```

By default, the path to the data folder is set to /workspace/bart/data for ease of use in all the scripts.

5. Start pre-training

BART is designed to pre-train language representations. The following scripts are to replicate pre-training on Wikipedia, Common Crawl, and OpenWebTextCorpus from the LAMB paper. These scripts are general and can be used for pre-training language representations on any corpus of choice.
From within the container, you can use the following script to run pre-training using LAMB.

```bash
bash scripts/run_pretraining.sh <train_batch_size_phase1> <train_batch_size_phase2> <learning_rate_phase1> <learning_rate_phase2> <precision> <use_preln> <num_gpus> <warmup_steps_phase1> <warmup_steps_phase2> <train_steps_phase1> <train_steps_phase2> <save_checkpoint_steps> <num_accumulation_phase1> <num_accumulation_steps_phase2> <config_path>
```

6. Start summarizing.

Pretrained BART representations can be fine tuned for a state-of-the-art summarization system. From within the container, you can use the following script to run summarization on CNN DM dataset.


```bash
bash scripts/run_summarization.sh <DATA_DIR> <CKPT_PATH> <CONFIG_PATH> <NUM_GPU> <LR> <BS> <ACCUM> <PREC> <TRAIN_STEPS> <WARMUP_STEPS> <MAX_SOURCE_LEN> <MAX_TARGET_LEN> <EVAL_BEAMS> <EVAL_BS> <PRED_BS> <PRELN>
```

This repository contains a number of predefined configurations to run the CNN+DM fine tuning on NVIDIA DGX-1 V100 or NVIDIA DGX A100 nodes in `scripts/params/cnn_dm_params.sh`. For example, to use the default DGX A100 8 gpu config, run:

```bash
bash scripts/run_summarization.sh $(source scripts/params/cnn_dm_params.sh && dgxa100_8gpu_bf16)
```

Similarly, configurations for XSum dataset are available in `scripts/params/xsum_params.sh`.


7. Start inference/predictions.

You can run the following script to run inference summarization using a fine-tuned checkpoint:

```bash
bash scripts/run_eval_summarization.sh <INIT_CKPT> <PRED_BS> <NUM_GPU> <PRECISION> <EVAL_BEAMS> <MAX_SOURCE_LEN> <MAX_TARGET_LEN> <DATA_DIR> <CONFIG_PATH> <PRELN>
```

This repository contains multiple predefined configurations in `scripts/params/cnn_dm_params.sh` and `scripts/params/xsum_params.sh`. For example, to run inference on CNN-DM with a checkpoint run:

```bash
bash scripts/run_eval_summarization.sh <INIT_CKPT> $(source scripts/params/cnn_dm_params.sh && dgxa100_8gpu_bf16_eval)
```

Now that you have your model trained and evaluated, you can choose to compare your training results with our [Training accuracy results](#training-accuracy-results). You can also choose to benchmark yours performance to [Training performance benchmark](#training-performance-results), or [Inference performance benchmark](#inference-performance-results). Following the steps in these sections will ensure that you achieve the same accuracy and performance results as stated in the [Results](#results) section.

8. Run Custom Inference with the fine-tuned checkpoint
We can write a simple few lines of code to run custom inference with the fine-tuned checkpoint.

```python
from bart.modeling.modeling_bart import BartForConditionalGeneration
from bart.tokenization.tokenization_bart import BartTokenizer
from bart.configuration.configuration_bart import BartConfig
import json
config = BartConfig(**json.load(open('configs/config.json', "r")))
config.dtype = None
config.pre_ln = True
model_path = 'results/_epoch1_step2000.ckpt' # The fine-tuned checkpoint path
model = BartForConditionalGeneration.from_pretrained(model_path, config=config)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
ARTICLE_TO_SUMMARIZE = "NVIDIA Geforce Won't Run or Uninstall"
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, truncation=True, return_tensors='pt')
summary_ids = model.generate(inputs['input_ids'],
              num_beams=4,
              max_length=50,
              num_beam_groups=1,
              output_scores=False,
              return_dict_in_generate=False,
              encoder_no_repeat_ngram_size=0,
              diversity_penalty=0.0,
              early_stopping=True)
print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])
```
## Advanced

The following sections provide greater details of the dataset, running training and inference, and the training results.

### Scripts and sample code

In the root directory, the most important files are:
* `pretrain.py` - Serves as entry point for pre-training
* `finetune.py` - Serves as entry point for fine-tuning
* `run_eval.py` - Serves as entry point for inference
* `Dockerfile` - Container with the basic set of dependencies to run BART

The `scripts/` folder encapsulates all the one-click scripts required for running various functionalities supported such as:
* `run_summarization.sh` - Runs summarization finetuning followed by inference using the `finetune.py` and `run_eval.py` files.
* `run_summarization_eval.sh` - Runs inference on fine tuned checkpoint using the `run_eval.py` file.
* `get_data.sh` - Preprocesses CNN-DM dataset as well as downloads and preprocesses XSum dataset.
* `get_pretraining_data.sh` - Downloads pre-train dataset.
* `preprocess_pretrain_data.sh` - Preprocesses pre-train dataset.

Other folders included in the root directory are:
* `data/` - Necessary folder to download datasets required for fine tuning of BART.
* `src/` - Modeling, tokenization and configuration functionality files for implementing the BART model.
* `utils/` - Necessary utility files for BART model.

### Parameters
Aside from the options to set hyperparameters, the relevant options to control the behaviour of the `pretrain.py` script are:

```
--config_path: The configuration file corresponding to BART Model
--warmup_steps: Number of WARMUP_STEPS
--max_steps: Number of MAX_STEPS
--data_dir: Location to DATA_DIR
--learning_rate: Learning Rate
--n_val: Number of validation examples to test for early stopping
--train_batch_size: Train batch size
--gradient_accumulation_steps: Number of accumulation steps
--max_source_length: Maximum source length
--max_target_length: Maximum target length
--val_max_target_length: Maximum length of validation tokens
--eval_max_gen_length: Maximum length while generating validation tokens
--weight_decay: weight decay
--dropout: drop out
--lamb: Whether to use LAMB optimizer
--pre_ln: Whether to use Pre-LN architecture
--allreduce_post_accumulation_half_precision: Whether to do fp16/bf16 allreduce post accumulation
```

Aside from the options to set hyperparameters, the relevant options to control the behaviour of the `finetune.py` script are:

```
--config_path: The configuration file corresponding to BART Model
--warmup_steps: Number of WARMUP_STEPS
--max_steps: Number of MAX_STEPS
--data_dir: Location to DATA_DIR
--learning_rate: Learning Rate
--n_val: Number of validation examples to test for early stopping
--train_batch_size: Train batch size
--gradient_accumulation_steps: Number of accumulation steps
--max_source_length: Maximum source length
--max_target_length: Maximum target length
--val_max_target_length: Maximum length of validation tokens
--eval_max_gen_length: Maximum length while generating validation tokens
--weight_decay: weight decay
--dropout: drop out
--pre_ln: Whether to use Pre-LN architecture
--allreduce_post_accumulation_half_precision: Whether to do fp16/bf16 allreduce post accumulation
```

### Command-line options

To see the full list of available options and their descriptions, use the `-h` or `--help` command-line option with the Python file, for example:

```bash
python pretrain.py --help
python finetune.py --help
python run_eval.py --help
```
### Getting the data

For pre-training BART, we use the concatenation of Wikipedia, Common Crawl, and OpenWebTextCorpus.

Common Crawl is an archieve of news articles from small and major publishers world wide, which is provided from commoncrawl.org.

OpenWebTextCorpus is an open source effort to reproduce OpenAI’s WebText dataset. The distribution was created by Aaron Gokaslan and Vanya Cohen of Brown University.

For fine-tuning BART, we have tested fine tuning the BART model on summarization benchmarks such as CNN-DM and XSum.

CNN-DM is a concatenation of CNN Stories as well as Daily Mail Stories. CNN consists of approximately 90k documents whereas Daily Mail consists of 197k documents.

These documents are preprocessed to have two features:
* Article: text of news article, used as the document to be summarized
* Highlights: joined text of highlights with and around each highlight, which is the target summary

XSum, on the other hand, is also a single-document summarization task dataset but one that favors abstractive modeling. It consists of BBC articles and single sentence summaries. It consists of approximately 230k articles.

#### Dataset guidelines

The repository contains scripts to preprocess and download data. It can be run as:

```bash
bash scripts/get_data.sh <path to output data folder>
```

The script downloads CNN and DM raw data from [here](https://cs.nyu.edu/~kcho/DMQA/). The raw data is preprocessed using scripts from [repository](https://github.com/abisee/cnn-dailymail). The stories are first tokenized, written to serialized binary files and split into train, test and validation sets.

The script also downloads the XSum dataset from the [HuggingFace storage](https://s3.amazonaws.com/datasets.huggingface.co/summarization/xsum.tar.gz).

```bash
bash scripts/get_pretraining_data.sh <path to data folder>
```
The script uses the LDDL downloader to download Wikipedia, Common Crawl, and OpenWebTextCorpus dataset. The Common Crawl is downloaded by [news-please](https://github.com/fhamborg/news-please). And OpenWebTextCorpus is downloaded from [here](https://skylion007.github.io/OpenWebTextCorpus/)

For downloading less dataset, you can change the date period of Common Crawl archive in the script to take less time. For example:
```bash
download_common_crawl \
    --outdir $data_folder/common_crawl \
    --warc-files-start-date 2016-09-01 \
    --warc-files-end-date 2016-10-31 \
    --start-date 2016-09-01 \
    --end-date 2016-10-31
```

```bash
bash scripts/preprocess_pretrain_data.sh <path to Wikipedia> <path to Common Crawl> <path to OpenWebTextCorpus> <path to data folder>
```
The script uses the LDDL preprocessor and load balancer to preprocess the pre-training dataset into Parquet shards which are then streamed during the pre-training by the LDDL data loader.

The script by default stores the data into the `/workspace/bart/data` folder.
### Training process

The training process consists of two steps: pre-training and fine-tuning.

#### Pre-training
Pre-training BART is done using `scripts/run_pretraining.sh` script that, in turn, uses the `pretrain.py` file to perform training.

For example, it can be invoked by calling:

```bash
bash scripts/run_pretraining.sh <train_batch_size_phase1> <train_batch_size_phase2> <learning_rate_phase1> <learning_rate_phase2> <precision> <use_preln> <num_gpus> <warmup_steps_phase1> <warmup_steps_phase2> <train_steps_phase1> <train_steps_phase2> <save_checkpoint_steps> <num_accumulation_phase1> <num_accumulation_steps_phase2> <config_path>
```

Where:
* train_batch_size_phase* - per-GPU batch size used for training in the respective phase
* learning_rate_phase* - Learning rate in the respective phase
* precision - fp16/bf16/fp32/tf32 precision for training
* use_preln - Whether to use Pre-LN architecture
* num_gpus - number of GPUs to run training with
* warmup_steps_phase* - Number of warmup steps for learning rate scheduler in the respective phase
* train_steps_phase* - Number of training steps in the respective phase
* save_checkpoint_steps - Number of steps for saving checkpoint
* num_accumulation_phase* - Number of accumulation steps for an effective larger training batch size in the respective phase
* config_path - path to configuration file of BART Model



By default, the training script stores results to `results/bart_pyt_pretraining` and runs with:

```bash
bash scripts/run_pretraining.sh 200 32 5e-3 4e-3 bf16 true 8 2166 200 95040 7560 100 40 120 configs/config.json
```

#### Fine-tuning
Training BART for summarization is done using `scripts/run_summarization.sh` script that, in turn, uses the `finetune.py` file to perform training.

For example, it can be invoked by calling:

```bash
bash scripts/run_summarization.sh <DATA_DIR> <CKPT_PATH> <CONFIG_PATH> <NUM_GPU> <LR> <BS> <ACCUM> <PREC> <TRAIN_STEPS> <WARMUP_STEPS> <MAX_SOURCE_LEN> <MAX_TARGET_LEN> <EVAL_BEAMS> <EVAL_BS> <PRED_BS> <PRELN>
```

Where:
* DATA_DIR - path to data directory with train/test/val files.
* CONFIG_PATH - path to configuration file of BART Model
* NUM_GPU - number of GPUs to run training with
* LR - Learning rate for training process
* BS - Training batch size
* ACCUM - Number of accumulation steps for an effective larger training batch size
* PREC - fp16/fp32/tf32 precision for training and inference
* TRAIN_STEPS - Maximum number of training steps
* WARMUP_STEPS - Number of warmup steps for learning rate scheduler
* MAX_SOURCE_LEN - Maximum source length of articles
* MAX_TARGET_LEN - Maximum target length of summaries
* EVAL_BEAMS - Number of beams to run during inference
* EVAL_BS - Batch size for inference during validation
* PRED_BS - Batch size for inference on test data
* PRELN - Whether to use Pre-LN architecture



By default, the training script stores results to `results/bart_pyt_${DATESTAMP}` and runs with:

```bash
bash scripts/run_summarization.sh data/cnn_dm/ data/nvidia_pretrained/bart_large/ configs/config.json 8 1.25e-4 40 1 bf16 2000 50 1024 142 4 128 true
```

These parameters train CNN-DM with reasonable rouge scores on a LUNA with 80GB A100 cards. Other tested configurations are available under `scripts/params/cnn_dm_params.sh` for CNN-DM and `scripts/params/xsum_params.sh` for XSum datasets.
### Inference process

Evaluating BART for summarization is done using `scripts/run_eval_summarization.sh` script that, in turn, uses the `run_eval.py` file to perform inference.

For example, it can be invoked by calling:

```bash
bash scripts/run_eval_summarization.sh <INIT_CKPT> <PRED_BS> <NUM_GPU> <PRECISION> <EVAL_BEAMS> <MAX_SOURCE_LEN> <MAX_TARGET_LEN> <DATA_DIR> <CONFIG_PATH> <PRELN>
```

Where:
* `INIT_CKPT` - Model name or path to initialize BART Model weights with.
* `PRED_BS` - Batch size for inference
* `NUM_GPU` - number of GPUs to run training with
* `PRECISION` - FP16/FP32/TF32 precision for training and inference
* `EVAL_BEAMS` - Number of beams to run during inference
* `MAX_SOURCE_LEN` - Maximum source length of articles
* `MAX_TARGET_LEN` - Maximum target length of summaries
* `DATA_DIR` - path to data directory with train/test/val files.
* `CONFIG_PATH` - path to configuration file of BART Model
* `PRELN` - Whether to use Pre-LN architecture

By default, the training script stores results to `results/bart_pyt_inference_${DATESTAMP}` and runs with:

```bash
bash scripts/run_eval_summarization.sh data/nvidia-pretrained/model.ckpt128 8 fp16 4 1024 142 data/cnn_dm/ configs/config.json
```

These parameters run inference on CNN-DM on a DGX A100 with 80GB A100 cards. For XSum, try `EVAL_BEAMS`=6, `MAX_SOURCE_LEN`=1024 and `MAX_TARGET_LEN`=60. For other GPUS/precisions, reduce PRED_BS as indicated in `scripts/params`.
## Performance

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in training and inference modes.

#### Training performance benchmark

To benchmark the training performance on a specific batch size, source length, target length and dataset for one epoch, run:

```bash
bash scripts/run_training_benchmark.sh <batch size> <max source length> <max target length> <data dir>
```

The resulting `NUM_GPU` and PRECISION vs Throughput is stored in `results/bart_pyt_training_benchmark_${DATESTAMP}/inference_benchmark.log`
#### Inference performance benchmark

To benchmark the inference performance on a specific batch size, source length, target length and dataset, run:

```bash
bash scripts/run_inference_benchmark.sh <predict batch size> <eval beams> <max source length> <max target length> <model name or path> <data dir> <config path>
```
The resulting `NUM_GPU` and PRECISION vs Throughput is stored in `results/bart_pyt_inference_benchmark_${DATESTAMP}/inference_benchmark.log`
### Results

The following sections provide details on how we achieved our performance and accuracy in training and inference.

#### Training accuracy results
##### Pre-training accuracy: NVIDIA DGX A100 (320x A100 80GB)
Our results were obtained by running the `run_pretraining.sh` training script in the PyTorch 22.08-py3 NGC container on 40 nodes NVIDIA DGX A100 (320x A100 80GB) GPUs.
| Nodes | Sequence Length | Batch size/GPU (BF16) | Accumulation Steps | Final loss - BF16 | Time to train (hrs) - BF16 |
|-------|-------|---------------------------------------|------------------------------------|----------------------------------|-----------------------------------|
|    40 | 128 | 200 | 1 | 0.5095 | 17.38 |
|    40 | 512 |  32 | 3 | 0.6085 |  3.28 |
##### Fine-tuning accuracy: NVIDIA DGX A100 (8x A100 80GB)

Our results for XSUM dataset were obtained by running the `run_summarization.sh` training script in the PyTorch 22.08-py3 NGC container on NVIDIA DGX A100 (8x A100 80GB) GPUs. Rogue1, rogue2 and rogueLSum scores list as accuracy.

| GPUs | Batch size (TF32, BF16) | R1 - TF32 | R2 - TF32 | RL - TF32 | R1 - BF16 | R2 - BF16 | RL - BF16 | Time to train (hrs) - TF32 | Time to train (hrs) - BF16 | Time to train (hrs) speedup (TF32 to BF16) |
|------|------------------|-----|-----|-----|-----|-----|-----|----------------------|---------------------------------|-------------------------------------------------|
|    1 |           24, 40 | 45.22 | 22.03 | 36.95 | 44.91 | 21.85 | 36.78 | 2.41 | 1.69 | 1.43 |
|    8 |         192, 320 | 45.04 | 21.92 | 36.82 | 45.01 | 21.86 | 36.81 | 0.64 | 0.39 | 1.64 |

In addition,results for CNN-DM dataset are:

| GPUs | Batch size (TF32, BF16) | R1 - TF32 | R2 - TF32 | RL - TF32 | R1 - BF16 | R2 - BF16 | RL - BF16 | Time to train (hrs) - TF32 | Time to train (hrs) - BF16 | Time to train (hrs) speedup (TF32 to BF16) |
|------|------------------|-----|-----|-----|-----|-----|-----|----------------------|---------------------------------|-------------------------------------------------|
|    1 |           24, 40 | 43.76 | 20.79 | 40.51 | 43.58 | 20.63 | 40.32 | 3.87 | 2.42 | 1.60 |
|    8 |         192, 320 | 43.77 | 20.77 | 40.53 | 43.76 | 20.73 | 40.50 | 0.73 | 0.45 | 1.62 |

##### Fine-tuning stability test

Our results for XSUM dataset were obtained by running the `run_summarization.sh` training script in the PyTorch 22.08-py3 NGC container on NVIDIA DGX A100 (8x A100 80GB) GPUs. Accuracy column lists rogue1 scores across 5 different training runs with different seeds on DGX A100.
| **FP16, 8x GPUs** | **seed 1** | **seed 2** | **seed 3** | **seed 4** | **seed 5** | **mean** | **std** |
|:-----------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|rogue1       | 45.08 | 44.98 | 45.10 | 44.91 | 44.95 | 45.00 |


#### Training performance results
##### Pre-training performance: Single-node on NVIDIA DGX A100 (8x A100 80GB)
Our results were obtained by running the `run_pretraining.sh` training script in the PyTorch 22.08-py3 NGC container on single node NVIDIA DGX A100 (8x A100 80GB) GPUs.
| GPUs | Sequence Length | Batch size / GPU (TF32, BF16) | Throughput - TF32 | Throughput - BF16 | Throughput speedup (TF32 - BF16) | Weak scaling - TF32 | Weak scaling - BF16 |
|------|------|------------------|-------------------|------------------------------|---------------------------------------------|---------------------|--------------------------------|
|    1 |       128       | 100, 200 | 202.53 | 326.53 | 1.61 | 1 | 1 |
|    8 |       128       | 100, 200 | 1556.23 | 2572.86 | 1.65 | 7.68 | 7.88 |
|    1 |       512       | 16, 32 | 41.35 | 69.31 | 1.68 | 1 | 1 |
|    8 |       512       | 16, 32 | 317.85 | 549.67 | 1.73 | 7.69 | 7.93 |
##### Pre-training performance: Multi-node on NVIDIA DGX A100 (8x A100 80GB)
Our results were obtained by running the `run_pretraining.sh` training script in the PyTorch 22.08-py3 NGC container on multi node NVIDIA DGX A100 (8x A100 80GB) GPUs.
| Nodes | Sequence Length |Batch size / GPU (TF32, BF16) | Throughput - TF32 | Throughput - BF16 | Throughput speedup (TF32 - BF16) | Weak scaling - TF32 | Weak scaling - BF16 |
|------|------|------------------|-------------------|------------------------------|---------------------------------------------|---------------------|--------------------------------|
|    1 |       128       | 100, 200 |    1556.23 |   2572.86 | 1.65 | 1 | 1 |
|   20 |       128       | 100, 200 |   31067.96 | 52,459.02 | 1.69 | 19.96 | 20.39 |
|   40 |       128       | 100, 200 |  61,538.46 |  97028.51 | 1.58 | 39.54 | 37.71 |
|    1 |       512       | 16, 32 |    317.85 |   549.67 | 1.73 | 1 | 1 |
|   20 |       512       | 16, 32 |   5953.49 | 10520.54 | 1.77 | 18.73 | 19.14 |
|   40 |       512       | 16, 32 | 11,636.36 | 19948.05 | 1.71 | 36.61 | 36.29 |
##### Fine-tuning performance: NVIDIA DGX A100 (8x A100 80GB)

Our results were obtained by running the `run_summarization.sh` training script in the PyTorch 22.08-py3 NGC container on NVIDIA DGX A100 (8x A100 80GB) GPUs. Performance numbers (in items per second) were averaged over an entire training epoch.

| GPUs | Batch size / GPU (TF32, BF16) | Throughput - TF32 | Throughput - BF16 | Throughput speedup (TF32 - BF16) | Weak scaling - TF32 | Weak scaling - BF16 |
|------|------------------|-------------------|------------------------------|---------------------------------------------|---------------------|--------------------------------|
|    1 |               24, 40 |    48.61  |   74.59 |                    1.53 |                1.00 |                           1.00 |
|    8 |               24, 40 |   243.03  |  390.24 |                    1.61 |                3.39 |                           4.08 |



To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

The performance metrics used are tokens per second computed from iterating through an entire epoch of XSum dataset with source length = 1024 and target length = 60.


#### Inference performance results

##### Inference performance: NVIDIA DGX A100 (1x A100 80GB)

Our results were obtained by running the `run_eval_summarization.sh` inferencing benchmarking script in the PyTorch 22.08-py3 NGC container on NVIDIA DGX A100 (1x A100 80GB) GPU.

BF16
| Batch size | Latency Avg | Latency 90% | Latency 95% | Latency 99% | Throughput |
|------------|-------------|:-----------:|:-----------:|:-----------:|------------|
|      1     |        0.28 |        0.35 |        0.38 |        0.46 |       3.54 |
|      4     |        0.44 |        0.52 |        0.56 |        0.71 |       9.16 |
|      8     |        0.63 |        0.75 |        0.83 |        0.98 |      12.79 |
|      16    |        0.98 |        1.2	 |        1.29 |        1.47 |      16.3  |
|      32    |        1.8  |        2.27 |        2.47 |        2.63 |      17.73 |
|      64    |        3.78 |        4.85 |        5.21 |        5.4  |      16.83 |
|      128   |        8.29 |       10.53 |       10.69 |       10.93 |      15.36 |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

The inference performance metrics used are milliseconds per iteration. They are computed by iterating through the XSum test data with source length = 1024, target length = 60 and beam search = 6.

## Release notes

### Changelog

June, 2021
- Initial release

December, 2022
- Add features for pre-training

### Known issues

There are no known issues with this model.

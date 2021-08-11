# BART 1.0 For PyTorch

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
            * [Training accuracy: NVIDIA DGX A100 (8x A100 80GB)](#training-accuracy-nvidia-dgx-a100-8x-a100-80gb)
            * [Training accuracy: NVIDIA DGX-1 V100 (8x V100 32GB)](#training-accuracy-nvidia-dgx-1-v100-8x-v100-32gb)
            * [Training stability test](#training-stability-test)
        * [Training performance results](#training-performance-results)
            * [Training performance: NVIDIA DGX A100 (8x A100 80GB)](#training-performance-nvidia-dgx-a100-8x-a100-80gb)
            * [Training performance: NVIDIA DGX-1 V100 (8x V100 32GB)](#training-performance-nvidia-dgx-1-v100-8x-v100-32gb)
        * [Inference performance results](#inference-performance-results)
            * [Inference performance: NVIDIA DGX A100 (1x A100 80GB)](#inference-performance-nvidia-dgx-a100-1x-a100-80gb)
            * [Inference performance: NVIDIA DGX-1 V100 (1x V100 32GB)](#inference-performance-nvidia-dgx-1-v100-1x-v100-16gb)
            * [Inference performance: NVIDIA T4](#inference-performance-nvidia-t4)
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

| **Feature** | **BERT** |
|:---------:|:----------:|
|APEX AMP|Yes|
|APEX DDP|Yes|

#### Features

[APEX](https://github.com/NVIDIA/apex) is a PyTorch extension with NVIDIA-maintained utilities to streamline mixed precision and distributed training, whereas [AMP](https://nvidia.github.io/apex/amp.html) is an abbreviation used for automatic mixed precision training.

[DDP](https://nvidia.github.io/apex/parallel.html) stands for DistributedDataParallel and is used for multi-GPU training.
### Mixed precision training

Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in Volta, and following with both the Turing and Ampere architectures, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using [mixed precision training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) previously required two steps:
1.  Porting the model to use the FP16 data type where appropriate.
2.  Adding loss scaling to preserve small gradient values.

For information about:
-   How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) documentation.
-   Techniques used for mixed precision training, see the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
-   APEX tools for mixed precision training, see the [NVIDIA Apex: Tools for Easy Mixed-Precision Training in PyTorch](https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/).

#### Enabling mixed precision

In this repository, mixed precision training is enabled by PyTorch Lightning with NVIDIA’s APEX library. The APEX library has an automatic mixed precision module that allows mixed precision to be enabled with minimal code changes.

Automatic mixed precision can be enabled with the following code changes:

```
if args.fp16:
        train_params["precision"] = 16
        train_params["amp_level"] = args.amp_level
```

Where `<amp_level>` is the optimization level. In the summarization, `O1` is set as the optimization level. Mixed precision training can be turned on by passing the `fp16` argument to the `finetune.py`. All shell scripts have a positional argument available to enable mixed precision training.

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
-   [PyTorch 21.02-py3+](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) NGC container
-   Supported GPUs:
- [NVIDIA Volta architecture](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
- [NVIDIA Turing architecture](https://www.nvidia.com/en-us/design-visualization/technologies/turing-architecture/)
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

By default, the path to the data folder is set to /workspace/bart/data for ease of use in all the scripts.

5. Start summarizing.

Pretrained BART representations can be fine tuned for a state-of-the-art summarization system. From within the container, you can use the following script to run summarization on CNN DM dataset.


```bash
bash scripts/run_summarization.sh <DATA_DIR> <CONFIG_PATH> <num_gpu> <LR> <BS> <ACCUM> <PREC> <TRAIN_STEPS> <WARMUP_STEPS> <MAX_SOURCE_LEN> <MAX_TARGET_LEN> <EVAL_BEAMS> <EVAL_BS> <PRED_BS> <VAL_CHECK_INTERVAL> <PATIENCE>
```

This repository contains a number of predefined configurations to run the CNN+DM fine tuning on NVIDIA DGX-1 V100 or NVIDIA DGX A100 nodes in `scripts/params/cnn_dm_params.sh`. For example, to use the default DGX A100 8 gpu config, run:

```bash
bash scripts/run_summarization.sh $(source scripts/params/cnn_dm_params.sh && dgxa100_8gpu_fp16)
```

Similarly, configurations for XSum dataset are available in `scripts/params/xsum_params.sh`.


6. Start inference/predictions.

You can run the following script to run inference summarization using a fine-tuned checkpoint:

```bash
bash scripts/run_eval_summarization.sh <INIT_CKPT> <PRED_BS> <NUM_GPU> <PRECISION> <EVAL_BEAMS> <MAX_SOURCE_LEN> <MAX_TARGET_LEN> <DATA_DIR> <CONFIG_PATH>
```

This repository contains multiple predefined configurations in `scripts/params/cnn_dm_params.sh` and `scripts/params/xsum_params.sh`. For example, to run inference on CNN-DM with a checkpoint run:

```bash
bash scripts/run_eval_summarization.sh <INIT_CKPT> $(source scripts/params/cnn_dm_params.sh && dgxa100_8gpu_fp16_eval)
```

Now that you have your model trained and evaluated, you can choose to compare your training results with our [Training accuracy results](#training-accuracy-results). You can also choose to benchmark yours performance to [Training performance benchmark](#training-performance-results), or [Inference performance benchmark](#inference-performance-results). Following the steps in these sections will ensure that you achieve the same accuracy and performance results as stated in the [Results](#results) section.

7. Run Custom Inference with the fine-tuned checkpoint
We can write a simple few lines of code to run custom inference with the fine-tuned checkpoint.

```python
from bart.modeling.modeling_bart import BartForConditionalGeneration
from bart.tokenization.tokenization_bart import BartTokenizer
from bart.configuration.configuration_bart import BartConfig
import json
config = BartConfig(**json.load(open('configs/config.json', "r")))
config.fp16 = False
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
* `finetune.py` - Serves as entry point for fine-tuning
* `run_eval.py` - Serves as entry point for inference
* `Dockerfile` - Container with the basic set of dependencies to run BART

The `scripts/` folder encapsulates all the one-click scripts required for running various functionalities supported such as:
* `run_summarization.sh` - Runs summarization finetuning followed by inference using the `finetune.py` and `run_eval.py` files.
* `run_summarization_eval.sh` - Runs inference on fine tuned checkpoint using the `run_eval.py` file.
* `get_data.sh` - Preprocesses CNN-DM dataset as well as downloads and preprocesses XSum dataset.

Other folders included in the root directory are:
* `data/` - Necessary folder to download datasets required for fine tuning of BART.
* `src/` - Modeling, tokenization and configuration functionality files for implementing the BART model.
* `utils/` - Necessary utility files for BART model.

### Parameters
Aside from the options to set hyperparameters, the relevant options to control the behaviour of the `run_pretraining.py` script are:

```
--config_path: The configuration file corresponding to BART Model
--warmup_steps: Number of WARMUP_STEPS
--max_steps: Number of MAX_STEPS
--data_dir: Location to DATA_DIR
--gpus: Number of GPUs
--learning_rate: Learning Rate
--n_val: Number of validation examples to test for early stopping
--train_batch_size: Train batch size
--gradient_accumulation_steps: Number of accumulation steps
--val_check_interval: Periodicity of checking validation score
--max_source_length: Maximum source length
--max_target_length: Maximum target length
--val_max_target_length: Maximum length of validation tokens
--eval_max_gen_length: Maximum length while generating validation tokens
--weight_decay: weight decay
--dropout: drop out
--early_stopping_patience: number of validation trials of no improvement before which to trigger early stopping
--amp_level: amp mode of optimization level to use if training with mixed precision
```

### Command-line options

To see the full list of available options and their descriptions, use the `-h` or `--help` command-line option with the Python file, for example:

```bash
python finetune.py --help
python run_eval.py --help
```
### Getting the data


We have tested fine tuning the BART model on summarization benchmarks such as CNN-DM and XSum.

CNN-DM is a concatenation of CNN Stories as well as Daily Mail Stories. CNN consists of approximately 90k documents whereas Daily Mail consists of 197k documents.

These documents are preprocessed to have two features:
* Article: text of news article, used as the document to be summarized
* Highlights: joined text of highlights with and around each highlight, which is the target summary

XSum, on the other hand, is also a single-document summarization task dataset but one that favors abstractive modeling. It consists of BBC articles and single sentence summaries. It consists of approximately 230k articles.

#### Dataset guidelines

The repository contains a script to preprocess and download data. It can be run as:

```bash
bash scripts/get_data.sh <path to output data folder>
```

The script downloads CNN and DM raw data from [here](https://cs.nyu.edu/~kcho/DMQA/). The raw data is preprocessed using scripts from [repository](https://github.com/abisee/cnn-dailymail). The stories are first tokenized, written to serialized binary files and split into train, test and validation sets.

The script also downloads the XSum dataset from the [HuggingFace storage](https://s3.amazonaws.com/datasets.huggingface.co/summarization/xsum.tar.gz).

The script by default stores the data into the `/workspace/bart/data` folder.
### Training process

Training BART for summarization is done using `scripts/run_summarization.sh` script that, in turn, uses the `finetune.py` file to perform training.

For example, it can be invoked by calling:

```bash
Bash scripts/run_summarization.sh <DATA_DIR> <CONFIG_PATH> <num_gpu> <LR> <BS> <ACCUM> <PREC> <TRAIN_STEPS> <WARMUP_STEPS> <MAX_SOURCE_LEN> <MAX_TARGET_LEN> <EVAL_BEAMS> <EVAL_BS> <PRED_BS> <VAL_CHECK_INTERVAL> <PATIENCE>
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
* VAL_CHECK_INTERVAL - Fraction of an epoch after which runs validation if <1. Or number of training samples after which to run validation if >1.
* PATIENCE - Number of validation checks of no improvement after which to stop training



By default, the training script stores results to `results/bart_pyt_${DATESTAMP}` and runs with:

```bash
bash scripts/run_summarization.sh data/cnn_dm/ configs/config.json 8 1e-4 24 1 fp16 20000 500 1024 142 4 128 64 0.3
```

These parameters train CNN-DM with reasonable rouge scores on a LUNA with 80GB A100 cards. Other tested configurations are available under `scripts/params/cnn_dm_params.sh` for CNN-DM and `scripts/params/xsum_params.sh` for XSum datasets.
### Inference process

Evaluating BART for summarization is done using `scripts/run_eval_summarization.sh` script that, in turn, uses the `run_eval.py` file to perform inference.

For example, it can be invoked by calling:

```bash
bash scripts/run_eval_summarization.sh <INIT_CKPT> <PRED_BS> <NUM_GPU> <PRECISION> <EVAL_BEAMS> <MAX_SOURCE_LEN> <MAX_TARGET_LEN> <DATA_DIR> <CONFIG_PATH>
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
##### Training accuracy: NVIDIA DGX A100 (8x A100 80GB)

Our results for XSUM dataset were obtained by running the `run_summarization.sh` training script in the PyTorch 21.02-py3 NGC container on NVIDIA DGX A100 (8x A100 80GB) GPUs. Accuracy column lists rogue1, rogue2 and rogueLSum scores.

| GPUs | Batch size (TF32, mixed precision)       | Accuracy - TF32 | Accuracy - mixed precision | Time to train (hrs) - TF32 | Time to train (hrs) - mixed precision | Time to train (hrs) speedup (TF32 to mixed precision) |
|------|------------------|-----------------|----------------------------|----------------------|---------------------------------|-------------------------------------------------|
|    1 |           24, 40 |           44.41, 21.02, 35.66 | 44.87, 21.49, 36.17 | 3.10 | 2.43 | 1.27 |
|    8 |         192, 320 |           45.34, 21.93, 36.61 | 45.31, 21.83, 36.60 | 0.58 | 0.45 | 1.27 |

In addition,results for CNN-DM dataset are:

| GPUs | Batch size (TF32, mixed precision)       | Accuracy - TF32 | Accuracy - mixed precision | Time to train (hrs) - TF32 | Time to train (hrs) - mixed precision | Time to train (hrs) speedup (TF32 to mixed precision) |
|------|------------------|-----------------|----------------------------|----------------------|---------------------------------|-------------------------------------------------|
|    1 |           24, 40 |           44.37, 21.36, 41.17 | 44.43, 21.43, 41.22 | 4.88 | 3.61 | 1.35 |
|    8 |         192, 320 |           44.49, 21.48, 41.28 | 44.19, 21.26, 40.97 | 0.73 | 0.56 | 1.30 |

##### Training accuracy: NVIDIA DGX-1 V100 (8x V100 32GB)

Our results were obtained by running the `run_summarization.sh` training script in the PyTorch 21.02-py3 NGC container on NVIDIA DGX-2 with (16x V100 32GB) GPUs. Accuracy column lists rogue1, rogue2 and rogueLSum scores.

| GPUs | Batch size (FP32, mixed precision)       | Accuracy - FP32 | Accuracy - mixed precision | Time to train (hrs) - FP32 | Time to train (hrs) - mixed precision | Time to train (hrs) speedup (FP32 to mixed precision) |
|------|------------------|-----------------|----------------------------|----------------------|---------------------------------|-------------------------------------------------|
|    1 |            8, 14 |           44.16, 20.66, 35.24 | 44.86, 21.41, 36.02 |                17.23 |                           6.12 |                                            2.82 |
|    8 |          64, 112 |           45.42, 21.91, 36.62 | 45.58, 22.01, 36.79 |                 2.56 |                           1.09 |                                            2.36 |


In addition,results for CNN-DM dataset are:

| GPUs | Batch size (FP32, mixed precision)       | Accuracy - FP32 | Accuracy - mixed precision | Time to train (hrs) - FP32 | Time to train (hrs) - mixed precision | Time to train (hrs) speedup (FP32 to mixed precision) |
|------|------------------|-----------------|----------------------------|----------------------|---------------------------------|-------------------------------------------------|
|    1 |            8, 14 |           44.49, 21.48, 41.26 | 44.55, 21.47, 41.32 |                26.17 |                           9.74 |                                            2.69 |
|    8 |          64, 112 |           44.34, 21.42, 41.12 | 44.27, 21.30, 41.06 |                 3.58 |                           1.45 |                                            2.46 |


##### Training stability test

Our results for XSUM dataset were obtained by running the `run_summarization.sh` training script in the PyTorch 21.02-py3 NGC container on NVIDIA DGX A100 (8x A100 80GB) GPUs. Accuracy column lists rogue1 scores across 5 different training runs with different seeds on DGX A100.
| **FP16, 8x GPUs** | **seed 1** | **seed 2** | **seed 3** | **seed 4** | **seed 5** | **mean** | **std** |
|:-----------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|rogue1       | 45.34 | 45.34 | 45.21 | 45.33 | 45.34 | 45.31 | 0.055 |


#### Training performance results
##### Training performance: NVIDIA DGX A100 (8x A100 80GB)

Our results were obtained by running the `run_summarization.sh` training script in the PyTorch 21.02-py3 NGC container on NVIDIA DGX A100 (8x A100 80GB) GPUs. Performance numbers (in items/images per second) were averaged over an entire training epoch.

| GPUs | Batch size / GPU (TF32, mixed precision) | Throughput - TF32 | Throughput - mixed precision | Throughput speedup (TF32 - mixed precision) | Weak scaling - TF32 | Weak scaling - mixed precision |
|------|------------------|-------------------|------------------------------|---------------------------------------------|---------------------|--------------------------------|
|    1 |               24, 40 |             31607 |                        42076 |                                        1.33 |                1.00 |                           1.00 |
|    8 |               24, 40 |            163054 |                       217514 |                                        1.33 |                5.16 |                           5.17 |



To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

The performance metrics used are tokens per second computed from iterating through an entire epoch of XSum dataset with source length = 1024 and target length = 60.


##### Training performance: NVIDIA DGX-1 V100 (8x V100 32GB)

Our results were obtained by running the `run_summarization.sh` training script in the PyTorch 21.02-py3 NGC container on NVIDIA DGX-2 with (16x V100 32GB) GPUs. Performance numbers (in items/images per second) were averaged over an entire training epoch.

| GPUs | Batch size / GPU (FP32, mixed precision) | Throughput - FP32 | Throughput - mixed precision | Throughput speedup (FP32 - mixed precision) | Weak scaling - FP32 | Weak scaling - mixed precision |
|------|------------------|-------------------|------------------------------|---------------------------------------------|---------------------|--------------------------------|
|    1 |            8, 14 |           7527 |                      19356 |                                        2.57 |                1.00 |                           1.00 |
|    8 |            8, 14 |          42024 |                     111720 |                                        2.65 |                5.58 |                           5.77 |


To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

The performance metrics used are tokens per second computed from iterating through an entire epoch of XSum dataset with source length = 1024 and target length = 60.

#### Inference performance results

##### Inference performance: NVIDIA DGX A100 (1x A100 80GB)

Our results were obtained by running the `run_eval_summarization.sh` inferencing benchmarking script in the PyTorch 20.11-py3 NGC container on NVIDIA DGX A100 (1x A100 80GB) GPU.

FP16
| Batch size | Latency Avg | Latency 90% | Latency 95% | Latency 99% | Throughput |
|------------|-------------|:-----------:|:-----------:|:-----------:|------------|
|      1     |        0.43 |        0.53 |        0.57 |        0.67 |       2.34 |
|      4     |        0.64 |        0.75 |        0.81 |        0.95 |       6.28 |
|      8     |        0.86 |        1.01 |        1.09 |        1.20 |       9.35 |
|     16     |        1.29 |        1.56 |        1.65 |        1.76 |      12.44 |
|     32     |        2.38 |        3.06 |        3.23 |        3.33 |      13.42 |
|     64     |        4.70 |        6.06 |        6.25 |        6.35 |      13.55 |
|     128    |       10.10 |       12.22 |       12.32 |       12.96 |      12.61 |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

The inference performance metrics used are milliseconds per iteration. They are computed by iterating through the XSum test data with source length = 1024, target length = 60 and beam search = 6.


##### Inference performance: NVIDIA DGX-1 V100 (1x V100 32GB)

Our results were obtained by running the `run_eval_summarization.sh` inferencing benchmarking script in the PyTorch 20.11-py3 NGC container on NVIDIA DGX-2 with (1x V100 32GB) GPU.

FP16
| Batch size | Latency Avg | Latency 90% | Latency 95% | Latency 99% | Throughput |
|------------|-------------|:-----------:|:-----------:|:-----------:|------------|
|      1     |        0.67 |        0.84 |        0.89 |        1.04 |       1.49 |
|      4     |        0.96 |        1.14 |        1.24 |        1.43 |       4.16 |
|      8     |        1.33 |        1.59 |        1.72 |        1.90 |       6.01 |
|     16     |        1.99 |        2.39 |        2.57 |        2.69 |       8.04 |
|     32     |        3.41 |        4.31 |        4.53 |        4.63 |       9.36 |
|     64     |        6.66 |        8.61 |        8.75 |        8.92 |       9.55 |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

The inference performance metrics used are milliseconds per iteration. They are computed by iterating through the XSum test data with source length = 1024, target length = 60 and beam search = 6.

##### Inference performance: NVIDIA T4

Our results were obtained by running the `run_eval_summarization.sh` inferencing benchmarking script in the PyTorch 21.02-py3 NGC container on NVIDIA T4 with GPU.

FP16
| Batch size | Latency Avg | Latency 90% | Latency 95% | Latency 99% | Throughput |
|------------|-------------|:-----------:|:-----------:|:-----------:|------------|
|      1     |        0.42 |        0.52 |        0.56 |        0.66 |       2.40 |
|      4     |        0.72 |        0.89 |        0.96 |        1.09 |       5.58 |
|      8     |        1.13 |        1.60 |        1.73 |        1.96 |       7.08 |
|     16     |        2.25 |        3.19 |        3.38 |        3.58 |       7.11 |
|     32     |        4.44 |        6.53 |        6.96 |        7.21 |       7.19 |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

The inference performance metrics used are milliseconds per iteration. They are computed by iterating through the XSum test data with source length = 1024, target length = 60 and beam search = 6.

## Release notes

### Changelog

June, 2021
- Initial release

### Known issues

There are no known issues with this model.

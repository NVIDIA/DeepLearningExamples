# BERT For PyTorch
 
This repository provides a script and recipe to train the BERT model for PyTorch to achieve state-of-the-art accuracy and is tested and maintained by NVIDIA.
 
## Table Of Contents
 
- [Model overview](#model-overview)
    * [Model architecture](#model-architecture)
    * [Default configuration](#default-configuration)
    * [Feature support matrix](#feature-support-matrix)
        * [Features](#features)
    * [Mixed precision training](#mixed-precision-training)
        * [Enabling mixed precision](#enabling-mixed-precision)
        * [Enabling TF32](#enabling-tf32)
    * [Glossary](#glossary)
- [Setup](#setup)
    * [Requirements](#requirements)
- [Quick Start Guide](#quick-start-guide)
- [Advanced](#advanced)
    * [Scripts and sample code](#scripts-and-sample-code)
    * [Parameters](#parameters)
        * [Pre-training parameters](#pre-training-parameters)
        * [Fine tuning parameters](#fine-tuning-parameters)    
        * [Multi-node](#multi-node)
    * [Command-line options](#command-line-options)
    * [Getting the data](#getting-the-data)
        * [Dataset guidelines](#dataset-guidelines)
    * [Training process](#training-process)
        * [Pre-training](#pre-training)
        * [Fine-tuning](#fine-tuning)   
    * [Inference process](#inference-process)
        * [Fine-tuning inference](#fine-tuning-inference)
    * [Deploying BERT using NVIDIA Triton Inference Server](#deploying-bert-using-nvidia-triton-inference-server)
- [Performance](#performance)
    * [Benchmarking](#benchmarking)
        * [Training performance benchmark](#training-performance-benchmark)
        * [Inference performance benchmark](#inference-performance-benchmark)
    * [Results](#results)
        * [Training accuracy results](#training-accuracy-results)
            * [Pre-training loss results: NVIDIA DGX A100 (8x A100 80GB)](#pre-training-loss-results-nvidia-dgx-a100-8x-a100-80gb)
            * [Pre-training loss curves](#pre-training-loss-curves)
            * [Fine-tuning accuracy results: NVIDIA DGX A100 (8x A100 80GB)](#fine-tuning-accuracy-results-nvidia-dgx-a100-8x-a100-80gb)
            * [Training stability test](#training-stability-test)
                * [Pre-training stability test](#pre-training-stability-test)
                * [Fine-tuning stability test](#fine-tuning-stability-test) 
        * [Training performance results](#training-performance-results)
            * [Training performance: NVIDIA DGX A100 (8x A100 80GB)](#training-performance-nvidia-dgx-a100-8x-a100-80gb)
                * [Pre-training NVIDIA DGX A100 (8x A100 80GB)](#pre-training-nvidia-dgx-a100-8x-a100-80gb)
                * [Pre-training NVIDIA DGX A100 (8x A100 80GB) Multi-node Scaling](#pre-training-nvidia-dgx-a100-8x-a100-80gb-multi-node-scaling)
                * [Fine-tuning NVIDIA DGX A100 (8x A100 80GB)](#fine-tuning-nvidia-dgx-a100-8x-a100-80gb)
            * [Training performance: NVIDIA DGX-1 (8x V100 32G)](#training-performance-nvidia-dgx-1-8x-v100-32g)
                * [Pre-training NVIDIA DGX-1 With 32G](#pre-training-nvidia-dgx-1-with-32g)
                * [Fine-tuning NVIDIA DGX-1 With 32G](#fine-tuning-nvidia-dgx-1-with-32g)   
        * [Inference performance results](#inference-performance-results)
            * [Inference performance: NVIDIA DGX A100 (1x A100 80GB)](#inference-performance-nvidia-dgx-a100-1x-a100-80gb)
                * [Fine-tuning inference on NVIDIA DGX A100 (1x A100 80GB)](#fine-tuning-inference-on-nvidia-dgx-a100-1x-a100-80gb)
- [Release notes](#release-notes)
    * [Changelog](#changelog)
    * [Known issues](#known-issues)
 
 
 
## Model overview
 
BERT, or Bidirectional Encoder Representations from Transformers, is a new method of pre-training language representations that obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks. This model is based on the [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) paper. NVIDIA's implementation of BERT is an optimized version of the [Hugging Face implementation](https://github.com/huggingface/pytorch-pretrained-BERT), leveraging mixed precision arithmetic and Tensor Cores on NVIDIA Volta V100 and NVIDIA Ampere A100 GPUs for faster training times while maintaining target accuracy.
 
This repository contains scripts to interactively launch data download, training, benchmarking, and inference routines in a Docker container for both pre-training and fine-tuning tasks such as question answering. The major differences between the original implementation of the paper and this version of BERT are as follows:
 
-   Scripts to download the Wikipedia dataset
-   Scripts to preprocess downloaded data into inputs and targets for pre-training in a modular fashion
-   Fused [LAMB](https://arxiv.org/pdf/1904.00962.pdf) optimizer to support training with larger batches
-   Fused Adam optimizer for fine-tuning tasks
-   Fused CUDA kernels for better performance LayerNorm
-   Automatic mixed precision (AMP) training support
-   Scripts to launch on multiple number of nodes
 
Other publicly available implementations of BERT include:
1. [NVIDIA TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT)
2. [Hugging Face](https://github.com/huggingface/pytorch-pretrained-BERT)
3. [codertimo](https://github.com/codertimo/BERT-pytorch)
4. [gluon-nlp](https://github.com/dmlc/gluon-nlp/tree/v0.10.x/scripts/bert)
5. [Google's implementation](https://github.com/google-research/bert)
    
This model trains with mixed precision Tensor Cores on NVIDIA Volta and provides a push-button solution to pre-training on a corpus of choice. As a result, researchers can get results 4x faster than training without Tensor Cores. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.
 
### Model architecture
 
The BERT model uses the same architecture as the encoder of the Transformer. Input sequences are projected into an embedding space before being fed into the encoder structure. Additionally, positional and segment encodings are added to the embeddings to preserve positional information. The encoder structure is simply a stack of Transformer blocks, which consist of a multi-head attention layer followed by successive stages of feed-forward networks and layer normalization. The multi-head attention layer accomplishes self-attention on multiple input representations.
 
An illustration of the architecture taken from the [Transformer paper](https://arxiv.org/pdf/1706.03762.pdf) is shown below.
 
 ![BERT](images/model.png)
 
### Default configuration
 
The architecture of the BERT model is almost identical to the Transformer model that was first introduced in the [Attention Is All You Need paper](https://arxiv.org/pdf/1706.03762.pdf). The main innovation of BERT lies in the pre-training step, where the model is trained on two unsupervised prediction tasks using a large text corpus. Training on these unsupervised tasks produces a generic language model, which can then be quickly fine-tuned to achieve state-of-the-art performance on language processing tasks such as question answering.
 
The BERT paper reports the results for two configurations of BERT, each corresponding to a unique model size. This implementation provides the same configurations by default, which are described in the table below.  
 
| **Model** | **Hidden layers** | **Hidden unit size** | **Attention heads** | **Feedforward filter size** | **Max sequence length** | **Parameters** |
|:---------:|:-----------------:|:--------------------:|:-------------------:|:---------------------------:|:-----------------------:|:--------------:|
| BERTBASE  |    12 encoder     |         768          |         12          |          4 x  768           |           512           |      110M      |
| BERTLARGE |    24 encoder     |         1024         |         16          |          4 x 1024           |           512           |      330M      |
 
 
 
### Feature support matrix
 
The following features are supported by this model.  
 
| **Feature** | **BERT** |
|:-----------:|:--------:|
|  PyTorch AMP   |   Yes    |
|  PyTorch DDP   |   Yes    |
|      LAMB      |   Yes    |
|   Multi-node   |   Yes    |
|      LDDL      |   Yes    |
|     NVFuser    |   Yes    |
 
#### Features
 
[APEX](https://github.com/NVIDIA/apex) is a PyTorch extension with NVIDIA-maintained utilities to streamline mixed precision and distributed training, whereas [AMP](https://nvidia.github.io/apex/amp.html) is an abbreviation used for automatic mixed precision training.
 
[DDP](https://nvidia.github.io/apex/parallel.html) stands for DistributedDataParallel and is used for multi-GPU training.
 
[LAMB](https://arxiv.org/pdf/1904.00962.pdf) stands for Layerwise Adaptive Moments based optimizer, is a large batch optimization technique that helps accelerate training of deep neural networks using large minibatches. It allows using a global batch size of 65536 and 32768 on sequence lengths 128 and 512 respectively, compared to a batch size of 256 for [Adam](https://arxiv.org/pdf/1412.6980.pdf). The optimized implementation accumulates 1024 gradient batches in phase 1 and 4096 steps in phase 2 before updating weights once. This results in a 15% training speedup. On multi-node systems, LAMB allows scaling up to 1024 GPUs resulting in training speedups of up to 72x in comparison to Adam. Adam has limitations on the learning rate that can be used since it is applied globally on all parameters whereas LAMB follows a layerwise learning rate strategy.
 
NVLAMB adds the necessary tweaks to [LAMB version 1](https://arxiv.org/abs/1904.00962v1), to ensure correct convergence. The algorithm is as follows:
 
  ![NVLAMB](images/nvlamb.png)

[LDDL](../../../Tools/lddl) is a library that enables scalable data preprocessing and loading. LDDL is used by this PyTorch BERT example.
 
NVFuser is NVIDIA's fusion backend for PyTorch.

### Mixed precision training
 
Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [tensor cores](https://developer.nvidia.com/tensor-cores) in the NVIDIA Volta, and following with both the NVIDIA Turing and NVIDIA Ampere architectures, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using mixed precision training requires two steps:
 
1.  Porting the model to use the FP16 data type where appropriate.
2.  Adding loss scaling to preserve small gradient values.
 
For information about:
-   How to train using mixed precision, refer to the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) documentation.
-   Techniques used for mixed precision training, refer to the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
-   APEX tools for mixed precision training, refer to the [NVIDIA APEX: Tools for Easy Mixed-Precision Training in PyTorch](https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/).
 
#### Enabling mixed precision
 
In this repository, mixed precision training is enabled by NVIDIA’s APEX library. The APEX library has an automatic mixed precision module that allows mixed precision to be enabled with minimal code changes.
 
Automatic mixed precision can be enabled with the following code changes:
 
```
from apex import amp
if fp16:
    # Wrap optimizer and model
    model, optimizer = amp.initialize(model, optimizer, opt_level=<opt_level>, loss_scale="dynamic")
 
if fp16:
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
   ```
 
Where `<opt_level>` is the optimization level. In the pre-training, `O2` is set as the optimization level. Mixed precision training can be turned on by passing the `fp16` argument to the `run_pretraining.py` and `run_squad.py`. All shell scripts have a positional argument available to enable mixed precision training.

#### Enabling TF32

TensorFloat-32 (TF32) is the new math mode in [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) GPUs for handling the matrix math, also called tensor operations. TF32 running on Tensor Cores in A100 GPUs can provide up to 10x speedups compared to single-precision floating-point math (FP32) on NVIDIA Volta GPUs. 

TF32 Tensor Cores can speed up networks using FP32, typically with no loss of accuracy. It is more robust than FP16 for models which require a high dynamic range for weights or activations.

For more information, refer to the [TensorFloat-32 in the A100 GPU Accelerates AI Training, HPC up to 20x](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) blog post.

TF32 is supported in the NVIDIA Ampere GPU architecture and is enabled by default.

### Glossary
 
**Fine-tuning**  
Training an already pre-trained model further using a task-specific dataset for subject-specific refinements by adding task-specific layers on top if required.
 
**Language Model**  
Assigns a probability distribution over a sequence of words. Given a sequence of words, it assigns a probability to the whole sequence.
 
**Pre-training**  
Training a model on vast amounts of data on the same (or different) task to build general understandings.
 
**Transformer**  
The paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) introduces a novel architecture called Transformer that uses an attention mechanism and transforms one sequence into another.
 
**Phase 1**  
Pre-training on samples of sequence length 128 and 20 masked predictions per sequence.
 
**Phase 2**  
Pre-training on samples of sequence length 512 and 80 masked predictions per sequence.
 
## Setup
 
The following section lists the requirements that you need to meet in order to start training the BERT model. 
 
### Requirements
 
This repository contains a Dockerfile that extends the PyTorch NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:
 
-   [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
-   [PyTorch 21.11-py3 NGC container or later](https://ngc.nvidia.com/registry/nvidia-pytorch)
-   Supported GPUs:
- [NVIDIA Volta architecture](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
- [NVIDIA Turing architecture](https://www.nvidia.com/en-us/geforce/turing/)
- [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)
 
For more information about how to get started with NGC containers, refer to the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
-   [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
-   [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/dgx/user-guide/index.html#accessing_registry)
-   [Running PyTorch](https://docs.nvidia.com/deeplearning/dgx/pytorch-release-notes/running.html#running)
 
For those unable to use the PyTorch NGC container, to set up the required environment or create your own container, refer to the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/dgx/support-matrix/index.html).
 
For multi-node, the sample provided in this repository requires [Enroot](https://github.com/NVIDIA/enroot) and [Pyxis](https://github.com/NVIDIA/pyxis) set up on a [SLURM](https://slurm.schedmd.com) cluster.
 
More information on how to set up and launch can be found in the [Multi-node Documentation](https://docs.nvidia.com/ngc/multi-node-bert-user-guide).
 
## Quick Start Guide
 
To train your model using mixed or TF32 precision with Tensor Cores or using FP32, perform the following steps using the default parameters of the BERT model. Training configurations to run on 8 x A100 80G, 8 x V100 16G, 16 x V100 32G cards and examples of usage are provided at the end of this section. For the specifics concerning training and inference, refer to the [Advanced](#advanced) section.
 
 
1. Clone the repository.
```
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/PyTorch/LanguageModeling/BERT
```
 
2. Download the NVIDIA pre-trained checkpoint.
 
If you want to use a pre-trained checkpoint, visit [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/models/bert_pyt_ckpt_large_pretraining_amp_lamb/files?version=20.03.0). This pre-trained checkpoint is used to fine-tune on SQuAD. Ensure you unzip the downloaded file and place the checkpoint in the `checkpoints/` folder. For a checkpoint already fine-tuned for QA on SQuAD v1.1 visit [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/models/bert_pyt_ckpt_large_qa_squad11_amp/files).

Find all trained and available checkpoints in the table below:
| Model                  | Description                                                               |
|------------------------|---------------------------------------------------------------------------|
| [bert-large-uncased-qa](https://catalog.ngc.nvidia.com/orgs/nvidia/models/bert_pyt_ckpt_large_qa_squad11_amp/files) | Large model fine-tuned on SQuAD v1.1                                       |
| [bert-large-uncased-sst2](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/bert_pyt_ckpt_large_ft_sst2_amp) |Large model fine-tuned on GLUE SST-2                                      |
| [bert-large-uncased-pretrained](https://catalog.ngc.nvidia.com/orgs/nvidia/models/bert_pyt_ckpt_large_pretraining_amp_lamb/files?version=20.03.0) | Large model pretrained checkpoint on Generic corpora like Wikipedia|
| [bert-base-uncased-qa](https://catalog.ngc.nvidia.com/orgs/nvidia/models/bert_pyt_ckpt_base_qa_squad11_amp/files) | Base model fine-tuned on SQuAD v1.1                                         |
| [bert-base-uncased-sst2](https://catalog.ngc.nvidia.com/orgs/nvidia/models/bert_pyt_ckpt_base_ft_sst2_amp_128/files) | Base model fine-tuned on GLUE SST-2                                       |
| [bert-base-uncased-pretrained](https://catalog.ngc.nvidia.com/orgs/nvidia/models/bert_pyt_ckpt_base_pretraining_amp_lamb/files) | Base model pretrained checkpoint on Generic corpora like Wikipedia. |
| [bert-dist-4L-288D-uncased-qa](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/bert_pyt_ckpt_distilled_4l_288d_qa_squad11_amp/files) | 4 layer distilled model fine-tuned on SQuAD v1.1                                         |
| [bert-dist-4L-288D-uncased-sst2](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/bert_pyt_ckpt_distilled_4l_288d_ft_sst2_amp/files) | 4 layer distilled model fine-tuned on GLUE SST-2                                       |
| [bert-dist-4L-288D-uncased-pretrained](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/bert_pyt_ckpt_distilled_4l_288d_pretraining_amp/files) | 4 layer distilled model pretrained checkpoint on Generic corpora like Wikipedia. |
| [bert-dist-6L-768D-uncased-qa](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/bert_pyt_ckpt_distilled_6l_768d_qa_squad11_amp/files) | 6 layer distilled model fine-tuned on SQuAD v1.1                                         |
| [bert-dist-6L-768D-uncased-sst2](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/bert_pyt_ckpt_distilled_6l_768d_ft_sst2_amp/files) | 6 layer distilled model fine-tuned on GLUE SST-2                                       |
| [bert-dist-6L-768D-uncased-pretrained](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/bert_pyt_ckpt_distilled_6l_768d_pretraining_amp/files) | 6 layer distilled model pretrained checkpoint on Generic corpora like Wikipedia. |




3. Build BERT on top of the  NGC container.
```
bash scripts/docker/build.sh
```
 
4. Start an interactive session in the NGC container to run training/inference.
```
bash scripts/docker/launch.sh
```
 
Resultant logs and checkpoints of pre-training and fine-tuning routines are stored in the `results/` folder.
 
`data` and `vocab.txt` are downloaded in the `data/` directory by default. Refer to the [Getting the data](#getting-the-data) section for more details on how to process a custom corpus as required for BERT pre-training. 

5. Download the dataset.

This repository provides scripts to download, verify, and extract the following datasets:
 
- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (fine-tuning for question answering)
- [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398) (fine-tuning for paraphrase detection)
- [SST-2](https://nlp.stanford.edu/sentiment/index.html) (fine-tuning for sentiment analysis)
- Wikipedia (pre-training)
 
To download, verify, extract the datasets, run:  
```
/workspace/bert/data/create_datasets_from_start.sh
```

Note: For fine-tuning only, downloading the Wikipedia dataset can be skipped by commenting it out.

Note: Ensure a complete Wikipedia download. But if the download failed in LDDL,
remove the output directory `data/wikipedia/` and start over again. 

6. Start pre-training.
 
To run on a single node 8 x V100 32G cards, from within the container, you can use the following script to run pre-training.  
```
bash scripts/run_pretraining.sh
```
 
The default hyperparameters are set to run on 8x V100 32G cards.
 
To run on multiple nodes, refer to the [Multi-node](#multi-node) section.  
 
7. Start fine-tuning with the SQuAD dataset.
 
The above pre-trained BERT representations can be fine-tuned with just one additional output layer for a state-of-the-art question answering system. Running the following script launches fine-tuning for question answering with the SQuAD dataset.
```
bash scripts/run_squad.sh /workspace/bert/checkpoints/<pre-trained_checkpoint>
```
  
8. Start fine-tuning with the GLUE tasks.
 
The above pre-trained BERT representations can be fine-tuned with just one additional output layer for GLUE tasks. Running the following scripts launch fine-tuning for paraphrase detection with the MRPC dataset:
```
bash scripts/run_glue.sh /workspace/bert/checkpoints/<pre-trained_checkpoint>
```

9. Run Knowledge Distillation (Optional).
 
To get setup to run distillation on BERT, follow steps provided [here](./distillation/README.md).

10. Start validation/evaluation.
 
For both SQuAD and GLUE, validation can be performed with the `bash scripts/run_squad.sh /workspace/bert/checkpoints/<pre-trained_checkpoint>` or `bash scripts/run_glue.sh /workspace/bert/checkpoints/<pre-trained_checkpoint>`, setting `mode` to `eval` in `scripts/run_squad.sh` or `scripts/run_glue.sh` as follows:

```
mode=${11:-"eval"}
```
 
11. Start inference/predictions.
 
Inference can be performed with the `bash scripts/run_squad.sh /workspace/bert/checkpoints/<pre-trained_checkpoint>`, setting `mode` to `prediction` in `scripts/run_squad.sh` or `scripts/run_glue.sh` as follows:

```
mode=${11:-"prediction"}
```

Inference predictions are saved to `<OUT_DIR>/predictions.json`, set in `scripts/run_squad.sh` or `scripts/run_glue.sh` as follows:

```
OUT_DIR=${10:-"/workspace/bert/results/SQuAD"} # For SQuAD.
# Or…
out_dir=${5:-"/workspace/bert/results/MRPC"} # For MRPC.
# Or... 
out_dir=${5:-"/workspace/bert/results/SST-2"} # For SST-2.
```

This repository contains a number of predefined configurations to run the SQuAD, GLUE and pre-training on NVIDIA DGX-1, NVIDIA DGX-2H or NVIDIA DGX A100 nodes in `scripts/configs/squad_config.sh`, `scripts/configs/glue_config.sh` and `scripts/configs/pretrain_config.sh`. For example, to use the default DGX A100 8 gpu config, run:

```
bash scripts/run_squad.sh $(source scripts/configs/squad_config.sh && dgxa100-80g_8gpu_fp16)  # For the SQuAD v1.1 dataset.
bash scripts/run_glue.sh $(source scripts/configs/glue_config.sh && mrpc_dgxa100-80g_8gpu_fp16)  # For the MRPC dataset.
bash scripts/run_glue.sh $(source scripts/configs/glue_config.sh && sst-2_dgxa100-80g_8gpu_fp16)  # For the SST-2 dataset.
bash scripts/run_pretraining.sh $(source scripts/configs/pretrain_config.sh && dgxa100-80g_8gpu_fp16) # For pre-training
```

## Advanced
 
The following sections provide greater details of the dataset, running training and inference, and the training results.
 
### Scripts and sample code
 
Descriptions of the key scripts and folders are provided below.
 
-   `data/` - Contains scripts for downloading and preparing individual datasets and will contain downloaded and processed datasets.
-   `scripts/` - Contains shell scripts to launch data download, pre-training, and fine-tuning.
-   `run_squad.sh`  - Interface for launching question answering fine-tuning with `run_squad.py`.
-   `run_glue.sh`  - Interface for launching paraphrase detection and sentiment analysis fine-tuning with `run_glue.py`.
-   `run_pretraining.sh`  - Interface for launching BERT pre-training with `run_pretraining.py`.
-   `create_pretraining_data.py` - Creates `.hdf5` files from shared text files in the final step of dataset creation.
-   `model.py` - Implements the BERT pre-training and fine-tuning model architectures with PyTorch.
-   `optimization.py` - Implements the LAMB optimizer with PyTorch.
-   `run_squad.py` - Implements fine-tuning training and evaluation for question answering on the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset.
-   `run_glue.py` - Implements fine-tuning training and evaluation for [GLUE](https://gluebenchmark.com/) tasks.
-   `run_pretraining.py` - Implements BERT pre-training.
 
### Parameters
 
#### Pre-training parameters
 
BERT is designed to pre-train deep bidirectional networks for language representations. The following scripts replicate pre-training on Wikipedia from this [paper](https://arxiv.org/pdf/1810.04805.pdf). These scripts are general and can be used for pre-training language representations on any corpus of choice.
 
The complete list of the available parameters for the `run_pretraining.py` script is :
 
```
  --input_dir INPUT_DIR       - The input data directory.
                                Should contain .hdf5 files for the task.
 
  --config_file CONFIG_FILE      - Path to a json file describing the BERT model
                                configuration. This file configures the model
                                architecture, such as the number of transformer
                                blocks, number of attention heads, etc.
 
  --bert_model BERT_MODEL        - Specifies the type of BERT model to use;
                                should be one of the following:
        bert-base-uncased
        bert-large-uncased
        bert-base-cased
        bert-base-multilingual
        bert-base-chinese
 
  --output_dir OUTPUT_DIR        - Path to the output directory where the model
                                checkpoints will be written.
 
  --init_checkpoint           - Initial checkpoint to start pre-training from (Usually a BERT pre-trained checkpoint)
 
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
                              - The initial learning rate for the LAMB  optimizer.
 
  --max_steps MAX_STEPS       - Total number of training steps to perform.
 
  --warmup_proportion WARMUP_PROPORTION
                              - Proportion of training to perform linear learning
                                rate warmup for. For example, 0.1 = 10% of training.
 
  --seed SEED                 - Sets the seed to use for random number generation.
 
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                              - Number of update steps to accumulate before
                                performing a backward/update pass.
 
  --allreduce_post_accumulation - If set to true, performs allreduce only after the defined number of gradient accumulation steps.
  
  --allreduce_post_accumulation_fp16 -  If set to true, performs allreduce after gradient accumulation steps in FP16.
 
  --amp or --fp16                      - If set, performs computations using
                                automatic mixed precision.
 
  --loss_scale LOSS_SCALE        - Sets the loss scaling value to use when
                                mixed precision is used. The default value (0)
                                tells the script to use dynamic loss scaling
                                instead of fixed loss scaling.
 
  --log_freq LOG_FREQ         - If set, the script outputs the training
                                loss every LOG_FREQ step.
 
  --resume_from_checkpoint       - If set, training resumes from a checkpoint
                                that currently exists in OUTPUT_DIR.
 
  --num_steps_per_checkpoint NUM_STEPS_PER_CHECKPOINT
                              - Number of update steps until a model checkpoint
                                is saved to disk.
  --phase2                 - Specified if training on phase 2 only. If not specified, default pre-training is on phase 1.
 
  --phase1_end_step        - The number of steps phase 1 was trained for. In order to  
                           resume phase 2 the correct way; phase1_end_step should correspond to the --max_steps phase 1 was trained for.
 
```

#### Fine tuning parameters

* SQuAD 
 
Default arguments are listed below in the order `scripts/run_squad.sh` expects:
 
-   Initial checkpoint - The default is `/workspace/checkpoints/bert_uncased.pt`.
-   Number of training Epochs - The default is `2`.
-   Batch size - The default is `3`.
-   Learning rate - The default is `3e-5`.
-   Precision (either `fp16`, `tf32` or `fp32`) - The default is `fp16`.
-   Number of GPUs - The default is `8`.
-   Seed - The default is `1`.
-   SQuAD directory -  The default is `/workspace/bert/data/v1.1`.
-   Vocabulary file (token to ID mapping) - The default is `/workspace/bert/vocab/vocab`.
-   Output directory for results - The default is `/results/SQuAD`.
-   Mode (`train`, `eval`, `train eval`, `predict`) - The default is `train`.
-   Config file for the BERT model (It should be the same as the pre-trained model) - The default is `/workspace/bert/bert_config.json`.
 
The script saves the final checkpoint to the `/results/SQuAD/pytorch_model.bin` file.

* GLUE

Default arguments are listed below in the order `scripts/run_glue.sh` expects:
 
-   Initial checkpoint - The default is `/workspace/bert/checkpoints/bert_uncased.pt`.
-   Data directory -  The default is `/workspace/bert/data/download/glue/MRPC/`.
-   Vocabulary file (token to ID mapping) - The default is `/workspace/bert/vocab/vocab`.
-   Config file for the BERT model (It should be the same as the pre-trained model) - The default is `/workspace/bert/bert_config.json`.
-   Output directory for result - The default is `/workspace/bert/results/MRPC`.
-   The name of the GLUE task (`mrpc` or `sst-2`) - The default is `mrpc`
-   Number of GPUs - The default is `8`.
-   Batch size per GPU - The default is `16`.
-   Number of update steps to accumulate before performing a backward/update pass (this option effectively normalizes the GPU memory footprint down by the same factor) - The default is `1`.
-   Learning rate - The default is `2.4e-5`.
-   The proportion of training samples used to warm up the learning  rate - The default is `0.1`.
-   Number of training Epochs - The default is `3`.
-   Total number of training steps to perform - The default is `-1.0`, which means it is determined by the number of epochs.
-   Precision (either `fp16`, `tf32` or `fp32`) - The default is `fp16`.
-   Seed - The default is `2`.
-   Mode (`train`, `eval`, `prediction`, `train eval`, `train prediction`, `eval prediction`, `train eval prediction`) - The default is `train eval`.

#### Multi-node
 
Multi-node runs can be launched on a pyxis/enroot Slurm cluster (refer to [Requirements](#requirements)) with the `run.sub` script with the following command for a 4-node DGX-1 example for both phase 1 and phase 2:
 
```
BATCHSIZE=2048 LR=6e-3 GRADIENT_STEPS=128 PHASE=1 sbatch -N4 --ntasks-per-node=8 run.sub
BATCHSIZE=1024 LR=4e-3 GRADIENT_STEPS=256 PHASE=2 sbatch -N4 --ntasks-per-node=8 run.sub
```
 
Checkpoints  after phase 1 will be saved in `checkpointdir` specified in `run.sub`. The checkpoint will be automatically picked up to resume training on phase 2. Note that phase 2 should be run after phase 1.
 
Variables to re-run the [Training performance results](#training-performance-results) are available in the `configurations.yml` file. 
 
The batch variables `BATCHSIZE`, `LR`, `GRADIENT_STEPS`,`PHASE` refer to the Python arguments `train_batch_size`, `learning_rate`, `gradient_accumulation_steps`, `phase2` respectively.
 
Note that the `run.sub` script is a starting point that has to be adapted depending on the environment. In particular, variables such as `datadir` handle the location of the files for each phase. 
 
Refer to the file’s contents to find the full list of variables to adjust for your system.
 
### Command-line options
 
To view the full list of available options and their descriptions, use the `-h` or `--help` command-line option, for example:
 
`python run_pretraining.py --help`
 
`python run_squad.py --help`

`python run_glue.py --help`
 
Detailed descriptions of command-line options can be found in the [Parameters](#parameters) section.
 
### Getting the data
 
For pre-training BERT, we use the Wikipedia (2500M words) dataset. We extract 
only the text passages and ignore headers, lists, and tables. BERT requires that
datasets are structured as a document level corpus rather than a shuffled 
sentence-level corpus because it is critical to extract long contiguous 
sentences. `data/create_datasets_from_start.sh` uses the LDDL downloader to 
download the Wikipedia dataset, and `scripts/run_pretraining.sh` uses the LDDL 
preprocessor and load balancer to preprocess the Wikipedia dataset into Parquet
shards which are then streamed during the pre-training by the LDDL data loader.
Refer to [LDDL's README](../../../Tools/lddl/README.md) for more 
information on how to use LDDL. Depending on the speed of your internet 
connection, downloading and extracting the Wikipedia dataset takes a few hours,
and running the LDDL preprocessor and load balancer takes half an hour on a 
single DGXA100 node.

For fine-tuning a pre-trained BERT model for specific tasks, by default, this repository prepares the following dataset:
 
-   [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/): for question answering
-   [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398): for paraphrase detection.
-   [SST-2](https://nlp.stanford.edu/sentiment/index.html): for sentiment analysis.
 
 
#### Dataset guidelines
 
The procedure to prepare a text corpus for pre-training is described in the above section. This section provides additional insight into how exactly raw text is processed so that it is ready for pre-training.
 
First, raw text is tokenized using [WordPiece tokenization](https://arxiv.org/pdf/1609.08144.pdf). A [CLS] token is inserted at the start of every sequence, and the two sentences in the sequence are separated by a [SEP] token.
 
Note: BERT pre-training looks at pairs of sentences at a time. A sentence embedding token [A] is added to the first sentence and token [B] to the next.
 
BERT pre-training optimizes for two unsupervised classification tasks. The first is Masked Language Modeling (Masked LM). One training instance of Masked LM is a single modified sentence. Each token in the sentence has a 15% chance of being replaced by a [MASK] token. The chosen token is replaced with [MASK] 80% of the time, 10% with a random token and the remaining 10% the token is retained. The task is then to predict the original token.
 
The second task is next sentence prediction. One training instance of BERT pre-training is two sentences (a sentence pair). A sentence pair may be constructed by simply taking two adjacent sentences from a single document or by pairing up two random sentences with equal probability. The goal of this task is to predict whether or not the second sentence followed the first in the original document.

### Training process
 
The training process consists of two steps: pre-training and fine-tuning.
 
#### Pre-training
 
Pre-training is performed using the `run_pretraining.py` script along with parameters defined in the `scripts/run_pretraining.sh`.
 
The `run_pretraining.sh` script runs a job on a single node that trains the BERT-large model from scratch using the Wikipedia dataset as training data using the LAMB optimizer. By default, the training script runs two phases of training with a hyperparameter recipe specific to 8x V100 32G cards:
 
Phase 1: (Maximum sequence length of 128)
-   Runs on 8 GPUs with a training batch size of 64 per GPU
-   Uses a learning rate of 6e-3
-   Has FP16 precision enabled
-   Runs for 7038 steps, where the first 28.43% (2000) are warm-up steps
-   Saves a checkpoint every 200 iterations (keeps only the latest three checkpoints) and at the end of training. All checkpoints and training logs are saved to the `/results` directory (in the container which can be mounted to a local directory).
-   Creates a log file containing all the output
 
Phase 2: (Maximum sequence length of 512)
-   Runs on 8 GPUs with a training batch size of 8 per GPU
-   Uses a learning rate of 4e-3
-   Has FP16 precision enabled
-   Runs for 1563 steps, where the first 12.8% are warm-up steps
-   Saves a checkpoint every 200 iterations (keeps only the latest three checkpoints) and at the end of training. All checkpoints and training logs are saved to the `/results` directory (in the container which can be mounted to a local directory).
-   Creates a log file containing all the output
 
These parameters will train on the Wikipedia dataset to state-of-the-art accuracy on a DGX-1 with 32GB V100 cards.
 
`bash run_pretraining.sh <training_batch_size> <learning-rate> <precision> <num_gpus> <warmup_proportion> <training_steps> <save_checkpoint_steps> <resume_training> <create_logfile> <accumulate_gradients> <gradient_accumulation_steps> <seed> <job_name> <allreduce_post_accumulation> <allreduce_post_accumulation_fp16> <accumulate_into_fp16> <train_bath_size_phase2> <learning_rate_phase2> <warmup_proportion_phase2> <train_steps_phase2> <gradient_accumulation_steps_phase2> `
 
Where:
 
-   `<training_batch_size>` is per-GPU batch size used for training. Larger batch sizes run more efficiently but require more memory.
-   `<learning_rate>` is the base learning rate for training
-   `<precision>` is the type of math in your model, which can  be either `fp32` or `fp16`. The options mean:
    -   FP32: 32-bit IEEE single precision floats.
    -   FP16: Mixed precision 16 and 32-bit floats.
-   `<num_gpus>` is the number of GPUs to use for training. Must be equal to or smaller than the number of GPUs attached to your node.
-   `<warmup_proportion>` is the percentage of training steps used for warm-up at the start of training.
-   `<training_steps>` is the total number of training steps.
-   `<save_checkpoint_steps>` controls how often checkpoints are saved.
-   `<resume_training>` if set to `true`, training should resume from the latest model in `/results/checkpoints`. Default is `false`.
-   `<create_logfile>` a flag indicating if output should be written to a log file or not (acceptable values are `true` or 'false`. `true` indicates output should be saved to a log file.)
-   `<accumulate_gradient>` a flag indicating whether a larger batch should be simulated with gradient accumulation.
-   `<gradient_accumulation_steps>` an integer indicating the number of steps to accumulate gradients over. Effective batch size = `training_batch_size` / `gradient_accumulation_steps`.
-   `<seed>` random seed for the run.
- `<allreduce_post_accumulation>` - If set to `true`, performs `allreduce` only after the defined number of gradient accumulation steps.
- `<allreduce_post_accumulation_fp16>` -  If set to `true`, performs `allreduce` after gradient accumulation steps in FP16.
 
    Note: The above two options need to be set to false when running either TF32 or  FP32. 
    
-  `<training_batch_size_phase2>` is per-GPU batch size used for training in phase 2. Larger batch sizes run more efficiently but require more memory.
-   `<learning_rate_phase2>` is the base learning rate for training phase 2.
-   `<warmup_proportion_phase2>` is the percentage of training steps used for warm-up at the start of training.
-   `<training_steps_phase2>` is the total number of training steps for phase 2, to be continued in addition to phase 1.
-   `<gradient_accumulation_steps_phase2>` an integer indicating the number of steps to accumulate gradients over in phase 2. Effective batch size = `training_batch_size_phase2` / `gradient_accumulation_steps_phase2`.
-   `<init_checkpoint>` A checkpoint to start the pre-training routine on (Usually a BERT pre-trained checkpoint).
 
For example:
 
`bash scripts/run_pretraining.sh`
 
Trains BERT-large from scratch on a DGX-1 32G using FP16 arithmetic. 90% of the training steps are done with sequence length 128 (phase 1 of training), and 10% of the training steps are done with sequence length 512 (phase 2 of training).
 
To train on a DGX-1 16G, set `gradient_accumulation_steps` to `512` and `gradient_accumulation_steps_phase2` to `1024` in `scripts/run_pretraining.sh`.
 
To train on a DGX-2 32G, set `train_batch_size` to `4096`, `train_batch_size_phase2` to `2048`, `num_gpus` to `16`, `gradient_accumulation_steps` to `64` and `gradient_accumulation_steps_phase2` to `256` in `scripts/run_pretraining.sh`
 
In order to run a pre-training routine on an initial checkpoint, perform the following in `scripts/run_pretraining.sh`:
-   point the `init_checkpoint` variable to the location of the checkpoint
-   set `resume_training` to `true`
-   Note: The parameter value assigned to `BERT_CONFIG` during training should remain unchanged. Also, to resume pre-training on your corpus of choice, the training dataset should be created using the same vocabulary file used in `data/create_datasets_from_start.sh`.
 
#### Fine-tuning
 
Fine-tuning is provided for a variety of tasks. The following tasks are included with this repository through the following scripts:
 
-   Question Answering (`scripts/run_squad.sh`)
-   Paraphrase Detection and Sentiment Analysis (`script/run_glue.sh`)
 
By default, each Python script implements fine-tuning a pre-trained BERT model for a specified number of training epochs as well as evaluation of the fine-tuned model. Each shell script invokes the associated Python script with the following default parameters:
 
-   Uses 8 GPUs
-   Has FP16 precision enabled
-   Saves a checkpoint at the end of training to the `results/<dataset_name>` folder
 
Fine-tuning Python scripts implement support for mixed precision and multi-GPU training through NVIDIA’s [APEX](https://github.com/NVIDIA/apex) library. For a full list of parameters and associated explanations, refer to the [Parameters](#parameters) section.
 
The fine-tuning shell scripts have positional arguments outlined below:
 
```
# For SQuAD.
bash scripts/run_squad.sh <checkpoint_to_load> <epochs> <batch_size per GPU> <learning rate> <precision (either `fp16` or `fp32`)> <number of GPUs to use> <seed> <SQuAD_DATA_DIR> <VOCAB_FILE> <OUTPUT_DIR> <mode (either `train`, `eval` or `train eval`)> <CONFIG_FILE>
# For GLUE
bash scripts/run_glue.sh <checkpoint_to_load> <data_directory> <vocab_file> <config_file> <out_dir> <task_name> <number of GPUs to use> <batch size per GPU> <gradient_accumulation steps> <learning_rate> <warmup_proportion> <epochs> <precision (either `fp16` or `fp32` or `tf32`)> <seed> <mode (either `train`, `eval`, `prediction`, `train eval`, `train prediction`, `eval prediction` or `train eval prediction`)>
```
 
By default, the mode positional argument is set to train eval. Refer to the [Quick Start Guide](#quick-start-guide) for explanations of each positional argument.
 
Note: The first positional argument (the path to the checkpoint to load) is required.
 
Each fine-tuning script assumes that the corresponding dataset files exist in the `data/` directory or separate path can be a command-line input to `run_squad.sh`.
 
### Inference process

Fine-tuning inference can be run in order to obtain predictions on fine-tuning tasks, for example, Q&A on SQuAD.
 
#### Fine-tuning inference
 
Evaluation fine-tuning is enabled by the same scripts as training:
 
-   Question Answering (`scripts/run_squad.sh`)
-   Paraphrase Detection and Sentiment Analysis (`scripts/run_glue.sh`)
 
The mode positional argument of the shell script is used to run in evaluation mode. The fine-tuned BERT model will be run on the evaluation dataset, and the evaluation loss and accuracy will be displayed.
 
Each inference shell script expects dataset files to exist in the same locations as the corresponding training scripts. The inference scripts can be run with default settings. By setting the `mode` variable in the script to either `eval` or `prediction` flag, you can choose between running predictions and evaluating them on a given dataset or just obtain the model predictions.
 
`bash scripts/run_squad.sh <path to fine-tuned model checkpoint>`
`bash scripts/run_glue.sh <path to fine-tuned model checkpoint>`

For SQuAD, to run inference interactively on question-context pairs, use the script `inference.py` as follows:
 
`python inference.py --bert_model "bert-large-uncased" --init_checkpoint=<fine_tuned_checkpoint> --config_file="bert_config.json" --vocab_file=<path to vocab file>  --question="What food does Harry like?" --context="My name is Harry and I grew up in Canada. I love apples."`


### Deploying BERT using NVIDIA Triton Inference Server
 
The [NVIDIA Triton Inference Server](https://github.com/NVIDIA/triton-inference-server) provides a cloud inferencing solution optimized for NVIDIA GPUs. The server provides an inference service via an HTTP or GRPC endpoint, allowing remote clients to request inferencing for any model being managed by the server. More information on how to perform inference using NVIDIA Triton Inference Server can be found in [triton/README.md](./triton/README.md).
 
## Performance
 
### Benchmarking
 
The following section shows how to run benchmarks measuring the model performance in training and inference modes.
 
#### Training performance benchmark
 
Training performance benchmarks for pre-training can be obtained by running `scripts/run_pretraining.sh`, and for fine-tuning can be obtained by running `scripts/run_squad.sh` or `scripts/run_glue.sh` for SQuAD or GLUE, respectively. The required parameters can be passed through the command-line as described in [Training process](#training-process).
 
As an example, to benchmark the training performance on a specific batch size for SQuAD, run:
`bash scripts/run_squad.sh <pre-trained checkpoint path> <epochs> <batch size> <learning rate> <fp16|fp32> <num_gpus> <seed> <path to SQuAD dataset> <path to vocab set> <results directory> train <BERT config path] <max steps>`
 
An example call used to generate throughput numbers:
`bash scripts/run_squad.sh /workspace/bert/bert_large_uncased.pt 2.0 4 3e-5 fp16 8 42 /workspace/bert/squad_data /workspace/bert/scripts/vocab/vocab /results/SQuAD train /workspace/bert/bert_config.json -1`
 
 
 
#### Inference performance benchmark
 
Inference performance benchmarks for both fine-tuning can be obtained by running `scripts/run_squad.sh` and `scripts/run_glue.sh` respectively. The required parameters can be passed through the command-line as described in [Inference process](#inference-process).
 
As an example, to benchmark the inference performance on a specific batch size for SQuAD, run:
`bash scripts/run_squad.sh <pre-trained checkpoint path> <epochs> <batch size> <learning rate> <fp16|fp32> <num_gpus> <seed> <path to SQuAD dataset> <path to vocab set> <results directory> eval <BERT config path> <max steps>`
 
An example call used to generate throughput numbers:
`bash scripts/run_squad.sh /workspace/bert/bert_large_uncased.pt 2.0 4 3e-5 fp16 8 42 /workspace/bert/squad_data /workspace/bert/scripts/vocab/vocab /results/SQuAD eval /workspace/bert/bert_config.json -1`
 
 
 
### Results
 
The following sections provide details on how we achieved our performance and accuracy in training and inference. 
 
#### Training accuracy results
 
Our results were obtained by running the `scripts/run_squad.sh` and `scripts/run_pretraining.sh` training scripts in the pytorch:21.11-py3 NGC container unless otherwise specified.
 
##### Pre-training loss results: NVIDIA DGX A100 (8x A100 80GB)

| DGX System         | GPUs / Node | Batch size / GPU (Phase 1 and Phase 2) | Accumulated Batch size / GPU (Phase 1 and Phase 2) | Accumulation steps (Phase 1 and Phase 2) | Final Loss - TF32 | Final Loss - mixed precision | Time to train(hours) - TF32 | Time to train(hours) - mixed precision | Time to train speedup (TF32 to mixed precision) |
|--------------------|-------------|----------------------------------------------------|------------------------------------------|-------------------|------------------------------|-----------------------------|----------------------------------------|-------------------------------------------------|-----|
| 32 x DGX A100 80GB | 8           | 256 and 32 | 256 and 128                                        | 1 and 4                                  | ---               | 1.2437                       | ---                         | 1.2                                    | 1.9                                             |
| 32 x DGX A100 80GB | 8           | 128 and 16 | 256 and 128                                        | 2 and 8                                  | 1.2465            | ---                          | 2.4                         | ---                                    | ---                                             |


##### Pre-training loss curves
![Pre-training Loss Curves](images/loss_curves.png)

##### Fine-tuning accuracy results: NVIDIA DGX A100 (8x A100 80GB)
* SQuAD

| GPUs | Batch size / GPU (TF32 and FP16) | Accuracy - TF32(% F1) | Accuracy - mixed precision(% F1) | Time to train(hours) - TF32 | Time to train(hours) - mixed precision | Time to train speedup (TF32 to mixed precision) |
|------|----------------------------------|-----------------------|----------------------------------|-----------------------------|----------------------------------------|-------------------------------------------------|
| 8    | 32                               | 90.93                 | 90.96                            | 0.102                       | 0.0574                                 | 1.78                                            |

* MRPC

| GPUs | Batch size / GPU (TF32 and FP16) | Accuracy - TF32(%) | Accuracy - mixed precision(%) | Time to train(seconds) - TF32 | Time to train(seconds) - mixed precision | Time to train speedup (TF32 to mixed precision) |
|------|----------------------------------|--------------------|-------------------------------|-------------------------------|------------------------------------------|-------------------------------------------------|
| 8    | 16                               | 87.25              | 88.24                         | 17.26                         | 7.31                                     | 2.36                                            |

* SST-2

| GPUs | Batch size / GPU (TF32 and FP16) | Accuracy - TF32(%) | Accuracy - mixed precision(%) | Time to train(seconds) - TF32 | Time to train(seconds) - mixed precision | Time to train speedup (TF32 to mixed precision) |
|------|----------------------------------|--------------------|-------------------------------|-------------------------------|------------------------------------------|-------------------------------------------------|
| 8    | 128                              | 91.97              | 92.78                         | 119.28                        | 62.59                                    | 1.91                                            |

 
##### Training stability test
 
###### Pre-training stability test
 
| Accuracy Metric | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean  | Standard Deviation |
|-----------------|--------|--------|--------|--------|--------|-------|--------------------|
| Final Loss      | 1.260  | 1.265  | 1.304  | 1.256  | 1.242  | 1.265 | 0.023              |
 
###### Fine-tuning stability test
* SQuAD
 
Training stability with 8 GPUs, FP16 computations, batch size of 4:
 
| Accuracy Metric | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 | Seed 6 | Seed 7 | Seed 8 | Seed 9 | Mean  | Standard Deviation |
|-----------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-------|--------------------|
| Exact Match %   | 83.64  | 84.05  | 84.51  | 83.69  | 83.87  | 83.94  | 84.27  | 83.97  | 83.75  | 83.92  | 83.96 | 0.266              |
| f1 %            | 90.60  | 90.65  | 90.96  | 90.44  | 90.58  | 90.78  | 90.81  | 90.82  | 90.51  | 90.68  | 90.68 | 0.160              |
 
* MRPC

Training stability with 8 A100 GPUs, FP16 computations, batch size of 16 per GPU:
 
| Accuracy Metric | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 | Seed 6 | Seed 7 | Seed 8 | Seed 9 | Mean  | Standard Deviation |
|-----------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-------|--------------------|
| Exact Match %   | 85.78  | 85.54  | 84.56  | 86.27  | 84.07  | 86.76  | 87.01  | 85.29  | 88.24  | 86.52  | 86.00 | 1.225              |

> Note: Since MRPC is a very small dataset where overfitting can often occur, the resulting validation accuracy can often have high variance. By repeating the above experiments for 100 seeds, the max accuracy is 88.73, and the average accuracy is 82.56 with a standard deviation of 6.01.

* SST-2

Training stability with 8 A100 GPUs, FP16 computations, batch size of 128 per GPU:
 
| Accuracy Metric | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 | Seed 6 | Seed 7 | Seed 8 | Seed 9 | Mean  | Standard Deviation |
|-----------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-------|--------------------|
| Exact Match %   | 91.86  | 91.28  | 91.86  | 91.74  | 91.28  | 91.86  | 91.40  | 91.97  | 91.40  | 92.78  | 91.74 | 0.449              |
 
#### Training performance results

##### Training performance: NVIDIA DGX A100 (8x A100 80GB)

Our results were obtained by running the `scripts run_pretraining.sh` training script in the pytorch:21.11-py3 NGC container on NVIDIA DGX A100 (8x A100 80GB) GPUs. Performance numbers (in items/images per second) were averaged over a few training iterations.

###### Pre-training NVIDIA DGX A100 (8x A100 80GB)

| GPUs | Batch size / GPU (TF32 and FP16) | Accumulated Batch size / GPU (TF32 and FP16) | Accumulation steps (TF32 and FP16) | Sequence length | Throughput - TF32(sequences/sec) | Throughput - mixed precision(sequences/sec) | Throughput speedup (TF32 - mixed precision) | Weak scaling - TF32 | Weak scaling - mixed precision |
|------|----------------------------------|------------------------------------|-----------------|----------------------------------|---------------------------------------------|---------------------------------------------|---------------------|--------------------------------|----|
| 1    | 128 and 256 | 8192 and 8192                    | 64 and 32                          | 128             | 317                              | 580                                         | 1.83                                        | 1.00                | 1.00                           |
| 8    | 128 and 256 | 8192 and 8192                    | 64 and 32                          | 128             | 2505                             | 4591                                        | 1.83                                        | 7.90                | 7.91                           |
| 1    | 16 and 32   | 4096 and 4096                    | 256 and 128                        | 512             | 110                              | 210                                         | 1.90                                        | 1.00                | 1.00                           |
| 8    | 16 and 32   | 4096 and 4096                    | 256 and 128                        | 512             | 860                              | 1657                                        | 1.92                                        | 7.81                | 7.89                           |

###### Pre-training NVIDIA DGX A100 (8x A100 80GB) Multi-node Scaling

| Nodes | GPUs / node | Batch size / GPU (TF32 and FP16) | Accumulated Batch size / GPU (TF32 and FP16) | Accumulation steps (TF32 and FP16) | Sequence length | Mixed Precision Throughput | Mixed Precision Strong Scaling | TF32 Throughput | TF32 Strong Scaling | Speedup (Mixed Precision to TF32) |
|-------|-------------|----------------------------------|------------------------------------|-----------------|----------------------------|--------------------------------|-----------------|---------------------|-----------------------------------|-----|
| 1     | 8           | 126 and 256 | 8192 and 8192                    | 64 and 32                          | 128             | 4553                       | 1                              | 2486            | 1                   | 1.83                              |
| 2     | 8           | 126 and 256 | 4096 and 4096                    | 32 and 16                          | 128             | 9191                       | 2.02                           | 4979            | 2.00                | 1.85                              |
| 4     | 8           | 126 and 256 | 2048 and 2048                    | 16 and 18                           | 128             | 18119                      | 3.98                           | 9859            | 3.97                | 1.84                              |
| 8     | 8           | 126 and 256 | 1024 and 1024                    | 8 and 4                            | 128             | 35774                      | 7.86                           | 19815           | 7.97                | 1.81                              |
| 16    | 8           | 126 and 256 | 512 and 512                      | 4 and 2                            | 128             | 70555                      | 15.50                          | 38866           | 15.63               | 1.82                              |
| 32    | 8           | 126 and 256 | 256 and 256                      | 2 and 1                            | 128             | 138294                     | 30.37                          | 75706           | 30.45               | 1.83                              |
| 1     | 8           | 16  and 32  | 4096 and 4096                    | 256 and 128                        | 512             | 1648                       | 1                              | 854             | 1                   | 1.93                              |
| 2     | 8           | 16  and 32  | 2048 and 2048                    | 128 and 64                         | 512             | 3291                       | 2.00                           | 1684            | 1.97                | 1.95                              |
| 4     | 8           | 16  and 32  | 1024 and 1024                    | 64 and 32                          | 512             | 6464                       | 3.92                           | 3293            | 3.86                | 1.96                              |
| 8     | 8           | 16  and 32  | 512 and 512                      | 32 and 16                          | 512             | 13005                      | 7.89                           | 6515            | 7.63                | 2.00                              |
| 16    | 8           | 16  and 32  | 256 and 256                      | 16 and 8                           | 512             | 25570                      | 15.51                          | 12131           | 14.21               | 2.11                              |
| 32    | 8           | 16  and 32  | 128 and 128                      | 8 and 4                            | 512             | 49663                      | 30.13                          | 21298           | 24.95               | 2.33                              |

###### Fine-tuning NVIDIA DGX A100 (8x A100 80GB)

* SQuAD
  
| GPUs | Batch size / GPU (TF32 and FP16) | Throughput - TF32(sequences/sec) | Throughput - mixed precision(sequences/sec) | Throughput speedup (TF32 - mixed precision) | Weak scaling - TF32 | Weak scaling - mixed precision |
|------|----------------------------------|----------------------------------|---------------------------------------------|---------------------------------------------|---------------------|--------------------------------|
| 1    | 32 and 32                        | 61.5                             | 110.5                                       | 1.79                                        | 1.00                | 1.00                           |
| 8    | 32 and 32                        | 469.8                            | 846.7                                       | 1.80                                        | 7.63                | 7.66                           |
 

##### Training performance: NVIDIA DGX-1 (8x V100 32G)

Our results were obtained by running the `scripts/run_pretraining.sh` and `scripts/run_squad.sh` training scripts in the pytorch:21.11-py3 NGC container on NVIDIA DGX-1 with (8x V100 32G) GPUs. Performance numbers (in sequences per second) were averaged over a few training iterations.
 
###### Pre-training NVIDIA DGX-1 With 32G
 
| GPUs | Batch size / GPU (FP32 and FP16) | Accumulation steps (FP32 and FP16) | Sequence length | Throughput - FP32(sequences/sec) | Throughput - mixed precision(sequences/sec) | Throughput speedup (FP32 - mixed precision) | Weak scaling - FP32 | Weak scaling - mixed precision |
|------|----------------------------------|------------------------------------|-----------------|----------------------------------|---------------------------------------------|---------------------------------------------|---------------------|--------------------------------|
| 1    | 4096 and 4096                    | 128 and 64                         | 128             | 50                               | 224                                         | 4.48                                        | 1.00                | 1.00                           |
| 8    | 4096 and 4096                    | 128 and 64                         | 128             | 387                              | 1746                                        | 4.51                                        | 7.79                | 7.79                           |
| 1    | 2048 and 2048                    | 512 and 256                        | 512             | 19                               | 75                                          | 3.94                                        | 1.00                | 1.00                           |
| 8    | 2048 and 2048                    | 512 and 256                        | 512             | 149.6                            | 586                                         | 3.92                                        | 7.87                | 7.81                           |

###### Fine-tuning NVIDIA DGX-1 With 32G

* SQuAD 
 
| GPUs | Batch size / GPU (FP32 and FP16) | Throughput - FP32(sequences/sec) | Throughput - mixed precision(sequences/sec) | Throughput speedup (FP32 - mixed precision) | Weak scaling - FP32 | Weak scaling - mixed precision |
|------|----------------------------------|----------------------------------|---------------------------------------------|---------------------------------------------|---------------------|--------------------------------|
| 1    | 8 and 16                         | 12                               | 52                                          | 4.33                                        | 1.00                | 1.00                           |
| 8    | 8 and 16                         | 85.5                             | 382                                         | 4.47                                        | 7.12                | 7.34                           |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).
 
#### Inference performance results

##### Inference performance: NVIDIA DGX A100 (1x A100 80GB) 
 
Our results were obtained by running `scripts/run_squad.sh` in the pytorch:21.11-py3 NGC container on NVIDIA DGX A100 with (1x A100 80G) GPUs.
 
###### Fine-tuning inference on NVIDIA DGX A100 (1x A100 80GB)

* SQuAD
 
| GPUs | Batch Size \(TF32/FP16\) | Sequence Length | Throughput \- TF32\(sequences/sec\) | Throughput \- Mixed Precision\(sequences/sec\) |
|------|--------------------------|-----------------|-------------------------------------|------------------------------------------------|
| 1    | 32/32                    | 384             | 216                                 | 312                                            |
 
To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).
 
The inference performance metrics used were sequences/second.
 
## Release notes
 
### Changelog
 
January 2022
- Knowledge Distillation support
- Pre-training with native AMP, native DDP, and TorchScript with NVFuser backend
- Pre-training using [Language Datasets and Data Loaders (LDDL)](../../../Tools/lddl)
- Binned pretraining for phase2 with LDDL using a bin size of 64

July 2020
-  Updated accuracy and performance tables to include A100 results
-  Fine-tuning with the MRPC and SST-2 datasets.
 
March 2020
- TRITON Inference Server support.
 
February 2020
- Integrate DLLogger.
 
November 2019
- Use LAMB from APEX.
- Code cleanup.
- Bug fix in BertAdam optimizer.
 
September 2019
- Scripts to support a multi-node launch.
- Update pre-training loss results based on the latest data preparation scripts.
 
August 2019
- Pre-training support with LAMB optimizer.
- Updated Data download and Preprocessing.
 
July 2019
- Initial release.
 
### Known issues
 
There are no known issues with this model.




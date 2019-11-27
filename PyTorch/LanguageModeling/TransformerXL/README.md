# Transformer-XL For PyTorch

This repository provides a script and recipe to train the Transformer-XL model
to achieve state-of-the-art accuracy, and is tested and maintained by NVIDIA.

## Table Of Contents

<!-- TOC GFM -->

* [Model overview](#model-overview)
  * [Model architecture](#model-architecture)
  * [Default configuration](#default-configuration)
  * [Feature support matrix](#feature-support-matrix)
    * [Features](#features)
  * [Mixed precision training](#mixed-precision-training)
    * [Enabling mixed precision](#enabling-mixed-precision)
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
  * [Inference process](#inference-process)
* [Performance](#performance)
  * [Benchmarking](#benchmarking)
    * [Training performance benchmark](#training-performance-benchmark)
    * [Inference performance benchmark](#inference-performance-benchmark)
  * [Results](#results)
    * [Training accuracy results](#training-accuracy-results)
      * [Training accuracy: NVIDIA DGX-1 (8x V100 16G)](#training-accuracy-nvidia-dgx-1-8x-v100-16g)
      * [Training accuracy: NVIDIA DGX-2 (16x V100 32G)](#training-accuracy-nvidia-dgx-2-16x-v100-32g)
      * [Training stability test](#training-stability-test)
    * [Training performance results](#training-performance-results)
      * [Training performance: NVIDIA DGX-1 (8x V100 16G)](#training-performance-nvidia-dgx-1-8x-v100-16g)
      * [Training performance: NVIDIA DGX-2 (16x V100 32G)](#training-performance-nvidia-dgx-2-16x-v100-32g)
    * [Inference performance results](#inference-performance-results)
      * [Inference performance: NVIDIA DGX-1 (1x V100 16G)](#inference-performance-nvidia-dgx-1-1x-v100-16g)
      * [Inference performance: NVIDIA T4](#inference-performance-nvidia-t4)
* [Release notes](#release-notes)
  * [Changelog](#changelog)
  * [Known issues](#known-issues)

<!-- /TOC -->

## Model overview

This repository provides an implementation of the Transformer-XL model in
PyTorch from the paper [Transformer-XL: Attentive Language Models Beyond a
Fixed-Length Context](https://arxiv.org/abs/1901.02860). Transformer-XL is
a transformer-based language model with a segment-level recurrence and a novel
relative positional encoding. Enhancements introduced in Transformer-XL help
capture better long-term dependencies by attending to tokens from multiple
previous segments.

Our implementation is based on the
[codebase](https://github.com/kimiyoung/transformer-xl) published by the
authors of the Transformer-XL paper.
Our implementation uses modified model architecture hyperparameters. Our
modifications were made to achieve better hardware utilization and to take
advantage of Tensor Cores. Similar modifications were also proposed in an
implementation available from
[github.com/cybertronai/transformer-xl](https://github.com/cybertronai/transformer-xl).
Refer to the [Model architecture](#model-architecture) section for more
details.

This model is trained with mixed precision using Tensor Cores on NVIDIA Volta
GPUs and evaluated on Volta and Turing GPUs. Therefore, researchers can get
results up to 2.5x faster than training without Tensor Cores, while
experiencing the benefits of mixed precision training. This model is tested
against each NGC monthly container release to ensure consistent accuracy and
performance over time.

### Model architecture

The Transformer-XL "base" model for WikiText-103 dataset available in this
repository was modified to use the following values of hyperparameters:

|**Hyperparameter**|**Description**|**Original setting**|**Our modification**|
|------------------|---------------|-------------------:|-------------------:|
| `d_model` | hidden size                                                      | 410  | 512  |
| `n_head`  | number of attention heads                                        | 10   | 8    |
| `d_head`  | size of each attention head                                      | 41   | 64   |
| `d_inner` | hidden size in fully-connected layers                            | 2100 | 2048 |
| `tgt_len` | number of tokens to predict during training                      | 150  | 192  |
| `mem_len` | number of tokens cached from previous iterations during training | 150  | 192  |

Changes described above were made to align certain hyperparameters with powers
of two, with this modification, the model is able to achieve better hardware
utilization, and therefore higher training throughput.


The Transformer-XL model addresses the limitations of vanilla transformer-based
language models, which are only able to use relatively short context, bounded
by the segment length. The Transformer-XL introduces a recurrence mechanism,
which is able to use a cached hidden state from previous segments. During
training, the context consists of a concatenation of current segment's hidden
state and cached states from previous iterations. Gradients are backpropagated
only through the current segment, although the model is able to take advantage
of the extra information stored in the cache and therefore is able to model
long-term dependencies.

An illustration of the recurrence mechanism taken from the [Transformer-XL
paper](https://arxiv.org/abs/1901.02860) is shown below.
![model](pytorch/img/model.png)


### Default configuration

The following features were implemented in this model:

* general
  * single-node, data-parallel multi-GPU training,
  * training and inference with mixed precision using Tensor Cores,
  * mixed precision training implemented using 
    [Apex AMP](https://nvidia.github.io/apex/amp.html), with `O2` optimization
    level and with a dynamic loss scaling,

* model
  * a 16-layer base Transformer-XL model with hidden size 512, 8 attention heads,
    each head with hidden size 64,
  * the model trained on
    [WikiText-103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)
    dataset, using word-level vocabulary and
    adaptive softmax,
  * embedding weights are tied with weights in the classifier,

* training
  * training with [LAMB](https://arxiv.org/abs/1904.00962) optimizer,
  * linear learning rate warmup for 1000 iterations, followed by cosine
    learning rate schedule, initial learning rate is set to 0.01, final
    learning rate is set to 0.001,
  * training for 40,000 steps, using batch size of 256,
  * support for a training with a gradient accumulation,

* inference
  * support for multi-gpu inference,
  * support for TorchScript and pure Python inference,
  * target length is set to 64, length of memory is set to 640,
  * positional embeddings are clamped after 400 time steps,
  * each token is using the same size of the context from previous time steps.

### Feature support matrix

The following features are supported by this model:

| **Feature** | **Transformer-XL** |
|:------------|------------:|
|[Apex AMP](https://nvidia.github.io/apex/amp.html) | Yes |
|[Apex DistributedDataParallel](https://nvidia.github.io/apex/parallel.html#apex.parallel.DistributedDataParallel) | Yes |

#### Features

[Apex AMP](https://nvidia.github.io/apex/amp.html) - a tool that enables Tensor
Core-accelerated training. Refer to the [Enabling mixed
precision](#enabling-mixed-precision) section for more details.

[Apex
DistributedDataParallel](https://nvidia.github.io/apex/parallel.html#apex.parallel.DistributedDataParallel) -
a module wrapper that enables easy multiprocess distributed data parallel
training, similar to
[torch.nn.parallel.DistributedDataParallel](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel).
`DistributedDataParallel` is optimized for use with
[NCCL](https://github.com/NVIDIA/nccl). It achieves high performance by
overlapping communication with computation during `backward()` and bucketing
smaller gradient transfers to reduce the total number of transfers required.

### Mixed precision training

Mixed precision is the combined use of different numerical precisions in a
computational method.
[Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant
computational speedup by performing operations in half-precision format, while
storing minimal information in single-precision to retain as much information
as possible in critical parts of the network. Since the introduction of [Tensor
Cores](https://developer.nvidia.com/tensor-cores) in the Volta and Turing
architectures, significant training speedups are experienced by switching to
mixed precision -- up to 3x overall speedup on the most arithmetically intense
model architectures. Using mixed precision training previously required two
steps:

1. Porting the model to use the FP16 data type where appropriate.
2. Manually adding loss scaling to preserve small gradient values.

The ability to train deep learning networks with lower precision was introduced
in the Pascal architecture and first supported in [CUDA
8](https://devblogs.nvidia.com/parallelforall/tag/fp16/) in the NVIDIA Deep
Learning SDK.

For information about:
* How to train using mixed precision, see the [Mixed Precision
  Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed
  Precision](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html)
  documentation.
* Techniques used for mixed precision training, see the [Mixed-Precision
  Training of Deep Neural
  Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/)
  blog.
* APEX tools for mixed precision training, see the [NVIDIA Apex: Tools for Easy
  Mixed-Precision Training in
  PyTorch](https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/)
  .

#### Enabling mixed precision
The `pytorch/train.py` training script launches mixed precision training
with Tensor Cores if the flag `--fp16` is set.

Mixed precision is enabled in PyTorch by using the Automatic Mixed Precision
(AMP), library from [APEX](https://github.com/NVIDIA/apex) that casts variables
to half-precision upon retrieval, while storing variables in single-precision
format. Furthermore, to preserve small gradient magnitudes in backpropagation,
a [loss
scaling](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#lossscaling)
step must be included when applying gradients. In PyTorch, loss scaling can be
easily applied by using `scale_loss()` method provided by AMP. The scaling
value to be used can be
[dynamic](https://nvidia.github.io/apex/amp.html#apex.amp.initialize) or fixed.

For an in-depth walk through on AMP, check out sample usage
[here](https://nvidia.github.io/apex/amp.html#).
[APEX](https://github.com/NVIDIA/apex) is a PyTorch extension that contains
utility libraries, such as AMP, which require minimal network code changes to
leverage Tensor Cores performance.

The following steps were needed to enable mixed precision training in
Transformer-XL:

1. Import AMP from APEX:

```
from apex import amp
```

2. Initialize AMP and wrap the model and the optimizer before starting the
  training:

```
model, optimizer = amp.initialize(
    model,
    optimizer,
    opt_level='O2',
    )
```

3. Apply `scale_loss` context manager:

```
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
```

4. Apply gradient clipping on single precision master weights:

```
torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip)
```

## Setup

The following section lists the requirements that you need to meet in order to
start training the Transformer-XL model.

### Requirements

This repository contains `Dockerfile` which extends the PyTorch NGC container
and encapsulates some dependencies.  Aside from these dependencies, ensure you
have the following components:

* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch 19.09-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch)
* [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
  or [Turing](https://www.nvidia.com/pl-pl/geforce/turing/) based GPU

For more information about how to get started with NGC containers, see the
following sections from the NVIDIA GPU Cloud Documentation and the Deep
Learning DGX Documentation:

* [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html),
* [Accessing And Pulling From The NGC container registry](https://docs.nvidia.com/deeplearning/dgx/user-guide/index.html#accessing_registry),
* [Running PyTorch](https://docs.nvidia.com/deeplearning/dgx/pytorch-release-notes/running.html#running).

For those unable to use the Pytorch NGC container, to set up the required
environment or create your own container, see the versioned [NVIDIA Container
Support
Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## Quick Start Guide

To train your model using mixed precision with Tensor Cores or using FP32,
perform the following steps using the default parameters of the Transformer-XL
base model on the
[WikiText-103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)
dataset. 

For the specifics concerning training
and inference, see the [Advanced](#advanced) section.

1. Clone the repository.

```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL
```

2. Download and preprocess the dataset.

```
bash getdata.sh
```

3. Build the Transformer-XL PyTorch NGC container.

From now on, all scripts should be executed from the `pytorch` directory.

```
cd pytorch
bash scripts/docker/build.sh
```

4. Start an interactive session in the NGC container to run training/inference.

```
bash scripts/docker/interactive.sh
```

5. Start training.

To start 8 GPU mixed precision training on DGX-1, run:

```
bash run_wt103_base.sh train 8 --vocab word --adaptive --fp16 --batch_chunk 1 
```

To start 8 GPU FP32 training on DGX-1, run:

```
bash run_wt103_base.sh train 8 --vocab word --adaptive --batch_chunk 2
```

To start 16 GPU mixed precision training on DGX-2, run:

```
bash run_wt103_base.sh train 16 --vocab word --adaptive --fp16 --batch_chunk 1 
```

To start 16 GPU FP32 training on DGX-2, run:

```
bash run_wt103_base.sh train 16 --vocab word --adaptive --batch_chunk 1
```

For more information on the available options, refer to the [Training
process](#training-process) section.

6. Start evaluation.

To start mixed precision inference on the test set using `<#GPUs>` GPUs, run:

```
bash run_wt103_base.sh eval <#GPUs> [--fp16] [--type {pytorch, torchscript}]
```

The `--fp16` flag is optional, however, if it's specified, then the script
launches mixed precision inference with Tensor Cores. If the flag is not
present, then the script launches FP32 inference.
By default, the script is loading the checkpoint from
`LM-TFM/checkpoint_best.pt`, which contains the model corresponding to the
lowest value of the validation loss from the previous training run. Path to the
checkpoint can be customized by setting the `--model` flag.

Inference can use pure Python execution or TorchScript from using the `--type`
flag.

Supported values for `<#GPUs>` are: 1, 2, 4, 8, 16.

Additionally, one can pass the input text directly from the command-line using
the `--manual` flag. This mode of operation supports only 1 GPU and batch size
of 1. The script outputs average loss and perplexity for the provided input
text.

Examples:


```
bash run_wt103_base.sh eval 1 \
  --model LM-TFM/checkpoint_best.pt \
  --fp16 \
  --manual "recognize speech"

===============================================================================
| test loss  6.20 | test ppl   494.291
===============================================================================
```

```
bash run_wt103_base.sh eval 1 \
  --model LM-TFM/checkpoint_best.pt \
  --fp16 \
  --manual "wreck a nice beach"

===============================================================================
| test loss  8.04 | test ppl  3099.706
===============================================================================
```

For more information on the available options, refer to the [Inference
process](#inference-process) section.

## Advanced

The following sections provide greater details of the dataset, running training
and inference, and the training results.

### Scripts and sample code

In the `pytorch` directory, the most important files are:

* `Dockerfile`: container with the basic set of dependencies to run Transformer-XL
* `data_utils.py`: data loading utilities
* `eval.py`: serves as the entry point to launch the evaluation and inference
* `lamb.py`: implementation of [LAMB](https://arxiv.org/abs/1904.00962) optimizer
* `mem_transformer.py`: implementation of the Transformer-XL model
* `requirements.txt`: set of extra requirements for running Transformer-XL
* `train.py`: serves as the entry point to launch the training

The `pytorch/utils` directory contains the following additional modules:

* `adaptive_softmax.py`: implementation of adaptive softmax
* `data_parallel.py`: implementation of `BalancedDataParallel` class
* `distributed.py`: utility functions for running distributed training
* `exp_utils.py`: utility functions for running training and benchmarking
* `log_uniform_sampler.py`: implementation of log-uniform sampler
* `proj_adaptive_softmax.py`: implementation of projected adaptive softmax
* `vocabulary.py`: implementation of word-level vocabulary and BPE-based vocabulary

### Parameters

**Training**

The complete list of available parameters for the `pytorch/train.py` training script
contains:

```
general setup:
  --work_dir WORK_DIR   Directory for the results (default: LM-TFM)
  --append_dataset      Automatically append dataset name to work_dir
                        (default: False)
  --append_time         Automatically append current time to work_dir
                        (default: False)
  --cuda                Use CUDA (default: False)
  --fp16                Run training in fp16/mixed precision (default: False)
  --restart RESTART     Restart training from the saved checkpoint (default: )
  --debug               Run in debug mode (do not create exp dir) (default:
                        False)
  --log_all_ranks       Enable logging from all distributed ranks (default:
                        False)
  --save-all            Save all checkpoints (default: False)
  --log_interval LOG_INTERVAL
                        Report interval (default: 10)
  --target_throughput TARGET_THROUGHPUT
                        Target training throughput (for benchmarking)
                        (default: None)
  --target_perplexity TARGET_PERPLEXITY
                        Target validation perplexity (for benchmarking)
                        (default: None)

dataset setup:
  --data DATA           Location of the data corpus (default:
                        ../data/wikitext-103)
  --dataset {wt103,lm1b,enwik8,text8}
                        Dataset name (default: wt103)
  --vocab {word,bpe}    Type of vocabulary (default: word)

model setup:
  --n_layer N_LAYER     Number of total layers (default: 16)
  --n_head N_HEAD       Number of heads (default: 8)
  --d_head D_HEAD       Head dimension (default: 64)
  --d_embed D_EMBED     Embedding dimension (default: -1)
  --d_model D_MODEL     Model dimension (default: 512)
  --d_inner D_INNER     Inner dimension in feedforward layer (default: 2048)
  --dropout DROPOUT     Global dropout rate (default: 0.1)
  --dropatt DROPATT     Attention probability dropout rate (default: 0.0)
  --pre_lnorm           Apply LayerNorm to the input instead of the output
                        (default: False)
  --attn_type ATTN_TYPE
                        Attention type. 0 for ours, 1 for Shaw et al,2 for
                        Vaswani et al, 3 for Al Rfou et al. (default: 0)
  --not_tied            Do not tie the word embedding and softmax weights
                        (default: False)
  --clamp_len CLAMP_LEN
                        Use the same pos embeddings after clamp_len (default:
                        -1)
  --adaptive            Use adaptive softmax (default: False)
  --div_val DIV_VAL     Dividend value for adaptive input and softmax
                        (default: 1)
  --sample_softmax SAMPLE_SOFTMAX
                        Number of samples in sampled softmax (default: -1)
  --init INIT           Parameter initializer to use (default: normal)
  --emb_init EMB_INIT   Parameter initializer to use (default: normal)
  --init_range INIT_RANGE
                        Parameters initialized by U(-init_range, init_range)
                        (default: 0.1)
  --emb_init_range EMB_INIT_RANGE
                        Parameters initialized by U(-init_range, init_range)
                        (default: 0.01)
  --init_std INIT_STD   Parameters initialized by N(0, init_std) (default:
                        0.02)
  --proj_init_std PROJ_INIT_STD
                        Parameters initialized by N(0, init_std) (default:
                        0.01)

optimizer setup:
  --optim {adam,sgd,adagrad,lamb}
                        Optimizer to use (default: lamb)
  --lr LR               Initial learning rate (default: 0.01)
  --mom MOM             Momentum for sgd (default: 0.0)
  --scheduler {cosine,inv_sqrt,dev_perf,constant}
                        LR scheduler to use (default: cosine)
  --max_step_scheduler MAX_STEP_SCHEDULER
                        Max number of training steps for LR scheduler
                        (default: None)
  --warmup_step WARMUP_STEP
                        Number of iterations for LR warmup (default: 1000)
  --decay_rate DECAY_RATE
                        Decay factor when ReduceLROnPlateau is used (default:
                        0.5)
  --lr_min LR_MIN       Minimum learning rate during annealing (default: 0.0)
  --clip CLIP           Gradient clipping (default: 0.25)
  --weight_decay WEIGHT_DECAY
                        Weight decay for adam|lamb (default: 0.0)
  --clip_nonemb         Only clip the gradient of non-embedding params
                        (default: False)
  --patience PATIENCE   Patience (default: 0)
  --eta_min ETA_MIN     Min learning rate for cosine scheduler (default:
                        0.001)

training setup:
  --max_step MAX_STEP   Max number of training steps (default: 40000)
  --batch_size BATCH_SIZE
                        Global batch size (default: 256)
  --batch_chunk BATCH_CHUNK
                        Split batch into chunks to save memory (default: 1)
  --roll                Enable random shifts within each data stream (default:
                        False)
  --tgt_len TGT_LEN     Number of tokens to predict (default: 192)
  --ext_len EXT_LEN     Length of the extended context (default: 0)
  --mem_len MEM_LEN     Length of the retained previous heads (default: 192)
  --seed SEED           Random seed (default: 1111)
  --multi_gpu {ddp,dp}  Use multiple GPU (default: None)
  --gpu0_bsz GPU0_BSZ   Batch size on gpu 0 (for "dp" backend) (default: -1)
  --same_length         Use the same attn length for all tokens (default:
                        False)
  --varlen              Use variable length (default: False)

validation setup:
  --eval_tgt_len EVAL_TGT_LEN
                        Number of tokens to predict for evaluation (default:
                        192)
  --eval_batch_size EVAL_BATCH_SIZE
                        Eval batch size (default: 16)
  --eval_max_steps EVAL_MAX_STEPS
                        Max eval steps (default: -1)
  --eval_interval EVAL_INTERVAL
                        Evaluation interval (default: 5000)
```

**Inference**

The complete list of available parameters for the `eval.py` inference
script contains:

```
  --work_dir WORK_DIR   experiment directory (default: LM-TFM)
  --debug               run in debug mode (do not create exp dir) (default:
                        False)
  --data DATA           location of the data corpus (default:
                        ../data/wikitext-103)
  --manual MANUAL [MANUAL ...]
                        run model on raw input data (default: None)
  --dataset {wt103,lm1b,enwik8,text8}
                        dataset name (default: wt103)
  --split {all,valid,test}
                        which split to evaluate (default: all)
  --type {pytorch,torchscript,onnx}
                        type of runtime to use (default: pytorch)
  --batch_size BATCH_SIZE
                        batch size (default: 16)
  --tgt_len TGT_LEN     number of tokens to predict (default: 64)
  --ext_len EXT_LEN     length of the extended context (default: 0)
  --mem_len MEM_LEN     length of the retained previous heads (default: 640)
  --clamp_len CLAMP_LEN
                        max positional embedding index (default: -1)
  --cuda                use CUDA (default: False)
  --model MODEL         path to the checkpoint (default: )
  --fp16                Run training in fp16/mixed precision (default: False)
  --log_all_ranks       Enable logging for all distributed ranks (default:
                        False)
  --same_length         set same length attention with masking (default:
                        False)
  --target_perplexity TARGET_PERPLEXITY
                        target perplexity (default: None)
  --target_throughput TARGET_THROUGHPUT
                        target throughput (default: None)
  --save_data           save latency and throughput data to a file (default:
                        False)
  --repeat REPEAT       loop over the dataset REPEAT times (default: 1)
  --max_size MAX_SIZE   run inference on up to MAX_SIZE batches (default:
                        None)
  --percentiles PERCENTILES [PERCENTILES ...]
                        percentiles for latency confidence intervals (default:
                        [90, 95, 99])
  --save_torchscript SAVE_TORCHSCRIPT
                        save torchscript model to a file (default: None)
  --load_torchscript LOAD_TORCHSCRIPT
                        load torchscript model from a file (default: None)
```


### Command-line options

To see the full list of available options and their descriptions, use the `-h`
or `--help` command-line option. For example, for training:

```
python3 train.py --help

usage: train.py [-h] [--work_dir WORK_DIR] [--append_dataset] [--append_time]
                [--cuda] [--fp16] [--restart RESTART] [--debug]
                [--log_all_ranks] [--save-all] [--log_interval LOG_INTERVAL]
                [--target_throughput TARGET_THROUGHPUT]
                [--target_perplexity TARGET_PERPLEXITY] [--data DATA]
                [--dataset {wt103,lm1b,enwik8,text8}] [--vocab {word,bpe}]
                [--n_layer N_LAYER] [--n_head N_HEAD] [--d_head D_HEAD]
                [--d_embed D_EMBED] [--d_model D_MODEL] [--d_inner D_INNER]
                [--dropout DROPOUT] [--dropatt DROPATT] [--pre_lnorm]
                [--attn_type ATTN_TYPE] [--not_tied] [--clamp_len CLAMP_LEN]
                [--adaptive] [--div_val DIV_VAL]
                [--sample_softmax SAMPLE_SOFTMAX] [--init INIT]
                [--emb_init EMB_INIT] [--init_range INIT_RANGE]
                [--emb_init_range EMB_INIT_RANGE] [--init_std INIT_STD]
                [--proj_init_std PROJ_INIT_STD]
                [--optim {adam,sgd,adagrad,lamb}] [--lr LR] [--mom MOM]
                [--scheduler {cosine,inv_sqrt,dev_perf,constant}]
                [--max_step_scheduler MAX_STEP_SCHEDULER]
                [--warmup_step WARMUP_STEP] [--decay_rate DECAY_RATE]
                [--lr_min LR_MIN] [--clip CLIP] [--weight_decay WEIGHT_DECAY]
                [--clip_nonemb] [--patience PATIENCE] [--eta_min ETA_MIN]
                [--max_step MAX_STEP] [--batch_size BATCH_SIZE]
                [--batch_chunk BATCH_CHUNK] [--roll] [--tgt_len TGT_LEN]
                [--ext_len EXT_LEN] [--mem_len MEM_LEN] [--seed SEED]
                [--multi_gpu {ddp,dp}] [--gpu0_bsz GPU0_BSZ] [--same_length]
                [--varlen] [--eval_tgt_len EVAL_TGT_LEN]
                [--eval_batch_size EVAL_BATCH_SIZE]
                [--eval_max_steps EVAL_MAX_STEPS]
                [--eval_interval EVAL_INTERVAL] [--local_rank LOCAL_RANK]
```

For example, for inference:

```
python3 eval.py --help

usage: eval.py [-h] [--work_dir WORK_DIR] [--debug] [--data DATA]
               [--manual MANUAL [MANUAL ...]]
               [--dataset {wt103,lm1b,enwik8,text8}]
               [--split {all,valid,test}] [--type {pytorch,torchscript,onnx}]
               [--batch_size BATCH_SIZE] [--tgt_len TGT_LEN]
               [--ext_len EXT_LEN] [--mem_len MEM_LEN] [--clamp_len CLAMP_LEN]
               [--cuda] [--model MODEL] [--fp16] [--log_all_ranks]
               [--same_length] [--target_perplexity TARGET_PERPLEXITY]
               [--target_throughput TARGET_THROUGHPUT] [--save_data]
               [--repeat REPEAT] [--max_size MAX_SIZE]
               [--percentiles PERCENTILES [PERCENTILES ...]]
               [--save_torchscript SAVE_TORCHSCRIPT]
               [--load_torchscript LOAD_TORCHSCRIPT] [--local_rank LOCAL_RANK]
```


### Getting the data

The Transformer-XL base model was trained on the
[WikiText-103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)
dataset. The WikiText-103 dataset is a collection of over 100 million tokens
extracted from the set of verified
[Good](https://en.wikipedia.org/wiki/Wikipedia:Good_articles) and
[Featured](https://en.wikipedia.org/wiki/Wikipedia:Featured_articles) articles
on Wikipedia.

This repository contains the `getdata.sh` download script which
automatically downloads and extracts the training, validation and test
datasets. By default, data is downloaded to the `data` directory.

In order to test with other datasets, the script needs to be customized
accordingly.

#### Dataset guidelines

The WikiText-103 dataset was already pre-tokenized with word-level tokens. The
dataset features a large vocabulary of 267,735 tokens and retains the original
case, punctuation and numbers.

The `getdata.sh` script downloads the data, extracts the archive and renames
the training, validation and test set to `train.txt`, `valid.txt`, `test.txt`
respectively.

#### Multi-dataset

Using other datasets requires changes in the following files:

* `pytorch/train.py`:
  * name of the new dataset should be added to the `dataset` argument in the `parse_args()` function
  * desired values of cutoffs for adaptive softmax should be added in the
    `main()` function, after the section which builds train/valid/test data
    iterators
* `pytorch/data_utils.py`:
  * support for the new dataset needs to be added to the `Corpus` class: names
    of files containing training, validation and test data, options for the
    tokenizer, and dataset iterator

The current codebase supports training with word-level vocabulary
(automatically generated based on the provided dataset) and with BPE vocabulary
(using pre-built vocabulary from pretrained GPT2 model imported from
[github.com/huggingface/transformers](https://github.com/huggingface/transformers).

Additionally, using other datasets may require changes in some hyperparameters
(for example, batch size, learning rate, number of training steps,
configuration of learning rate scheduler). 

### Training process

The default training configuration can be launched by running the
`run_wt103_base.sh` script with the first argument set to `train`. By default,
the training results are saved to the `LM-TFM` directory; this can be
customized by setting the `--work_dir` parameter.

The training script launches a single node data-parallel training with a fixed
global batch size of 256, optionally with gradient accumulation to allow
training on configurations with less than 8 GPUs. Logs from the training are
automatically saved to the `LM-TFT/log.log` file.

**Command-line**

```
bash run_wt103_base.sh train <#GPUs> --vocab word --adaptive [--fp16] [--batch_chunk CHUNK]
```

Launches training of Transformer-XL base model on WikiText-103 dataset with word-based vocabulary and adaptive softmax using `<#GPUs>` GPUs.

The `--fp16` flag is optional, if it's specified, then the script launches mixed
precision training with Tensor Cores, if the flag is not present, then the
script launches FP32 training.

The `--batch_chunk CHUNK` parameter controls gradient accumulation. With gradient
accumulation the batch size is split into `CHUNK` chunks of equal size, the
training script executes the forward and backward pass using each chunk and
then executes the optimizer using accumulated gradients.

**Examples**

```
bash run_wt103_base.sh train 16 --fp16 --vocab word --adaptive --batch_chunk 1
```

Launches mixed precision training of Transformer-XL base model on WikiText-103
using 16 GPUs. Batch size per GPU is equal to the default global batch size of 256
divided by the product of the number of GPUs times the number of chunks, in this
case batch size per GPU is equal to `256 / (16 * 1) = 16`.

```
bash run_wt103_base.sh train 8 --vocab word --adaptive --batch_chunk 2
```

Launches a FP32 training using 8 GPUs, the batch size per GPU is equal to 16
(`--batch_chunk` was set to `2` because a local batch size of 32 runs out
of memory on a DGX-1 with Tesla V100 16G in FP32 training).

A summary of the training progress is printed after every 10 training
iterations; this can be customized by setting the `--log_interval` parameter.


The summary is printed in the following format:

```
| epoch  18 step    36000 | batches    283 / 2101 | lr 1.220e-03 | ms/batch 185.1 | tok/s  265585 | loss  3.12 | ppl     22.71
```

which contains information about a current training epoch, current training
step, number of batches processed within the current epoch, current learning
rate, execution time in milliseconds per batch, throughput in tokens per
second, current training loss and training perplexity.

The script saves two checkpoints: `checkpoint_best.pt` which contains the model
corresponding to the lowest value of the validation loss and
`checkpoint_last.pt` which contains the model corresponding to the last
execution of the validation step. By default, the validation is executed every
5000 training steps, this can be customized by setting the `--eval_interval`
parameter. The summary of results on the validation dataset is printed in the
following format:

```
| Eval   7 at step    35000 | time:  1.37s | valid loss  3.14 | valid ppl    23.132
```

which contains information about the current epoch, current training step, time
needed to execute the validation, current validation loss and validation
perplexity.

### Inference process

Inference can be run by launching the `run_wt103_base.sh` script with the first
argument set to `eval`. Running inference requires a pre-trained model
checkpoint.

The script supports single node multi-GPU inference, each batch is split
equally among all GPUs running the inference and the loss is averaged over the
global batch.

**Command-line**

```
bash run_wt103_base.sh eval <#GPUs> --model <PATH TO THE CHECKPOINT> [--fp16] [--type {pytorch, torchscript}]
```

The `--fp16` flag is optional, if it's specified, then the script launches inference
with Tensor Cores, if the flag is not present, then the script launches FP32
inference.

The `--type` flag selects between pure Python pytorch execution and TorchScript execution.

Supported values for `<#GPUs>` are: 1, 2, 4, 8, 16.

**Examples**

```
bash run_wt103_base.sh eval 8 --model LM-TFM/checkpoint_best.pt --fp16 --type torchscript
```

Launches TorchScript mixed precision inference on 8 GPUs using a checkpoint loaded from
`LM-TFM/checkpoint_best.pt`.

```
bash run_wt103_base.sh eval 1 --model LM-TFM/checkpoint_best.pt --type pytorch
```

Launches pure Python FP32 inference on a single GPU using a checkpoint loaded from
`LM-TFM/checkpoint_best.pt`.

After the execution, the script prints a summary in the following format:

```
Evaluating with math fp16 type torchscript bsz 16 tgt_len 64 ext_len 0 mem_len 640 clamp_len 400
Time : 5.29s, 22.05ms/segment
====================================================================================================
| test loss  3.15 | test ppl    23.304
====================================================================================================
```

which contains information about runtime parameters, execution time, loss and
perplexity on the test dataset.

## Performance

### Benchmarking

The following section shows how to run benchmarks measuring the model
performance in training and inference modes.

#### Training performance benchmark

To benchmark the training performance on a specific global batch size `<BS>`,
with a specific number of GPUs `<#GPUs>` for a specific number of training
iterations `<ITER>` run:

```
bash run_wt103_base.sh train <#GPUs> --batch_size <BS> --max_step <ITER> --vocab word --adaptive --log_interval 1 --debug [--fp16] [--batch_chunk CHUNK]
```

It's recommended to launch at least 500 training steps to get a reliable
estimate of training performance. For more information about the available
options, refer to the [Training process](#training-process) section.

The training script prints information in the following format:

```
(...)
| epoch   1 step      499 | batches    499 / 16802 | lr 4.990e-03 | ms/batch 219.9 | tok/s   27947 | loss  6.43 | ppl    620.80
| epoch   1 step      500 | batches    500 / 16802 | lr 5.000e-03 | ms/batch 221.4 | tok/s   27747 | loss  6.42 | ppl    611.70
-------------------------------------------------------------------------------
(...)
Training time: 1.81 minutes
Training throughput: 28508.91 tok/s
```

The last two lines contain information on the total training time and on the
average training throughput measured in tokens per second.

#### Inference performance benchmark

The inference performance and accuracy benchmarks require a checkpoint from a
trained model.

To benchmark the inference performance on a specific global batch size `<BS>`
with a specific number of GPUs `<#GPUs>`, run:

```
bash run_wt103_base.sh eval <#GPUs> --model <CHECKPOINT> --batch_size <BS> --save_data [--fp16] [--type {pytorch, torchscript}]
```

The inference script prints information in the following format:

```
Evaluating with math fp16 type torchscript bsz 16 tgt_len 64 ext_len 0 mem_len 640 clamp_len 400
Time : 5.25s, 21.88ms/segment
====================================================================================================
| test loss  3.15 | test ppl    23.304
====================================================================================================
Throughput Avg: 46316.64 tok/s
Latency Avg: 22.09 ms
Latency 90%: 22.22 ms
Latency 95%: 22.25 ms
Latency 99%: 22.37 ms
====================================================================================================
```

The output contains information on the achieved test loss and test perplexity,
average inference throughput (measured in tokens per second), average inference
latency and latency at 90%, 95% and 99% confidence intervals (measured in
milliseconds).

The `scripts/inference_benchmark.sh` benchmarking script is provided for
convenience, it automatically launches FP32 and FP16 inference for various
batch sizes.

### Results

The following sections provide details on how we achieved our performance and
accuracy in training and inference.

#### Training accuracy results

##### Training accuracy: NVIDIA DGX-1 (8x V100 16G)

Our results were obtained by running the `pytorch/run_wt103_base.sh`
training script in the in the pytorch-19.09-py3 NGC container on NVIDIA DGX-1
with 8x V100 16G GPUs.

|**GPUs**|**Batch Size / GPU**|**Accuracy - FP32 (perplexity)**|**Accuracy - Mixed precision (perplexity)**|**Time to Train - FP32 (minutes)**|**Time to Train - Mixed precision (minutes)**|**Time to Train Speedup (FP32 to Mixed precision)**|
|-------:|-------------------:|-------------------------------:|------------------------------------------:|---------------------------------:|--------------------------------------------:|--------------------------------------------------:|
| 1 | 16 | 23.24 | 23.42 | 2542.0 | 1037.8 | 2.45 |
| 8 | 16 | 23.38 | 23.44 | 366.9  | 168.9  | 2.17 |
| 1 | 32 | N/A   | 23.38 | N/A    | 894.3  | 2.84 |
| 8 | 32 | N/A   | 23.38 | N/A    | 140.7  | 2.61 |

##### Training accuracy: NVIDIA DGX-2 (16x V100 32G)

Our results were obtained by running the `pytorch/run_wt103_base.sh`
training script in the in the pytorch-19.09-py3 NGC container on NVIDIA DGX-2
with 16x V100 32G GPUs.

|**GPUs**|**Batch Size / GPU**|**Accuracy - FP32 (perplexity)**|**Accuracy - Mixed precision (perplexity)**|**Time to Train - FP32 (minutes)**|**Time to Train - Mixed precision (minutes)**|**Time to Train Speedup (FP32 to Mixed precision)**|
|-------:|-------------------:|-------------------------------:|------------------------------------------:|---------------------------------:|--------------------------------------------:|--------------------------------------------------:|
| 16 | 16 | 23.36 | 23.32 | 184.4 | 91.2 | 2.02 |

![TrainingLoss](pytorch/img/training_loss.png)

##### Training stability test

The Transformer-XL model was trained for 40000 training steps, starting from 20
different initial random seeds. After every 5000 training steps, the model was
evaluated on the validation dataset and validation perplexity was recorded. The
training was performed in the pytorch-19.09-py3 NGC container on NVIDIA
DGX-1 with 8x V100 16G GPUs. The following table summarizes the perplexity
on our validation dataset.

|**Training step**|**Average**|**Standard deviation**|**Minimum**|**Maximum**|**Median**|
|----------------:|----------:|---------------------:|----------:|----------:|---------:|
| 5000  | 42.58 | 0.28639 | 41.98 | 43.11 | 42.62 |
| 10000 | 32.39 | 0.19765 | 32.09 | 32.78 | 32.41 |
| 15000 | 28.49 | 0.15000 | 28.28 | 28.78 | 28.49 |
| 20000 | 26.22 | 0.11862 | 26.06 | 26.52 | 26.22 |
| 25000 | 24.73 | 0.11190 | 24.45 | 24.88 | 24.74 |
| 30000 | 23.88 | 0.10489 | 23.67 | 24.04 | 23.87 |
| 35000 | 23.31 | 0.10010 | 23.09 | 23.45 | 23.33 |
| 40000 | 23.10 | 0.09857 | 22.86 | 23.23 | 23.11 |

After training, the models were evaluated on the test dataset. The following
table summarizes the final perplexity on the test set.

|**Average**|**Standard deviation**|**Minimum**|**Maximum**|**Median**|
|----------:|---------------------:|----------:|----------:|---------:|
| 23.39 | 0.06817 | 23.26 | 23.51 | 23.39 |

#### Training performance results

##### Training performance: NVIDIA DGX-1 (8x V100 16G)

Our results were obtained by running the `pytorch/run_wt103_base.sh`
training script in the pytorch-19.09-py3 NGC container on NVIDIA DGX-1 with 8x
V100 16G GPUs. Performance numbers (in tokens per second) were averaged 500
training iterations.

|**GPUs**|**Batch Size / GPU**|**Throughput - FP32 (tok/s)**|**Throughput - Mixed precision (tok/s)**|**Throughput speedup (FP32 to Mixed precision)**|**Weak Scaling - FP32**|**Weak Scaling - Mixed precision**|
|-------:|-------------------:|----------------------------:|----------------------------------------:|-----------------------------------------------:|------------------------:|-----------------------------------:|
| 1 | 16 | 11,499.8 | 24,028.9  | 2.089 | 1.000 | 1.000 |
| 2 | 16 | 19,574.0 | 40,001.7  | 2.044 | 1.702 | 1.665 |
| 4 | 16 | 42,184.9 | 85,391.2  | 2.024 | 3.668 | 3.554 |
| 8 | 16 | 84,803.6 | 159,122.2 | 1.876 | 7.374 | 6.622 |
| 1 | 32 | N/A      | 31,072.4  | 2.702 | N/A   | 1.000 |
| 2 | 32 | N/A      | 55,534.1  | 2.837 | N/A   | 1.787 |
| 4 | 32 | N/A      | 117,200.6 | 2.778 | N/A   | 3.772 |
| 8 | 32 | N/A      | 234,437.3 | 2.764 | N/A   | 7.545 |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

##### Training performance: NVIDIA DGX-2 (16x V100 32G)

Our results were obtained by running the `pytorch/run_wt103_base.sh`
training script in the pytorch-19.09-py3 NGC container on NVIDIA DGX-2 with 16x
V100 32G GPUs. Performance numbers (in tokens per second) were averaged 500
training iterations.

|**GPUs**|**Batch Size / GPU**|**Throughput - FP32 (tok/s)**|**Throughput - Mixed precision (tok/s)**|**Throughput speedup (FP32 to Mixed precision)**|**Weak Scaling - FP32**|**Weak Scaling - Mixed precision**|
|-------:|-------------------:|----------------------------:|---------------------------------------:|-----------------------------------------------:|----------------------:|---------------------------------:|
| 1  | 16 | 12,204.0  | 25,337.5  | 2.076 | 1.000  | 1.000  |
| 2  | 16 | 22,995.2  | 46,605.1  | 2.027 | 1.884  | 1.839  |
| 4  | 16 | 45,321.1  | 91,537.4  | 2.020 | 3.714  | 3.613  |
| 8  | 16 | 89,427.3  | 179,920.5 | 2.012 | 7.328  | 7.101  |
| 16 | 16 | 177,245.0 | 357,343.6 | 2.016 | 14.524 | 14.103 |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

#### Inference performance results

##### Inference performance: NVIDIA DGX-1 (1x V100 16G)

Our results were obtained by running the
`pytorch/scripts/inference_benchmark.sh` inferencing benchmarking script in the
pytorch-19.09-py3 NGC container on NVIDIA DGX-1 with 1x V100 16G GPU.

The command to launch the inference performance benchmark is provided in the
[Inference performance benchmark](#inference-performance-benchmark) section.

**FP16, pure Python**

|**Batch size**|**Sequence length**|**Memory length**|**Throughput Avg (tok/s)**|**Latency Avg (ms)**|**Latency 90% (ms)**|**Latency 95% (ms)**|**Latency 99% (ms)**|
|-------------:|------------------:|----------------:|-------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
| 1  | 64 | 640 | 3,346.3  | 19.13 | 19.47 | 19.64 | 20.67 |
| 2  | 64 | 640 | 6,486.5  | 19.74 | 20.04 | 20.20 | 21.52 |
| 4  | 64 | 640 | 13,007.2 | 19.68 | 19.93 | 20.12 | 21.45 |
| 8  | 64 | 640 | 24,783.1 | 20.65 | 20.95 | 21.22 | 22.78 |
| 16 | 64 | 640 | 42,777.7 | 23.93 | 24.09 | 24.41 | 25.82 |
| 32 | 64 | 640 | 52,961.6 | 38.64 | 38.86 | 39.77 | 41.25 |

**FP16, TorchScript**

|**Batch size**|**Sequence length**|**Memory length**|**Throughput Avg (tok/s)**|**Latency Avg (ms)**|**Latency 90% (ms)**|**Latency 95% (ms)**|**Latency 99% (ms)**|
|-------------:|------------------:|----------------:|-------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
| 1  | 64 | 640 | 5,117.4  | 12.52 | 12.76 | 12.87 | 13.39 |
| 2  | 64 | 640 | 9,703.5  | 13.20 | 13.39 | 13.52 | 14.78 |
| 4  | 64 | 640 | 18,259.5 | 14.02 | 14.22 | 14.38 | 15.72 |
| 8  | 64 | 640 | 35,758.7 | 14.32 | 14.52 | 14.66 | 16.04 |
| 16 | 64 | 640 | 50,159.9 | 20.41 | 20.44 | 20.69 | 21.98 |
| 32 | 64 | 640 | 57,223.4 | 35.76 | 35.95 | 36.25 | 37.57 |

**FP32, pure Python**

|**Batch size**|**Sequence length**|**Memory length**|**Throughput Avg (tok/s)**|**Latency Avg (ms)**|**Latency 90% (ms)**|**Latency 95% (ms)**|**Latency 99% (ms)**|
|-------------:|------------------:|----------------:|-------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
| 1  | 64 | 640 | 3,216.4  | 19.91  | 20.31  | 20.45  | 21.61  |
| 2  | 64 | 640 | 6,314.0  | 20.28  | 20.63  | 20.80  | 21.95  |
| 4  | 64 | 640 | 10,991.5 | 23.28  | 23.56  | 23.74  | 25.20  |
| 8  | 64 | 640 | 16,398.5 | 31.20  | 31.57  | 31.83  | 33.38  |
| 16 | 64 | 640 | 18,845.5 | 54.29  | 54.71  | 54.89  | 56.05  |
| 32 | 64 | 640 | 19,209.5 | 106.51 | 107.45 | 107.69 | 108.81 |

**FP32, TorchScript**

|**Batch size**|**Sequence length**|**Memory length**|**Throughput Avg (tok/s)**|**Latency Avg (ms)**|**Latency 90% (ms)**|**Latency 95% (ms)**|**Latency 99% (ms)**|
|-------------:|------------------:|----------------:|-------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
| 1  | 64 | 640 | 4,915.5  | 13.03  | 13.37  | 13.50  | 14.00  |
| 2  | 64 | 640 | 8,644.5  | 14.81  | 15.10  | 15.19  | 16.39  |
| 4  | 64 | 640 | 13,480.2 | 18.98  | 19.20  | 19.29  | 20.55  |
| 8  | 64 | 640 | 17,075.5 | 29.96  | 30.18  | 30.26  | 31.64  |
| 16 | 64 | 640 | 19,201.7 | 53.29  | 53.74  | 53.95  | 54.78  |
| 32 | 64 | 640 | 19,724.4 | 103.73 | 104.40 | 104.59 | 105.73 |


To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

##### Inference performance: NVIDIA T4

Our results were obtained by running the
`pytorch/scripts/inference_benchmark.sh` inferencing benchmarking script in the
pytorch-19.09-py3 NGC container on NVIDIA T4.

The command to launch the inference performance benchmark is provided in the
[Inference performance benchmark](#inference-performance-benchmark) section.

**FP16, pure Python**

|**Batch size**|**Sequence length**|**Memory length**|**Throughput Avg (tok/s)**|**Latency Avg (ms)**|**Latency 90% (ms)**|**Latency 95% (ms)**|**Latency 99% (ms)**|
|-------------:|------------------:|----------------:|-------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
| 1  | 64 | 640 | 4,067.0  | 15.75  | 16.26  | 16.43  | 16.74  |
| 2  | 64 | 640 | 7,559.5  | 16.94  | 17.37  | 17.55  | 17.93  |
| 4  | 64 | 640 | 13,203.1 | 19.38  | 19.91  | 20.02  | 20.46  |
| 8  | 64 | 640 | 16,101.8 | 31.78  | 32.45  | 32.53  | 33.00  |
| 16 | 64 | 640 | 17,375.8 | 58.89  | 59.87  | 60.23  | 60.63  |
| 32 | 64 | 640 | 17,946.2 | 114.03 | 115.33 | 116.17 | 119.87 |

**FP16, TorchScript**

|**Batch size**|**Sequence length**|**Memory length**|**Throughput Avg (tok/s)**|**Latency Avg (ms)**|**Latency 90% (ms)**|**Latency 95% (ms)**|**Latency 99% (ms)**|
|-------------:|------------------:|----------------:|-------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
| 1  | 64 | 640 | 5,834.6  | 10.99  | 11.45  | 11.61  | 11.94  |
| 2  | 64 | 640 | 11,167.8 | 11.47  | 11.87  | 12.08  | 12.80  |
| 4  | 64 | 640 | 14,890.8 | 17.19  | 17.70  | 17.89  | 18.19  |
| 8  | 64 | 640 | 16,862.5 | 30.35  | 31.09  | 31.32  | 31.81  |
| 16 | 64 | 640 | 18,281.2 | 55.98  | 56.82  | 57.00  | 58.52  |
| 32 | 64 | 640 | 18,912.0 | 108.21 | 109.54 | 110.20 | 113.80 |

**FP32, pure Python**

|**Batch size**|**Sequence length**|**Memory length**|**Throughput Avg (tok/s)**|**Latency Avg (ms)**|**Latency 90% (ms)**|**Latency 95% (ms)**|**Latency 99% (ms)**|
|-------------:|------------------:|----------------:|-------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
| 1  | 64 | 640 | 3,457.2 | 18.56  | 19.40  | 19.74  | 20.35  |
| 2  | 64 | 640 | 4,746.1 | 26.98  | 27.84  | 28.12  | 28.56  |
| 4  | 64 | 640 | 5,687.6 | 44.98  | 45.93  | 46.35  | 47.24  |
| 8  | 64 | 640 | 6,223.5 | 82.21  | 83.37  | 83.72  | 84.22  |
| 16 | 64 | 640 | 6,522.6 | 156.87 | 159.63 | 160.43 | 161.13 |
| 32 | 64 | 640 | 6,608.2 | 309.63 | 313.21 | 314.07 | 315.32 |

**FP32, TorchScript**

|**Batch size**|**Sequence length**|**Memory length**|**Throughput Avg (tok/s)**|**Latency Avg (ms)**|**Latency 90% (ms)**|**Latency 95% (ms)**|**Latency 99% (ms)**|
|-------------:|------------------:|----------------:|-------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
| 1  | 64 | 640 | 3,859.7 | 16.64  | 17.71  | 17.98  | 18.53  |
| 2  | 64 | 640 | 4,823.6 | 26.55  | 27.41  | 27.70  | 28.05  |
| 4  | 64 | 640 | 5,790.0 | 44.18  | 45.07  | 45.30  | 45.91  |
| 8  | 64 | 640 | 6,306.4 | 81.12  | 82.26  | 82.54  | 83.11  |
| 16 | 64 | 640 | 6,599.5 | 155.04 | 157.54 | 158.15 | 159.88 |
| 32 | 64 | 640 | 6,707.0 | 305.06 | 307.94 | 308.54 | 309.44 |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

## Release notes

### Changelog

* October 2019
  * Initial release
    *  Support for FP32 and mixed precision training on NVIDIA DGX-1, NVIDIA
       DGX-2 and inference on NVIDIA Tesla V100 16G and NVIDIA T4

### Known issues
There are no known issues with this model.

# GNMT v2 For PyTorch

This repository provides a script and recipe to train the GNMT v2 model to
achieve state of the art accuracy, and is tested and maintained by NVIDIA.

## Table Of Contents

<!-- TOC GFM -->

* [Model overview](#model-overview)
  * [Model architecture](#model-architecture)
  * [Default configuration](#default-configuration)
  * [Feature support matrix](#feature-support-matrix)
    * [Features](#features)
  * [Mixed precision training](#mixed-precision-training)
    * [Enabling mixed precision](#enabling-mixed-precision)
    * [Enabling TF32](#enabling-tf32)
* [Setup](#setup)
  * [Requirements](#requirements)
* [Quick Start Guide](#quick-start-guide)
* [Advanced](#advanced)
  * [Scripts and sample code](#scripts-and-sample-code)
  * [Parameters](#parameters)
  * [Command-line options](#command-line-options)
  * [Getting the data](#getting-the-data)
    * [Dataset guidelines](#dataset-guidelines)
  * [Training process](#training-process)
  * [Inference process](#inference-process)
* [Performance](#performance)
  * [Benchmarking](#benchmarking)
    * [Training performance benchmark](#training-performance-benchmark)
    * [Inference performance benchmark](#inference-performance-benchmark)
  * [Results](#results)
    * [Training accuracy results](#training-accuracy-results)
      * [Training accuracy: NVIDIA DGX A100 (8x A100 40GB)](#training-accuracy-nvidia-dgx-a100-8x-a100-40gb)
      * [Training accuracy: NVIDIA DGX-1 (8x V100 16GB)](#training-accuracy-nvidia-dgx-1-8x-v100-16gb)
      * [Training accuracy: NVIDIA DGX-2H (16x V100 32GB)](#training-accuracy-nvidia-dgx-2h-16x-v100-32gb)
      * [Training stability test](#training-stability-test)
    * [Training throughput results](#training-throughput-results)
      * [Training throughput: NVIDIA DGX A100 (8x A100 40GB)](#training-throughput-nvidia-dgx-a100-8x-a100-40gb)
      * [Training throughput: NVIDIA DGX-1 (8x V100 16GB)](#training-throughput-nvidia-dgx-1-8x-v100-16gb)
      * [Training throughput: NVIDIA DGX-2H (16x V100 32GB)](#training-throughput-nvidia-dgx-2h-16x-v100-32gb)
    * [Inference accuracy results](#inference-accuracy-results)
      * [Inference accuracy: NVIDIA A100 40GB](#inference-accuracy-nvidia-a100-40gb)
      * [Inference accuracy: NVIDIA Tesla V100 16GB](#inference-accuracy-nvidia-tesla-v100-16gb)
      * [Inference accuracy: NVIDIA T4](#inference-accuracy-nvidia-t4)
    * [Inference throughput results](#inference-throughput-results)
      * [Inference throughput: NVIDIA A100 40GB](#inference-throughput-nvidia-a100-40gb)
      * [Inference throughput: NVIDIA T4](#inference-throughput-nvidia-t4)
    * [Inference latency results](#inference-latency-results)
      * [Inference latency: NVIDIA A100 40GB](#inference-latency-nvidia-a100-40gb)
      * [Inference latency: NVIDIA T4](#inference-latency-nvidia-t4)
* [Release notes](#release-notes)
  * [Changelog](#changelog)
  * [Known issues](#known-issues)

<!-- /TOC -->

## Model overview
The GNMT v2 model is similar to the one discussed in the [Google's Neural
Machine Translation System: Bridging the Gap between Human and Machine
Translation](https://arxiv.org/abs/1609.08144) paper.

The most important difference between the two models is in the attention
mechanism. In our model, the output from the first LSTM layer of the decoder
goes into the attention module, then the re-weighted context is concatenated
with inputs to all subsequent LSTM layers in the decoder at the current
time step.

The same attention mechanism is also implemented in the default GNMT-like
models from [TensorFlow Neural Machine Translation
Tutorial](https://github.com/tensorflow/nmt) and [NVIDIA OpenSeq2Seq
Toolkit](https://github.com/NVIDIA/OpenSeq2Seq).

### Model architecture
![ModelArchitecture](./img/diagram.png)

### Default configuration

The following features were implemented in this model:

* general:
  * encoder and decoder are using shared embeddings
  * data-parallel multi-GPU training
  * dynamic loss scaling with backoff for Tensor Cores (mixed precision)
    training
  * trained with label smoothing loss (smoothing factor 0.1)
* encoder:
  * 4-layer LSTM, hidden size 1024, first layer is bidirectional, the rest are
    unidirectional
  * with residual connections starting from 3rd layer
  * uses standard PyTorch nn.LSTM layer
  * dropout is applied on input to all LSTM layers, probability of dropout is
    set to 0.2
  * hidden state of LSTM layers is initialized with zeros
  * weights and bias of LSTM layers is initialized with uniform(-0.1,0.1)
    distribution
* decoder:
  * 4-layer unidirectional LSTM with hidden size 1024 and fully-connected
    classifier
  * with residual connections starting from 3rd layer
  * uses standard PyTorch nn.LSTM layer
  * dropout is applied on input to all LSTM layers, probability of dropout is
    set to 0.2
  * hidden state of LSTM layers is initialized with zeros
  * weights and bias of LSTM layers is initialized with uniform(-0.1,0.1)
    distribution
  * weights and bias of fully-connected classifier is initialized with
    uniform(-0.1,0.1) distribution
* attention:
  * normalized Bahdanau attention
  * output from first LSTM layer of decoder goes into attention, then
    re-weighted context is concatenated with the input to all subsequent LSTM
    layers of the decoder at the current timestep
  * linear transform of keys and queries is initialized with uniform(-0.1,
    0.1), normalization scalar is initialized with 1.0/sqrt(1024),
    normalization bias is initialized with zero
* inference:
  * beam search with default beam size of 5
  * with coverage penalty and length normalization, coverage penalty factor is
    set to 0.1, length normalization factor is set to 0.6 and length
    normalization constant is set to 5.0
  * de-tokenized BLEU computed by
    [SacreBLEU](https://github.com/mjpost/sacrebleu)
  * [motivation](https://github.com/mjpost/sacrebleu#motivation) for choosing
    SacreBLEU

When comparing the BLEU score, there are various tokenization approaches and
BLEU calculation methodologies; therefore, ensure you align similar metrics.

Code from this repository can be used to train a larger, 8-layer GNMT v2 model.
Our experiments show that a 4-layer model is significantly faster to train and
yields comparable accuracy on the public [WMT16
English-German](http://www.statmt.org/wmt16/translation-task.html) dataset. The
number of LSTM layers is controlled by the `--num-layers` parameter in the
`train.py` training script.

### Feature support matrix
The following features are supported by this model.

| **Feature** | **GNMT v2** |
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
Cores](https://developer.nvidia.com/tensor-cores) in Volta, and following with
both the Turing and Ampere architectures, significant training speedups are
experienced by switching to mixed precision -- up to 3x overall speedup on the
most arithmetically intense model architectures. Using mixed precision training
previously required two steps:

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

The following steps were needed to enable mixed precision training in GNMT:

* Import AMP from APEX (file: `seq2seq/train/trainer.py`):

```
from apex import amp
```

* Initialize AMP and wrap the model and the optimizer (file:
  `seq2seq/train/trainer.py`, class: `Seq2SeqTrainer`):

```
self.model, self.optimizer = amp.initialize(
    self.model,
    self.optimizer,
    cast_model_outputs=torch.float16,
    keep_batchnorm_fp32=False,
    opt_level='O2')
```

* Apply `scale_loss` context manager (file: `seq2seq/train/fp_optimizers.py`,
  class: `AMPOptimizer`):

```
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
```

* Apply gradient clipping on single precision master weights (file:
  `seq2seq/train/fp_optimizers.py`, class: `AMPOptimizer`):

```
if self.grad_clip != float('inf'):
    clip_grad_norm_(amp.master_params(optimizer), self.grad_clip)
```

#### Enabling TF32
TensorFloat-32 (TF32) is the new math mode in [NVIDIA
A100](https://www.nvidia.com/en-us/data-center/a100/) GPUs for handling the
matrix math also called tensor operations. TF32 running on Tensor Cores in A100
GPUs can provide up to 10x speedups compared to single-precision floating-point
math (FP32) on Volta GPUs.

TF32 Tensor Cores can speed up networks using FP32, typically with no loss of
accuracy. It is more robust than FP16 for models which require high dynamic
range for weights or activations.

For more information, refer to the [TensorFloat-32 in the A100 GPU Accelerates
AI Training, HPC up to
20x](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/)
blog post.

TF32 is supported in the NVIDIA Ampere GPU architecture and is enabled by
default.

## Setup

The following section lists the requirements in order to start training the
GNMT v2 model.

### Requirements

This repository contains `Dockerfile` which extends the PyTorch NGC container
and encapsulates some dependencies. Aside from these dependencies, ensure you
have the following components:
* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch 20.06-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch)
* GPU architecture:
  * [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
  * [NVIDIA Turing](https://www.nvidia.com/en-us/geforce/turing/)
  * [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)

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
To train your model using mixed or TF32 precision with Tensor Cores or using
FP32, perform the following steps using the default parameters of the GNMT v2
model on the WMT16 English German dataset. For the specifics concerning
training and inference, see the [Advanced](#advanced) section.

**1. Clone the repository.**

```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/PyTorch/Translation/GNMT
```

**2. Build the GNMT v2 Docker container.**

```
bash scripts/docker/build.sh
```

**3. Start an interactive session in the container to run training/inference.**

```
bash scripts/docker/interactive.sh
```

**4. Download and preprocess the dataset.**

Data will be downloaded to the `data` directory (on the host). The `data`
directory is mounted to the `/workspace/gnmt/data` location in the Docker
container.

```
bash scripts/wmt16_en_de.sh
```

**5. Start training.**

The training script saves only one checkpoint with the lowest value of the loss
function on the validation dataset. All results and logs are saved to the
`gnmt` directory (on the host) or to the `/workspace/gnmt/gnmt` directory
(in the container). By default, the `train.py` script will launch mixed
precision training with Tensor Cores. You can change this behavior by setting:
* the `--math fp32` flag to launch single precision training (for NVIDIA Volta
  and NVIDIA Turing architectures) or
* the `--math tf32` flag to launch TF32 training with Tensor Cores (for NVIDIA
  Ampere architecture) 

for the `train.py` training script.

To launch mixed precision training on 1, 4 or 8 GPUs, run:

```
python3 -m torch.distributed.launch --nproc_per_node=<#GPUs> train.py --seed 2 --train-global-batch-size 1024
```

To launch mixed precision training on 16 GPUs, run:

```
python3 -m torch.distributed.launch --nproc_per_node=16 train.py --seed 2 --train-global-batch-size 2048
```

By default, the training script will launch training with batch size 128 per
GPU. If `--train-global-batch-size` is specified and larger than 128 times the
number of GPUs available for the training then the training script will
accumulate gradients over consecutive iterations and then perform the weight
update. For example, 1 GPU training with `--train-global-batch-size 1024` will
accumulate gradients over 8 iterations before doing the weight update with
accumulated gradients.

**6. Start evaluation.**

The training process automatically runs evaluation and outputs the BLEU score
after each training epoch. Additionally, after the training is done, you can
manually run inference on the test dataset with the checkpoint saved during the
training.

To launch FP16 inference on the `newstest2014.en` test set, run:

```
python3 translate.py \
  --input data/wmt16_de_en/newstest2014.en \
  --reference data/wmt16_de_en/newstest2014.de \
  --output /tmp/output \
  --model gnmt/model_best.pth
```

The script will load the checkpoint specified by the `--model` option, then it
will launch inference on the file specified by the `--input` option, and
compute BLEU score against the reference translation specified by the
`--reference` option. Outputs will be stored to the location specified by the
`--output` option.


Additionally, one can pass the input text directly from the command-line:

```
python3 translate.py \
  --input-text "The quick brown fox jumps over the lazy dog" \
  --model gnmt/model_best.pth
```

Translated output will be printed to the console:

```
(...)
0: Translated output:
Der schnelle braune Fuchs springt über den faulen Hund
```

By default, the `translate.py` script will launch FP16 inference with Tensor
Cores. You can change this behavior by setting:
* the `--math fp32` flag to launch single precision inference (for NVIDIA Volta
  and NVIDIA Turing architectures) or
* the `--math tf32` flag to launch TF32 inference with Tensor Cores (for NVIDIA
  Ampere architecture)

for the `translate.py` inference script.

## Advanced
The following sections provide greater details of the dataset, running training
and inference, and the training results.

### Scripts and sample code
In the `root` directory, the most important files are:

* `train.py`: serves as the entry point to launch the training
* `translate.py`: serves as the entry point to launch inference
* `Dockerfile`: container with the basic set of dependencies to run GNMT v2
* `requirements.txt`: set of extra requirements for running GNMT v2

The `seq2seq/model` directory contains the implementation of GNMT v2 building
blocks:

* `attention.py`: implementation of normalized Bahdanau attention
* `encoder.py`: implementation of recurrent encoder
* `decoder.py`: implementation of recurrent decoder with attention
* `seq2seq_base.py`: base class for seq2seq models
* `gnmt.py`: implementation of GNMT v2 model

The `seq2seq/train` directory encapsulates the necessary tools to execute
training:

* `trainer.py`: implementation of training loop
* `smoothing.py`: implementation of cross-entropy with label smoothing
* `lr_scheduler.py`: implementation of exponential learning rate warmup and
  step decay
* `fp_optimizers.py`: implementation of optimizers for various floating point
  precisions

The `seq2seq/inference` directory contains scripts required to run inference:

* `beam_search.py`: implementation of beam search with length normalization and
  length penalty
* `translator.py`: implementation of auto-regressive inference

The `seq2seq/data` directory contains implementation of components needed for
data loading:

* `dataset.py`: implementation of text datasets
* `sampler.py`: implementation of batch samplers with bucketing by sequence
  length
* `tokenizer.py`: implementation of tokenizer (maps integer vocabulary indices
  to text)

### Parameters
Training


The complete list of available parameters for the `train.py` training script
contains:

```
dataset setup:
  --dataset-dir DATASET_DIR
                        path to the directory with training/test data
                        (default: data/wmt16_de_en)
  --src-lang SRC_LANG   source language (default: en)
  --tgt-lang TGT_LANG   target language (default: de)
  --vocab VOCAB         path to the vocabulary file (relative to DATASET_DIR
                        directory) (default: vocab.bpe.32000)
  -bpe BPE_CODES, --bpe-codes BPE_CODES
                        path to the file with bpe codes (relative to
                        DATASET_DIR directory) (default: bpe.32000)
  --train-src TRAIN_SRC
                        path to the training source data file (relative to
                        DATASET_DIR directory) (default:
                        train.tok.clean.bpe.32000.en)
  --train-tgt TRAIN_TGT
                        path to the training target data file (relative to
                        DATASET_DIR directory) (default:
                        train.tok.clean.bpe.32000.de)
  --val-src VAL_SRC     path to the validation source data file (relative to
                        DATASET_DIR directory) (default:
                        newstest_dev.tok.clean.bpe.32000.en)
  --val-tgt VAL_TGT     path to the validation target data file (relative to
                        DATASET_DIR directory) (default:
                        newstest_dev.tok.clean.bpe.32000.de)
  --test-src TEST_SRC   path to the test source data file (relative to
                        DATASET_DIR directory) (default:
                        newstest2014.tok.bpe.32000.en)
  --test-tgt TEST_TGT   path to the test target data file (relative to
                        DATASET_DIR directory) (default: newstest2014.de)
  --train-max-size TRAIN_MAX_SIZE
                        use at most TRAIN_MAX_SIZE elements from training
                        dataset (useful for benchmarking), by default uses
                        entire dataset (default: None)

results setup:
  --save-dir SAVE_DIR   path to directory with results, it will be
                        automatically created if it does not exist (default:
                        gnmt)
  --print-freq PRINT_FREQ
                        print log every PRINT_FREQ batches (default: 10)

model setup:
  --hidden-size HIDDEN_SIZE
                        hidden size of the model (default: 1024)
  --num-layers NUM_LAYERS
                        number of RNN layers in encoder and in decoder
                        (default: 4)
  --dropout DROPOUT     dropout applied to input of RNN cells (default: 0.2)
  --share-embedding     use shared embeddings for encoder and decoder (use '--
                        no-share-embedding' to disable) (default: True)
  --smoothing SMOOTHING
                        label smoothing, if equal to zero model will use
                        CrossEntropyLoss, if not zero model will be trained
                        with label smoothing loss (default: 0.1)

general setup:
  --math {fp16,fp32,tf32,manual_fp16}
                        precision (default: fp16)
  --seed SEED           master seed for random number generators, if "seed" is
                        undefined then the master seed will be sampled from
                        random.SystemRandom() (default: None)
  --prealloc-mode {off,once,always}
                        controls preallocation (default: always)
  --dllog-file DLLOG_FILE
                        Name of the DLLogger output file (default:
                        train_log.json)
  --eval                run validation and test after every epoch (use '--no-
                        eval' to disable) (default: True)
  --env                 print info about execution env (use '--no-env' to
                        disable) (default: True)
  --cuda                enables cuda (use '--no-cuda' to disable) (default:
                        True)
  --cudnn               enables cudnn (use '--no-cudnn' to disable) (default:
                        True)
  --log-all-ranks       enables logging from all distributed ranks, if
                        disabled then only logs from rank 0 are reported (use
                        '--no-log-all-ranks' to disable) (default: True)

training setup:
  --train-batch-size TRAIN_BATCH_SIZE
                        training batch size per worker (default: 128)
  --train-global-batch-size TRAIN_GLOBAL_BATCH_SIZE
                        global training batch size, this argument does not
                        have to be defined, if it is defined it will be used
                        to automatically compute train_iter_size using the
                        equation: train_iter_size = train_global_batch_size //
                        (train_batch_size * world_size) (default: None)
  --train-iter-size N   training iter size, training loop will accumulate
                        gradients over N iterations and execute optimizer
                        every N steps (default: 1)
  --epochs EPOCHS       max number of training epochs (default: 6)
  --grad-clip GRAD_CLIP
                        enables gradient clipping and sets maximum norm of
                        gradients (default: 5.0)
  --train-max-length TRAIN_MAX_LENGTH
                        maximum sequence length for training (including
                        special BOS and EOS tokens) (default: 50)
  --train-min-length TRAIN_MIN_LENGTH
                        minimum sequence length for training (including
                        special BOS and EOS tokens) (default: 0)
  --train-loader-workers TRAIN_LOADER_WORKERS
                        number of workers for training data loading (default:
                        2)
  --batching {random,sharding,bucketing}
                        select batching algorithm (default: bucketing)
  --shard-size SHARD_SIZE
                        shard size for "sharding" batching algorithm, in
                        multiples of global batch size (default: 80)
  --num-buckets NUM_BUCKETS
                        number of buckets for "bucketing" batching algorithm
                        (default: 5)

optimizer setup:
  --optimizer OPTIMIZER
                        training optimizer (default: Adam)
  --lr LR               learning rate (default: 0.002)
  --optimizer-extra OPTIMIZER_EXTRA
                        extra options for the optimizer (default: {})

mixed precision loss scaling setup:
  --init-scale INIT_SCALE
                        initial loss scale (default: 8192)
  --upscale-interval UPSCALE_INTERVAL
                        loss upscaling interval (default: 128)

learning rate scheduler setup:
  --warmup-steps WARMUP_STEPS
                        number of learning rate warmup iterations (default:
                        200)
  --remain-steps REMAIN_STEPS
                        starting iteration for learning rate decay (default:
                        0.666)
  --decay-interval DECAY_INTERVAL
                        interval between learning rate decay steps (default:
                        None)
  --decay-steps DECAY_STEPS
                        max number of learning rate decay steps (default: 4)
  --decay-factor DECAY_FACTOR
                        learning rate decay factor (default: 0.5)

validation setup:
  --val-batch-size VAL_BATCH_SIZE
                        batch size for validation (default: 64)
  --val-max-length VAL_MAX_LENGTH
                        maximum sequence length for validation (including
                        special BOS and EOS tokens) (default: 125)
  --val-min-length VAL_MIN_LENGTH
                        minimum sequence length for validation (including
                        special BOS and EOS tokens) (default: 0)
  --val-loader-workers VAL_LOADER_WORKERS
                        number of workers for validation data loading
                        (default: 0)

test setup:
  --test-batch-size TEST_BATCH_SIZE
                        batch size for test (default: 128)
  --test-max-length TEST_MAX_LENGTH
                        maximum sequence length for test (including special
                        BOS and EOS tokens) (default: 150)
  --test-min-length TEST_MIN_LENGTH
                        minimum sequence length for test (including special
                        BOS and EOS tokens) (default: 0)
  --beam-size BEAM_SIZE
                        beam size (default: 5)
  --len-norm-factor LEN_NORM_FACTOR
                        length normalization factor (default: 0.6)
  --cov-penalty-factor COV_PENALTY_FACTOR
                        coverage penalty factor (default: 0.1)
  --len-norm-const LEN_NORM_CONST
                        length normalization constant (default: 5.0)
  --intra-epoch-eval N  evaluate within training epoch, this option will
                        enable extra N equally spaced evaluations executed
                        during each training epoch (default: 0)
  --test-loader-workers TEST_LOADER_WORKERS
                        number of workers for test data loading (default: 0)

checkpointing setup:
  --start-epoch START_EPOCH
                        manually set initial epoch counter (default: 0)
  --resume PATH         resumes training from checkpoint from PATH (default:
                        None)
  --save-all            saves checkpoint after every epoch (default: False)
  --save-freq SAVE_FREQ
                        save checkpoint every SAVE_FREQ batches (default:
                        5000)
  --keep-checkpoints KEEP_CHECKPOINTS
                        keep only last KEEP_CHECKPOINTS checkpoints, affects
                        only checkpoints controlled by --save-freq option
                        (default: 0)

benchmark setup:
  --target-perf TARGET_PERF
                        target training performance (in tokens per second)
                        (default: None)
  --target-bleu TARGET_BLEU
                        target accuracy (default: None)
```

Inference


The complete list of available parameters for the `translate.py` inference
script contains:

```
data setup:
  -o OUTPUT, --output OUTPUT
                        full path to the output file if not specified, then
                        the output will be printed (default: None)
  -r REFERENCE, --reference REFERENCE
                        full path to the file with reference translations (for
                        sacrebleu, raw text) (default: None)
  -m MODEL, --model MODEL
                        full path to the model checkpoint file (default: None)
  --synthetic           use synthetic dataset (default: False)
  --synthetic-batches SYNTHETIC_BATCHES
                        number of synthetic batches to generate (default: 64)
  --synthetic-vocab SYNTHETIC_VOCAB
                        size of synthetic vocabulary (default: 32320)
  --synthetic-len SYNTHETIC_LEN
                        sequence length of synthetic samples (default: 50)
  -i INPUT, --input INPUT
                        full path to the input file (raw text) (default: None)
  -t INPUT_TEXT [INPUT_TEXT ...], --input-text INPUT_TEXT [INPUT_TEXT ...]
                        raw input text (default: None)
  --sort                sorts dataset by sequence length (use '--no-sort' to
                        disable) (default: False)

inference setup:
  --batch-size BATCH_SIZE [BATCH_SIZE ...]
                        batch size per GPU (default: [128])
  --beam-size BEAM_SIZE [BEAM_SIZE ...]
                        beam size (default: [5])
  --max-seq-len MAX_SEQ_LEN
                        maximum generated sequence length (default: 80)
  --len-norm-factor LEN_NORM_FACTOR
                        length normalization factor (default: 0.6)
  --cov-penalty-factor COV_PENALTY_FACTOR
                        coverage penalty factor (default: 0.1)
  --len-norm-const LEN_NORM_CONST
                        length normalization constant (default: 5.0)

general setup:
  --math {fp16,fp32,tf32} [{fp16,fp32,tf32} ...]
                        precision (default: ['fp16'])
  --env                 print info about execution env (use '--no-env' to
                        disable) (default: False)
  --bleu                compares with reference translation and computes BLEU
                        (use '--no-bleu' to disable) (default: True)
  --cuda                enables cuda (use '--no-cuda' to disable) (default:
                        True)
  --cudnn               enables cudnn (use '--no-cudnn' to disable) (default:
                        True)
  --batch-first         uses (batch, seq, feature) data format for RNNs
                        (default: True)
  --seq-first           uses (seq, batch, feature) data format for RNNs
                        (default: True)
  --save-dir SAVE_DIR   path to directory with results, it will be
                        automatically created if it does not exist (default:
                        gnmt)
  --dllog-file DLLOG_FILE
                        Name of the DLLogger output file (default:
                        eval_log.json)
  --print-freq PRINT_FREQ, -p PRINT_FREQ
                        print log every PRINT_FREQ batches (default: 1)

benchmark setup:
  --target-perf TARGET_PERF
                        target inference performance (in tokens per second)
                        (default: None)
  --target-bleu TARGET_BLEU
                        target accuracy (default: None)
  --repeat REPEAT [REPEAT ...]
                        loops over the dataset REPEAT times, flag accepts
                        multiple arguments, one for each specified batch size
                        (default: [1])
  --warmup WARMUP       warmup iterations for performance counters (default:
                        0)
  --percentiles PERCENTILES [PERCENTILES ...]
                        Percentiles for confidence intervals for
                        throughput/latency benchmarks (default: (90, 95, 99))
  --tables              print accuracy, throughput and latency results in
                        tables (use '--no-tables' to disable) (default: False)
```

### Command-line options
To see the full list of available options and their descriptions, use the `-h`
or `--help` command line option. For example, for training:

```
python3 train.py --help

usage: train.py [-h] [--dataset-dir DATASET_DIR] [--src-lang SRC_LANG]
                [--tgt-lang TGT_LANG] [--vocab VOCAB] [-bpe BPE_CODES]
                [--train-src TRAIN_SRC] [--train-tgt TRAIN_TGT]
                [--val-src VAL_SRC] [--val-tgt VAL_TGT] [--test-src TEST_SRC]
                [--test-tgt TEST_TGT] [--save-dir SAVE_DIR]
                [--print-freq PRINT_FREQ] [--hidden-size HIDDEN_SIZE]
                [--num-layers NUM_LAYERS] [--dropout DROPOUT]
                [--share-embedding] [--smoothing SMOOTHING]
                [--math {fp16,fp32,tf32,manual_fp16}] [--seed SEED]
                [--prealloc-mode {off,once,always}] [--dllog-file DLLOG_FILE]
                [--eval] [--env] [--cuda] [--cudnn] [--log-all-ranks]
                [--train-max-size TRAIN_MAX_SIZE]
                [--train-batch-size TRAIN_BATCH_SIZE]
                [--train-global-batch-size TRAIN_GLOBAL_BATCH_SIZE]
                [--train-iter-size N] [--epochs EPOCHS]
                [--grad-clip GRAD_CLIP] [--train-max-length TRAIN_MAX_LENGTH]
                [--train-min-length TRAIN_MIN_LENGTH]
                [--train-loader-workers TRAIN_LOADER_WORKERS]
                [--batching {random,sharding,bucketing}]
                [--shard-size SHARD_SIZE] [--num-buckets NUM_BUCKETS]
                [--optimizer OPTIMIZER] [--lr LR]
                [--optimizer-extra OPTIMIZER_EXTRA] [--init-scale INIT_SCALE]
                [--upscale-interval UPSCALE_INTERVAL]
                [--warmup-steps WARMUP_STEPS] [--remain-steps REMAIN_STEPS]
                [--decay-interval DECAY_INTERVAL] [--decay-steps DECAY_STEPS]
                [--decay-factor DECAY_FACTOR]
                [--val-batch-size VAL_BATCH_SIZE]
                [--val-max-length VAL_MAX_LENGTH]
                [--val-min-length VAL_MIN_LENGTH]
                [--val-loader-workers VAL_LOADER_WORKERS]
                [--test-batch-size TEST_BATCH_SIZE]
                [--test-max-length TEST_MAX_LENGTH]
                [--test-min-length TEST_MIN_LENGTH] [--beam-size BEAM_SIZE]
                [--len-norm-factor LEN_NORM_FACTOR]
                [--cov-penalty-factor COV_PENALTY_FACTOR]
                [--len-norm-const LEN_NORM_CONST] [--intra-epoch-eval N]
                [--test-loader-workers TEST_LOADER_WORKERS]
                [--start-epoch START_EPOCH] [--resume PATH] [--save-all]
                [--save-freq SAVE_FREQ] [--keep-checkpoints KEEP_CHECKPOINTS]
                [--target-perf TARGET_PERF] [--target-bleu TARGET_BLEU]
                [--local_rank LOCAL_RANK]
```
For example, for inference:

```
python3 translate.py --help

usage: translate.py [-h] [-o OUTPUT] [-r REFERENCE] [-m MODEL] [--synthetic]
                    [--synthetic-batches SYNTHETIC_BATCHES]
                    [--synthetic-vocab SYNTHETIC_VOCAB]
                    [--synthetic-len SYNTHETIC_LEN]
                    [-i INPUT | -t INPUT_TEXT [INPUT_TEXT ...]] [--sort]
                    [--batch-size BATCH_SIZE [BATCH_SIZE ...]]
                    [--beam-size BEAM_SIZE [BEAM_SIZE ...]]
                    [--max-seq-len MAX_SEQ_LEN]
                    [--len-norm-factor LEN_NORM_FACTOR]
                    [--cov-penalty-factor COV_PENALTY_FACTOR]
                    [--len-norm-const LEN_NORM_CONST]
                    [--math {fp16,fp32,tf32} [{fp16,fp32,tf32} ...]] [--env]
                    [--bleu] [--cuda] [--cudnn] [--batch-first | --seq-first]
                    [--save-dir SAVE_DIR] [--dllog-file DLLOG_FILE]
                    [--print-freq PRINT_FREQ] [--target-perf TARGET_PERF]
                    [--target-bleu TARGET_BLEU] [--repeat REPEAT [REPEAT ...]]
                    [--warmup WARMUP]
                    [--percentiles PERCENTILES [PERCENTILES ...]] [--tables]
                    [--local_rank LOCAL_RANK]
```

### Getting the data
The GNMT v2 model was trained on the [WMT16
English-German](http://www.statmt.org/wmt16/translation-task.html) dataset.
Concatenation of the newstest2015 and newstest2016 test sets are used as a
validation dataset and the newstest2014 is used as a testing dataset.

This repository contains the `scripts/wmt16_en_de.sh` download script which
automatically downloads and preprocesses the training, validation and test
datasets. By default, data is downloaded to the `data` directory.

Our download script is very similar to the `wmt16_en_de.sh` script from the
[tensorflow/nmt](https://github.com/tensorflow/nmt/blob/master/nmt/scripts/wmt16_en_de.sh)
repository. Our download script contains an extra preprocessing step, which
discards all pairs of sentences which can't be decoded by *latin-1* encoder.
The `scripts/wmt16_en_de.sh` script uses the
[subword-nmt](https://github.com/rsennrich/subword-nmt) package to segment text
into subword units (Byte Pair Encodings -
[BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding)). By default, the
script builds the shared vocabulary of 32,000 tokens.

In order to test with other datasets, the script needs to be customized
accordingly.

#### Dataset guidelines
The process of downloading and preprocessing the data can be found in the
`scripts/wmt16_en_de.sh` script.

Initially, data is downloaded from [www.statmt.org](www.statmt.org). Then
`europarl-v7`, `commoncrawl` and `news-commentary` corpora are concatenated to
form the training dataset, similarly `newstest2015` and `newstest2016` are
concatenated to form the validation dataset. Raw data is preprocessed with
[Moses](https://github.com/moses-smt/mosesdecoder), first by launching [Moses
tokenizer](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl)
(tokenizer breaks up text into individual words), then by launching
[clean-corpus-n.perl](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/training/clean-corpus-n.perl)
which removes invalid sentences and does initial filtering by sequence length.

Second stage of preprocessing is done by launching the
`scripts/filter_dataset.py` script, which discards all pairs of sentences that
can't be decoded by latin-1 encoder.

Third state of preprocessing uses the
[subword-nmt](https://github.com/rsennrich/subword-nmt) package. First it
builds shared [byte pair
encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding) vocabulary with
32,000 merge operations (command `subword-nmt learn-bpe`), then it applies
generated vocabulary to training, validation and test corpora (command
`subword-nmt apply-bpe`).

### Training process
The default training configuration can be launched by running the `train.py`
training script. By default, the training script saves only one checkpoint with
the lowest value of the loss function on the validation dataset. An evaluation
is then performed after each training epoch. Results are stored in the
`gnmt` directory.

The training script launches data-parallel training with batch size 128 per GPU
on all available GPUs. We have tested reliance on up to 16 GPUs on a single
node.
After each training epoch, the script runs an evaluation on the validation
dataset and outputs a BLEU score on the test dataset (newstest2014). BLEU is
computed by the [SacreBLEU](https://github.com/mjpost/sacreBLEU) package. Logs
from the training and evaluation are saved to the `gnmt` directory.

The summary after each training epoch is printed in the following format:

```
0: Summary: Epoch: 3	Training Loss: 3.1336	Validation Loss: 2.9587	Test BLEU: 23.18
0: Performance: Epoch: 3	Training: 418772 Tok/s	Validation: 1445331 Tok/s
```

The training loss is averaged over an entire training epoch, the validation
loss is averaged over the validation dataset and the BLEU score is computed on
the test dataset. Performance is reported in total tokens per second. The
result is averaged over an entire training epoch and summed over all GPUs
participating in the training.

By default, the `train.py` script will launch mixed precision training with
Tensor Cores. You can change this behavior by setting:
* the `--math fp32` flag to launch single precision training (for NVIDIA Volta
  and NVIDIA Turing architectures) or
* the `--math tf32` flag to launch TF32 training with Tensor Cores (for NVIDIA
  Ampere architecture)

for the `train.py` training script.

To view all available options for training, run `python3 train.py --help`.

### Inference process
Inference can be run by launching the `translate.py` inference script,
although, it requires a pre-trained model checkpoint and tokenized input.

The inference script, `translate.py`, supports batched inference. By default,
it launches beam search with beam size of 5, coverage penalty term and length
normalization term. Greedy decoding can be enabled by setting the beam size to
1.

To view all available options for inference, run `python3 translate.py --help`.


## Performance

The performance measurements in this document were conducted at the time of
publication and may not reflect the performance achieved from NVIDIA’s latest
software release. For the most up-to-date performance measurements, go to
[NVIDIA Data Center Deep Learning Product
Performance](https://developer.nvidia.com/deep-learning-performance-training-inference).

### Benchmarking
The following section shows how to run benchmarks measuring the model
performance in training and inference modes.

#### Training performance benchmark
Training is launched on batches of text data, different batches have different
sequence lengths (number of tokens in the longest sequence). Sequence length
and batch efficiency (ratio of non-pad tokens to total number of tokens) affect
performance of the training, therefore it's recommended to run the training on
a large chunk of training dataset to get a stable and reliable average training
performance. Ideally at least one full epoch of training should be launched to
get a good estimate of training performance.

The following commands will launch one epoch of training:

To launch mixed precision training on 1, 4 or 8 GPUs, run:
```
python3 -m torch.distributed.launch --nproc_per_node=<#GPUs> train.py --seed 2 --train-global-batch-size 1024 --epochs 1 --math fp16
```

To launch mixed precision training on 16 GPUs, run:
```
python3 -m torch.distributed.launch --nproc_per_node=16 train.py --seed 2 --train-global-batch-size 2048 --epochs 1 --math fp16
```

Change `--math fp16` to `--math fp32` to launch single precision training (for
NVIDIA Volta and NVIDIA Turing architectures) or to `--math tf32` to launch
TF32 training with Tensor Cores (for NVIDIA Ampere architecture).

After the training is completed, the `train.py` script prints a summary to
standard output. Performance results are printed in the following format:

```
(...)
0: Performance: Epoch: 0	Training: 418926 Tok/s	Validation: 1430828 Tok/s
(...)
```

`Training: 418926 Tok/s` represents training throughput averaged over an entire
training epoch and summed over all GPUs participating in the training.

#### Inference performance benchmark
The inference performance and accuracy benchmarks require a checkpoint from a
fully trained model.

Command to launch the inference accuracy benchmark on NVIDIA Volta or on NVIDIA
Turing architectures:

```
python3 translate.py \
  --model gnmt/model_best.pth \
  --input data/wmt16_de_en/newstest2014.en \
  --reference data/wmt16_de_en/newstest2014.de \
  --output /tmp/output \
  --math fp16 fp32 \
  --batch-size 128 \
  --beam-size 1 2 5 \
  --tables
```

Command to launch the inference accuracy benchmark on NVIDIA Ampere architecture:

```
python3 translate.py \
  --model gnmt/model_best.pth \
  --input data/wmt16_de_en/newstest2014.en \
  --reference data/wmt16_de_en/newstest2014.de \
  --output /tmp/output \
  --math fp16 tf32 \
  --batch-size 128 \
  --beam-size 1 2 5 \
  --tables
```

Command to launch the inference throughput and latency benchmarks on NVIDIA
Volta or NVIDIA Turing architectures:

```
python3 translate.py \
  --model gnmt/model_best.pth \
  --input data/wmt16_de_en/newstest2014.en \
  --reference data/wmt16_de_en/newstest2014.de \
  --output /tmp/output \
  --math fp16 fp32 \
  --batch-size 1 2 4 8 32 128 512 \
  --repeat 1 1 1 1 2 8 16 \
  --beam-size 1 2 5 \
  --warmup 5 \
  --tables
```

Command to launch the inference throughput and latency benchmarks on NVIDIA
Ampere architecture:

```
python3 translate.py \
  --model gnmt/model_best.pth \
  --input data/wmt16_de_en/newstest2014.en \
  --reference data/wmt16_de_en/newstest2014.de \
  --output /tmp/output \
  --math fp16 tf32 \
  --batch-size 1 2 4 8 32 128 512 \
  --repeat 1 1 1 1 2 8 16 \
  --beam-size 1 2 5 \
  --warmup 5 \
  --tables
```

### Results
The following sections provide details on how we achieved our performance and
accuracy in training and inference.

#### Training accuracy results

##### Training accuracy: NVIDIA DGX A100 (8x A100 40GB)
Our results were obtained by running the `train.py` script with the default
batch size = 128 per GPU in the pytorch-20.06-py3 NGC container on NVIDIA DGX
A100 with 8x A100 40GB GPUs.

Command to launch the training:

```
python3 -m torch.distributed.launch --nproc_per_node=<#GPUs> train.py --seed 2 --train-global-batch-size 1024 --math fp16
```

Change `--math fp16` to `--math tf32` to launch TF32 training with Tensor Cores.

| **GPUs** | **Batch Size / GPU** | **Accuracy - TF32 (BLEU)** | **Accuracy - Mixed precision (BLEU)** | **Time to Train - TF32 (minutes)** | **Time to Train - Mixed precision (minutes)** | **Time to Train Speedup (TF32 to Mixed precision)** |
| --- | --- | ----- | ----- | ----- | ------ | ---- |
| 8   | 128 | 24.46 | 24.60 | 34.7  | 22.7   | 1.53 |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide)
outlined above.

##### Training accuracy: NVIDIA DGX-1 (8x V100 16GB)
Our results were obtained by running the `train.py` script with the default
batch size = 128 per GPU in the pytorch-20.06-py3 NGC container on NVIDIA DGX-1
with 8x V100 16GB GPUs.

Command to launch the training:

```
python3 -m torch.distributed.launch --nproc_per_node=<#GPUs> train.py --seed 2 --train-global-batch-size 1024 --math fp16
```

Change `--math fp16` to `--math fp32` to launch single precision training.

| **GPUs** | **Batch Size / GPU** | **Accuracy - FP32 (BLEU)** | **Accuracy - Mixed precision (BLEU)** | **Time to Train - FP32 (minutes)** | **Time to Train - Mixed precision (minutes)** | **Time to Train Speedup (FP32 to Mixed precision)** |
| --- | --- | ----- | ----- | ----- | ------ | ---- |
| 1   | 128 | 24.41 | 24.42 | 810.0 | 224.0  | 3.62 |
| 4   | 128 | 24.40 | 24.33 | 218.2 | 69.5   | 3.14 |
| 8   | 128 | 24.45 | 24.38 | 112.0 | 38.6   | 2.90 |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide)
outlined above.

##### Training accuracy: NVIDIA DGX-2H (16x V100 32GB)
Our results were obtained by running the `train.py` script with the default
batch size = 128 per GPU in the pytorch-20.06-py3 NGC container on NVIDIA DGX-2H
with 16x V100 32GB GPUs.

To launch mixed precision training on 16 GPUs, run:
```
python3 -m torch.distributed.launch --nproc_per_node=16 train.py --seed 2 --train-global-batch-size 2048 --math fp16
```

Change `--math fp16` to `--math fp32` to launch single precision training.

| **GPUs** | **Batch Size / GPU** | **Accuracy - FP32 (BLEU)** | **Accuracy - Mixed precision (BLEU)** | **Time to Train - FP32 (minutes)** | **Time to Train - Mixed precision (minutes)** | **Time to Train Speedup (FP32 to Mixed precision)** |
| --- | --- | ----- | ----- | ------ | ----- | ---- |
| 16  | 128 | 24.41 | 24.38 | 52.1   | 19.4  | 2.69 |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide)
outlined above.

![TrainingLoss](./img/training_loss.png)

##### Training stability test
The GNMT v2 model was trained for 6 epochs, starting from 32 different initial
random seeds. After each training epoch, the model was evaluated on the test
dataset and the BLEU score was recorded. The training was performed in the
pytorch-20.06-py3 Docker container on NVIDIA DGX A100 with 8x A100 40GB GPUs.
The following table summarizes the results of the stability test.

In the following table, the BLEU scores after each training epoch for different
initial random seeds are displayed.

| **Epoch** | **Average** | **Standard deviation** | **Minimum** | **Maximum** | **Median** |
| --- | ------ | ----- | ------ | ------ | ------ |
| 1   | 19.959 | 0.238 | 19.410 | 20.390 | 19.970 |
| 2   | 21.772 | 0.293 | 20.960 | 22.280 | 21.820 |
| 3   | 22.435 | 0.264 | 21.740 | 22.870 | 22.465 |
| 4   | 23.167 | 0.166 | 22.870 | 23.620 | 23.195 |
| 5   | 24.233 | 0.149 | 23.820 | 24.530 | 24.235 |
| 6   | 24.416 | 0.131 | 24.140 | 24.660 | 24.390 |

#### Training throughput results

##### Training throughput: NVIDIA DGX A100 (8x A100 40GB)
Our results were obtained by running the `train.py` training script in the
pytorch-20.06-py3 NGC container on NVIDIA DGX A100 with 8x A100 40GB GPUs.
Throughput performance numbers (in tokens per second) were averaged over an
entire training epoch.

| **GPUs** | **Batch size / GPU** | **Throughput - TF32 (tok/s)** | **Throughput - Mixed precision (tok/s)** | **Throughput speedup (TF32 to Mixed precision)** | **Strong Scaling - TF32** | **Strong Scaling - Mixed precision** |
| --- | --- | ------ | ------ | ----- | ----- | ----- |
| 1   | 128 | 83214  | 140909 | 1.693 | 1.000 | 1.000 |
| 4   | 128 | 278576 | 463144 | 1.663 | 3.348 | 3.287 |
| 8   | 128 | 519952 | 822024 | 1.581 | 6.248 | 5.834 |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide)
outlined above.

##### Training throughput: NVIDIA DGX-1 (8x V100 16GB)
Our results were obtained by running the `train.py` training script in the
pytorch-20.06-py3 NGC container on NVIDIA DGX-1 with 8x V100 16GB GPUs.
Throughput performance numbers (in tokens per second) were averaged over an
entire training epoch.

| **GPUs** | **Batch size / GPU** | **Throughput - FP32 (tok/s)** | **Throughput - Mixed precision (tok/s)** | **Throughput speedup (FP32 to Mixed precision)** | **Strong Scaling - FP32** | **Strong Scaling - Mixed precision** |
| --- | --- | ------ | ------ | ----- | ----- | ----- |
| 1   | 128 | 21860  | 76438  | 3.497 | 1.000 | 1.000 |
| 4   | 128 | 80224  | 249168 | 3.106 | 3.670 | 3.260 |
| 8   | 128 | 154168 | 447832 | 2.905 | 7.053 | 5.859 |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide)
outlined above.

##### Training throughput: NVIDIA DGX-2H (16x V100 32GB)
Our results were obtained by running the `train.py` training script in the
pytorch-20.06-py3 NGC container on NVIDIA DGX-2H with 16x V100 32GB GPUs.
Throughput performance numbers (in tokens per second) were averaged over an
entire training epoch.

| **GPUs** | **Batch size / GPU** | **Throughput - FP32 (tok/s)** | **Throughput - Mixed precision (tok/s)** | **Throughput speedup (FP32 to Mixed precision)** | **Strong Scaling - FP32** | **Strong Scaling - Mixed precision** |
| --- | --- | ------ | ------ | ----- | ------ | ------ |
| 1  | 128 | 25583  | 87829   | 3.433 | 1.000  | 1.000  |
| 4  | 128 | 91400  | 290640  | 3.180 | 3.573  | 3.309  |
| 8  | 128 | 176616 | 522008  | 2.956 | 6.904  | 5.943  |
| 16 | 128 | 351792 | 1010880 | 2.874 | 13.751 | 11.510 |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide)
outlined above.

#### Inference accuracy results

##### Inference accuracy: NVIDIA A100 40GB
Our results were obtained by running the `translate.py` script in the
pytorch-20.06-py3 NGC Docker container with NVIDIA A100 40GB GPU. Full
command to launch the inference accuracy benchmark was provided in the
[Inference performance benchmark](#inference-performance-benchmark) section.

| **Batch Size** | **Beam Size** | **Accuracy - TF32 (BLEU)** | **Accuracy - FP16 (BLEU)** |
| -------------: | ------------: | -------------------------: | -------------------------: |
| 128            | 1             | 23.07                      | 23.07                      |
| 128            | 2             | 23.81                      | 23.81                      |
| 128            | 5             | 24.41                      | 24.43                      |

##### Inference accuracy: NVIDIA Tesla V100 16GB
Our results were obtained by running the `translate.py` script in the
pytorch-20.06-py3 NGC Docker container with NVIDIA Tesla V100 16GB GPU. Full
command to launch the inference accuracy benchmark was provided in the
[Inference performance benchmark](#inference-performance-benchmark) section.

| **Batch Size** | **Beam Size** | **Accuracy - FP32 (BLEU)** | **Accuracy - FP16 (BLEU)** |
| -------------: | ------------: | -------------------------: | -------------------------: |
| 128            | 1             | 23.07                      | 23.07                      |
| 128            | 2             | 23.81                      | 23.79                      |
| 128            | 5             | 24.40                      | 24.43                      |

##### Inference accuracy: NVIDIA T4
Our results were obtained by running the `translate.py` script in the
pytorch-20.06-py3 NGC Docker container with NVIDIA Tesla T4. Full command to
launch the inference accuracy benchmark was provided in the [Inference
performance benchmark](#inference-performance-benchmark) section.

| **Batch Size** | **Beam Size** | **Accuracy - FP32 (BLEU)** | **Accuracy - FP16 (BLEU)** |
| -------------: | ------------: | -------------------------: | -------------------------: |
| 128            | 1             | 23.07                      | 23.08                      |
| 128            | 2             | 23.81                      | 23.80                      |
| 128            | 5             | 24.40                      | 24.39                      |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide)
outlined above.

#### Inference throughput results
Tables presented in this section show the average inference throughput (columns
**Avg (tok/s)**) and inference throughput for various confidence intervals
(columns **N% (ms)**, where `N` denotes the confidence interval). Inference
throughput is measured in tokens per second. Speedups reported in FP16
subsections are relative to FP32 (for NVIDIA Volta and NVIDIA Turing) and
relative to TF32 (for NVIDIA Ampere) numbers for corresponding configuration.

##### Inference throughput: NVIDIA A100 40GB
Our results were obtained by running the `translate.py` script in the
pytorch-20.06-py3 NGC Docker container with NVIDIA A100 40GB.
Full command to launch the inference throughput benchmark was provided in the
[Inference performance benchmark](#inference-performance-benchmark) section.

**FP16**

|**Batch Size**|**Beam Size**|**Avg (tok/s)**|**Speedup**|**90% (tok/s)**|**Speedup**|**95% (tok/s)**|**Speedup**|**99% (tok/s)**|**Speedup**|
|-------------:|------------:|--------------:|----------:|--------------:|----------:|--------------:|----------:|--------------:|----------:|
|             1|            1|         1291.6|      1.031|         1195.7|      1.029|         1165.8|      1.029|         1104.7|      1.030|
|             1|            2|          882.7|      1.019|          803.4|      1.015|          769.2|      1.015|          696.7|      1.017|
|             1|            5|          848.3|      1.042|          753.0|      1.037|          715.0|      1.043|          636.4|      1.033|
|             2|            1|         2060.5|      1.034|         1700.8|      1.032|         1621.8|      1.032|         1487.4|      1.022|
|             2|            2|         1445.7|      1.026|         1197.6|      1.024|         1132.5|      1.023|         1043.7|      1.033|
|             2|            5|         1402.3|      1.063|         1152.4|      1.056|         1100.5|      1.053|          992.9|      1.053|
|             4|            1|         3465.6|      1.046|         2838.3|      1.040|         2672.7|      1.043|         2392.8|      1.043|
|             4|            2|         2425.4|      1.041|         2002.5|      1.028|         1898.3|      1.033|         1690.2|      1.028|
|             4|            5|         2364.4|      1.075|         1930.0|      1.067|         1822.0|      1.065|         1626.1|      1.058|
|             8|            1|         6151.1|      1.099|         5078.0|      1.087|         4786.5|      1.096|         4206.9|      1.090|
|             8|            2|         4241.9|      1.075|         3494.1|      1.066|         3293.6|      1.066|         2970.9|      1.064|
|             8|            5|         4117.7|      1.118|         3430.9|      1.103|         3224.5|      1.104|         2833.5|      1.110|
|            32|            1|        18830.4|      1.147|        16210.0|      1.152|        15563.9|      1.138|        13973.2|      1.135|
|            32|            2|        12698.2|      1.133|        10812.3|      1.114|        10256.1|      1.145|         9330.2|      1.101|
|            32|            5|        11802.6|      1.355|         9998.8|      1.318|         9671.6|      1.329|         9058.4|      1.335|
|           128|            1|        53394.5|      1.350|        48867.6|      1.342|        46898.5|      1.414|        40670.6|      1.305|
|           128|            2|        34876.4|      1.483|        31687.4|      1.491|        30025.4|      1.505|        27677.1|      1.421|
|           128|            5|        28201.3|      1.986|        25660.5|      1.997|        24306.0|      1.967|        23326.2|      2.007|
|           512|            1|       119675.3|      1.904|       112400.5|      1.971|       109694.8|      1.927|       108781.3|      1.919|
|           512|            2|        74514.7|      2.126|        69578.9|      2.209|        69348.1|      2.210|        69253.7|      2.212|
|           512|            5|        47003.2|      2.760|        43348.2|      2.893|        43080.3|      2.884|        42878.4|      2.881|

##### Inference throughput: NVIDIA T4
Our results were obtained by running the `translate.py` script in the
pytorch-20.06-py3 NGC Docker container with NVIDIA T4.
Full command to launch the inference throughput benchmark was provided in the
[Inference performance benchmark](#inference-performance-benchmark) section.

**FP16**

|**Batch Size**|**Beam Size**|**Avg (tok/s)**|**Speedup**|**90% (tok/s)**|**Speedup**|**95% (tok/s)**|**Speedup**|**99% (tok/s)**|**Speedup**|
|-------------:|------------:|--------------:|----------:|--------------:|----------:|--------------:|----------:|--------------:|----------:|
|             1|            1|         1133.8|      1.266|         1059.1|      1.253|         1036.6|      1.251|          989.5|      1.242|
|             1|            2|          793.9|      1.169|          728.3|      1.165|          698.1|      1.163|          637.1|      1.157|
|             1|            5|          766.8|      1.343|          685.6|      1.335|          649.3|      1.335|          584.1|      1.318|
|             2|            1|         1759.8|      1.233|         1461.6|      1.239|         1402.3|      1.242|         1302.1|      1.242|
|             2|            2|         1313.3|      1.186|         1088.7|      1.185|         1031.6|      1.180|          953.2|      1.178|
|             2|            5|         1257.2|      1.301|         1034.1|      1.316|          990.3|      1.313|          886.3|      1.265|
|             4|            1|         2974.0|      1.261|         2440.3|      1.255|         2294.6|      1.257|         2087.7|      1.261|
|             4|            2|         2204.7|      1.320|         1826.3|      1.283|         1718.9|      1.260|         1548.4|      1.260|
|             4|            5|         2106.1|      1.340|         1727.8|      1.345|         1625.7|      1.353|         1467.7|      1.346|
|             8|            1|         5076.6|      1.423|         4207.9|      1.367|         3904.4|      1.360|         3475.3|      1.355|
|             8|            2|         3761.7|      1.311|         3108.1|      1.285|         2931.6|      1.300|         2628.7|      1.300|
|             8|            5|         3578.2|      1.660|         2998.2|      1.614|         2812.1|      1.609|         2447.6|      1.523|
|            32|            1|        14637.8|      1.636|        12702.5|      1.644|        12070.3|      1.634|        11036.9|      1.647|
|            32|            2|        10627.3|      1.818|         9198.3|      1.818|         8431.6|      1.725|         8000.0|      1.773|
|            32|            5|         8205.7|      2.598|         7117.6|      2.476|         6825.2|      2.497|         6293.2|      2.437|
|           128|            1|        33800.5|      2.755|        30824.5|      2.816|        27685.2|      2.661|        26580.9|      2.694|
|           128|            2|        20829.4|      2.795|        18665.2|      2.778|        17372.1|      2.639|        16820.5|      2.821|
|           128|            5|        11753.9|      3.309|        10658.1|      3.273|        10308.7|      3.205|         9630.7|      3.328|
|           512|            1|        44474.6|      3.327|        40108.1|      3.394|        39816.6|      3.378|        39708.0|      3.381|
|           512|            2|        26057.9|      3.295|        23197.3|      3.294|        23019.8|      3.284|        22951.4|      3.284|
|           512|            5|        12161.5|      3.428|        10777.5|      3.418|        10733.1|      3.414|        10710.5|      3.420|

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide)
outlined above.

#### Inference latency results
Tables presented in this section show the average inference latency (columns **Avg
(ms)**) and inference latency for various confidence intervals (columns **N%
(ms)**, where `N` denotes the confidence interval). Inference latency is
measured in milliseconds. Speedups reported in FP16 subsections are relative to
FP32 (for NVIDIA Volta and NVIDIA Turing) and relative to TF32 (for NVIDIA
Ampere) numbers for corresponding configuration.

##### Inference latency: NVIDIA A100 40GB
Our results were obtained by running the `translate.py` script in the
pytorch-20.06-py3 NGC Docker container with NVIDIA A100 40GB.
Full command to launch the inference latency benchmark was provided in the
[Inference performance benchmark](#inference-performance-benchmark) section.

**FP16**

|**Batch Size**|**Beam Size**|**Avg (ms)**|**Speedup**|**90% (ms)**|**Speedup**|**95% (ms)**|**Speedup**|**99% (ms)**|**Speedup**|
|-------------:|------------:|-----------:|----------:|-----------:|----------:|-----------:|----------:|-----------:|----------:|
|             1|            1|       44.69|      1.032|       74.04|      1.035|       84.61|      1.034|       99.14|      1.042|
|             1|            2|       64.76|      1.020|      105.18|      1.018|      118.92|      1.019|      139.42|      1.023|
|             1|            5|       67.06|      1.043|      107.56|      1.049|      121.82|      1.054|      143.85|      1.054|
|             2|            1|       56.57|      1.034|       85.59|      1.037|       92.55|      1.038|      107.59|      1.046|
|             2|            2|       80.22|      1.027|      119.22|      1.027|      128.43|      1.030|      150.06|      1.028|
|             2|            5|       82.54|      1.063|      121.37|      1.067|      132.35|      1.069|      156.34|      1.059|
|             4|            1|       67.29|      1.047|       92.69|      1.048|      100.08|      1.056|      112.63|      1.064|
|             4|            2|       95.86|      1.041|      129.83|      1.040|      139.48|      1.044|      162.34|      1.045|
|             4|            5|       98.34|      1.075|      133.83|      1.076|      142.70|      1.068|      168.30|      1.075|
|             8|            1|       75.60|      1.099|       97.87|      1.103|      104.13|      1.099|      117.40|      1.102|
|             8|            2|      109.38|      1.074|      137.71|      1.079|      147.69|      1.069|      168.79|      1.065|
|             8|            5|      112.71|      1.116|      143.50|      1.104|      153.17|      1.118|      172.60|      1.113|
|            32|            1|       98.40|      1.146|      117.02|      1.153|      123.42|      1.150|      129.01|      1.128|
|            32|            2|      145.87|      1.133|      171.71|      1.159|      184.01|      1.127|      188.64|      1.141|
|            32|            5|      156.82|      1.357|      189.10|      1.374|      194.95|      1.392|      196.65|      1.419|
|           128|            1|      137.97|      1.350|      150.04|      1.348|      151.52|      1.349|      154.52|      1.434|
|           128|            2|      211.58|      1.484|      232.96|      1.490|      237.46|      1.505|      239.86|      1.567|
|           128|            5|      261.44|      1.990|      288.54|      2.017|      291.63|      2.052|      298.73|      2.136|
|           512|            1|      245.93|      1.906|      262.51|      1.998|      264.24|      1.999|      265.23|      2.000|
|           512|            2|      395.61|      2.129|      428.54|      2.219|      431.58|      2.224|      433.86|      2.227|
|           512|            5|      627.21|      2.767|      691.72|      2.878|      696.01|      2.895|      702.13|      2.887|

##### Inference latency: NVIDIA T4
Our results were obtained by running the `translate.py` script in the
pytorch-20.06-py3 NGC Docker container with NVIDIA T4.
Full command to launch the inference latency benchmark was provided in the
[Inference performance benchmark](#inference-performance-benchmark) section.

**FP16**

|**Batch Size**|**Beam Size**|**Avg (ms)**|**Speedup**|**90% (ms)**|**Speedup**|**95% (ms)**|**Speedup**|**99% (ms)**|**Speedup**|
|-------------:|------------:|-----------:|----------:|-----------:|----------:|-----------:|----------:|-----------:|----------:|
|             1|            1|       51.08|      1.261|       84.82|      1.254|       97.45|      1.251|       114.6|      1.257|
|             1|            2|       72.05|      1.168|      117.41|      1.165|      132.33|      1.170|       155.8|      1.174|
|             1|            5|       74.20|      1.345|      119.45|      1.352|      135.07|      1.354|       160.3|      1.354|
|             2|            1|       66.31|      1.232|      100.90|      1.232|      108.52|      1.235|       126.9|      1.238|
|             2|            2|       88.35|      1.185|      131.47|      1.188|      141.46|      1.185|       164.7|      1.191|
|             2|            5|       92.12|      1.305|      136.30|      1.310|      148.66|      1.309|       174.8|      1.320|
|             4|            1|       78.54|      1.260|      108.53|      1.256|      117.19|      1.259|       133.7|      1.259|
|             4|            2|      105.54|      1.315|      142.74|      1.317|      154.36|      1.307|       178.7|      1.303|
|             4|            5|      110.43|      1.351|      150.62|      1.388|      161.61|      1.397|       191.2|      1.427|
|             8|            1|       91.65|      1.418|      117.92|      1.421|      126.60|      1.405|       144.0|      1.411|
|             8|            2|      123.39|      1.315|      156.00|      1.337|      167.34|      1.347|       193.4|      1.340|
|             8|            5|      129.69|      1.666|      165.01|      1.705|      178.18|      1.723|       200.3|      1.765|
|            32|            1|      126.53|      1.641|      153.23|      1.689|      159.58|      1.692|       167.0|      1.700|
|            32|            2|      174.37|      1.822|      209.04|      1.899|      219.59|      1.877|       228.6|      1.878|
|            32|            5|      226.15|      2.598|      277.38|      2.636|      290.27|      2.648|       299.4|      2.664|
|           128|            1|      218.29|      2.755|      238.94|      2.826|      243.18|      2.843|       267.1|      2.828|
|           128|            2|      354.83|      2.796|      396.63|      2.832|      410.53|      2.803|       433.2|      2.866|
|           128|            5|      628.32|      3.311|      699.57|      3.353|      723.98|      3.323|       771.0|      3.337|
|           512|            1|      663.07|      3.330|      748.62|      3.388|      753.20|      3.388|       758.0|      3.378|
|           512|            2|     1134.04|      3.295|     1297.85|      3.283|     1302.25|      3.304|      1306.9|      3.308|
|           512|            5|     2428.82|      3.428|     2771.72|      3.415|     2801.32|      3.427|      2817.6|      3.422|

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide)
outlined above.

## Release notes
### Changelog
* July 2020
  * Added support for NVIDIA DGX A100
  * Default container updated to NGC PyTorch 20.06-py3
* June 2019
  * Default container updated to NGC PyTorch 19.05-py3
  * Mixed precision training implemented using APEX AMP
  * Added inference throughput and latency results on NVIDIA T4 and NVIDIA
    Tesla V100 16GB
  * Added option to run inference on user-provided raw input text from command
    line
* February 2019
  * Different batching algorithm (bucketing with 5 equal-width buckets)
  * Additional dropouts before first LSTM layer in encoder and in decoder
  * Weight initialization changed to uniform (-0.1,0.1)
  * Switched order of dropout and concatenation with attention in decoder
  * Default container updated to NGC PyTorch 19.01-py3
* December 2018
  * Added exponential warm-up and step learning rate decay
  * Multi-GPU (distributed) inference and validation
  * Default container updated to NGC PyTorch 18.11-py3
  * General performance improvements
* August 2018
  * Initial release

### Known issues
There are no known issues in this release.

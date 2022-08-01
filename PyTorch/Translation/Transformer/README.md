# Transformer For PyTorch

This repository provides a script and recipe to train the Transformer model to achieve state of the art accuracy, and is tested and maintained by NVIDIA.

## Table Of Contents
* [Model overview](#model-overview)
    * [Model architecture](#model-architecture)
    * [Default configuration](#default-configuration)
    * [Feature support matrix](#feature-support-matrix)
	    * [Features](#features)
    * [Mixed precision training](#mixed-precision-training)
	    * [Enabling mixed precision](#enabling-mixed-precision)
        * [Enabling TF32](#enabling-tf32)
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
    * [Inference process](#inference-process)
* [Performance](#performance)
    * [Benchmarking](#benchmarking)
        * [Training performance benchmark](#training-performance-benchmark)
        * [Inference performance benchmark](#inference-performance-benchmark)
    * [Results](#results)
        * [Training accuracy results](#training-accuracy-results)
            * [Training accuracy: NVIDIA DGX A100 (8x A100 40GB)](#training-accuracy-nvidia-dgx-a100-8x-a100-40gb)
            * [Training accuracy: NVIDIA DGX-1 (8x V100 16GB)](#training-accuracy-nvidia-dgx-1-8x-v100-16gb)
            * [Training stability test](#training-stability-test)
        * [Training performance results](#training-performance-results)
            * [Training performance: NVIDIA DGX A100 (8x A100 40GB)](#training-performance-nvidia-dgx-a100-8x-a100-40gb)
            * [Training performance: NVIDIA DGX-1 (8x V100 16GB)](#training-performance-nvidia-dgx-1-8x-v100-16gb)
            * [Training performance: NVIDIA DGX-2 (16x V100 32GB)](#training-performance-nvidia-dgx-2-16x-v100-32gb)
        * [Inference performance results](#inference-performance-results)
            * [Inference performance: NVIDIA DGX A100 (1x A100 40GB)](#inference-performance-nvidia-dgx-a100-1x-a100-40gb)
            * [Inference performance: NVIDIA DGX-1 (1x V100 16GB)](#inference-performance-nvidia-dgx-1-1x-v100-16gb)
* [Release notes](#release-notes)
    * [Changelog](#changelog)
    * [Known issues](#known-issues)


## Model overview

The Transformer is a Neural Machine Translation (NMT) model which uses attention mechanism to boost training speed and overall accuracy. The Transformer model was introduced in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) and improved in [Scaling Neural Machine Translation](https://arxiv.org/abs/1806.00187).
This implementation is based on the optimized implementation in [Facebook's Fairseq NLP toolkit](https://github.com/pytorch/fairseq), built on top of PyTorch.

This model is trained with mixed precision using Tensor Cores on NVIDIA Volta, Turing and Ampere GPU architectures. Therefore, researchers can get results 6.5x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.

### Model architecture

The Transformer model uses standard NMT encoder-decoder architecture. This model unlike other NMT models, uses no recurrent connections and operates on fixed size context window.
The encoder stack is made up of N identical layers. Each layer is composed of the following sublayers:
    1. Self-attention layer
    2. Feedforward network (which is 2 fully-connected layers)
Like the encoder stack, the decoder stack is made up of N identical layers. Each layer is composed of the sublayers:
    1. Self-attention layer
    2. Multi-headed attention layer combining encoder outputs with results from
       the previous self-attention layer.
    3. Feedforward network (2 fully-connected layers)

The encoder uses self-attention to compute a representation of the input sequence. The decoder generates the output sequence one token at a time, taking the encoder output and previous decoder-outputted tokens as inputs.
The model also applies embeddings on the input and output tokens, and adds a constant positional encoding. The positional encoding adds information about the position of each token.

<p align="center">
    <img width="50%" src="./transformer.png" />
    <br>
    Figure 1. The architecture of a Transformer model.
</p>

The complete description of the Transformer architecture can be found in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper.
### Default configuration

The Transformer uses Byte Pair Encoding tokenization scheme using [Moses decoder](https://github.com/moses-smt/mosesdecoder). This is a lossy compression method (we drop information about white spaces). Tokenization is applied over whole [WMT14](http://statmt.org/wmt14/translation-task.html#Download) en-de dataset including test set. Default vocabulary size is 33708, excluding all special tokens. Encoder and decoder are using shared embeddings.
We use 6 blocks in each encoder and decoder stacks. Self attention layer computes it's outputs according to the following formula $`Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V`$. At each attention step, the model computes 16 different attention representations (which we will call attention heads) and concatenates them.
We trained the Transformer model using the Adam optimizer with betas `(0.9, 0.997)`, epsilon `1e-9` and learning rate `6e-4`. We used the inverse square root training schedule preceded with linear warmup of 4000 steps.
The implementation allows to perform training in mixed precision. We use dynamic loss scaling and custom mixed precision optimizer. Distributed multi-GPU and multi-Node is implemented with `torch.distirbuted` module with NCCL backend.
For inference, we use beam search with default beam size of 5. Model performance is evaluated with BLEU4 metrics. For clarity, we report internal (legacy) BLEU implementation as well as external [SacreBleu](https://github.com/mjpost/sacreBLEU)  score.

### Feature support matrix

The following features are supported by this model.<br>

| Feature                  | Yes column                
|--------------------------|--------------------------
| Multi-GPU training with [Distributed Communication Package](https://pytorch.org/docs/stable/distributed.html)  | Yes          
| Nvidia APEX              | Yes         
| AMP                      | Yes
| TorchScript              | Yes

#### Features

* Multi-GPU training with [Distributed Communication Package](https://pytorch.org/docs/stable/distributed.html): Our model uses torch.distributed package to implement efficient multi-GPU training with NCCL.
To enable multi-GPU training with torch.distributed, you have to initialize your model identically in every process spawned by torch.distributed.launch. Distributed strategy is implemented with APEX's DistributedDataParallel.
For details, see example sources in this repo or see the [pytorch tutorial](https://pytorch.org/docs/stable/distributed.html)

* Nvidia APEX: The purpose of the APEX is to provide easy and intuitive framework for distributed training and mixed precision training. For details, see official [APEX repository](https://github.com/NVIDIA/apex).

* AMP: This implementation uses Apex's AMP to perform mixed precision training.

* TorchScript: Transformer can be converted to TorchScript format offering ease of deployment on platforms without Python dependencies. For more information see official [TorchScript](https://pytorch.org/docs/stable/jit.html) documentation.


### Mixed precision training
Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format, while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in the Volta and Turing architecture, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using mixed precision training requires two steps:
1.  Porting the model to use the FP16 data type where appropriate.    
2.  Adding loss scaling to preserve small gradient values.

The ability to train deep learning networks with lower precision was introduced in the Pascal architecture and first supported in [CUDA 8](https://devblogs.nvidia.com/parallelforall/tag/fp16/) in the NVIDIA Deep Learning SDK.

For information about:
-   How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html) documentation.
-   Techniques used for mixed precision training, see the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
-   APEX tools for mixed precision training, see the [NVIDIA Apex: Tools for Easy Mixed-Precision Training in PyTorch](https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/).

#### Enabling mixed precision

Mixed precision is enabled using the `--amp` option in the `train.py` script. The default is optimization level `O2` but can be overriden with `--amp-level $LVL` option (for details see [amp documentation](https://nvidia.github.io/apex/amp.html)). Forward and backward pass are computed with FP16 precision with exclusion of a loss function which is computed in FP32 precision. Default optimization level keeps a copy of a model in higher precision in order to perform accurate weight update. After the update FP32 weights are again copied to FP16 model. We use dynamic loss scaling with initial scale of 2^7 increasing it by a factor of 2 every 2000 successful iterations. Overflow is being checked after reducing gradients from all of the workers. If we encounter infs or nans the whole batch is dropped.

#### Enabling TF32
TensorFloat-32 (TF32) is the new math mode in [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) GPUs for handling the matrix math also called tensor operations. TF32 running on Tensor Cores in A100 GPUs can provide up to 10x speedups compared to single-precision floating-point math (FP32) on Volta GPUs. 

TF32 Tensor Cores can speed up networks using FP32, typically with no loss of accuracy. It is more robust than FP16 for models which require high dynamic range for weights or activations.

For more information, refer to the [TensorFloat-32 in the A100 GPU Accelerates AI Training, HPC up to 20x](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) blog post.

TF32 is supported in the NVIDIA Ampere GPU architecture and is enabled by default.


### Glossary

Attention layer - Layer that computes which elements of input sequence or it's hidden representation contribute the most to the currently considered output element.  
Beam search - A heuristic search algorithm which at each step of predictions keeps N most possible outputs as a base to perform further prediction.  
BPE - Binary Pair Encoding, compression algorithm that find most common pair of symbols in a data and replaces them with new symbol absent in the data.  
EOS - End of a sentence.  
Self attention layer - Attention layer that computes hidden representation of input using the same tensor as query, key and value.  
Token - A  string that is representable within the model. We also refer to the token's position in the dictionary as a token. There are special non-string tokens: alphabet tokens (all characters in a dataset), EOS token, PAD token.  
Tokenizer - Object that converts raw strings to sequences of tokens.  
Vocabulary embedding - Layer that projects one-hot token representations to a high dimensional space which preserves some information about correlations between tokens.  

## Setup

The following section lists the requirements in order to start training the Transformer model.

### Requirements

This repository contains Dockerfile which extends the PyTorch NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:

-   [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
-   [PyTorch 22.06-py3+ NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch)
-   GPU-based architecture:
	- [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
	- [NVIDIA Turing](https://www.nvidia.com/en-us/geforce/turing/)
	- [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)

For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
-   [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
-   [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/dgx/user-guide/index.html#accessing_registry)
-   Running [PyTorch NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
  
For those unable to use the PyTorch NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## Quick Start Guide
To train your model using mixed or TF32 precision with Tensor Cores or using FP32, perform the following steps using the default parameters of the Transformer model on the [WMT14 English-German](http://statmt.org/wmt14/translation-task.html#Download) dataset. For the specifics concerning training and inference, see the [Advanced](#advanced) section.

1. Clone the repository 
```
git clone https://github.com/NVIDIA/DeepLearningExamples.git 
cd DeepLearningExamples/PyTorch/Translation/Transformer
```

2. Build and launch the Transformer PyTorch NGC  container
```bash
docker build . -t your.repository:transformer
nvidia-docker run -it --rm --ipc=host your.repository:transformer bash
```
If you already have preprocessed data, use:
```bash
nvidia-docker run -it --rm --ipc=host -v <path to your preprocessed data>:/data/wmt14_en_de_joined_dict your.repository:transformer bash
```
If you already have data downloaded, but it has not yet been preprocessed, use:
```bash
nvidia-docker run -it --rm --ipc=host -v <path to your unprocessed data>:/workspace/translation/examples/translation/orig your.repository:transformer bash
```
3. Download and preprocess dataset: Download and preprocess the WMT14 English-German dataset.

```bash 
scripts/run_preprocessing.sh
```
After running this command, data will be downloaded to `/workspace/translation/examples/translation/orig` directory and this data will be processed and put into `/data/wmt14_en_de_joined_dict` directory.

4. Start training
```bash
python -m torch.distributed.run --nproc_per_node 8 /workspace/translation/train.py /data/wmt14_en_de_joined_dict \
  --arch transformer_wmt_en_de_big_t2t \
  --share-all-embeddings \
  --optimizer adam \
  --adam-betas '(0.9, 0.997)' \
  --adam-eps "1e-9" \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 0.0 \
  --warmup-updates 4000 \
  --lr 0.0006 \
  --min-lr 0.0 \
  --dropout 0.1 \
  --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-tokens 5120 \
  --seed 1 \
  --fuse-layer-norm \
  --amp \
  --amp-level O2 \
  --save-dir /workspace/checkpoints
```

The script saves checkpoints every epoch to the directory specified in the `--save-dir` option. In addition, the best performing checkpoint (in terms of loss) and the latest checkpoints are saved separately.
**WARNING**: If you don't have access to sufficient disk space, use the `--save-interval $N` option. The checkpoints are ~3.4GB large. For example, it takes the Transformer model 30 epochs for the validation loss to plateau. The default option is to save last checkpoint, the best checkpoint and a checkpoint for every epoch, which means (30+1+1)*3.4GB = 108.8GB of a disk space used. Specifying `--save-interval 10` reduces this to (30/10+1+1)*3.4GB = 17GB. 

5. Start interactive inference
```bash
python inference.py \ 
  --buffer-size 5000 \
  --path /path/to/your/checkpoint.pt \
  --max-tokens 10240 \
  --fuse-dropout-add \
  --remove-bpe \
  --bpe-codes /path/to/bpe_code_file \
  --fp16
```
where, 
* `--path` option is the location of the checkpoint file.  
* `--bpe-codes` option is the location of the `code` file. If the default training command mentioned above is used, this file can be found in the preprocessed data ( i.e., `/data/wmt14_en_de_joined_dict` ) directory.

## Advanced
The following sections provide greater details of the dataset, running training and inference, and the training results.

### Scripts and sample code

The `preprocess.py` script performs binarization of the dataset obtained and tokenized by the `examples/translation/prepare-wmt14en2de.sh` script. The `train.py` script contains training loop as well as statistics gathering code. Steps performed in single training step can be found in `fairseq/ddp_trainer.py`. Model definition is placed in the file `fairseq/models/transformer.py`. Model specific modules including multiheaded attention and sinusoidal positional embedding are inside the `fairseq/modules/` directory. Finally, the data wrappers are placed inside the `fairseq/data/` directory.

### Parameters

In this section we give a user friendly description of the most common options used in the `train.py` script.
### Command-line options
`--arch` - select the specific configuration for the model. You can select between various predefined hyper parameters values like number of encoder/decoder blocks, dropout value or size of hidden state representation.<br/>
`--share-all-embeddings` - use the same set of weights for encoder and decoder words embedding.<br/>
`--optimizer` - choose optimization algorithm.<br/>
`--clip-norm` - set a value that gradients will be clipped to.<br/>
`--lr-scheduler` - choose learning rate change strategy.<br/>
`--warmup-init-lr` - start linear warmup with a learning rate at this value.<br/>
`--warmup-updates` - set number of optimization steps after which linear warmup will end.<br/>
`--lr` - set learning rate.<br/>
`--min-lr` - prevent learning rate to fall below this value using arbitrary learning rate schedule.<br/>
`--dropout` - set dropout value.<br/>
`--weight-decay` - set weight decay value.<br/>
`--criterion` - select loss function.<br/>
`--label-smoothing` - distribute value of one-hot labels between all entries of a dictionary. Value set by this option will be a value subtracted from one-hot label.<br/>
`--max-tokens` - set batch size in terms of tokens.<br/>
`--max-sentences` - set batch size in terms of sentences. Note that then the actual batchsize will vary a lot more than when using `--max-tokens` option.<br/>
`--seed` - set random seed for NumPy and PyTorch RNGs.<br/>
`--max-epochs` - set the maximum number of epochs.<br/>
`--online-eval` - perform inference on test set and then compute BLEU score after every epoch.<br/>
`--target-bleu` - works like `--online-eval` and sets a BLEU score threshold which after being attained will cause training to stop.<br/>
`--amp` - use mixed precision.<br/>
`--save-dir` - set directory for saving checkpoints.<br/>
`--distributed-init-method` - method for initializing torch.distributed package. You can either provide addresses with the `tcp` method or use the envionment variables initialization with `env` method<br/>
`--update-freq` - use gradient accumulation. Set number of training steps across which gradient will be accumulated.<br/>

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option, for example:
```
python train.py --help
```

The following (partial) output is printed when running the sample:
```
usage: train.py [-h] [--no-progress-bar] [--log-interval N]
                [--log-format {json,none,simple,tqdm}] [--seed N] [--fp16]
                [--task TASK] [--skip-invalid-size-inputs-valid-test] [--max-tokens N]
                [--max-sentences N] [--sentencepiece] [--train-subset SPLIT]
                [--valid-subset SPLIT] [--max-sentences-valid N]
                [--gen-subset SPLIT] [--num-shards N] [--shard-id ID]
                [--distributed-world-size N]
                [--distributed-rank DISTRIBUTED_RANK]
                [--local_rank LOCAL_RANK]
                [--distributed-backend DISTRIBUTED_BACKEND]
                [--distributed-init-method DISTRIBUTED_INIT_METHOD]
                [--distributed-port DISTRIBUTED_PORT] [--device-id DEVICE_ID]
                --arch ARCH [--criterion CRIT] [--max-epoch N]
                [--max-update N] [--target-bleu TARGET] [--clip-norm NORM]
                [--sentence-avg] [--update-freq N] [--optimizer OPT]
                [--lr LR_1,LR_2,...,LR_N] [--momentum M] [--weight-decay WD]
                [--lr-scheduler LR_SCHEDULER] [--lr-shrink LS] [--min-lr LR]
                [--min-loss-scale D] [--enable-parallel-backward-allred-opt]
                [--parallel-backward-allred-opt-threshold N]
                [--enable-parallel-backward-allred-opt-correctness-check]
                [--save-dir DIR] [--restore-file RESTORE_FILE]
                [--save-interval N] [--save-interval-updates N]
                [--keep-interval-updates N] [--no-save]
                [--no-epoch-checkpoints] [--validate-interval N] [--path FILE]
                [--remove-bpe [REMOVE_BPE]] [--cpu] [--quiet] [--beam N]
                [--nbest N] [--max-len-a N] [--max-len-b N] [--min-len N]
                [--no-early-stop] [--unnormalized] [--no-beamable-mm]
                [--lenpen LENPEN] [--unkpen UNKPEN]
                [--replace-unk [REPLACE_UNK]] [--score-reference]
                [--prefix-size PS] [--sampling] [--sampling-topk PS]
                [--sampling-temperature N] [--print-alignment]
                [--model-overrides DICT] [--online-eval] 
                [--bpe-codes CODES] [--fuse-dropout-add] [--fuse-relu-dropout]
```

### Getting the data

The Transformer model was trained on the [WMT14 English-German](http://statmt.org/wmt14/translation-task.html#Download) dataset. Concatenation of the *commoncrawl*, *europarl* and *news-commentary* is used as train and validation dataset and *newstest2014* is used as test dataset.<br/>
This repository contains the `run_preprocessing.sh` script which will automatically downloads and preprocesses the training and test datasets. By default, data will be stored in the `/data/wmt14_en_de_joined_dict` directory.<br/>
Our download script utilizes [Moses decoder](https://github.com/moses-smt/mosesdecoder) to perform tokenization of the dataset and [subword-nmt](https://github.com/rsennrich/subword-nmt) to segment text into subword units (BPE). By default, the script builds a shared vocabulary of 33708 tokens, which is consistent with [Scaling Neural Machine Translation](https://arxiv.org/abs/1806.00187).

#### Dataset guidelines

The Transformer model works with a fixed sized vocabulary. Prior to the training, we need to learn a data representation that allows us to store the entire dataset as a sequence of tokens. To achieve this we use Binary Pair Encoding. This algorithm builds a vocabulary by iterating over a dataset, looking for the most frequent pair of symbols and replacing them with a new symbol, yet absent in the dataset. After identifying the desired number of encodings (new symbols can also be merged together) it outputs a code file that is used as an input for the `Dictionary` class.
This approach does not minimize the length of the encoded dataset, however this is allowed using [SentencePiece](https://github.com/google/sentencepiece/) to tokenize the dataset with the unigram model. This approach tries to find encoding that is close to the theoretical entropy limit.
Data is then sorted by length (in terms of tokens) and examples with similar length are batched together, padded if necessary.

#### Multi-dataset

The model has been tested oni the [wmt14 en-fr](http://www.statmt.org/wmt14/translation-task.html) dataset. Achieving state of the art accuracy of 41.4 BLEU.

### Training process

The default training configuration can be launched by running the `train.py` training script. By default, the script saves one checkpoint every epoch in addition to the latest and the best ones. The best checkpoint is considered the one with the lowest value of loss, not the one with the highest BLEU score. To override this behavior use the `--save-interval $N` option to save epoch checkpoints every N epoch or `--no-epoch-checkpoints` to disable them entirely (with this option the latest and the best checkpoints still will be saved). Specify save the directory with `--save-dir` option.<br/>
In order to run multi-GPU training, launch the training script with `python -m torch.distributed.launch --nproc_per_node $N` prepended, where N is the number of GPUs.
We have tested reliance on up to 16 GPUs on a single node.<br/>
After each training epoch, the script runs a loss validation on the validation split of the dataset and outputs the validation loss. By default the evaluation after each epoch is disabled. To enable it, use the `--online-eval` option or to use the BLEU score value as the training stopping condition use the `--target-bleu $TGT` option. The BLEU scores computed are case insensitive. The BLEU is computed by the internal fairseq algorithm which implementation can be found in the `fairseq/bleu.py` script.<br/>
By default, the `train.py` script will launch FP32 training without Tensor Cores. To use mixed precision with Tensor Cores use the `--fp16` option.<br/>

To reach the BLEU score reported in [Scaling Neural Machine Translation](https://arxiv.org/abs/1806.00187) research paper, we used mixed precision training with a batch size of 5120 per GPU and learning rate of 6e-4 on a DGX-1V system with 8 Tesla V100s 16G. If you use a different setup, we recommend you scale your hyperparameters by applying the following rules:
1. To use FP32, reduce the batch size to 2560 and set the `--update-freq 2` option.
2. To train on a fewer GPUs, multiply `--update-freq` by the reciprocal of the scaling factor.

For example, when training in FP32 mode on 4 GPUs, use the `--update-freq=4` option.

### Inference process

Inference on a raw input can be performed by piping file to be translated into the `inference.py` script. It requires a pre-trained model checkpoint, BPE codes file and dictionary file (both are produced by the `run_preprocessing.sh` script and can be found in the dataset directory).<br/>
In order to run interactive inference, run command:
```
python inference.py --path /path/to/your/checkpoint.pt --fuse-dropout-add --remove-bpe --bpe-codes /path/to/code/file
```
The `--buffer-size` option allows the batching of input sentences up to `--max_token` length.

To test model checkpoint accuracy on wmt14 test set run following command:

```bash
sacrebleu -t wmt14/full -l en-de --echo src | python inference.py --buffer-size 5000 --path /path/to/your/checkpoint.pt --max-tokens 10240 --fuse-dropout-add --remove-bpe --bpe-codes /data/code --fp16 | sacrebleu -t wmt14/full -l en-de -lc
```

## Performance

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in training and inference modes.

#### Training performance benchmark

To benchmark the training performance on a specific batch size, run `train.py` training script. Performance in tokens/s will be printed to standard output every N iterations, specified by the `--log-interval` option. Additionally performance and loss values will be logged by [dllogger](https://github.com/NVIDIA/dllogger) to the file specified in `--stat-file` option. Every line in the output file will be a valid JSON file prepended with `DLLL` prefix.

#### Inference performance benchmark

To benchmark the inference performance on a specific batch size, run following command to start the benchmark
```bash
for i in {1..10}; do sacrebleu -t wmt14/full -l en-de --echo src; done | python inference.py --buffer-size 5000 --path /path/to/your/checkpoint.pt --max-tokens 10240 --fuse-dropout-add --remove-bpe --bpe-codes /data/code --fp16 > /dev/null
```
Results will be printed to stderr.

### Results

The following sections provide details on how we achieved our performance and accuracy in training and inference.

#### Training accuracy results

Following the spirit of the paper [A Call for Clarity in Reporting BLEU Scores](https://arxiv.org/pdf/1804.08771.pdf) we decided to change evaluation metric implemented in fairseq to [SacreBleu](https://github.com/mjpost/sacreBLEU) score. We have calculated that the new metric has almost linear relationship with the old one. We run linear regression on nearly 2000 checkpoints to discover that the SacreBleu score almost perfectly follows the formula: newScore = 0.978 * oldScore - 0.05.
<p align="center">
    <img src="./bleu_relationship.png" />
    <br>
    Figure 2. Linear relationship between old and new BLEU metric.
</p>
To take into account the varibaility of the results we computed basic statistics that help us verify whether a model trains correctly. Evaluating nearly 2000 checkpoints from 20 runs, the best score we achieved is 28.09 BLEU (which corresponds to 28.77 old score). Variance of the score of the best performing model between those 20 runs is 0.011. Knowing that max statistic is skewed toward higher values we have also run studies which calculate threshold beyond which validation loss is no longer correlated with BLEU score.
Of course our hope is that dev's set distribution is similar to test's set distribution and when validation loss drops, BLEU score rises. But due to the finiteness of the validation and test sets we expect that there is such a loss value that makes performance on both sets decoupled from each other. To find this point we used Pearson correlation coefficient as a metric. The results indicate that optimizing beyond 4.02 validation loss value is no longer beneficial for the BLEU score. Further optimization does not cause overfitting but results become stochastic.
Mean BLEU score after reaching 4.02 validation loss is 27.38. We observe variance of 0.08, which translate to nearly 0.3 BLEU average difference between mean score and obtained score.
<p align="center">
    <img src="./decorrelation_threshold.png" />
    <br>
    Figure 3. Validation loss vs BLEU score. Plots are trimmed to certain validation loss threshold.
</p>

##### Training accuracy: NVIDIA DGX A100 (8x A100 40GB)
Our results were obtained by running the `run_DGXA100_AMP_8GPU.sh` and `run_DGXA100_TF32_8GPU.sh` training scripts in the pytorch-22.06-py3 NGC container on NVIDIA DGX A100 (8x A100 40GB) GPUs. We report average accuracy over 6 runs. We consider a model trained when it reaches minimal validation loss. Time to train contains only training time without validation. Depending on a configuration and frequency of validation it can take up to additional minute per epoch. 

| GPUs    | Batch size / GPU    | Accuracy - TF32  | Accuracy - mixed precision  |   Time to train - TF32  |  Time to train - mixed precision | Time to train speedup (TF32 to mixed precision)        
|---------|---------------------|------------------|-----------------------------|-------------------------|----------------------------------|------------------------------------
| 8       | 10240               | 27.92            | 27.76                       | 2.74 hours              | 2.64 hours                       | x1.04

##### Training accuracy: NVIDIA DGX-1 (8x V100 16GB)

Our results were obtained by running the `run_DGX1_AMP_8GPU.sh` and `run_DGX1_FP32_8GPU.sh` training scripts in the pytorch-22.06-py3 NGC container on NVIDIA DGX-1 (8x V100 16GB) GPUs. We report average accuracy over 6 runs. We consider a model trained when it reaches minimal validation loss. Time to train contains only training time without validation. Depending on a configuration and frequency of validation it can take up to additional minute per epoch. Using mixed precision we could fit a larger batch size in the memory, further speeding up the training.

| GPUs    | Batch size / GPU    | Accuracy - FP32  | Accuracy - mixed precision  |   Time to train - FP32  |  Time to train - mixed precision | Time to train speedup (FP32 to mixed precision)        
|---------|---------------------|------------------|-----------------------------|-------------------------|----------------------------------|------------------------------------
| 8       | 5120/2560           | 27.66            | 27.82                       | 11.8 hours              | 4.5  hours                       | x2.62

#### Training performance results

##### Training performance: NVIDIA DGX A100 (8x A100 40GB)

Our results were obtained by running the `run_DGXA100_AMP_8GPU.sh` and `run_DGXA100_TF32_8GPU.sh` training scripts in the pytorch-22.06-py3 NGC container on NVIDIA DGX A100 (8x A100 40GB) GPUs. Performance numbers (in tokens per second) were averaged over an entire training epoch.

| GPUs   | Batch size / GPU   | Throughput - TF32    | Throughput - mixed precision    | Throughput speedup (TF32 - mixed precision)   | Weak scaling - TF32    | Weak scaling - mixed precision        
|--------|--------------------|----------------------|---------------------------------|-----------------------------------------------|------------------------|-----
| 8      | 10240              | 347936               | 551599                          | x1.59                                         | 6.81                   | 6.72 
| 4      | 10240              | 179245               | 286081                          | x1.60                                         | 3.51                   | 3.49
| 1      | 10240              | 51057                | 82059                           | x1.60                                         | 1                      | 1


To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

##### Training stability test

The following plot shows average validation loss curves for different configs. We can see that training with AMP O2 converges slightly slower that FP32 and TF32 training. In order to mitigate this, you can use option `--amp-level O1` at the cost of 20% performance drop compared to the default AMP setting.

<p align="center">
    <img width="75%" hight="75%" src="./average_valid_loss.png" />
    <br>
    Figure 4. Validation loss curves
</p>

##### Training performance: NVIDIA DGX-1 (8x V100 16GB)

Our results were obtained by running the `run_DGX1_AMP_8GPU.sh` and `run_DGX1_FP32_8GPU.sh` training scripts in the pytorch-22.06-py3 NGC container on NVIDIA DGX-1 with (8x V100 16GB) GPUs. Performance numbers (in tokens per second) were averaged over an entire training epoch. Using mixed precision we could fit a larger batch size in the memory, further speeding up the training.

| GPUs   | Batch size / GPU   | Throughput - FP32    | Throughput - mixed precision    | Throughput speedup (FP32 - mixed precision)   | Weak scaling - FP32    | Weak scaling - mixed precision        
|--------|--------------------|----------------------|---------------------------------|-----------------------------------------------|------------------------|-----
| 8      | 5120/2560          | 59316                | 214656                          | x3.62                                         | 6.79                   | 6.52
| 4      | 5120/2560          | 30204                | 109726                          | x3.63                                         | 3.46                   | 3.33
| 1      | 5120/2560          | 8742                 | 32942                           | x3.77                                         | 1                      | 1


To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

##### Training performance: NVIDIA DGX-2 (16x V100 32GB)

Our results were obtained by running the `run_DGX1_AMP_8GPU.sh` and `run_DGX1_FP32_8GPU.sh` training scripts setting number of GPUs to 16 in the pytorch-22.06-py3 NGC container on NVIDIA DGX-2 with (16x V100 32GB) GPUs. Performance numbers (in tokens per second) were averaged over an entire training epoch. Using mixed precision we could fit a larger batch size in the memory, further speeding up the training.

| GPUs   | Batch size / GPU   | Throughput - FP32    | Throughput - mixed precision    | Throughput speedup (FP32 - mixed precision)   | Weak scaling - FP32    | Weak scaling - mixed precision        
|--------|--------------------|----------------------|---------------------------------|-----------------------------------------------|------------------------|-----
| 16     | 10240/5120         | 136253               | 517227                          | x3.80                                         | 13.87                  | 12.96
| 8      | 10240/5120         | 68929                | 267815                          | x3.89                                         | 7.01                   | 6.71
| 4      | 10240/5120         | 35216                | 137703                          | x3.91                                         | 3.58                   | 3.45
| 1      | 10240/5120         | 9827                 | 39911                           | x4.06                                         | 1                      | 1   


To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

#### Inference performance results

Our implementation of the Transformer has dynamic batching algorithm, which batches sentences together in such a way that there are no more than `N` tokens in each batch or no more than `M` sentences in each batch. In this benchmark we use the first option in order to get the most stable results.

##### Inference performance: NVIDIA DGX A100 (1x A100 40GB)

Our results were obtained by running the `inference.py` inferencing benchmarking script in the pytorch-22.06-py3 NGC container on NVIDIA DGX A100 (1x A100 40GB) GPU.

| Precision | Batch size | Throughput Avg | Latency Avg | Latency 90% |Latency 95% |Latency 99% |
|-----------|------------|----------------|-------------|-------------|------------|------------|
| TF32      |  10240     | 7105           | 1.22s       | 1.67s       | 1.67s      | 1.67s      |
| FP16      |  10240     | 7988           | 1.09s       | 1.73s       | 1.73s      | 1.73s      |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

##### Inference performance: NVIDIA DGX-1 (1x V100 16GB)

Our results were obtained by running the `inference.py` inferencing benchmarking script in the pytorch-22.06-py3 NGC container on NVIDIA DGX-1 with (1x V100 16GB) GPU.

| Precision | Batch size | Throughput Avg | Latency Avg | Latency 90% | Latency 95% | Latency 99% |
|-----------|------------|----------------|-------------|-------------|-------------|-------------|
| FP32      | 10240      | 3461           | 2.51s       | 3.19 s      | 3.19s       | 3.19s       |
| FP16      | 10240      | 5983           | 1.45s       | 2.03 s      | 2.03s       | 2.03s       |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).


## Release notes

### Changelog

February 2022:
- Update depricated calls in PyTorch CPP and Python API
- Update README with latest performance measurements

June 2020
- add TorchScript support
- Ampere support

March 2020
- remove language modeling from the repository
- one inference script for large chunks of data as well as for interactive demo
- change custom distributed strategy to APEX's DDP
- replace custom fp16 training with AMP
- major refactoring of the codebase

December 2019
- Change evaluation metric

August 2019
- add basic AMP support

July 2019
- Replace custom fused operators with jit functions

June 2019
- New README

March 2019
- Add mid-training [SacreBLEU](https://pypi.org/project/sacrebleu/1.2.10/) evaluation. Better handling of OOMs.

Initial commit, forked from [fairseq](https://github.com/pytorch/fairseq/commit/ac5fddfc691267285a84c81d39475411da5ed1c6)

## Known issues

- Using batch size greater than 16k causes indexing error in strided_batched_gemm module

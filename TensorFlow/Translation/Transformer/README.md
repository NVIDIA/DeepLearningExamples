# Transformer For Tensorflow

This repository provides a script and recipe to train the Transformer model to achieve state of the art accuracy, and is tested and maintained by NVIDIA.

**Table Of Contents**
- [Model overview](#model-overview)
    * [Model architecture](#model-architecture)
    * [Default configuration](#default-configuration)
    * [Feature support matrix](#feature-support-matrix)
	    * [Features](#features)
    * [Mixed precision training](#mixed-precision-training)
	    * [Enabling mixed precision](#enabling-mixed-precision)
    * [Glossary](#glossary)
- [Setup](#setup)
	* [Requirements](#requirements)
- [Quick Start Guide](#quick-start-guide)
- [Advanced](#advanced)
	* [Scripts and sample code](#scripts-and-sample-code)
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
		    * [NVIDIA DGX-1 (8x V100 16G)](#nvidia-dgx-1-(8x-v100-16G))
	    * [Training performance results](#training-performance-results)
		    * [NVIDIA DGX-1 (8x V100 16G)](#nvidia-dgx-1-(8x-v100-16G))
	    * [Inference performance results](#inference-performance-results)
- [Release notes](#release-notes)
    * [Changelog](#changelog)
    * [Known issues](#known-issues)

## Model overview

The Transformer is a Neural Machine Translation model which uses attention mechanism to boost training speed and overall accuracy. The Transformer has been introduced in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) and improved in [Scaling Neural Machine Translation](https://arxiv.org/abs/1806.00187).
This implementation is based on [MLPerf](https://mlperf.org/) reference implementation for TensorFlow. In order to enhance training performance we use [Horovod](https://github.com/horovod/horovod) and Automatic Mixed Precision.
Alternative implementation can be found in the official [TensorFlow repository](https://github.com/tensorflow/models/tree/master/official/transformer).

This model is trained with mixed precision using Tensor Cores on NVIDIA Volta and Turing GPUs. Therefore, researchers can get results 1.7x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.


### Model architecture

The Transformer model uses standard NMT encoder-decoder architecture. This model unlike other NMT models uses no recurrent connections and operates on fixed size context window.
The encoder stack is made up of N identical layers. Each layer is composed of the sublayers:
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

Full description of the Transformer architecture can be found in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper.

### Default configuration

The Transformer uses lossless tokenization scheme simillar to the one described in [SentencePiece](https://www.aclweb.org/anthology/D18-2012) paper. Tokenizer is trained upon joined source and target training files of [WMT14](http://statmt.org/wmt14/) dataset. Default vocabulary size is 33708, including all special tokens. Encoder and decode are using shared embeddings.

We use 6 blocks in each encoder and decoder stacks. Self attention layer computes it's outputs according to the following formula $`Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt(d_k)})V`$. At each attention step the model computes 16 different attention representations (which we will call attention heads) and concatenates them.

We trained the Transformer model using Lazy Adam optimizer with betas `(0.9, 0.997)` and epsilon `1e-9`. We used inverse square root training schedule preceded with liniar warmup of 6000 steps.
The implementation allows to perform training in mixed precision. We use dynamic loss scaling and Automatic Mixed Precision. Distributed multi-GPU and multi-Node is implemented with [Horovod](https://github.com/horovod/horovod) module with NCCL backend.

For inference we use beam search with default beam size of 4. Model performance is evaluated with BLEU4 metrics. For clarity we report internal (legacy) BLEU implementation as well as external [SacreBleu](https://github.com/mjpost/sacreBLEU) score.

### Feature support matrix

The following features are supported by this model.

| Feature                   | Transformer_TF
|---------------------------|--------------------------
| Horovod Multi-GPU (NCCL)  | Yes

#### Features

Horovod - Horovod is a distributed training framework for TensorFlow, Keras, PyTorch and MXNet. The goal of Horovod is to make distributed deep learning fast and easy to use. For more information about how to get started with Horovod, see the [Horovod: Official repository](https://github.com/horovod/horovod).

### Mixed precision training

Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format, while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [tensor cores](https://developer.nvidia.com/tensor-cores) in the Volta and Turing architecture, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using mixed precision training requires two steps:
1.  Porting the model to use the FP16 data type where appropriate.
2.  Adding loss scaling to preserve small gradient values.

The ability to train deep learning networks with lower precision was introduced in the Pascal architecture and first supported in [CUDA 8](https://devblogs.nvidia.com/parallelforall/tag/fp16/) in the NVIDIA Deep Learning SDK.

For information about:
-   How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html) documentation.
-   Techniques used for mixed precision training, see the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
-   How to access and enable AMP for TensorFlow, see [Using TF-AMP](https://docs.nvidia.com/deeplearning/dgx/tensorflow-user-guide/index.html#tfamp) from the TensorFlow User Guide.
-   APEX tools for mixed precision training, see the [NVIDIA Apex: Tools for Easy Mixed-Precision Training in PyTorch](https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/).

#### Enabling mixed precision

By supplying the --enable_amp flag to the transformer_main.py script while training in FP32, the following variables are set to their correct value for mixed precision training.

```python
if FLAGS.enable_amp:
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE'] = '1'
```

### Glossary

Attention layer - Layer that computes which elements of input sequence or it's hidden representation contribute the most to the currently considered output element.
Beam search - A heuristic search algorithm which at each step of predictions keeps N most possible outputs as a base to perform further prediction.
BPE - Binary Pair Encoding, compression algorithm that find most common pair of symbols in a data and replaces them with new symbol absent in the data.
EOS - End of a sentence.
Self attention layer - Attention layer that computes hidden representation of input using the same tensor as query, key and value.
Token - A  string that is representable within the model. We also refer to the token's position in the dictionary as a token. There are special non-string tokens: alphabet tokens (all characters in a dataset), EOS token, PAD token.
Tokenizer - Object that converts raw strings to seqiences of tokens.
Vocabulary embedding - Layer that projects one-hot token representations to a high dimensional space which preserves some information about corelations between tokens.

## Setup

The following section lists the requirements in order to start training the Transformer model.

### Requirements

This repository contains Dockerfile which extends the TensorFlow NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:

-   [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
-   TensorFlow 19.07-py3 NGC container
-   [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) based GPU

For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
-   [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
-   [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/dgx/user-guide/index.html#accessing_registry)
-   Running [framework name - link to topic]

For those unable to use the Tensorflow NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/dgx/support-matrix/index.html).

## Quick Start Guide

To train your model using mixed precision with tensor cores or using FP32, perform the following steps using the default parameters of the Transformer model on the WMT14 English German dataset. For the specifics concerning training and inference, see the [Advanced](#advanced) section.

1. Clone the repository.
	```
	git clone [https://github.com/NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples)
	cd [DeepLearningExamples/TensorFlow/Translation/Transformer](https://github.com/NVIDIA/DeepLearningExamples)/Translation/Transformer
	```

2.  Build the Transformer Tensorflow NGC container.
    ```
	docker build . -t transformer_tf
	```

3.  Start an interactive session in the NGC container to run training/inference.
    ```
	nvidia-docker run -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/research/transformer transformer_tf bash
	```

4.  Download and preprocess the dataset.
	Data will be downloaded to the `data/raw_data` directory and be preprocessed to the `data/processed_data` directory.
	```
	scripts/data_helper.sh
	```

5. 	Start training.
	To run training for a default configuration with a certain number of GPUs, run:
   	```
	scripts/run_training.sh <GPU nums>
   	```
	To run training for a custom configuration:
	```
	scripts/run_training.sh <GPU nums> <Learning rate> <steps> <warmup steps> <fp32, fp16> <enable xla> <target bleu score> <random seeds>
	```
	For example:
	```
	scripts/run_training.sh 8 1.0 60000 6000 fp32 false 28 1
   	```

6. Start validation/evaluation.
   The training process automatically runs evaluation and outputs the BLEU score after each training epoch. Additionally, after the training is done, you can manually run inference on test dataset with the checkpoint saved during the training.
   To launch mixed precision inference run
   ```
   export TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=1
   ```
   before running the script.
   ```
   python transformer/translate.py \
  	--data_dir=data/processed_data --model_dir=/results \
  	--file=newstest2014.en --file_out=trans.de --sentencepiece
   ```
   To run inference with multiple GPU, run:
   ```
   mpiexec --allow-run-as-root --bind-to socket -np 8 python transformer/translate.py \
  	--data_dir=data/processed_data --model_dir=/results \
  	--file=newstest2014.en --file_out=trans.de --report_throughput \
  	--batch_size=4096 --sentencepiece --enable_horovod
   ```

## Advanced
The following sections provide greater details of the dataset, running training and inference, and the training results.

### Scripts and sample code

In `transformer/` directory, the most important files are:
- `transformer_main.py`: Serves as the entry point to the application.
- `Dockerfile`: Container with the basic set of dependencies to run Transformer.
- `translate.py`: Perform inference using Transformer.

In `transformer/utils/` folder contains necessary tools to process data and train using Transformer. The main components are:
- `compute_bleu.py`: Implement bleu score computing.
- `dataset.py`: Implement dataset loading.
- `tokenizer`: Implement dictionary andtokenizer.

### Command-line options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option, for example:
`python3 transformer/transformer_main.py --help`

To summarize, the most useful arguments are as follows:
```
  --data_dir <DD>, -dd <DD>
                        [default: /tmp/translate_ende] Directory containing
                        training and evaluation data, and vocab file used for
                        encoding.
  --model_dir <MD>, -md <MD>
                        [default: /tmp/transformer_model] Directory to save
                        Transformer model training checkpoints
  --save_checkpoints_steps <SC>, -sc <SC>
                        Save checkpoint every <SC> train iterations.
  --params <P>, -p <P>  [default: big] Parameter set to use when creating and
                        training the model.
  --batch_size <B>, -b <B>
                        Override default batch size parameter in prams
  --learning_rate <LR>, -lr <LR>
                        Override default learning rate parameter in params
  --warmup_steps <WS>, -ws <WS>
                        Override default warmup_steps parameter in params
  --train_steps <TS>, -ts <TS>
                        Total number of training steps. If both --train_epochs
                        and --train_steps are not set, the model will train
                        for 10 epochs.
  --steps_between_eval <SBE>, -sbe <SBE>
                        [default: 1000] Number of training steps to run
                        between evaluations.
  --bleu_source <BS>, -bs <BS>
                        Path to source file containing text translate when
                        calculating the official BLEU score. Both
                        --bleu_source and --bleu_ref must be set. The BLEU
                        score will be calculated during model evaluation.
  --bleu_ref <BR>, -br <BR>
                        Path to file containing the reference translation for
                        calculating the official BLEU score. Both
                        --bleu_source and --bleu_ref must be set. The BLEU
                        score will be calculated during model evaluation.
  --bleu_threshold <BT>, -bt <BT>
                        Stop training when the uncased BLEU score reaches this
                        value. Setting this overrides the total number of
                        steps or epochs set by --train_steps or
                        --train_epochs.
  --sentencepiece, -sp  Use SentencePiece tokenizer. Warning: In order to use
                        SP you have to preprocess dataset with SP as well
  --random_seed <SEED>, -rs <SEED>
                        The random seed to use.
  --enable_xla, -enable_xla
                        Enable JIT compile with XLA.
  --enable_amp, -enable_amp
                        Enable mixed-precision, fp16 where possible.
  --enable_horovod, -enable_hvd
                        Enable multi-gpu training with horovod
  --report_loss, -rl    Report throughput and loss in alternative format
```

### Getting the data

This section needs to cover the following:
-   Longer description of the training and evaluation data, for example:
	-   What is the name of the dataset?
	-   Provide context about the dataset download scripts.
	-   Provide context about preprocessing the data.
 -   Some guidance on what to change if training with a different dataset (as SAs have indicated that this will be a common mode of usage in the wild)

The Transformer model was trained on the [WMT14 English-German](http://statmt.org/wmt14/translation-task.html#Download) dataset. Concatenation of the *commoncrawl*, *europarl* and *news-commentary* is used as train and vaidation dataset and *newstest2014* is used as test dataset.

This repository contains `scripts/data_helper.sh` script which will automatically download and preprocess the training and test datasets. By default data will be stored in `/data/wmt14_en_de_joined_dict` directory.
There are two kind of tokenizer in our scripts, [SentencePiece](https://github.com/google/sentencepiece) or custom tokenizer, to perform tokenization of the dataset and segment text into subword units. By default, script builds shared vocabulary of 33708 tokens, which is constistent withbuilds shared vocabulary of 33708 tokens, which is constistent with [Scaling Neural Machine Translation](https://arxiv.org/abs/1806.00187).

#### Dataset guidelines

The process of loading, normalizing and augmenting the data contained in the dataset can be found in the `utils/dataset.py` script.

Initially, data is loaded from `TFRecord` file and convert to 256x? Tensors, which has totally 4096 tokens.

### Training process

The default training configuration can be launched ny running the `transformer/transformer_main.py` training script. By default, the script saves one checkpoint evety 2000 iter. Specify save directory with `--model_dir` option.

In order to run multi-GPU training launch the training script with `mpiexec --allow-run-as-root --bind-to socket -np $N` prepended and use`--enable_horovod` option, where N is the number of GPUs.
We have tested reliance on up to 8 GPUs on a single node.

After each training epoch, the script runs a loss validation on the validation split of the dataset and outputs validation loss. By default the evaluation after each epoch is disabled. To enable it use `--bleu_source $file` and `--bleu_ref $file` option or to use BLEU score value as training stopping condition use `--bleu_threshold $score` option. BLEU is computed by the internal (legacy) BLEU implementation in `utils/compute_bleu.py` as well as external [SacreBleu](https://github.com/mjpost/sacreBLEU).

By default, the `transformer_main.py` script will launch fp32 training without Tensor Cores. To use mixed precision with Tensor Cores use `--enable_amp` option.

### Inference process

Inference on a raw input can be performed by launching `translate.py` inference script. It requires pre-trained model checkpoint and dictionary file (which is produced by `data/process_data.py` script and can be found in the dataset directory).

The scripts run inference with a default beam size of 4 and give text output.

In order to run inference run command:
```
python3 transformer/translate.py --data_dir=<path/to/data> --params=<big, base> --batch_size=<batch size> --file=<path/to/source/file> --file_out=<path/to/target/output> <--sentencepiece>
```

## Benchmarking

The following section shows how to run benchmarks measuring the model performance in training and inference modes.

### Training performance benchmark

To benchmark the training performance on a specific batch size, run:
```
mpiexec --allow-run-as-root --bind-to socket -np <gpu number> python3 transformer/transformer_main.py --data_dir=<path/to/data> --enable_horovod --params=<big, base> --batch_size=<batch size> --train_steps=<step number> --steps_between_eval=<step number> --report_loss <--enable_amp> <--sentencepiece>
```

### Inference performance benchmark

To benchmark the inference performance on a specific batch size, run:
```
mpiexec --allow-run-as-root --bind-to socket -np <gpu number> python3 transformer/translate.py --data_dir=<path/to/data> --enable_horovod --params=<big, base> --batch_size=<batch size> --file=<path/to/source/file> --file_out=<path/to/target/output> --report_throughput <--sentencepiece>
```

## Results

The following sections provide details on how we achieved our performance and accuracy in training and inference.

### Training accuracy results

In order to test accuracy of our implementation we have run experiments with batch size 4096 per GPU and learining rate 0.5 in the tensorflow-19.07-py3 Docker container. Plot below shows BLEU score changes.

Running this code with the provided hyperparameters will allow you to achieve the following results. Our setup is a DGX-1 with 8x Tesla V100 16GB.

| GPU count | Mixed precision BLEU | fp32 BLEU
|-----------|----------------------|-----------
| 8         | 28.13                | 28.06

### Training performance results

#### NVIDIA DGX-1 (8x V100 16G)

Our results were obtained by running the `scripts/run_training.sh` training script in the TensorFlow 19.07-py3 NGC container on NVIDIA DGX-1 with (8x V100 16G) GPUs. Performance numbers (in tokens per second) were averaged over an entire training epoch.

| GPUs   | Batch size / GPU   | Throughput - FP32    | Throughput - mixed precision    | Throughput speedup (FP32 - mixed precision)   | Weak scaling - FP32    | Weak scaling - mixed precision
|--------|--------------------|----------------------|---------------------------------|-----------------------------------------------|------------------------|------------------------------
| 8      | 4096               | 62783.1              | 104703                          | 1.67                                          |  7.64                  | 7.37                 |
| 4      | 4096               | 31599.2              | 52718.7                         | 1.67                                          |  3.84                  | 3.71                 |
| 1      | 4096               | 8212.7               | 14196.1                         | 1.72                                          |  1                     | 1                    |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

### Inference performance results

#### NVIDIA DGX-1 (8x V100 16G)

We have benchmarked the inference performance by running `transformer/translate.py` script using TensorFlow 19.07-py3 NGC Docker container. Inference was run on a single GPU.
```
python3 transformer/translate.py --data_dir=data/processed_data --batch_size=100 --file=newstest2014.en --file_out=trans.de --report_throughput --sentencepiece
```
And for mixed precision inference run
```
export TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=1
```
before running the script.

GPU | Throughput - Mixed precision | Throughput - FP32 | FP16/Mixed speedup
---|---|---|---
Tesla V100-SXM2-16GB | 24.7 | 14.36 | 1.72

## Release notes

### Changelog

June 2019
- Initial release

## Known issues

There are no known issues in this release.
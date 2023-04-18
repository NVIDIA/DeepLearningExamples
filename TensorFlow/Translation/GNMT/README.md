# GNMT v2 For TensorFlow

This repository provides a script and recipe to train the GNMT v2 model to achieve state-of-the-art accuracy and is tested and maintained by NVIDIA.
GNMT model for TensorFlow1 is no longer maintained and will soon become unavailable, please consider PyTorch or TensorFlow2 models as a substitute for your requirements.

## Table Of Contents
- [Model overview](#model-overview)
    * [Model architecture](#model-architecture)
    * [Default configuration](#default-configuration)
    * [Feature support matrix](#feature-support-matrix)
        * [Features](#features)
    * [Mixed precision training](#mixed-precision-training)
        * [Enabling mixed precision](#enabling-mixed-precision)
        * [Enabling TF32](#enabling-tf32)
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
        * [Validation process](#validation-process)
        * [Translation process](#translation-process)
- [Performance](#performance)
    * [Benchmarking](#benchmarking)
        * [Training performance benchmark](#training-performance-benchmark)
        * [Inference performance benchmark](#inference-performance-benchmark)
    * [Results](#results)
        * [Training accuracy results](#training-accuracy-results)
            * [Training accuracy: NVIDIA DGX A100 (8x A100 40GB)](#training-accuracy-nvidia-dgx-a100-8x-a100-40gb)
            * [Training accuracy: NVIDIA DGX-1 (8x V100 16GB)](#training-accuracy-nvidia-dgx-1-8x-v100-16gb)
            * [Training stability test](#training-stability-test)
        * [Inference accuracy results](#inference-accuracy-results)
            * [Inference accuracy: NVIDIA DGX-1 (8x V100 16GB)](#inference-accuracy-nvidia-dgx-1-8x-v100-16gb)
        * [Training performance results](#training-performance-results)
            * [Training performance: NVIDIA DGX A100 (8x A100 40GB)](#training-performance-nvidia-dgx-a100-8x-a100-40gb)
            * [Training performance: NVIDIA DGX-1 (8x V100 16GB)](#training-performance-nvidia-dgx-1-8x-v100-16gb)
        * [Inference performance results](#inference-performance-results)
            * [Inference performance: NVIDIA DGX A100 (1x A100 40GB)](#inference-performance-nvidia-dgx-a100-1x-a100-40gb)
            * [Inference performance: NVIDIA DGX-1 (1x V100 16GB)](#inference-performance-nvidia-dgx-1-1x-v100-16gb)
            * [Inference performance: NVIDIA T4](#inference-performance-nvidia-t4)
- [Release notes](#release-notes)
    * [Changelog](#changelog)
    * [Known issues](#known-issues)


## Model overview

The GNMT v2 model is similar to the one discussed in the [Google's Neural Machine
Translation System: Bridging the Gap between Human and Machine
Translation](https://arxiv.org/abs/1609.08144) paper.

The most important difference between the two models is in the attention
mechanism. In our model, the output from the first LSTM layer of the decoder
goes into the attention module, then the re-weighted context is concatenated
with inputs to all subsequent LSTM layers in the decoder at the current
timestep.

The same attention mechanism is also implemented in the default
GNMT-like models from
[TensorFlow Neural Machine Translation Tutorial](https://github.com/tensorflow/nmt)
and
[NVIDIA OpenSeq2Seq Toolkit](https://github.com/NVIDIA/OpenSeq2Seq).

This model is trained with mixed precision using Tensor Cores on Volta, Turing, and the NVIDIA Ampere GPU architectures.  Therefore, researchers can get results 2x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.

### Model architecture

The following image shows the GNMT model architecture:

![TrainingLoss](./img/diagram.png)

### Default configuration

The following features were implemented in this model:

* general:
  * encoder and decoder are using shared embeddings
  * data-parallel multi-GPU training
  * dynamic loss scaling with backoff for Tensor Cores (mixed precision) training
  * trained with label smoothing loss (smoothing factor 0.1)

* encoder:
  * 4-layer LSTM, hidden size 1024, first layer is bidirectional, the rest are
    unidirectional
  * with residual connections starting from 3rd layer
  * dropout is applied on input to all LSTM layers, probability of dropout is
    set to 0.2
  * hidden state of LSTM layers is initialized with zeros
  * weights and bias of LSTM layers is initialized with uniform (-0.1, 0.1)
    Distribution

* decoder:
  * 4-layer unidirectional LSTM with hidden size 1024 and fully-connected
    classifier
  * with residual connections starting from 3rd layer
  * dropout is applied on input to all LSTM layers, probability of dropout is
    set to 0.2
  * hidden state of LSTM layers is initialized with the last hidden state from
    encoder
  * weights and bias of LSTM layers is initialized with uniform (-0.1, 0.1)
    distribution
  * weights and bias of fully-connected classifier is initialized with
    uniform (-0.1, 0.1) distribution

* attention:
  * normalized Bahdanau attention
  * output from first LSTM layer of decoder goes into attention,
  then re-weighted context is concatenated with the input to all subsequent
  LSTM layers of the decoder at the current timestep
  * linear transform of keys and queries is initialized with uniform (-0.1, 0.1), normalization scalar is initialized with 1.0 / sqrt(1024),     normalization bias is initialized with zero

* inference:
  * beam search with default beam size of 5
  * with coverage penalty and length normalization, coverage penalty factor is
    set to 0.1, length normalization factor is set to 0.6 and length
    normalization constant is set to 5.0
  * de-tokenized BLEU computed by [SacreBLEU](https://github.com/awslabs/sockeye/tree/master/sockeye_contrib/sacrebleu)
  * [motivation](https://github.com/awslabs/sockeye/tree/master/sockeye_contrib/sacrebleu#motivation) for choosing SacreBLEU

When comparing the BLEU score, there are various tokenization approaches and
BLEU calculation methodologies; therefore, ensure you align similar metrics.

Code from this repository can be used to train a larger, 8-layer GNMT v2 model.
Our experiments show that a 4-layer model is significantly faster to train and
yields comparable accuracy on the public
[WMT16 English-German](http://www.statmt.org/wmt16/translation-task.html)
dataset. The number of LSTM layers is controlled by the `--num_layers` parameter
in the `nmt.py` script.

### Feature support matrix

The following features are supported by this model.

| **Feature** | **GNMT TF** |
|:---:|:--------:|
| Automatic Mixed Precision | yes |


#### Features

The following features are supported by this model.

* Automatic Mixed Precision (AMP) - Computation graphs can be modified by TensorFlow on runtime to support mixed precision training. Detailed explanation of mixed precision can be found in the next section.



### Mixed precision training

Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format, while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in Volta, and following with both the Turing and Ampere architectures, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using [mixed precision training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) previously required two steps:
1.  Porting the model to use the FP16 data type where appropriate.
2.  Adding loss scaling to preserve small gradient values.

This can now be achieved using Automatic Mixed Precision (AMP) for TensorFlow to enable the full [mixed precision methodology](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#tensorflow) in your existing TensorFlow model code.  AMP enables mixed precision training on Volta, Turing, and NVIDIA Ampere GPU architectures automatically. The TensorFlow framework code makes all necessary model changes internally.

In TF-AMP, the computational graph is optimized to use as few casts as necessary and maximize the use of FP16, and the loss scaling is automatically applied inside of supported optimizers. AMP can be configured to work with the existing tf.contrib loss scaling manager by disabling the AMP scaling with a single environment variable to perform only the automatic mixed-precision optimization. It accomplishes this by automatically rewriting all computation graphs with the necessary operations to enable mixed precision training and automatic loss scaling.

For information about:
-   How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) documentation.
-   Techniques used for mixed precision training, see the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
-   How to access and enable AMP for TensorFlow, see [Using TF-AMP](https://docs.nvidia.com/deeplearning/dgx/tensorflow-user-guide/index.html#tfamp) from the TensorFlow User Guide.
-   APEX tools for mixed precision training, see the [NVIDIA Apex: Tools for Easy Mixed-Precision Training in PyTorch](https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/).

#### Enabling mixed precision

Mixed precision is enabled in TensorFlow by using the Automatic Mixed Precision (TF-AMP) extension which casts variables to half-precision upon retrieval, while storing variables in single-precision format. Furthermore, to preserve small gradient magnitudes in backpropagation, a [loss scaling](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#lossscaling) step must be included when applying gradients. In TensorFlow, loss scaling can be applied statically by using simple multiplication of loss by a constant value or automatically, by TF-AMP. Automatic mixed precision makes all the adjustments internally in TensorFlow, providing two benefits over manual operations. First, programmers need not modify network model code, reducing development and maintenance effort. Second, using AMP maintains forward and backward compatibility with all the APIs for defining and running TensorFlow models.

To enable mixed precision, you can simply add the values to the environmental variables inside your training script:
- Enable TF-AMP graph rewrite:
  ```
  os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
  ```

- Enable Automated Mixed Precision:
  ```
  os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
  ```

#### Enabling TF32

TensorFloat-32 (TF32) is the new math mode in [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) GPUs for handling the matrix math also called tensor operations. TF32 running on Tensor Cores in A100 GPUs can provide up to 10x speedups compared to single-precision floating-point math (FP32) on Volta GPUs.

TF32 Tensor Cores can speed up networks using FP32, typically with no loss of accuracy. It is more robust than FP16 for models which require high dynamic range for weights or activations.

For more information, refer to the [TensorFloat-32 in the A100 GPU Accelerates AI Training, HPC up to 20x](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) blog post.

TF32 is supported in the NVIDIA Ampere GPU architecture and is enabled by default.

## Setup

The following section lists the requirements that you need to meet in order to start training the GNMT v2 model.

### Requirements

This repository contains Dockerfile which extends the TensorFlow NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:
-   [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
-   [TensorFlow 20.06-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)
-   Supported GPUs:
    - [NVIDIA Volta architecture](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
    - [NVIDIA Turing architecture](https://www.nvidia.com/en-us/geforce/turing/)
    - [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)

For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
-   [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
-   [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)
-   [Running TensorFlow](https://docs.nvidia.com/deeplearning/dgx/tensorflow-release-notes/running.html#running).

For those unable to use the TensorFlow NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).


## Quick Start Guide

To train your model using mixed or TF32 precision with Tensor Cores or using FP32,
perform the following steps using the default parameters of the GNMT v2 model
on the WMT16 English German dataset.
For the specifics concerning training and inference, see the [Advanced](#advanced) section.

**1. Clone the repository.**
```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/TensorFlow/Translation/GNMT
```

**2. Build the GNMT v2 TensorFlow container.**
```
bash scripts/docker/build.sh
```

**3. Start an interactive session in the NGC container to run.** training/inference.
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

All results and logs are saved to the `results` directory (on the host) or to
the `/workspace/gnmt/results` directory (in the container). The training script
saves the checkpoint after every training epoch and after every 2000 training steps
within each epoch. You can modify the results directory using the `--output_dir`
argument.


To launch mixed precision training on 1 GPU, run:

```
python nmt.py --output_dir=results --batch_size=128 --learning_rate=5e-4 --amp
```

To launch mixed precision training on 8 GPUs, run:

```
python nmt.py --output_dir=results --batch_size=1024 --num_gpus=8 --learning_rate=2e-3 --amp
```

To launch FP32 (TF32 on NVIDIA Ampere GPUs) training on 1 GPU, run:

```
python nmt.py --output_dir=results --batch_size=128 --learning_rate=5e-4
```

To launch FP32 (TF32 on NVIDIA Ampere GPUs) training on 8 GPUs, run:

```
python nmt.py --output_dir=results --batch_size=1024 --num_gpus=8 --learning_rate=2e-3
```

**6. Start evaluation.**

The training process automatically runs evaluation and outputs the BLEU score
after each training epoch. Additionally, after the training is done, you can
manually run inference on test dataset with the checkpoint saved during the
training.

To launch mixed precision inference on 1 GPU, run:

```
python nmt.py --output_dir=results --infer_batch_size=128 --mode=infer --amp
```

To launch FP32 (TF32 on NVIDIA Ampere GPUs) inference on 1 GPU, run:

```
python nmt.py --output_dir=results --infer_batch_size=128 --mode=infer
```

**7. Start translation.**

After the training is done, you can translate custom sentences with the checkpoint saved during the training.

```bash
echo "The quick brown fox jumps over the lazy dog" >file.txt
python nmt.py --output_dir=results --mode=translate --translate-file=file.txt
cat file.txt.trans
```
```
Der schnelle braune Fuchs springt über den faulen Hund
```

## Advanced

The following sections provide greater details of the dataset, running training
and inference, and the training results.

### Scripts and sample code
In the root directory, the most important files are:

* `nmt.py`: serves as the entry point to launch the training
* `Dockerfile`: container with the basic set of dependencies to run GNMT v2
* `requirements.txt`: set of extra requirements for running GNMT v2
* `attention_wrapper.py`, `gnmt_model.py`, `model.py`: model definition
* `estimator.py`: functions for training and inference

In the `script` directory, the most important files are:
* `translate.py`: wrapped on `nmt.py` for benchmarking and running inference
* `parse_log.py`: script for retrieving information in JSON format from the training log
* `wmt16_en_de.sh`: script for downloading and preprocessing the dataset

In the `script/docker` directory, the files are:
* `build.sh`: script for building the GNMT container
* `interactive.sh`: script for running the GNMT container interactively

### Parameters

The most useful arguments are as follows:

```
  --learning_rate LEARNING_RATE
                        Learning rate.
  --warmup_steps WARMUP_STEPS
                        How many steps we inverse-decay learning.
  --max_train_epochs MAX_TRAIN_EPOCHS
                        Max number of epochs.
  --target_bleu TARGET_BLEU
                        Target bleu.
  --data_dir DATA_DIR   Training/eval data directory.
  --translate_file TRANSLATE_FILE
                        File to translate, works only with translate mode
  --output_dir OUTPUT_DIR
                        Store log/model files.
  --batch_size BATCH_SIZE
                        Total batch size.
  --log_step_count_steps LOG_STEP_COUNT_STEPS
                        The frequency, in number of global steps, that the
                        global step and the loss will be logged during training
  --num_gpus NUM_GPUS   Number of gpus in each worker.
  --random_seed RANDOM_SEED
                        Random seed (>0, set a specific seed).
  --ckpt CKPT           Checkpoint file to load a model for inference.
                        (defaults to newest checkpoint)
  --infer_batch_size INFER_BATCH_SIZE
                        Batch size for inference mode.
  --beam_width BEAM_WIDTH
                        beam width when using beam search decoder. If 0, use
                        standard decoder with greedy helper.
  --amp                 use amp for training and inference
  --mode {train_and_eval,infer,translate}
```

### Command-line options

To see the full list of available options and their descriptions, use the `-h`
or `--help` command line option, for example:

```
python nmt.py --help
```



### Getting the data

The GNMT v2 model was trained on the
[WMT16 English-German](http://www.statmt.org/wmt16/translation-task.html)
dataset and newstest2014 is used as a testing dataset.

This repository contains the `scripts/wmt16_en_de.sh` download script which
automatically downloads and preprocesses the training and test datasets. By
default, data is downloaded to the `data` directory.

Our download script is very similar to the `wmt16_en_de.sh` script from the
[tensorflow/nmt](https://github.com/tensorflow/nmt/blob/master/nmt/scripts/wmt16_en_de.sh)
repository. Our download script contains an extra preprocessing step, which
discards all pairs of sentences which can't be decoded by latin-1 encoder.
The `scripts/wmt16_en_de.sh` script uses the
[subword-nmt](https://github.com/rsennrich/subword-nmt)
package to segment text into subword units (Byte Pair Encodings - [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding)). By default, the script builds
the shared vocabulary of 32,000 tokens.

In order to test with other datasets, the scripts need to be customized accordingly.

#### Dataset guidelines

The process of downloading and preprocessing the data can be found in the
`scripts/wmt16_en_de.sh` script.

Initially, data is downloaded from [www.statmt.org](www.statmt.org). Then, `europarl-v7`,
`commoncrawl` and `news-commentary` corpora are concatenated to form the
training dataset, similarly `newstest2015` and `newstest2016` are concatenated
to form the validation dataset. Raw data is preprocessed with
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

The training configuration can be launched by running the `nmt.py` script.
By default, the training script saves the checkpoint after every training epoch
and after every 2000 training steps within each epoch.
Results are stored in the `results` directory.

The training script launches data-parallel training on multiple GPUs. We have
tested reliance on up to 8 GPUs on a single node.

After each training epoch, the script runs an evaluation and outputs a BLEU
score on the test dataset (*newstest2014*). BLEU is computed by the
[SacreBLEU](https://github.com/awslabs/sockeye/tree/master/sockeye_contrib/sacrebleu)
package. Logs from the training and evaluation are saved to the `results`
directory.

The training script automatically runs testing after each training epoch. The
results from the testing are printed to the standard output and saved to the
log files.

The summary after each training epoch is printed in the following format:

```
training time for epoch 1: 29.37 mins (2918.36 sent/sec, 139640.48 tokens/sec)
[...]
bleu is 20.50000
eval time for epoch 1: 1.57 mins (78.48 sent/sec, 4283.88 tokens/sec)
```
The BLEU score is computed on the test dataset.
Performance is reported in total sentences per second and in total tokens per
second. The performance result is averaged over an entire training epoch and
summed over all GPUs participating in the training.


To view all available options for training, run `python nmt.py --help`.

### Inference process

Validation and translation can be run by launching the `nmt.py` script, although, it requires a
pre-trained model checkpoint and tokenized input (for validation) and  non-tokenized input (for translation).

#### Validation process

The `nmt.py` script supports batched validation (`--mode=infer` flag). By
default, it launches beam search with beam size of 5, coverage penalty term and
length normalization term. Greedy decoding can be enabled by setting the
`--beam_width=1` flag for the `nmt.py` inference script. To control the
batch size use the `--infer_batch_size` flag.

To view all available options for validation, run `python nmt.py --help`.

#### Translation process

The `nmt.py` script supports batched translation (`--mode=translate` flag). By
default, it launches beam search with beam size of 5, coverage penalty term and
length normalization term. Greedy decoding can be enabled by setting the
`--beam_width=1` flag for the `nmt.py` prediction script. To control the
batch size use the `--infer_batch_size` flag.

The input file may contain many sentences, each on a new line. The file can be specified
by the `--translate_file <file>` flag. This script will create a new file called `<file>.trans`,
with translation of the input file.

To view all available options for translation, run `python nmt.py --help`.


## Performance

The performance measurements in this document were conducted at the time of publication and may not reflect the performance achieved from NVIDIA’s latest software release. For the most up-to-date performance measurements, go to [NVIDIA Data Center Deep Learning Product Performance](https://developer.nvidia.com/deep-learning-performance-training-inference).

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in training and inference modes.

#### Training performance benchmark

To benchmark training performance, run:

* `python nmt.py --output_dir=results --max_train_epochs=1 --num_gpus <num GPUs> --batch_size <total batch size> --amp` for mixed precision
* `python nmt.py --output_dir=results --max_train_epochs=1 --num_gpus <num GPUs> --batch_size <total batch size>` for FP32/TF32


The log file will contain training performance in the following format:

```
training time for epoch 1: 25.75 mins (3625.19 sent/sec, 173461.27 tokens/sec)
```

#### Inference performance benchmark

To benchmark inference performance, run the `scripts/translate.py` script:

* For FP32/TF32:
    `python scripts/translate.py --output_dir=/path/to/trained/model --beam_width <comma separated beam widths> --infer_batch_size <comma separated batch sizes>`

* For mixed precision
    `python scripts/translate.py --output_dir=/path/to/trained/model --amp --beam_width <comma separated beam widths> --infer_batch_size <comma separated batch sizes>`

The benchmark requires a checkpoint from a fully trained model.

### Results

The following sections provide details on how we achieved our performance and
accuracy in training and inference.


#### Training accuracy results

##### Training accuracy: NVIDIA DGX A100 (8x A100 40GB)

Our results were obtained by running the `examples/DGXA100_{TF32,AMP}_8GPU.sh`
training script in the tensorflow-20.06-tf1-py3 NGC container
on NVIDIA DGX A100 (8x A100 40GB) GPUs.

| **GPUs** | **Batch size / GPU** |**Accuracy - mixed precision (BLEU)** | **Accuracy - TF32 (BLEU)** | **Time to train - mixed precision** | **Time to train - TF32** | **Time to train speedup (TF32 to mixed precision)** |
| --- | --- | ----- | ----- | -------- | -------- | ---- |
|  8  | 128 | 25.1 | 24.31 | 96 min  | 139 min  | 1.45 |

##### Training accuracy: NVIDIA DGX-1 (8x V100 16GB)

Our results were obtained by running the `nmt.py` script in the
tensorflow-19.07-py3 NGC container on NVIDIA DGX-1 with (8x V100 16GB)  GPUs.

| **GPUs** | **Batch size / GPU** |**Accuracy - mixed precision (BLEU)** | **Accuracy - FP32 (BLEU)** | **Time to train - mixed precision** | **Time to train - FP32** | **Time to train speedup (FP32 to mixed precision)** |
| --- | --- | ----- | ----- | -------- | -------- | ---- |
|  1  | 128 | 24.90 | 24.84 | 763 min  | 1237 min | 1.62 |
|  8  | 128 | 24.33 | 24.34 | 168 min  | 237 min  | 1.41 |


In the following plot, the BLEU scores after each training epoch for different
configurations are displayed.

![BLEUScore](./img/bleu_score.png)


##### Training stability test

The GNMT v2 model was trained for 6 epochs, starting from 6 different initial
random seeds. After each training epoch, the model was evaluated on the test
dataset and the BLEU score was recorded. The training was performed in the
tensorflow-20.06-tf1-py3 NGC container.

In the following tables, the BLEU scores after each training epoch for different
initial random seeds are displayed.

###### NVIDIA DGX A100 with 8 Ampere A100 40GB GPUs with TF32.

| Epoch | Average | Standard deviation | Minimum | Median | Maximum |
| ----- | ------- | ------------------ | ------- | ------ | ------- |
| 1     | 20.272  | 0.165              | 19.760  | 20.295 | 20.480  |
| 2     | 21.911  | 0.145              | 21.650  | 21.910 | 22.230  |
| 3     | 22.731  | 0.140              | 22.490  | 22.725 | 23.020  |
| 4     | 23.142  | 0.164              | 22.930  | 23.090 | 23.440  |
| 5     | 23.967  | 0.137              | 23.760  | 23.940 | 24.260  |
| 6     | 24.358  | 0.143              | 24.120  | 24.360 | 24.610  |

###### NVIDIA DGX-1 with 8 Tesla V100 16GB GPUs with FP32.

| Epoch | Average | Standard deviation | Minimum | Median | Maximum |
| ----- | ------- | ------------------ | ------- | ------ | ------- |
| 1     | 20.259  | 0.225              | 19.820  | 20.300 | 20.590  |
| 2     | 21.954  | 0.194              | 21.540  | 21.955 | 22.370  |
| 3     | 22.729  | 0.150              | 22.480  | 22.695 | 23.110  |
| 4     | 23.218  | 0.210              | 22.820  | 23.225 | 23.470  |
| 5     | 23.921  | 0.114              | 23.680  | 23.910 | 24.080  |
| 6     | 24.381  | 0.131              | 24.160  | 24.375 | 24.590  |

#### Inference accuracy results

##### Inference accuracy: NVIDIA DGX-1 (8x V100 16GB)

Our results were obtained by running the `scripts/translate.py` script in the tensorflow-19.07-py3 NGC container on NVIDIA DGX-1 8x V100 16GB GPUs.

* For mixed precision: `python scripts/translate.py --output_dir=/path/to/trained/model --beam_width 1,2,5 --infer_batch_size 128 --amp`

* For FP32: `python scripts/translate.py --output_dir=/path/to/trained/model --beam_width 1,2,5 --infer_batch_size 128`

| **Batch size** | **Beam size** | **Mixed precision BLEU** | **FP32 BLEU** |
|:---:|:---:|:---:|:---:|
|128|1|23.80|23.80|
|128|2|24.58|24.59|
|128|5|25.10|25.09|

#### Training performance results

##### Training performance: NVIDIA DGX A100 (8x A100 40GB)

Our results were obtained by running the `examples/DGXA100_{TF32,AMP}_{1,8}GPU.sh`
training script in the tensorflow-20.06-tf1-py3 NGC container
on NVIDIA DGX A100 (8x A100 40GB) GPUs.
Performance numbers (in items/images per second)
were averaged over an entire training epoch.

| **GPUs** | **Batch size / GPU** | **Throughput - mixed precision (tokens/s)** | **Throughput - TF32 (tokens/s)** | **Throughput speedup (TF32 - mixed precision)** | **Weak scaling - mixed precision** | **Weak scaling - TF32** |
| --- | --- | ------- | ------- | ---- | ---- | ---- |
|  1  | 128 |  29 911 |  31 110 | 0.96 | 1.00 | 1.00 |
|  8  | 128 | 181 384 | 175 292 | 1.03 | 6.06 | 5.63 |



To achieve these same results, follow the steps in the
[Quick Start Guide](#quick-start-guide).

##### Training performance: NVIDIA DGX-1 (8x V100 16GB)

Our results were obtained by running the `nmt.py` script in the tensorflow-19.07-py3 NGC container on NVIDIA DGX-1 with 8x V100 16G GPUs.
Performance numbers (in tokens per second) were averaged over an entire
training epoch.

| **GPUs** | **Batch size / GPU** | **Throughput - mixed precision (tokens/s)** | **Throughput - FP32 (tokens/s)** | **Throughput speedup (FP32 - mixed precision)** | **Weak scaling - mixed precision** | **Weak scaling - FP32** |
| --- | --- | ------- | ------ | ---- | ---- | ---- |
|  1  | 128 | 23 011  | 14 106 | 1.63 | 1.00 | 1.00 |
|  8  | 128 | 138 106 | 93 688 | 1.47 | 6.00 | 6.64 |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide)
outlined above.

#### Inference performance results

The benchmark requires a checkpoint from a fully trained model.

To launch the inference benchmark in mixed precision on 1 GPU, run:

```
python scripts/translate.py --output_dir=/path/to/trained/model --beam_width 1,2,5 --infer_batch_size 1,2,4,8,32,128,512 --amp
```

To launch the inference benchmark in FP32/TF32 on 1 GPU, run:

```
python scripts/translate.py --output_dir=/path/to/trained/model --beam_width 1,2,5 --infer_batch_size 1,2,4,8,32,128,512
```

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide)
outlined above.

##### Inference performance: NVIDIA DGX A100 (1x A100 40GB)

Our results were obtained by running the
`python scripts/translate.py --infer_batch_size 1,2,4,8,32,128,512 --beam_width 1,2,5 {--amp}`
inferencing benchmarking script in the tensorflow-20.06-tf1-py3 NGC container
on NVIDIA DGX A100 (1x A100 40GB) GPU.

FP16

| **Batch size**     | **Beam width**     | **Bleu**           | **Sentences/sec**  | **Tokens/sec**     | **Latency Avg**    | **Latency 50%**     | **Latency 90%**     | **Latency 95%**     | **Latency 99%**     | **Latency 100%**    |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1              | 1              | 23.80          | 13.67          | 737.89         | 73.15          | 67.69          | 121.98         | 137.20         | 162.74         | 201.06         |
| 1              | 2              | 24.58          | 13.40          | 721.18         | 74.65          | 69.12          | 123.99         | 138.82         | 169.58         | 198.49         |
| 1              | 5              | 25.10          | 12.12          | 647.78         | 82.53          | 76.53          | 136.35         | 152.59         | 196.09         | 216.55         |
| 2              | 1              | 23.80          | 21.55          | 1163.16        | 92.82          | 88.15          | 139.88         | 152.49         | 185.18         | 208.35         |
| 2              | 2              | 24.58          | 21.07          | 1134.42        | 94.91          | 89.62          | 142.08         | 158.12         | 188.00         | 205.08         |
| 2              | 5              | 25.10          | 19.59          | 1047.21        | 102.10         | 96.20          | 152.36         | 172.46         | 211.96         | 219.87         |
| 4              | 1              | 23.80          | 36.98          | 1996.27        | 108.16         | 105.07         | 150.42         | 161.56         | 200.99         | 205.87         |
| 4              | 2              | 24.57          | 34.92          | 1880.48        | 114.53         | 111.42         | 160.29         | 177.14         | 205.32         | 211.80         |
| 4              | 5              | 25.10          | 31.56          | 1687.34        | 126.74         | 122.06         | 179.68         | 201.38         | 225.08         | 229.14         |
| 8              | 1              | 23.80          | 64.52          | 3482.81        | 123.99         | 122.89         | 159.89         | 174.66         | 201.12         | 205.59         |
| 8              | 2              | 24.57          | 59.04          | 3178.17        | 135.50         | 135.23         | 180.50         | 191.66         | 214.95         | 216.84         |
| 8              | 5              | 25.09          | 55.51          | 2967.82        | 144.11         | 141.98         | 198.39         | 218.88         | 223.55         | 225.61         |
| 32             | 1              | 23.80          | 193.54         | 10447.04       | 165.34         | 163.56         | 211.67         | 215.37         | 221.07         | 221.14         |
| 32             | 2              | 24.57          | 182.00         | 9798.09        | 175.82         | 176.04         | 220.33         | 224.25         | 226.45         | 227.05         |
| 32             | 5              | 25.10          | 141.63         | 7572.02        | 225.94         | 225.59         | 278.38         | 279.56         | 281.61         | 282.13         |
| 128            | 1              | 23.80          | 556.57         | 30042.59       | 229.98         | 226.81         | 259.05         | 260.26         | 260.74         | 260.85         |
| 128            | 2              | 24.57          | 400.02         | 21535.38       | 319.98         | 328.23         | 351.31         | 352.82         | 353.01         | 353.06         |
| 128            | 5              | 25.10          | 235.14         | 12570.95       | 544.35         | 576.62         | 581.95         | 582.64         | 583.61         | 583.85         |
| 512            | 1              | 23.80          | 903.83         | 48786.58       | 566.48         | 570.44         | 579.74         | 580.66         | 581.39         | 581.57         |
| 512            | 2              | 24.58          | 588.63         | 31689.07       | 869.81         | 894.90         | 902.65         | 902.85         | 903.00         | 903.04         |
| 512            | 5              | 25.10          | 285.86         | 15283.40       | 1791.06        | 1835.19        | 1844.29        | 1845.59        | 1846.63        | 1846.89        |

TF32

| **Batch size**     | **Beam width**     | **Bleu**           | **Sentences/sec**  | **Tokens/sec**     | **Latency Avg**    | **Latency 50%**     | **Latency 90%**     | **Latency 95%**     | **Latency 99%**     | **Latency 100%**    |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1              | 1              | 23.82          | 13.25          | 715.47         | 75.45          | 69.81          | 125.63         | 141.89         | 169.70         | 209.78         |
| 1              | 2              | 24.59          | 13.21          | 711.16         | 75.72          | 70.06          | 124.75         | 140.20         | 173.23         | 201.39         |
| 1              | 5              | 25.08          | 12.38          | 661.99         | 80.76          | 74.90          | 131.93         | 148.91         | 187.05         | 208.39         |
| 2              | 1              | 23.82          | 21.61          | 1166.56        | 92.55          | 87.25          | 139.54         | 151.77         | 180.24         | 209.05         |
| 2              | 2              | 24.59          | 21.24          | 1143.63        | 94.17          | 88.78          | 139.70         | 156.61         | 189.09         | 205.06         |
| 2              | 5              | 25.10          | 19.49          | 1042.17        | 102.62         | 96.14          | 153.38         | 172.89         | 213.99         | 219.54         |
| 4              | 1              | 23.81          | 35.84          | 1934.49        | 111.62         | 108.73         | 154.52         | 165.42         | 207.88         | 211.29         |
| 4              | 2              | 24.58          | 34.71          | 1869.20        | 115.24         | 111.24         | 161.24         | 177.73         | 208.12         | 212.74         |
| 4              | 5              | 25.09          | 32.24          | 1723.86        | 124.07         | 119.35         | 177.54         | 196.69         | 221.10         | 223.52         |
| 8              | 1              | 23.80          | 64.08          | 3459.74        | 124.84         | 123.61         | 161.92         | 177.06         | 205.47         | 206.47         |
| 8              | 2              | 24.61          | 59.31          | 3193.52        | 134.89         | 133.44         | 182.92         | 192.71         | 216.04         | 218.78         |
| 8              | 5              | 25.10          | 56.60          | 3026.29        | 141.35         | 138.61         | 194.52         | 213.65         | 220.24         | 221.45         |
| 32             | 1              | 23.80          | 195.31         | 10544.22       | 163.85         | 162.80         | 212.71         | 215.41         | 216.92         | 217.34         |
| 32             | 2              | 24.61          | 185.66         | 9996.59        | 172.36         | 171.07         | 216.46         | 221.64         | 223.68         | 225.25         |
| 32             | 5              | 25.11          | 147.24         | 7872.61        | 217.34         | 214.97         | 269.75         | 270.71         | 271.44         | 272.87         |
| 128            | 1              | 23.81          | 576.54         | 31123.19       | 222.02         | 219.25         | 249.44         | 249.75         | 249.88         | 249.91         |
| 128            | 2              | 24.57          | 419.87         | 22609.82       | 304.86         | 314.47         | 332.18         | 334.13         | 336.22         | 336.74         |
| 128            | 5              | 25.10          | 245.76         | 13138.84       | 520.83         | 552.68         | 558.89         | 559.09         | 559.13         | 559.13         |
| 512            | 1              | 23.80          | 966.24         | 52156.34       | 529.89         | 534.82         | 558.30         | 559.33         | 560.16         | 560.36         |
| 512            | 2              | 24.58          | 642.41         | 34590.81       | 797.00         | 812.40         | 824.23         | 825.92         | 827.27         | 827.61         |
| 512            | 5              | 25.10          | 289.33         | 15468.09       | 1769.61        | 1817.19        | 1849.83        | 1855.17        | 1859.45        | 1860.51        |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

##### Inference performance: NVIDIA DGX-1 (1x V100 16GB)

Our results were obtained by running the
`python scripts/translate.py --infer_batch_size 1,2,4,8,32,128,512 --beam_width 1,2,5 {--amp}`
inferencing benchmarking script in the tensorflow-20.06-tf1-py3 NGC container
on NVIDIA DGX-1 with (1x V100 16GB) GPU.

FP16

| **Batch size** | **Sequence length** | **Throughput Avg** | **Latency Avg** | **Latency 90%** |**Latency 95%** |**Latency 99%** |
|------------|-----------------|-----|-----|-----|-----|-----|
| 1              | 1              | 23.78          | 9.06           | 489.00         | 110.41         | 102.80         | 183.54         | 206.33         | 242.44         | 306.21         |
| 1              | 2              | 24.58          | 8.68           | 467.35         | 115.22         | 107.17         | 188.75         | 212.36         | 258.15         | 306.15         |
| 1              | 5              | 25.09          | 8.39           | 448.32         | 119.25         | 109.79         | 195.68         | 220.56         | 276.41         | 325.65         |
| 2              | 1              | 23.82          | 14.59          | 787.70         | 137.04         | 129.38         | 206.35         | 224.94         | 267.30         | 318.60         |
| 2              | 2              | 24.57          | 14.44          | 777.60         | 138.51         | 131.07         | 206.67         | 228.95         | 275.56         | 311.23         |
| 2              | 5              | 25.11          | 13.78          | 736.99         | 145.11         | 136.76         | 216.01         | 243.24         | 299.28         | 315.88         |
| 4              | 1              | 23.82          | 23.79          | 1284.24        | 168.14         | 164.13         | 234.70         | 248.42         | 308.38         | 325.46         |
| 4              | 2              | 24.59          | 22.67          | 1220.66        | 176.45         | 171.40         | 243.76         | 271.92         | 314.79         | 330.19         |
| 4              | 5              | 25.08          | 22.33          | 1194.00        | 179.12         | 174.04         | 253.36         | 281.88         | 318.76         | 340.01         |
| 8              | 1              | 23.81          | 43.33          | 2338.68        | 184.63         | 183.25         | 237.66         | 266.73         | 305.89         | 315.03         |
| 8              | 2              | 24.60          | 39.12          | 2106.44        | 204.49         | 200.96         | 276.05         | 294.53         | 327.61         | 335.50         |
| 8              | 5              | 25.10          | 37.16          | 1987.05        | 215.26         | 210.92         | 295.65         | 323.83         | 337.09         | 343.03         |
| 32             | 1              | 23.82          | 129.52         | 6992.15        | 247.06         | 245.81         | 317.71         | 325.54         | 330.09         | 335.04         |
| 32             | 2              | 24.55          | 123.28         | 6637.86        | 259.57         | 261.07         | 319.13         | 333.45         | 338.75         | 342.57         |
| 32             | 5              | 25.05          | 88.74          | 4744.33        | 360.61         | 359.27         | 446.65         | 448.40         | 455.93         | 461.86         |
| 128            | 1              | 23.80          | 332.81         | 17964.83       | 384.60         | 382.14         | 434.46         | 436.71         | 439.64         | 440.37         |
| 128            | 2              | 24.59          | 262.87         | 14153.59       | 486.93         | 506.45         | 528.87         | 530.90         | 533.09         | 533.64         |
| 128            | 5              | 25.08          | 143.91         | 7695.36        | 889.42         | 932.93         | 965.67         | 966.26         | 966.53         | 966.59         |
| 512            | 1              | 23.80          | 613.57         | 33126.42       | 834.46         | 848.06         | 868.21         | 869.04         | 869.70         | 869.86         |
| 512            | 2              | 24.59          | 387.72         | 20879.62       | 1320.54        | 1343.05        | 1354.40        | 1356.50        | 1358.19        | 1358.61        |
| 512            | 5              | 25.10          | 199.48         | 10664.34       | 2566.67        | 2628.50        | 2642.59        | 2644.73        | 2646.44        | 2646.86        |


FP32

| **Batch size** | **Sequence length** | **Throughput Avg** | **Latency Avg** | **Latency 90%** |**Latency 95%** |**Latency 99%** |
|------------|-----------------|-----|-----|-----|-----|-----|
| 1              | 1              | 23.80          | 8.37           | 451.86         | 119.46         | 111.26         | 199.36         | 224.49         | 269.03         | 330.72         |
| 1              | 2              | 24.59          | 8.83           | 475.11         | 113.31         | 104.54         | 187.79         | 210.64         | 260.42         | 317.45         |
| 1              | 5              | 25.09          | 7.74           | 413.92         | 129.15         | 119.44         | 212.84         | 239.52         | 305.47         | 349.09         |
| 2              | 1              | 23.80          | 13.96          | 753.79         | 143.22         | 135.73         | 213.96         | 235.89         | 284.62         | 330.71         |
| 2              | 2              | 24.59          | 12.96          | 697.63         | 154.33         | 145.01         | 230.88         | 255.31         | 306.71         | 340.36         |
| 2              | 5              | 25.09          | 12.67          | 677.23         | 157.88         | 148.24         | 236.50         | 266.91         | 322.94         | 349.55         |
| 4              | 1              | 23.80          | 22.42          | 1209.97        | 178.44         | 172.70         | 247.51         | 266.07         | 326.95         | 343.86         |
| 4              | 2              | 24.59          | 20.55          | 1106.07        | 194.68         | 188.83         | 271.75         | 295.08         | 345.76         | 364.00         |
| 4              | 5              | 25.09          | 21.19          | 1132.58        | 188.81         | 182.77         | 268.18         | 298.53         | 331.96         | 357.36         |
| 8              | 1              | 23.80          | 39.32          | 2122.26        | 203.48         | 201.89         | 263.28         | 286.71         | 332.70         | 348.93         |
| 8              | 2              | 24.59          | 37.51          | 2019.43        | 213.26         | 211.55         | 283.67         | 302.28         | 338.47         | 356.51         |
| 8              | 5              | 25.09          | 31.69          | 1694.02        | 252.46         | 245.33         | 348.95         | 378.16         | 392.72         | 401.73         |
| 32             | 1              | 23.80          | 118.51         | 6396.93        | 270.02         | 269.22         | 337.17         | 352.12         | 361.36         | 361.40         |
| 32             | 2              | 24.59          | 100.23         | 5395.33        | 319.28         | 318.89         | 399.80         | 403.12         | 414.51         | 423.41         |
| 32             | 5              | 25.09          | 68.59          | 3666.77        | 466.55         | 466.84         | 581.77         | 586.42         | 589.04         | 593.41         |
| 128            | 1              | 23.80          | 256.49         | 13845.09       | 499.04         | 492.36         | 562.12         | 567.20         | 571.18         | 572.18         |
| 128            | 2              | 24.59          | 176.83         | 9519.12        | 723.86         | 754.89         | 792.12         | 793.86         | 796.44         | 797.09         |
| 128            | 5              | 25.09          | 96.21          | 5143.17        | 1330.48        | 1420.94        | 1427.91        | 1431.02        | 1435.23        | 1436.28        |
| 512            | 1              | 23.80          | 366.07         | 19759.97       | 1398.63        | 1421.81        | 1457.81        | 1461.04        | 1463.63        | 1464.27        |
| 512            | 2              | 24.59          | 225.48         | 12137.77       | 2270.75        | 2323.62        | 2338.62        | 2340.94        | 2342.80        | 2343.27        |
| 512            | 5              | 25.09          | 106.02         | 5667.78        | 4829.31        | 4946.65        | 4956.15        | 4957.85        | 4959.21        | 4959.55        |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

##### Inference performance: NVIDIA T4

Our results were obtained by running the `scripts/translate.py` script in the tensorflow-19.07-py3 NGC container on NVIDIA T4.

Reported mixed precision speedups are relative to FP32 numbers for corresponding configuration.

| **Batch size** | **Beam size** | **Mixed precision tokens/s** | **Speedup** | **Mixed precision average latency (ms)** | **Average latency speedup** | **Mixed precision latency 50% (ms)** | **Latency 50% speedup** | **Mixed precision latency 90% (ms)** | **Latency 90% speedup** | **Mixed precision latency 95% (ms)** | **Latency 95% speedup** | **Mixed precision latency 99% (ms)** | **Latency 99% speedup** | **Mixed precision latency 100% (ms)** | **Latency 100% speedup** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1     | 1     | 643   | 1.278 | 84    | 1.278 | 78    | 1.279 | 138   | 1.309 | 154   | 1.312 | 180   | 1.304 | 220   | 1.296 |
| 1     | 2     | 584   | 1.693 | 92    | 1.692 | 86    | 1.686 | 150   | 1.743 | 168   | 1.737 | 201   | 1.770 | 236   | 1.742 |
| 1     | 5     | 552   | 1.702 | 97    | 1.701 | 90    | 1.696 | 158   | 1.746 | 176   | 1.738 | 218   | 1.769 | 244   | 1.742 |
| 2     | 1     | 948   | 1.776 | 114   | 1.776 | 108   | 1.769 | 170   | 1.803 | 184   | 1.807 | 218   | 1.783 | 241   | 1.794 |
| 2     | 2     | 912   | 1.761 | 118   | 1.760 | 112   | 1.763 | 175   | 1.776 | 192   | 1.781 | 226   | 1.770 | 246   | 1.776 |
| 2     | 5     | 832   | 1.900 | 128   | 1.900 | 121   | 1.910 | 192   | 1.912 | 214   | 1.922 | 258   | 1.922 | 266   | 1.905 |
| 4     | 1     | 1596  | 1.792 | 135   | 1.792 | 132   | 1.791 | 187   | 1.799 | 197   | 1.815 | 241   | 1.784 | 245   | 1.796 |
| 4     | 2     | 1495  | 1.928 | 144   | 1.927 | 141   | 1.926 | 201   | 1.927 | 216   | 1.936 | 250   | 1.956 | 264   | 1.890 |
| 4     | 5     | 1308  | 1.702 | 164   | 1.702 | 159   | 1.702 | 230   | 1.722 | 251   | 1.742 | 283   | 1.708 | 288   | 1.699 |
| 8     | 1     | 2720  | 1.981 | 159   | 1.981 | 158   | 1.992 | 204   | 1.975 | 219   | 1.986 | 249   | 1.987 | 252   | 1.966 |
| 8     | 2     | 2554  | 1.809 | 169   | 1.808 | 168   | 1.829 | 224   | 1.797 | 237   | 1.783 | 260   | 1.807 | 262   | 1.802 |
| 8     | 5     | 1979  | 1.768 | 216   | 1.768 | 213   | 1.780 | 292   | 1.797 | 319   | 1.793 | 334   | 1.760 | 336   | 1.769 |
| 32    | 1     | 7449  | 1.775 | 232   | 1.774 | 231   | 1.777 | 292   | 1.789 | 300   | 1.760 | 301   | 1.768 | 301   | 1.768 |
| 32    | 2     | 5569  | 1.670 | 309   | 1.669 | 311   | 1.672 | 389   | 1.652 | 392   | 1.665 | 401   | 1.651 | 404   | 1.644 |
| 32    | 5     | 3079  | 1.867 | 556   | 1.867 | 555   | 1.865 | 692   | 1.858 | 695   | 1.860 | 702   | 1.847 | 703   | 1.847 |
| 128   | 1     | 12986 | 1.662 | 532   | 1.662 | 529   | 1.667 | 607   | 1.643 | 608   | 1.645 | 609   | 1.647 | 609   | 1.647 |
| 128   | 2     | 7856  | 1.734 | 878   | 1.734 | 911   | 1.755 | 966   | 1.742 | 967   | 1.741 | 968   | 1.744 | 968   | 1.744 |
| 128   | 5     | 3361  | 1.683 | 2036  | 1.682 | 2186  | 1.678 | 2210  | 1.673 | 2210  | 1.674 | 2211  | 1.674 | 2211  | 1.674 |
| 512   | 1     | 14932 | 1.825 | 1851  | 1.825 | 1889  | 1.808 | 1927  | 1.801 | 1928  | 1.800 | 1929  | 1.800 | 1930  | 1.799 |
| 512   | 2     | 8109  | 1.786 | 3400  | 1.786 | 3505  | 1.783 | 3520  | 1.782 | 3523  | 1.781 | 3525  | 1.781 | 3525  | 1.781 |
| 512   | 5     | 3370  | 1.802 | 8123  | 1.801 | 8376  | 1.798 | 8391  | 1.804 | 8394  | 1.804 | 8396  | 1.805 | 8397  | 1.805 |


## Release notes

### Changelog
1. Mar 18, 2019
  * Initial release
2. June, 2019
  * Performance improvements
3. June, 2020
  * Updated performance tables to include A100 results
4. April 2023
  * Ceased maintenance of this model in TensorFlow1

### Known issues
There are no known issues in this release.

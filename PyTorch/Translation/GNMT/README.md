# GNMT v2

## The model
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

## Default configuration of the GNMT v2 model

* general:
  * encoder and decoder are using shared embeddings
  * data-parallel multi-gpu training
  * dynamic loss scaling with backoff for Tensor Cores (mixed precision) training
  * trained with label smoothing loss (smoothing factor 0.1)
* encoder:
  * 4-layer LSTM, hidden size 1024, first layer is bidirectional, the rest are
    unidirectional
  * with residual connections starting from 3rd layer
  * uses LSTM layer accelerated by cuDNN
* decoder:
  * 4-layer unidirectional LSTM with hidden size 1024 and fully-connected
    classifier
  * with residual connections starting from 3rd layer
  * uses LSTM layer accelerated by cuDNN
* attention:
  * normalized Bahdanau attention
  * output from first LSTM layer of decoder goes into attention,
  then re-weighted context is concatenated with the input to all subsequent
  LSTM layers of the decoder at the current timestep
* inference:
  * beam search with default beam size of 5
  * with coverage penalty and length normalization terms
  * detokenized BLEU computed by [SacreBLEU](https://github.com/awslabs/sockeye/tree/master/contrib/sacrebleu)
  * [motivation](https://github.com/awslabs/sockeye/tree/master/contrib/sacrebleu#motivation) for choosing SacreBLEU

When comparing the BLEU score there are various tokenization approaches and BLEU
calculation methodologies, ensure to align similar metrics.

Code from this repository can be used to train a larger, 8-layer GNMT v2 model.
Our experiments show that a 4-layer model is significantly faster to train and
yields comparable accuracy on the public
[WMT16 English-German](http://www.statmt.org/wmt16/translation-task.html)
dataset. The number of LSTM layers is controlled by the `num_layers` parameter
in the `scripts/train.sh` training script.

# Setup
## Requirements
* [PyTorch 18.06-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch)
(or newer)
* [SacreBLEU 1.2.10](https://pypi.org/project/sacrebleu/1.2.10/)

This repository contains `Dockerfile` which extends the PyTorch NGC container
and encapsulates all dependencies.

For more information about how to get started with NGC containers, see the
following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning
DGX Documentation:
[Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html),
[Accessing And Pulling From The NGC container registry](https://docs.nvidia.com/deeplearning/dgx/user-guide/index.html#accessing_registry)
and
[Running PyTorch](https://docs.nvidia.com/deeplearning/dgx/pytorch-release-notes/running.html#running).

## Training using mixed precision with Tensor Cores
Before you can train using mixed precision with Tensor Cores, ensure that you
have a
[NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
based GPU.
For information about how to train using mixed precision, see the
[Mixed Precision Training paper](https://arxiv.org/abs/1710.03740)
and
[Training With Mixed Precision documentation](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html).

Another option for adding mixed-precision support is available from NVIDIA’s
[APEX](https://github.com/NVIDIA/apex), A PyTorch Extension, that contains
utility libraries, such as AMP, which require minimal network code changes to
leverage Tensor Core performance.

# Quick start guide
Perform the following steps to run the training using the default parameters of
the GNMT v2 model on the *WMT16 English-German* dataset.
### 1. Build and launch the GNMT Docker container
```
bash scripts/docker/build.sh
bash scripts/docker/interactive.sh
```

### 2. Download the training dataset
Download and preprocess the WMT16 English-German dataset. Data will be
downloaded to the `data` directory (on the host). The `data` directory is
mounted to the `/workspace/gnmt/data` location in the Docker container.
```
bash scripts/wmt16_en_de.sh
```

### 3. Run training
By default, the training script will use all available GPUs. The training script
saves only one checkpoint with the lowest value of the loss function on the
validation dataset. All results and logs are saved to the `results` directory
(on the host) or to the `/workspace/gnmt/results` directory (in the container).
By default, the `scripts/train.sh` script will launch mixed precision training
with Tensor Cores. You can change this behaviour by setting the `--math fp32`
flag in the `scripts/train.sh` script.
```
bash scripts/train.sh
```
The training script automatically runs the validation and testing after each
training epoch. The results from the validation and testing are printed to
the standard error (stderr) and saved to log files.

The summary after each training epoch is printed in the following format:
```
Summary: Epoch: 3	Training Loss: 3.1735	Validation Loss: 3.0511	Test BLEU: 21.89
Performance: Epoch: 3  Training: 300155 Tok/s  Validation: 156066 Tok/s
```
The training loss is averaged over an entire training epoch, the validation loss
is averaged over the validation dataset and the BLEU score is computed by
the SacreBLEU package on the test dataset.
Performance is reported in total tokens per second. The result is averaged over
an entire training epoch and summed over all GPUs participating in the training.

# Details
## Getting the data
The GNMT v2 model was trained on the
[WMT16 English-German](http://www.statmt.org/wmt16/translation-task.html) dataset.
Concatenation of the *newstest2015* and *newstest2016* test sets are used as a
validation dataset and the *newstest2014* is used as a testing dataset.

This repository contains the `scripts/wmt16_en_de.sh` download script which will
automatically download and preprocess the training, validation and test
datasets. By default, data will be downloaded to the `data` directory.

Our download script is very similar to the `wmt16_en_de.sh` script from the
[tensorflow/nmt](https://github.com/tensorflow/nmt/blob/master/nmt/scripts/wmt16_en_de.sh)
repository. Our download script contains an extra preprocessing step, which
discards all pairs of sentences which can't be decoded by *latin-1* encoder.

The `scripts/wmt16_en_de.sh` script uses the
[subword-nmt](https://github.com/rsennrich/subword-nmt)
package to segment text into subword units (BPE). By default, the script builds
the shared vocabulary of 32,000 tokens.

## Running training
The default training configuration can be launched by running the
`scripts/train.sh` training script.
By default, the training script saves only one checkpoint with the lowest value
of the loss function on the validation dataset, an evaluation is performed after
each training epoch. Results are stored in the `results/gnmt_wmt16` directory.

The training script launches data-parallel training with batch size 128 per GPU
on all available GPUs. After each training epoch, the script runs an evaluation
on the validation dataset and outputs a BLEU score on the test dataset
(*newstest2014*). BLEU is computed by the
[SacreBLEU](https://github.com/awslabs/sockeye/tree/master/contrib/sacrebleu)
package. Logs from the training and evaluation are saved to the `results`
directory.

Even though the training script uses all available GPUs, you can change this
behavior by setting the `CUDA_VISIBLE_DEVICES` variable in your environment or
by setting the `NV_GPU` variable at the Docker container launch
([see section "GPU isolation"](https://github.com/NVIDIA/nvidia-docker/wiki/nvidia-docker#gpu-isolation)).

By default, the `scripts/train.sh` script will launch mixed precision training
with Tensor Cores. You can change this behaviour by setting the `--math fp32`
flag in the `scripts/train.sh` script.

Internally, the `scripts/train.sh` script uses `train.py`. To view all available
options for training, run `python3 train.py --help`.

## Running inference
Inference can be run by launching the `translate.py` inference script, although,
it requires a pre-trained model checkpoint and tokenized input.

The inference script, `translate.py`, supports batched inference. By default, it
launches beam search with beam size of 5, coverage penalty term and length
normalization term. Greedy decoding can be enabled by setting the beam size to 1.

To view all available options for inference, run `python3 translate.py --help`.

## Benchmarking scripts
### Training performance benchmark
The `scripts/benchmark_training.sh` benchmarking script runs a few, relatively
short training sessions and automatically collects performance numbers. The
benchmarking script assumes that the `scripts/wmt16_en_de.sh` data download
script was launched and the datasets are available in the default location
(`data` directory).

Results from the benchmark are stored in the `results` directory. After the
benchmark is done, you can launch the `scripts/parse_train_benchmark.sh` script
to generate a short summary which will contain launch configuration, performance
(in tokens per second), and estimated training time needed for one epoch (in
seconds).

### Inference performance and accuracy benchmark
The `scripts/benchmark_inference.sh` benchmarking script launches a number of
inference runs with different hyperparameters (beam size, batch size, arithmetic
type) on sorted and unsorted *newstest2014* test dataset. Performance and
accuracy results are stored in the `results/inference_benchmark` directory.
BLEU score is computed by the SacreBLEU package.

The `scripts/benchmark_inference.sh` script assumes that the
`scripts/wmt16_en_de.sh` data download script was
launched and the datasets are available in the default location (`data`
directory).

The `scripts/benchmark_inference.sh` script requires a pre-trained
model checkpoint. By default, the script is loading a checkpoint from the
`results/gnmt_wmt16/model_best.pth` location.

## Training Accuracy Results
All results were obtained by running the `scripts/train.sh` script in
the pytorch-18.06-py3 Docker container on NVIDIA DGX-1 with 8 V100 16G GPUs.


| **number of GPUs** | **mixed precision BLEU** | **fp32 BLEU** | **mixed precision training time** | **fp32 training time** |
| ------------------ | ------------------------ | ------------- | --------------------------------- | ---------------------- |
|         1          |        22.54		          |     22.25		  |          412 minutes              |       948 minutes      |
|         4          |        22.45		          |     22.46		  |          118 minutes              |       264 minutes      |
|         8          |        22.41		          |     22.43		  |          64 minutes               |       139 minutes      |

![TrainingLoss](./img/training_loss.png)

### Training Stability Test
The GNMT v2 model was trained for 10 epochs, starting from 96 different initial
random seeds. After each training epoch the model was evaluated on the test
dataset and the BLEU score was recorded. The training was performed in the
pytorch-18.06-py3 Docker container on NVIDIA DGX-1 with 8 V100 16G GPUs. The
following table summarizes results of the stability test.

![TrainingAccuracy](./img/training_accuracy.png)

## Training Performance Results
All results were obtained by running the `scripts/train.sh` training script in
the pytorch-18.06-py3 Docker container on NVIDIA DGX-1 with 8 V100 16G GPUs.
Performance numbers (in tokens per second) were averaged over an entire training
epoch.

| **number of GPUs** | **mixed precision tokens/s** | **fp32 tokens/s** | **mixed precision speedup** | **mixed precision multi-gpu weak scaling** | **fp32 multi-gpu weak scaling** |
| -------- | ------------- | ------------- | ------------ | --------------------------- | --------------------------- |
|    1     |    42337      |   18581       |   2.279      |        1.000                |        1.000                |
|    4     |    153433     |   67586       |   2.270      |        3.624                |        3.637                |
|    8     |    300181     |   132734      |   2.262      |        7.090                |        7.144                |

## Inference Performance Results
All results were obtained by running the `scripts/benchmark_inference.sh`
benchmarking script in the pytorch-18.06-py3 Docker container on NVIDIA DGX-1.
Inference was run on a single V100 16G GPU.

| **batch size** | **beam size** | **mixed precision BLEU** | **fp32 BLEU** | **mixed precision tokens/s** | **fp32 tokens/s** |
| -------------- | ------------- | ------------- | ------------- | ----------------- | ------------ |
|      512       |       1       |     20.63     |    20.63      |     62009  	     |    31229     |
|      512       |       2       |     21.55     |    21.60      |     32669  	     |    16454     |
|      512       |       5       |     22.34     |    22.36      |     21105  	     |    8562      |
|      512       |       10      |     22.34     |    22.40      |     12967  	     |    4720      |
|      128       |       1       |     20.62     |    20.63      |     27095  	     |    19505     |
|      128       |       2       |     21.56     |    21.60      |     13224  	     |    9718      |
|      128       |       5       |     22.38     |    22.36      |     10987  	     |    6575      |
|      128       |       10      |     22.35     |    22.40      |     8603          |    4103      |
|      32        |       1       |     20.62     |    20.63      |     9451   	     |    8483      |
|      32        |       2       |     21.56     |    21.60      |     4818          |    4333      |
|      32        |       5       |     22.34     |    22.36      |     4505          |    3655      |
|      32        |       10      |     22.37     |    22.40      |     4086          |    2822      |

# Changelog
1. Aug 7, 2018
  * Initial release

# Known issues
None

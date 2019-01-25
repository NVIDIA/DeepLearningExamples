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
in the `train.py` training script.

# Setup
## Requirements
* [PyTorch 18.11-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch)
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
based GPU. Other platforms might likely work but aren't officially supported.
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
By default, the `train.py` script will launch mixed precision training
with Tensor Cores. You can change this behaviour by setting the `--math fp32`
flag for the `train.py` training script.

Launching training on 1, 4 or 8 GPUs:

```
python3 -m launch train.py --seed 2 --train-global-batch-size 1024
```

Launching training on 16 GPUs:

```
python3 -m launch train.py --seed 2 --train-global-batch-size 2048
```

By default the training script will launch training with batch size 128 per GPU.
If specified `--train-global-batch-size` is larger than 128 times the number of
GPUs available for the training then the training script will accumulate
gradients over consecutive iterations and then perform the weight update.
For example 1 GPU training with `--train-global-batch-size 1024` will accumulate
gradients over 8 iterations before doing the weight update with accumulated
gradients.

The training script automatically runs the validation and testing after each
training epoch. The results from the validation and testing are printed to
the standard output (stdout) and saved to log files.

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

In order to test with other datasets, scripts need to be customized accordingly.

## Running training
The default training configuration can be launched by running the
`train.py` training script.
By default, the training script saves only one checkpoint with the lowest value
of the loss function on the validation dataset, an evaluation is performed after
each training epoch. Results are stored in the `results/gnmt_wmt16` directory.

The training script launches data-parallel training with batch size 128 per GPU
on all available GPUs. We have tested reliance on up to 16 GPUs on a single
node.
After each training epoch, the script runs an evaluation
on the validation dataset and outputs a BLEU score on the test dataset
(*newstest2014*). BLEU is computed by the
[SacreBLEU](https://github.com/awslabs/sockeye/tree/master/contrib/sacrebleu)
package. Logs from the training and evaluation are saved to the `results`
directory.

Even though the training script uses all available GPUs, you can change this
behavior by setting the `CUDA_VISIBLE_DEVICES` variable in your environment or
by setting the `NV_GPU` variable at the Docker container launch
([see section "GPU isolation"](https://github.com/NVIDIA/nvidia-docker/wiki/nvidia-docker#gpu-isolation)).

By default, the `train.py` script will launch mixed precision training
with Tensor Cores. You can change this behaviour by setting the `--math fp32`
flag for the `train.py` script.

To view all available options for training, run `python3 train.py --help`.

## Running inference
Inference can be run by launching the `translate.py` inference script, although,
it requires a pre-trained model checkpoint and tokenized input.

The inference script, `translate.py`, supports batched inference. By default, it
launches beam search with beam size of 5, coverage penalty term and length
normalization term. Greedy decoding can be enabled by setting the beam size to 1.

To view all available options for inference, run `python3 translate.py --help`.

## Training Accuracy Results
Results were obtained by running the `train.py` script with the default
batch size = 128 per GPU in the pytorch-18.11-py3 Docker container.

### NVIDIA DGX-1 (8x Tesla V100 16G)
Command used to launch the training:

```
python3 -m launch train.py --seed 2 --train-global-batch-size 1024
```

| **number of GPUs** | **batch size/GPU** | **mixed precision BLEU** | **fp32 BLEU** | **mixed precision training time** | **fp32 training time** |
| --- | --- | ----- | ----- | ------------- | ------------- |
|  1  | 128 | 22.97 | 23.03 | 281.7 minutes | 880.6 minutes |
|  4  | 128 | 23.32 | 23.34 | 92.2 minutes  | 260.8 minutes |
|  8  | 128 | 23.25 | 23.00 | 48.7 minutes  | 129.2 minutes |

### NVIDIA DGX-2 (16x Tesla V100 32G)
Commands used to launch the training:

```
for 1,4,8 GPUs:
python3 -m launch train.py --seed 2 --train-global-batch-size 1024
for 16 GPUs:
python3 -m launch train.py --seed 2 --train-global-batch-size 2048
```

| **number of GPUs** | **batch size/GPU** | **mixed precision BLEU** | **fp32 BLEU** | **mixed precision training time** | **fp32 training time** |
| --- | --- | ----- | ----- | ------------- | ------------- |
| 1   | 128 | 22.97 | 23.03 | 277.1 minutes | 834.8 minutes |
| 4   | 128 | 23.08 | 23.42 | 91.7 minutes  | 247.0 minutes |
| 8   | 128 | 22.99 | 23.25 | 52.1 minutes  | 130.2 minutes |
| 16  | 128 | 23.37 | 23.19 | 27.2 minutes  | 65.6 minutes  |

![TrainingLoss](./img/training_loss.png)

### Training Stability Test
The GNMT v2 model was trained for 6 epochs, starting from 50 different initial
random seeds. After each training epoch the model was evaluated on the test
dataset and the BLEU score was recorded. The training was performed in the
pytorch-18.11-py3 Docker container on NVIDIA DGX-1 with 8 Tesla V100 16G GPUs.
The following table summarizes results of the stability test.

![TrainingAccuracy](./img/training_accuracy.png)

#### BLEU scores after each training epoch for different initial random seeds
| **epoch** | **average** | **stdev** | **minimum** | **maximum** | **median** |
| --- | ------ | ----- | ------ | ------ | ------ |
|  1  | 19.066 | 0.260 | 18.170 | 19.670 | 19.055 |
|  2  | 20.799 | 0.314 | 19.560 | 21.390 | 20.840 |
|  3  | 21.520 | 0.199 | 20.970 | 21.830 | 21.565 |
|  4  | 21.901 | 0.234 | 21.120 | 22.260 | 21.925 |
|  5  | 23.081 | 0.144 | 22.740 | 23.360 | 23.105 |
|  6  | 23.235 | 0.135 | 22.930 | 23.480 | 23.225 |


## Training Performance Results
All results were obtained by running the `train.py` training script in the
pytorch-18.11-py3 Docker container. Performance numbers (in tokens per second)
were averaged over an entire training epoch.

### NVIDIA DGX-1 (8x Tesla V100 16G)

| **number of GPUs** | **batch size/GPU** | **mixed precision tokens/s** | **fp32 tokens/s** | **mixed precision speedup** | **mixed precision multi-gpu strong scaling** | **fp32 multi-gpu strong scaling** |
| --- | --- | ------ | ------ | ----- | ----- | ----- |
|  1  | 128 | 62305  | 20076  | 3.103 | 1.000 | 1.000 |
|  4  | 128 | 192918 | 67715  | 2.849 | 3.096 | 3.373 |
|  8  | 128 | 371795 | 137727 | 2.700 | 5.967 | 6.860 |


### NVIDIA DGX-2 (16x Tesla V100 32G)

| **number of GPUs** | **batch size/GPU** | **mixed precision tokens/s** | **fp32 tokens/s** | **mixed precision speedup** | **mixed precision multi-gpu strong scaling** | **fp32 multi-gpu strong scaling** |
| --- | --- | ------ | ------- | ----- | ------ | ------ |
|  1  | 128 | 63538  |  21115  | 3.009 | 1.000  | 1.000  |
|  4  | 128 | 192112 |  71251  | 2.696 | 3.024  | 3.374  |
|  8  | 128 | 344795 |  135949 | 2.536 | 5.427  | 6.439  |
| 16  | 128 | 719137 |  280924 | 2.560 | 11.318 | 13.304 |

## Inference Performance Results
All results were obtained by running the `translate.py` script in the
pytorch-18.11-py3 Docker container on NVIDIA DGX-1. Inference benchmark was run
on a single Tesla V100 16G GPU. The benchmark requires a checkpoint from a fully
trained model.

Command to launch the inference benchmark:
```
python3 translate.py --input data/wmt16_de_en/newstest2014.tok.bpe.32000.en \
  --reference data/wmt16_de_en/newstest2014.de --output /tmp/output \
  --model results/gnmt/model_best.pth --batch-size 32 128 512 \
  --beam-size 1 2 5 10 --math fp16 fp32
```

| **batch size** | **beam size** | **mixed precision BLEU** | **fp32 BLEU** | **mixed precision tokens/s** | **fp32 tokens/s** |
| ---- | ----- | ------- | ------- | ---------|-------- |
|  32  |   1   | 21.90   | 21.87   | 17394    | 14820   |
|  32  |   2   | 22.83   | 22.88   | 7649     | 6834    |
|  32  |   5   | 23.40   | 23.47   | 6956     | 5199    |
|  32  |   10  | 23.32   | 23.40   | 6277     | 4057    |
|  128 |   1   | 21.91   | 21.87   | 56101    | 37177   |
|  128 |   2   | 22.83   | 22.88   | 23785    | 16453   |
|  128 |   5   | 23.41   | 23.47   | 19160    | 10430   |
|  128 |   10  | 23.34   | 23.40   | 14009    | 6260    |
|  512 |   1   | 21.90   | 21.87   | 119005   | 48205   |
|  512 |   2   | 22.83   | 22.88   | 52099    | 24540   |
|  512 |   5   | 23.43   | 23.47   | 31793    | 11879   |
|  512 |   10  | 23.29   | 23.41   | 18813    | 6420    |


# Changelog
1. Aug 7, 2018
  * Initial release
2. Dec 4, 2018
  * Added exponential warm-up and step learning rate decay
  * Multi-GPU (distributed) inference and validation
  * Default container updated to PyTorch 18.11-py3
  * General performance improvements

# Known issues
None

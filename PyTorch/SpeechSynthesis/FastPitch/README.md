# FastPitch 1.0 for PyTorch

This repository provides a script and recipe to train the FastPitch model to achieve state-of-the-art accuracy and is tested and maintained by NVIDIA.

## Table Of Contents

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
    * [Parameters](#parameters)
        * [Training parameters](#training-parameters)
        * [Audio and SFST parameters](#audio-and-sfst-parameters)
        * [FastPitch parameters](#fastpitch parameters)
    * [Command-line options](#command-line-options)
    * [Getting the data](#getting-the-data)
        * [Dataset guidelines](#dataset-guidelines)
        * [Multi-dataset](#multi-dataset)
    * [Training process](#training-process)
    * [Inference process](#inference-process)
    * [Deploying the FastPitch model using Triton Inference Server](#deploying-the-fastpitch-model-using-triton-inference)
        * [Performance analysis for Triton Inference Server](#performance-analysis-for-triton-inference-server)
        * [Running the Triton Inference Server and client](#running-the-triton-inference-server-and-client)
- [Performance](#performance)
    * [Benchmarking](#benchmarking)
        * [Training performance benchmark](#training-performance-benchmark)
        * [Inference performance benchmark](#inference-performance-benchmark)
    * [Results](#results)
        * [Training accuracy results](#training-accuracy-results)
            * [Training accuracy: NVIDIA DGX-1 (8x V100 16G)](#training-accuracy-nvidia-dgx-1-8x-v100-16g)
            * [Training stability test](#training-stability-test)
        * [Training performance results](#training-performance-results)
            * [Training performance: NVIDIA DGX-1 (8x V100 16G)](#training-performance-nvidia-dgx-1-8x-v100-16g)
            * [Expected training time](#expected-training-time)
            * [Training performance: NVIDIA DGX-2 (16x V100 32G)](#training-performance-nvidia-dgx-2-16x-v100-32g)
        * [Inference performance results](#inference-performance-results)
            * [Inference performance: NVIDIA DGX-1 (1x V100 16G)](#inference-performance-nvidia-dgx-1-1x-v100-16g)
            * [Inference performance: NVIDIA T4](#inference-performance-nvidia-t4)
- [Release notes](#release-notes)
    * [Changelog](#changelog)
    * [Known issues](#known-issues)

## Model overview

A full text-to-speech (TTS) system is a pipeline of two neural network
models:
* a mel-spectrogram generator such as [FastPitch](#) or [Tacotron 2](https://arxiv.org/abs/1712.05884), and
* a waveform synthesizer such as [WaveGlow](https://arxiv.org/abs/1811.00002) (see [NVIDIA example code](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2)).

It enables users to synthesize natural sounding speech from raw transcripts.

The FastPitch model generates mel-spectrograms from raw input text and allows to exert additional control over the synthesized utterances, such as:
* supply pitch cues to control the prosody
* alter the pace of speech
The FastPitch model is based on the [FastSpeech](https://arxiv.org/abs/1905.09263) model(?). The main differences between FastPitch and FastSpeech are that FastPitch: 
* explicitly learns to predict pitch (f0),
* achieves higher quality, trains faster and no longer needs knowledge distillation from a teacher model,
* character durations are extracted with a Tacotron 2 model.

The model is trained on a publicly
available [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/).

This model is trained with mixed precision using Tensor Cores on NVIDIA Volta and Turing GPUs. Therefore, researchers can get results 2.2x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.

### Model architecture

FastPitch is a fully feedforward Transformer model that predicts mel-spectrograms
from raw text. The model is composed of an encoder, pitch predictor, duration predictor, and a decoder.
After encoding, the signal is augmented with pitch information and discretely upsampled.
The goal of the decoder is to smooth out the upsampled signal, and construct a mel-spectrogram.
The entire process is parallel.




<div style="text-align:center" align="center">
  <img src="./img/fastpitch_model.png" alt="FastPitch model architecture" />
  </br>
  <em>Figure 1. Architecture of FastPitch (source: [FastPitch: Paper title TODO](#))</em>
</div>

### Default configuration

The FastPitch model supports multi-GPU and mixed precision training with dynamic loss
scaling (see Apex code
[here](https://github.com/NVIDIA/apex/blob/master/apex/fp16_utils/loss_scaler.py)),
as well as mixed precision inference.

The following features were implemented in this model:

* data-parallel multi-GPU training,
* dynamic loss scaling with backoff for Tensor Cores (mixed precision)
training,
* gradient accumulation for reproducible results regardless of the number of GPUs.

To speed-up FastPitch training,
reference mel-spectrograms, character durations, and pitch cues
are generated during the pre-processing step and read
directly from the disk during training. For more information on data pre-processing refer to [Dataset guidelines
](#dataset-guidelines) and the [paper](#).

### Feature support matrix

The following features are supported by this model.

| Feature                                                            | FastPitch     |
| :------------------------------------------------------------------|------------:|
|[AMP](https://nvidia.github.io/apex/amp.html)                               | Yes |
|[Apex DistributedDataParallel](https://nvidia.github.io/apex/parallel.html) | Yes |
         
#### Features

AMP - a tool that enables Tensor Core-accelerated training. For more information,
refer to [Enabling mixed precision](#enabling-mixed-precision).

Apex DistributedDataParallel - a module wrapper that enables easy multiprocess
distributed data parallel training, similar to `torch.nn.parallel.DistributedDataParallel`.
`DistributedDataParallel` is optimized for use with NCCL. It achieves high
performance by overlapping communication with computation during `backward()`
and bucketing smaller gradient transfers to reduce the total number of transfers
required.

### Mixed precision training

Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in the Volta and Turing architecture, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using mixed precision training requires two steps:
1.  Porting the model to use the FP16 data type where appropriate.    
2.  Adding loss scaling to preserve small gradient values.

The ability to train deep learning networks with lower precision was introduced in the Pascal architecture and first supported in [CUDA 8](https://devblogs.nvidia.com/parallelforall/tag/fp16/) in the NVIDIA Deep Learning SDK.

For information about:
-   How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) documentation.
-   Techniques used for mixed precision training, see the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
-   How to access and enable AMP for TensorFlow, see [Using TF-AMP](https://docs.nvidia.com/deeplearning/dgx/tensorflow-user-guide/index.html#tfamp) from the TensorFlow User Guide.
-   APEX tools for mixed precision training, see the [NVIDIA Apex: Tools for Easy Mixed-Precision Training in PyTorch](https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/).

#### Enabling mixed precision

Mixed precision is enabled in PyTorch by using the Automatic Mixed Precision
(AMP)  library from [APEX](https://github.com/NVIDIA/apex) that casts variables
to half-precision upon retrieval, while storing variables in single-precision
format. Furthermore, to preserve small gradient magnitudes in backpropagation,
a [loss scaling](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#lossscaling)
step must be included when applying gradients. In PyTorch, loss scaling can be
easily applied by using the `scale_loss()` method provided by AMP. The scaling value
to be used can be [dynamic](https://nvidia.github.io/apex/fp16_utils.html#apex.fp16_utils.DynamicLossScaler) or fixed.

By default, the `train_fastpitch.sh` script will
launch mixed precision training with Tensor Cores. You can change this
behaviour by removing the `--amp-run` flag from the `train.py` script.

To enable mixed precision, the following steps were performed:
* Import AMP from APEX:
    ```bash
    from apex import amp
    ```

* Initialize AMP:
    ```bash
	model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    ```

* If running on multi-GPU, wrap the model with `DistributedDataParallel`:
    ```bash
    from apex.parallel import DistributedDataParallel as DDP
    model = DDP(model)
	```

* Scale loss before backpropagation (assuming loss is stored in a variable
called `losses`):

    * Default backpropagate for FP32:
        ```bash
        losses.backward()
        ```

    * Scale loss and backpropagate with AMP:
        ```bash
        with optimizer.scale_loss(losses) as scaled_losses:
            scaled_losses.backward()
        ```


### Glossary

**Forced alignment**
Segmentation of a recording into lexical units like characters, words, or phonemes. The segmentation is hard and defines exact starting end ending times for every unit.

**Fundamental frequency**
The lowest vibration frequency of a periodic soundwave, for example, produced by a vibrating instrument. It is perceived as the loudest. In the context of speech, it refers to the frequency of vibration of vocal chords.  Abbreviated as *f0*.

**Pitch**
A perceived frequency of vibration of music or sound.

**Transformer**  
The paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) introduces a novel architecture called Transformer, which repeatedly applies the attention mechanism. It transforms one sequence into another.

## Setup

The following section lists the requirements that you need to meet in order to start training the FastPitch model.

### Requirements

This repository contains Dockerfile which extends the PyTorch NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:
-   [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
-   [PyTorch 20.03-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch)
or newer
-   [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) based GPU

For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
-   [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
-   [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)
-   [Running PyTorch](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html#running)
  
For those unable to use the PyTorch NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## Quick Start Guide

To train your model using mixed precision with Tensor Cores or using FP32, perform the following steps using the default parameters of the FastPitch model on the LJSpeech 1.1 dataset. For the specifics concerning training and inference, see the [Advanced](#advanced) section.

1. Clone the repository.
   ```bash
   git clone https://github.com/NVIDIA/DeepLearningExamples.git
   cd DeepLearningExamples/PyTorch/SpeechSynthesis/FastPitch
   ```

2. Build and run the FastPitch PyTorch NGC container.

   By default the container will use the first available GPU. Modify the script to include other available devices.
   ```bash
   bash scripts/docker/build.sh
   bash scripts/docker/interactive.sh
   ```

3. Download and preprocess the dataset.

   Use the scripts to automatically download and preprocess the training, validation and test datasets:
   ```bash
   bash scripts/download_dataset.sh
   bash scripts/prepare_dataset.sh
   ```

   The data is downloaded to the `./LJSpeech-1.1` directory (on the host).  The
   `./LJSpeech-1.1` directory is mounted under the `/workspace/fastpitch/LJSpeech-1.1`
   location in the NGC container. The complete dataset has the following structure:
   ```bash
   ./LJSpeech-1.1
   ├── durations        # Character durations estimates for forced alignment training
   ├── mels             # Pre-calculated target mel-spectrograms
   ├── metadata.csv     # Mapping of waveforms to utterances
   ├── pitch_char       # Average fundamental frequencies, aligned and averaged for every character
   ├── pitch_char_stats__ljs_audio_text_train_filelist.json    # Mean and std of pitch for training data
   ├── README
   └── wavs             # Raw waveforms
   ```

4. Start training.
   ```bash
   bash scripts/train.sh
   ```
   The training will produce a FastPitch model capable of generating mel-spectrograms from raw text.
   It will be serialized as a single `.pt` checkpoint file, along with a series of intermediate checkpoints.

5. Start validation/evaluation.

   Ensure your training loss values are comparable to those listed in the table in the
   [Results](#results) section. Note that the validation loss is evaluated with ground truth durations for letters (not the predicted ones). The loss values are stored in the `./output/nvlog.json` log file, `./output/{train,val,test}` as TensorBoard logs, and printed to the standard output (`stdout`) during training.
   The main reported loss is a weighted sum of losses for mel-, pitch-, and duration- predicting modules.

   The audio can be generated by following the [Inference process](#inference-process) section below.
   The synthesized audio should be similar to the samples in the `./audio` directory.

6. Start inference/predictions.

   To synthesize audio, you will need to train a WaveGlow model, which generates waveforms based on mel-spectrograms generated with FastPitch. To train WaveGlow, follow the instructions in [NVIDIA/DeepLearningExamples/Tacotron2](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2). A pre-trained WaveGlow checkpoint should be placed into the `./pretrained_models` directory.

   You can perform inference using the respective `.pt` checkpoints that are passed as `--fastpitch`
   and `--waveglow` arguments:
   ```bash
   python inference.py --cuda --wn-channels 256 --amp-run \
                       --fastpitch output/<FastPitch checkpoint> \
                       --waveglow pretrained_models/waveglow/<waveglow checkpoint> \
                       -i phrases/devset10.tsv \
                       -o output/wavs_devset10
   ```

   The speech is generated from lines of text in the file that is passed with
   `-i` argument. To run
   inference in mixed precision, use the `--amp-run` flag. The output audio will
   be stored in the path specified by the `-o` argument. Consult the `inference.py` to learn more options, such as setting the batch size.


## Advanced

The following sections provide greater details of the dataset, running training and inference, and the training results.

### Scripts and sample code

The repository holds code for FastPitch (training and inference) and WaveGlow (inference only).
The code specific to a particular model is located in that model’s directory - `./fastpitch` and `./waveglow` - and common functions live in the `./common` directory. The model-specific scripts are as follows:

* `<model_name>/model.py` - the model architecture, definition of forward and
inference functions
* `<model_name>/arg_parser.py` - argument parser for parameters specific to a
given model
* `<model_name>/data_function.py` - data loading functions
* `<model_name>/loss_function.py` - loss function for the model

The common scripts contain layer definitions common to both models
(`common/layers.py`), some utility scripts (`common/utils.py`) and scripts
for audio processing (`common/audio_processing.py` and `common/stft.py`). 

In the root directory `./` of this repository, the `./train.py` script is used for
training while inference can be executed with the `./inference.py` script. The
scripts `./models.py`, `./data_functions.py` and `./loss_functions.py` call
the respective scripts in the `<model_name>` directory, depending on what
model is trained using the `train.py` script.

The structure of the repository follows closely to that of the [NVIDIA Tacotron2 Deep Learning example](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2). It allows to combine both models within a single project in more advanced use cases.

### Parameters

In this section, we list the most important hyperparameters and command-line arguments,
together with their default values that are used to train FastPitch.

#### Training parameters

* `--epochs` - number of epochs (default: 1500)
* `--learning-rate` - learning rate (default: 0.1)
* `--batch-size` - batch size (default: 32)
* `--amp-run` - use mixed precision training

#### Audio and STFT parameters

* `--sampling-rate` - sampling rate in Hz of input and output audio (22050)
* `--filter-length` - (1024)
* `--hop-length` - hop length for FFT, i.e., sample stride between consecutive FFTs (256)
* `--win-length` - window size for FFT (1024)
* `--mel-fmin` - lowest frequency in Hz (0.0)
* `--mel-fmax` - highest frequency in Hz (8.000)

#### FastPitch parameters

* `--pitch-predictor-loss-scale` - rescale the loss of the pitch predictor module to dampen
its influence on the shared encoder
* `--duration-predictor-loss-scale` - rescale the loss of the duration predictor module to dampen
its influence on the shared encoder
* `--pitch` - enable pitch conditioning and prediction

### Command-line options

To see the full list of available options and their descriptions, use the `-h`
or `--help` command line option, for example:
```bash
python train.py --help
```

The following example output is printed when running the model:

```bash
DLL 2020-03-30 10:41:12.562594 - epoch    1 | iter   1/19 | loss 36.99 | mel loss 35.25 |  142370.52 items/s | elapsed 2.50 s | lrate 1.00E-01 -> 3.16E-06
DLL 2020-03-30 10:41:13.202835 - epoch    1 | iter   2/19 | loss 37.26 | mel loss 35.98 |  561459.27 items/s | elapsed 0.64 s | lrate 3.16E-06 -> 6.32E-06
DLL 2020-03-30 10:41:13.831189 - epoch    1 | iter   3/19 | loss 36.93 | mel loss 35.41 |  583530.16 items/s | elapsed 0.63 s | lrate 6.32E-06 -> 9.49E-06
```

### Getting the data

The FastPitch and WaveGlow models were trained on the LJSpeech-1.1 dataset.
The `./scripts/download_dataset.sh` script will automatically download and extract the dataset to the `./LJSpeech-1.1` directory.

#### Dataset guidelines

The LJSpeech dataset has 13,100 clips that amount to about 24 hours of speech of a single, female speaker. Since the original dataset does not define a train/dev/test split of the data, we provide a split in the form of three file lists:
```bash
./filelists
├── ljs_mel_ali_pitch_text_test_filelist.txt
├── ljs_mel_ali_pitch_text_train_filelist.txt
└── ljs_mel_ali_pitch_text_val_filelist.txt
```

***NOTE: When combining FastPitch/WaveGlow with external models trained on LJSpeech-1.1, make sure that your train/dev/test split matches. Different organizations may use custom splits. A mismatch poses a risk of leaking the training data through model weights during validation and testing.***

FastPitch predicts character durations just like [FastSpeech](https://arxiv.org/abs/1905.09263) does.
This calls for training with forced alignments, expressed as the number of output mel-spectrogram frames for every input character.
To this end, a pre-trained
[Tacotron 2 model](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2)
is used. Its attention matrix
relates the input characters with the output mel-spectrogram frames.

For every mel-spectrogram frame, its fundamental frequency in Hz is estimated with [Praat](http://praat.org).
Those values are then averaged over every character, in order to provide sparse
pitch cues for the model. Character boundaries are calculated from durations
extracted previously with Tacotron 2.

<div style="text-align:center" align="center">
  <img src="./img/pitch.png" alt="Pitch estimates extracted with Praat" />
  <br>
  <em>Figure 2. Pitch estimates for mel-spectrogram frames of phrase "in being comparatively"
averaged over characters. Silent letters have duration 0 and are omitted.</em>
</div>

#### Multi-dataset

Follow these steps to use datasets different from the default LJSpeech dataset.

1. Prepare a directory with .wav files.
   ```bash
   ./my_dataset
   └── wavs
   ```

2. Prepare filelists with transcripts and paths to .wav files. They define training/validation split of the data (test is currently unused):
   ```
   ./filelists
   ├── my_dataset_mel_ali_pitch_text_train_filelist.txt
   └── my_dataset_mel_ali_pitch_text_val_filelist.txt
   ```

Those filelists should list a single utterance per line as:
   ```bash
   `<audio file path>|<transcript>`
   ```
The `<audio file path>` is the relative path to the path provided by the `--dataset-path` option of `train.py`.

3. Run the pre-processing script to calculate mel-spectrograms, durations and pitch:
   ```bash
   python extract_mels.py --cuda \
                          --dataset-path ./my_dataset \
                          --wav-text-filelist ./filelists/my_dataset_mel_ali_pitch_text_train_filelist.txt \
                          --extract-mels \
                          --extract-durations \
                          --extract-pitch-char \
                          --tacotron2-checkpoint ./pretrained_models/tacotron2/state_dict.pt"

   python extract_mels.py --cuda \
                          --dataset-path ./my_dataset \
                          --wav-text-filelist ./filelists/my_dataset_mel_ali_pitch_text_val_filelist.txt \
                          --extract-mels \
                          --extract-durations \
                          --extract-pitch-char \
                          --tacotron2-checkpoint ./pretrained_models/tacotron2/state_dict.pt"
   ```

In order to use the prepared dataset, pass the following to the `train.py` script:
   ```bash
   --dataset-path ./my_dataset` \
   --training-files ./filelists/my_dataset_mel_ali_pitch_text_train_filelist.txt \
   --validation files ./filelists/my_dataset_mel_ali_pitch_text_val_filelist.txt
   ```
   
### Training process

FastPitch is trained to generate mel-spectrograms from raw text input. It uses short time Fourier transform (STFT)
to generate target mel-spectrograms from audio waveforms to be the training targets.

The training loss is averaged over an entire training epoch, whereas the
validation loss is averaged over the validation dataset. Performance is
reported in total output mel-spectrogram frames per second and recorded as `train_frames/s` (after each iteration) and `avg_train_frames/s` (averaged over epoch) in the output log file `./output/nvlog.json`.
The result is averaged over an entire training epoch and summed over all GPUs that were
included in the training.

Even though the training script uses all available GPUs, you can change
this behavior by setting the `CUDA_VISIBLE_DEVICES` variable in your
environment or by setting the `NV_GPU` variable at the Docker container launch
([see section "GPU isolation"](https://github.com/NVIDIA/nvidia-docker/wiki/nvidia-docker#gpu-isolation)).

### Inference process

You can run inference using the `./inference.py` script. This script takes
text as input and runs FastPitch and then WaveGlow inference to produce an
audio file. It requires pre-trained checkpoints of both models
and input text as a text file, with one phrase per line.

Having pre-trained models in place, run the sample inference on LJSpeech-1.1 test-set with:
```bash
bash scripts/inference_example.sh
```
Examine the `inference_examples.sh` script to adjust paths to pre-trained models,
and call `python inference.py --help` to learn all available options.
Synthesized audio samples will be saved in the output folder.
The audio files
<a href="./audio/audio_fp16.wav">audio/audio_fp16.wav</a>
and <a href="./audio/audio_fp32.wav">audio/audio_fp32.wav</a> were generated using checkpoints from
mixed precision and FP32 training, respectively.
The audio files
<audio src="./audio/audio_fp16.wav"”/>
and <audio src="./audio/audio_fp32.wav"/> were generated using checkpoints from
mixed precision and FP32 training, respectively.

FastPitch allows us to linearly adjust the pace of synthesized speech like [FastSpeech](https://arxiv.org/abs/1905.09263).
For instance, pass `--pace 0.5` to slow down twice.

For every input character, the model predicts a pitch cue - an average pitch over a character in Hz.
Pitch can be adjusted by transforming those pitch cues. A few simple examples are provided below.

| Transformation                    | Flag                          | Samples   |
| :---------------------------------|:------------------------------|:---------:|
| Amplify pitch                     |`--pitch-transform-amplify`    | [link](./audio/sample_fp16_amplify.wav) |
| Invert pitch                      |`--pitch-transform-invert`     | [link](./audio/sample_fp16_invert.wav)  |
| Raise/lower pitch by <hz>         |`--pitch-transform-shift <hz>` | [link](./audio/sample_fp16_shift.wav)   |
| Flatten the pitch                 |`--pitch-transform-flatten`    | [link](./audio/sample_fp16_flatten.wav) |
| Change the pace (1.0 = unchanged) |`--pace <value>`               | [link](./audio/sample_fp16_pace.wav)    |

Modify these examples directly in `inference.py` to achieve more interesting results.

You can find all the available options by calling `python inference.py --help`.


## Performance

### Benchmarking

The following section shows how to run benchmarks measuring the model
performance in training and inference mode.

#### Training performance benchmark

To benchmark the training performance on a specific batch size, run:

**FastPitch**

* For 1 GPU
    * FP16
        ```bash
        python train.py \
            --amp-run \
            --cuda \
            --cudnn-enabled \
            -o <output-dir> \
            --log-file <output-dir>/nvlog.json \
            --dataset-path <dataset-path> \
            --training-files <train-filelist-path> \
            --validation-files <val-filelist-path> \
            --pitch-mean-std <pitch-stats-path> \
            --load-mel-from-disk \
            --epochs 10 \
            --warmup-steps 1000 \
            -lr 0.1 \
            -bs 64 \
            --optimizer lamb \
            --grad-clip-thresh 1000.0 \
            --dur-predictor-loss-scale 0.1 \
            --pitch-predictor-loss-scale 0.1 \
            --weight-decay 1e-6 \
            --gradient-accumulation-steps 4
        ```

    * FP32
        ```bash
        python train.py \
            --cuda \
            --cudnn-enabled \
            -o <output-dir> \
            --log-file <output-dir>/nvlog.json \
            --dataset-path <dataset-path> \
            --training-files <train-filelist-path> \
            --validation-files <val-filelist-path> \
            --pitch-mean-std <pitch-stats-path> \
            --load-mel-from-disk \
            --epochs 10 \
            --warmup-steps 1000 \
            -lr 0.1 \
            -bs 32 \
            --optimizer lamb \
            --grad-clip-thresh 1000.0 \
            --dur-predictor-loss-scale 0.1 \
            --pitch-predictor-loss-scale 0.1 \
            --weight-decay 1e-6 \
            --gradient-accumulation-steps 8
        ```

* For multiple GPUs
    * FP16
        ```bash
        python -m multiproc train.py \
            --amp-run \
            --cuda \
            --cudnn-enabled \
            -o <output-dir> \
            --log-file <output-dir>/nvlog.json \
            --dataset-path <dataset-path> \
            --training-files <train-filelist-path> \
            --validation-files <val-filelist-path> \
            --pitch-mean-std <pitch-stats-path> \
            --load-mel-from-disk \
            --epochs 10 \
            --warmup-steps 1000 \
            -lr 0.1 \
            -bs 32 \
            --optimizer lamb \
            --grad-clip-thresh 1000.0 \
            --dur-predictor-loss-scale 0.1 \
            --pitch-predictor-loss-scale 0.1 \
            --weight-decay 1e-6 \
            --gradient-accumulation-steps 1
        ```

    * FP32
        ```bash
        python -m multiproc train.py \
            --cuda \
            --cudnn-enabled \
            -o <output-dir> \
            --log-file <output-dir>/nvlog.json \
            --dataset-path <dataset-path> \
            --training-files <train-filelist-path> \
            --validation-files <val-filelist-path> \
            --pitch-mean-std <pitch-stats-path> \
            --load-mel-from-disk \
            --epochs 10 \
            --warmup-steps 1000 \
            -lr 0.1 \
            -bs 32 \
            --optimizer lamb \
            --grad-clip-thresh 1000.0 \
            --dur-predictor-loss-scale 0.1 \
            --pitch-predictor-loss-scale 0.1 \
            --weight-decay 1e-6 \
            --gradient-accumulation-steps 1
        ```

Each of these scripts runs for 10 epochs and for each epoch measures the
average number of items per second. The performance results can be read from
the `nvlog.json` files produced by the commands.

#### Inference performance benchmark

To benchmark the inference performance, run:

* For FP16
    ```
    python inference.py --cuda --amp-run \
                         --fastpitch output/checkpoint_FastPitch_1500.pt \
                         --waveglow pretrained_models/waveglow/waveglow_256channels_ljs_v2.pt \
                         --wn-channels 256 \
                         --include-warmup \
                         --batch-size 1 \
                         --repeats 100 \
                         --input phrases/benchmark_8_128.tsv \
                         --log_file output/nvlog_inference.json
    ```

* For FP32
    ```
    python inference.py --cuda \
                         --fastpitch output/checkpoint_FastPitch_1500.pt \
                         --waveglow pretrained_models/waveglow/waveglow_256channels_ljs_v2.pt \
                         --wn-channels 256 \
                         --include-warmup \
                         --batch-size 1 \
                         --pitch \
                         --repeats 100 \
                         --input phrases/benchmark_8_128.tsv \
                         --log_file output/nvlog_inference.json
   ```

The output log files will contain performance numbers for the FastPitch model
(number of output mel-spectrogram frames per second, reported as `generator_frames/s 
`)
and for WaveGlow (number of output samples per second, reported as ` waveglow_samples/s 
`).
The `inference.py` script will run a few warm-up iterations before running the benchmark. Inference will be averaged over 100 runs, as set by the `--repeats` flag.

### Results

The following sections provide details on how we achieved our performance
and accuracy in training and inference.

#### Training accuracy results

##### Training accuracy: NVIDIA DGX-1 (8x V100 16G)

Our results were obtained by running the `./platform/train_fastpitch_{AMP,FP32}_DGX1_16GB_8GPU.sh` training script in the PyTorch 20.03-py3 NGC container on NVIDIA DGX-1 with 8x V100 16G GPUs.

All of the results were produced using the `train.py` script as described in the
[Training process](#training-process) section of this document.

| Loss (Model/Epoch)   |      0 |   250 |   500 |   750 |   1000 |   1250 |   1500 |
| :------------------: | -----: | ----: | ----: | ----: | -----: | -----: | -----: |
| FastPitch FP16       | 35.839 | 0.491 | 0.339 | 0.29  |  0.265 |  0.249 |  0.239 |
| FastPitch FP32       | 34.781 | 0.497 | 0.340 | 0.292 |  0.266 |  0.250 |  0.239 |

##### Training stability test

#### Training performance results

##### Training performance: NVIDIA DGX-1 (8x V100 16G)

Our results were obtained by running the `./platform/train_fastpitch_{AMP,FP32}_DGX1_16GB_8GPU.sh`
training script in the PyTorch 20.03-py3 NGC container on NVIDIA DGX-1 with
8x V100 16G GPUs. Performance numbers, in output mel-spectrograms per second, were averaged over
an entire training epoch.

|Number of GPUs|Batch size per GPU|Number of mels used with mixed precision|Number of mels used with FP32|Speed-up with mixed precision|Multi-GPU strong scaling with mixed precision|Multi-GPU strong scaling with FP32|
|---:|----------------:|-------:|-------:|-----:|-----:|-----:|
|  1 |64@FP16, 32@FP32 | 109769 |  40636 | 2.70 | 1.00 | 1.00 |
|  4 |64@FP16, 32@FP32 | 361195 | 150921 | 2.39 | 3.29 | 3.71 |
|  8 |32@FP16, 32@FP32 | 562136 | 278778 | 2.02 | 5.12 | 6.86 |

To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

##### Expected training time

The following table shows the expected training time for convergence for 1500 epochs:

|Number of GPUs|Batch size per GPU|Time to train with mixed precision (Hrs)|Time to train with FP32 (Hrs)|Speed-up with mixed precision|
|---:|----------------:|-----:|-----:|-----:|
|  1 |64@FP16, 32@FP32 | 27.0 | 73.7 | 2.73 |
|  4 |64@FP16, 32@FP32 |  8.4 | 19.7 | 2.36 |
|  8 |32@FP16, 32@FP32 |  5.5 | 10.8 | 1.97 |


Note that most of the quality is achieved after the initial 500 epochs.

#### Inference performance results

The following tables show inference statistics for the FastPitch and WaveGlow
text-to-speech system, gathered from 1000 inference runs, on a single V100 and a single T4,
respectively. Latency is measured from the start of FastPitch inference to
the end of WaveGlow inference. Throughput is measured
as the number of generated audio samples per second at 22KHz. RTF is the real-time factor
which tells how many seconds of speech are generated in 1 second of wall-clock time.
The used WaveGlow model is a 256-channel model [published on NGC](https://ngc.nvidia.com/catalog/models/nvidia:waveglow_ljs_256channels).

Our results were obtained by running the `./scripts/inference_benchmark.sh` script in
the PyTorch 20.03-py3 NGC container. Note that to reproduce the results,
you need to provide pre-trained checkpoints for FastPitch and WaveGlow. Edit the script to provide your checkpoint filenames.

Note that performance numbers are related to the length of input. The numbers reported below were taken with a moderate length of 128 characters. For longer utterances even better numbers are expected, as the generator is fully parallel.

##### Inference performance: NVIDIA DGX-1 (1x V100 16G)

The input utterance has 128 characters, synthesized audio has 8.05 s.

|Batch size|Precision|Avg latency (s)|Latency tolerance interval 90% (s)|Latency tolerance interval 95% (s)|Latency tolerance interval 99% (s)|Throughput (samples/sec)|Speed-up with mixed precision|Avg RTF|
|------:|------------:|--------------:|--------------:|--------------:|--------------:|----------------:|---------------:|----------:|
|     1 | FP16        |         0.253 |         0.254 |         0.255 |         0.255 | 702,735         | 1.51           |     31.87 |
|     4 | FP16        |         0.572 |         0.575 |         0.575 |         0.576 | 1,243,094       | 2.55           |     14.09 |
|     8 | FP16        |         1.118 |         1.121 |         1.121 |         1.123 | 1,269,479       | 2.70           |      7.20 |
|     1 | FP32        |         0.382 |         0.384 |         0.384 |         0.385 | 464,920         | -              |     21.08 |
|     4 | FP32        |         1.458 |         1.461 |         1.461 |         1.462 | 486,756         | -              |      5.52 |
|     8 | FP32        |         3.015 |         3.023 |         3.024 |         3.027 | 470,741         | -              |      2.67 |


##### Inference performance: NVIDIA T4

The input utterance has 128 characters, synthesized audio has 8.05 s.

|Batch size|Precision|Avg latency (s)|Latency tolerance interval 90% (s)|Latency tolerance interval 95% (s)|Latency tolerance interval 99% (s)|Throughput (samples/sec)|Speed-up with mixed precision|Avg RTF|
|------:|------------:|--------------:|--------------:|--------------:|--------------:|----------------:|---------------:|----------:|
|     1 | FP16        |         0.952 |         0.958 |         0.960 |         0.962 | 186,349         | 1.30           |      8.45 |
|     4 | FP16        |         4.187 |         4.209 |         4.213 |         4.221 | 169,473         | 1.21           |      1.92 |
|     8 | FP16        |         7.799 |         7.824 |         7.829 |         7.839 | 181,978         | 1.38           |      1.03 |
|     1 | FP32        |         1.238 |         1.245 |         1.247 |         1.250 | 143,292         | -              |      6.50 |
|     4 | FP32        |         5.083 |         5.109 |         5.114 |         5.124 | 139,613         | -              |      1.58 |
|     8 | FP32        |        10.756 |        10.797 |        10.805 |        10.820 | 131,951         | -              |      0.75 |


## Release notes

### Changelog

May 2020
- Initial release

### Known issues

There are no known issues in this release.

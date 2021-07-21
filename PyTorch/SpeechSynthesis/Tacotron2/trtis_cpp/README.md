# Tacotron2+WaveGlow Inference Using TensorRT Inference Server with TensorRT

This is a subfolder of the Tacotron2 for PyTorch repository that provides
scripts to deploy high-performance inference using NVIDIA TensorRT Inference
Server with a custom TensorRT
[backend](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/build.html#building-a-custom-backend).

## Table of contents
* [Model overview](#model-overview)
  - [Tacotron2 plugins](#tacotron2-plugins)
* [Setup](#setup)
  - [Requirements](#requirements)
* [Quick Start Guide](#quick-start-guide)
  - [Export the models](#export-the-models)
  - [Setup the Triton server](#setup-the-trtis-server)
  - [Setup the Triton client](#setup-the-trtis-client)
  - [Starting the Triton server](#starting-the-trtis-server)
  - [Running the Triton client](#running-the-trtis-client)
* [Advanced](#advanced)
  - [Code structure](#code-structure)
  - [Precision](#precision)
* [Performance](#performance)
  - [Performance on NVIDIA T4](#performance-on-nvidia-t4)
  - [Running the benchmark](#running-the-benchmark)


## Model overview

The Tacotron2 and WaveGlow models form a text-to-speech system that enables
users to synthesize natural sounding speech from raw transcripts without any
additional information such as patterns and/or rhythms of speech.
In this implementation, the Tacotron2 network is split into three sub-networks,
the encoder, decoder, and postnet.
This is followed by WaveGlow as a vocoder, and a Denoiser network using a
[STFT](https://en.wikipedia.org/wiki/Short-time_Fourier_transform)
to remove noise from the audio output.
More information on the Tacotron2 and WaveGlow architectures can be found in
[Tacotron2 PyTorch README](../README.md), as well as information about
training.

### Tacotron2 plugins

Because the size of the layers in Tacotron2's decoder, are quite small, many
deep learning frameworks fail achieve high throughput for a batch size of one,
as the overhead
associated with executing each of these small layers can dominate the runtime. 

TensorRT supports custom layers through its 
[plugin](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#pluginv2-layer)
interface, which not only allows custom operations, but also allows
developers to manually tune and/or fuse specific layers in their
networks while still using TensorRT to perform automated optimizations on the
other layers, and to manage and execute the entire network.
This implementation uses several plugins for Tacotron2's decoder,
including fusing layers of the Prenet and Attention, as well as creating LSTM
Cell kernels optimized specifically for the dimensions used in Tacotron2.


## Setup

### Requirements

Building and running the container requires `docker`, `nvidia-docker` and `bash`.
In addition to this, the host machine must have a Volta or Turing based GPU.


## Quick Start Guide

### Clone the repository

```bash
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/trtis_cpp
```

### Export the models

You can either train models yourself, or download pretrained checkpoints from [NGC](https://ngc.nvidia.com/catalog/models) and copy them to the `./checkpoints` directory:

- [Tacotron2 checkpoint](https://ngc.nvidia.com/models/nvidia:tacotron2pyt_fp16)
- [WaveGlow checkpoint](https://ngc.nvidia.com/models/nvidia:waveglow256pyt_fp16)

```bash
mkdir checkpoints
cp <Tacotron2_checkpoint> ./checkpoints/
cp <WaveGlow_checkpoint> ./checkpoints/
```

Next you will need to export the PyTorch checkpoints so that they can be used to build TensorRT engines. This can be done via the script `export_weights.sh` script:

```bash
mkdir models
./export_weights.sh checkpoints/tacotron2_1032590_6000_amp checkpoints/waveglow_1076430_14000_amp models/
```

### Setup the Triton server
```bash
./build_trtis.sh models/tacotron2.json models/waveglow.onnx models/denoiser.json
```
This will take some time as TensorRT tries out different tactics for best
performance while building the engines.

### Setup the Triton client

Next you need to build the client docker container. To do this, enter the
`trtis_client` directory and run the script `build_trtis_client.sh`.

```bash
cd trtis_client
./build_trtis_client.sh
cd ..
```

### Run the Triton server

To run the server locally, use the script `run_trtis_server.sh`:
```bash
./run_trtis_server.sh
```

You can use the environment variable `NVIDIA_VISIBLE_DEVICES` to set which GPUs
the Triton server sees.


### Run the Triton client

Leave the server running. In another terminal, type:
```bash
cd trtis_client/
./run_trtis_client.sh phrases.txt
```

This will generate one WAV file per line in the file `phrases.txt`, named after
the line number (e.g., 1.wav through 8.wav for a 8 line file) in the `audio/`
directory. It is
important that each line in the file end with a period, or Tacotron2 may fail
to detect the end of the phrase.

## Advanced


### Code structure

The `src/` contains the following sub-directories:
* `trtis`: The directory containing code for the custom Triton backend.
* `trt/tacotron2`: The directory containing the Tacotron2 implementation in TensorRT.
* `trt/waveglow`: The directory containing the WaveGlow implementation in TensorRT.
* `trt/denoiser`: The directory containing the Denoiser (STFT) implementation in TensorRT.
* `trt/plugins`: The directory containing plugins used by the TensorRT engines.

The `trtis_client/` directory contains the code for running the client.

### Precision

By default the `./build_trtis.sh` script builds the TensorRT engines with FP16 mode enabled, which allows some operations to be performed in lower precision, in order to increase throughput. To use engines with only FP32 precision, add `0` to `./build_trtis.sh`’s arguments:

```bash
./build_trtis.sh models/tacotron2.json models/waveglow.onnx models/denoiser.json 0
```

## Performance

The performance measurements in this document were conducted at the time of publication and may not reflect the performance achieved from NVIDIA’s latest software release. For the most up-to-date performance measurements, go to [NVIDIA Data Center Deep Learning Product Performance](https://developer.nvidia.com/deep-learning-performance-training-inference).

The following tables show inference statistics for the Tacotron2 and WaveGlow
text-to-speech system.
The tables include average latency, latency standard deviation,
and latency confidence intervals. Throughput is measured as the number of
generated audio samples per second. RTF is the real-time factor which
tells how many seconds of speech are generated in 1 second of processing time.
For all tests in these tables, we used WaveGlow with 256 residual channels.

### Performance on NVIDIA T4


#### TensorRT \w Plugins vs. PyTorch

Latency in this table is measured from just before the input sequence starts
being copied from host memory to the GPU,
to just after the generated audio finishes being copied back to the host
memory.
That is, what is taking place in the custom backend inside of Triton.

|Framework|Batch size|Input length|Precision|Avg latency (s)|Latency std (s)| Latency interval 90% (s)|Latency interval 95% (s)|Latency interval 99% (s)| Throughput (samples/sec) | Speed-up vs. PyT FP32 | Speed-up vs. PyT FP16 | Avg mels generated |Avg audio length (s)|Avg RTF|
|------:|----:|-----:|-----------:|--------:|------:|------:|------:|------:|------:|------:|----:|------:|-------:|---:|
| TRT \w plugins | 1  | 128 | FP16 | 0.40 | 0.00 | 0.40 | 0.40 | 0.40 | 369,862 | __4.27x__ | __3.90x__ | 579 | 6.72 | 16.77 |
| TRT \w plugins | 1  | 128 | FP32 | 1.20 | 0.01 | 1.21 | 1.21 | 1.21 | 123,922 | __1.43x__ | __1.31x__ | 581 | 6.74 |  5.62 |
| PyTorch        | 1  | 128 | FP16 | 1.63 | 0.07 | 1.71 | 1.73 | 1.81 | 94,758 | __1.10x__ | __1.00x__ | 601 | 6.98 |  4.30 |
| PyTorch        | 1  | 128 | FP32 | 1.77 | 0.08 | 1.88 | 1.92 | 2.00 | 86,705 | __1.00x__ | __0.91x__ | 600 | 6.96 |  3.92 |

That is a __3.72x__ speedup when using TensorRT FP16 with plugins when compared to
PyTorch FP32, and still a __3.39x__ speedup when compared to PyTorch FP16.

The TensorRT entries in this table can be reproduced by using the output of
the Triton server, when performing the steps for [Running the
benchmark](#running-the-benchmark) below.
The PyTorch entries can be reproduced by following the instructions
[here](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2).



#### TensorRT \w Plugins in Triton

Latency in this table is measured from the client sending the request, to it
receiving back the generated audio. This includes network time,
request/response formatting time, as well as the backend time shown in the
above section.

|Batch size|Input length|Precision|Avg latency (s)|Latency std (s)| Latency interval 90% (s)|Latency interval 95% (s)|Latency interval 99% (s)|Avg mels generated |Avg audio length (s)|Avg RTF|
|---:|----:|-----:|------:|------:|------:|------:|------:|----:|------:|-------:|
| 1  | 128 | FP16 | 0.42 | 0.00 | 0.42 | 0.42 | 0.42 | 579 | 6.72 | 15.95 |
| 8  | 128 | FP16 | 2.55 | 0.01 | 2.56 | 2.56 | 2.57 | 571 | 6.62 |  2.60 |
| 1  | 128 | FP32 | 1.22 | 0.01 | 1.22 | 1.23 | 1.23 | 581 | 6.75 |  5.54 |
| 8  | 128 | FP32 | 8.64 | 0.01 | 8.68 | 8.69 | 8.71 | 569 | 6.61 |  0.72 |

To reproduce this table, see [Running the benchmark](#running-the-benchmark)
below.



### Running the benchmark

Once you have performed the steps in [Setup the Triton server](#setup-the-trtis-server) and
[Setup the Triton client](#setup-the-trtis-client), you can run the benchmark by starting the Triton server via:
```bash
./run_trtis_server.sh
```

Leave the server running, and in another terminal run the script `trtis_client/run_trtis_benchmark_client.sh`:

```bash
cd trtis_client/
./run_trtis_benchmark_client.sh <batch size>
```

Replace <batch size> with the desired batch size between 1 and 32. The engines are built with a maximum batch size of 32 in the `./build_trtis.sh` script.

After some time this should produce output like:
```
Performed 1000 runs.
batch size =  1
avg latency (s) = 0.421375
latency std (s) = 0.00170839
latency interval 50% (s) = 0.421553
latency interval 90% (s) = 0.422805
latency interval 95% (s) = 0.423273
latency interval 99% (s) = 0.424153
average mels generated = 582
average audio generated (s) = 6.72218
average real-time factor = 15.953
```

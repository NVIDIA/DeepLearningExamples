# BERT Inference Using TensorRT

This subfolder of the BERT TensorFlow repository, tested and maintained by NVIDIA, provides scripts to perform high-performance inference using NVIDIA TensorRT.


## Table Of Contents

- [Model Overview](#model-overview)
   * [Model Architecture](#model-architecture)
   * [TensorRT Inference Pipeline](#tensorrt-inference-pipeline)
   * [Version Info](#version-info)
- [Setup](#setup)
   * [Requirements](#requirements)
- [Quick Start Guide](#quick-start-guide)
   * [(Optional) Trying a different configuration](#optional-trying-a-different-configuration)
- [Advanced](#advanced)
   * [Scripts and sample code](#scripts-and-sample-code)
   * [Command-line options](#command-line-options)
   * [TensorRT inference process](#tensorrt-inference-process)
- [Performance](#performance)
   * [Benchmarking](#benchmarking)
      * [TensorRT inference benchmark](#tensorrt-inference-benchmark)
   * [Results](#results)
      * [Inference performance: NVIDIA T4](#inference-performance-nvidia-t4)
      * [BERT Base](#bert-base)
      * [BERT Large](#bert-large)
   * [Inference performance: NVIDIA V100 (32GB)](#inference-performance-nvidia-v100-(32gc))
      * [BERT Base](#bert-base)
      * [BERT Large](#bert-large)



## Model overview

BERT, or Bidirectional Encoder Representations from Transformers, is a new method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks. This model is based on the [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) paper. NVIDIA's BERT is an optimized version of [Google's official implementation](https://github.com/google-research/bert), leveraging mixed precision arithmetic and Tensor Cores for faster inference times while maintaining target accuracy.

Other publicly available implementations of BERT include:
1. [NVIDIA PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)
2. [Hugging Face](https://github.com/huggingface/pytorch-pretrained-BERT)
3. [codertimo](https://github.com/codertimo/BERT-pytorch)
4. [gluon-nlp](https://github.com/dmlc/gluon-nlp/tree/master/scripts/bert)
5. [Google's official implementation](https://github.com/google-research/bert)


### Model architecture

BERT's model architecture is a multi-layer bidirectional Transformer encoder. Based on the model size, we have the following two default configurations of BERT:

| **Model** | **Hidden layers** | **Hidden unit size** | **Attention heads** | **Feed-forward filter size** | **Max sequence length** | **Parameters** |
|:---------:|:----------:|:----:|:---:|:--------:|:---:|:----:|
|BERT-Base |12 encoder| 768| 12|4 x  768|512|110M|
|BERT-Large|24 encoder|1024| 16|4 x 1024|512|330M|

Typically, the language model is followed by a few task-specific layers. The model used here includes layers for question answering.

### TensorRT Inference Pipeline

BERT inference consists of three main stages: tokenization, the BERT model, and finally a projection of the tokenized prediction onto the original text.
Since the tokenizer and projection of the final predictions are not nearly as compute-heavy as the model itself, we run them on the host. The BERT model is GPU-accelerated via TensorRT.

The tokenizer splits the input text into tokens that can be consumed by the model. For details on this process, see [this tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/).

To run the BERT model in TensorRT, we construct the model using TensorRT APIs and import the weights from a pre-trained TensorFlow checkpoint from [NGC](https://ngc.nvidia.com/models/nvidian:bert_tf_v2_large_fp16_128). Finally, a TensorRT engine is generated and serialized to the disk. The various inference scripts then load this engine for inference.

Lastly, the tokens predicted by the model are projected back to the original text to get a final result.

### Version Info

The following software version configuration has been tested:

|Software|Version|
|--------|-------|
|Python|3.6.9|
|TensorFlow|1.13.1|
|TensorRT|6.0.1.8|
|CUDA|10.2.89|


## Setup

The following section lists the requirements that you need to meet in order to run the BERT model.

### Requirements

This repository contains a `Dockerfile` which extends the TensorRT 19.12-py3 NGC container and installs some dependencies. Ensure you have the following components:

* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [TensorRT 19.10-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt)
* [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) based GPU with NVIDIA Driver 440.33 or later

Required Python packages are listed in `requirements.txt`. These packages are automatically installed inside the container.

## Quick Start Guide

1. Create and launch the BERT container:
    ```bash
    bash trt/scripts/build.sh && bash trt/scripts/launch.sh
    ```

    **Note:** After this point, all commands should be run from within the container.

2. Download checkpoints for a pre-trained BERT model:
    ```bash
    bash scripts/download_model.sh
    ```
    This script downloads checkpoints for a BERT Large FP16 model with a sequence length of 128 by default.

**Note:** Since the checkpoints are stored in the directory mounted from the host, they do *not* need to be downloaded each time the container is launched.  

3. Build a TensorRT engine. To build an engine, run the `builder.py` script. For example:
    ```bash
    mkdir -p /workspace/bert/engines && python builder.py -m /workspace/bert/models/fine-tuned/bert_tf_v2_large_fp16_128_v2/model.ckpt-8144 -o /workspace/bert/engines/bert_large_128.engine -b 1 -s 128 --fp16 -c /workspace/bert/models/fine-tuned/bert_tf_v2_large_fp16_128_v2
    ```

    This will build an engine with a maximum batch size of 1 (`-b 1`), and sequence length of 128 (`-s 128`) using mixed precision (`--fp16`) using the BERT Large V2 FP16 Sequence Length 128 checkpoint (`-c /workspace/bert/models/fine-tuned/bert_tf_v2_large_fp16_128_v2`).

4. Run inference. Two options are provided for running the model.

    a. `inference.py` script
    This script accepts a passage and question and then runs the engine to generate an answer.
    For example:
    ```bash
    python inference.py -e /workspace/bert/engines/bert_large_128.engine -p "TensorRT is a high performance deep learning inference platform that delivers low latency and high throughput for apps such as recommenders, speech and image/video on NVIDIA GPUs. It includes parsers to import models, and plugins to support novel ops and layers before applying optimizations for inference. Today NVIDIA is open-sourcing parsers and plugins in TensorRT so that the deep learning community can customize and extend these components to take advantage of powerful TensorRT optimizations for your apps." -q "What is TensorRT?" -v /workspace/bert/models/fine-tuned/bert_tf_v2_large_fp16_128_v2/vocab.txt
    ```

    b. `inference.ipynb` Jupyter Notebook
    The Jupyter Notebook includes a passage and various example questions and allows you to interactively make modifications and see the outcome.
    To launch the Jupyter Notebook from inside the container, run:
    ```bash
    jupyter notebook --ip 0.0.0.0 inference.ipynb
    ```
    Then, use your browser to open the link displayed. The link should look similar to: `http://127.0.0.1:8888/?token=<TOKEN>`


### (Optional) Trying a different configuration

If you would like to run another configuration, you can manually download checkpoints using the included script. For example, run:
```bash
bash scripts/download_model.sh base
```

This will download a BERT Base model instead of the default BERT Large model.
To view all available model options, run:
```bash
bash scripts/download_model.sh -h
```

## Advanced

The following sections provide greater details on inference with TensorRT.

### Scripts and sample code

In the `root` directory, the most important files are:

- `builder.py` - Builds an engine for the specified BERT model
- `Dockerfile` - Container which includes dependencies and model checkpoints to run BERT
- `inference.ipynb` - Runs inference interactively
- `inference.py` - Runs inference with a given passage and question
- `perf.py` - Runs inference benchmarks

The `scripts/` folder encapsulates all the one-click scripts required for running various supported functionalities, such as:

- `build.sh` - Builds a Docker container that is ready to run BERT
- `launch.sh` - Launches the container created by the `build.sh` script.
- `download_model.sh` - Downloads pre-trained model checkpoints from NGC
- `inference_benchmark.sh` - Runs an inference benchmark and prints results

Other folders included in the `root` directory are:

- `helpers` - Contains helpers for tokenization of inputs

### Command-line options

To view the available parameters for each script, you can use the help flag (`-h`).

### TensorRT inference process

As mentioned in the [Quick Start Guide](#quick-start-guide), two options are provided for running inference:
1. The `inference.py` script which accepts a passage and a question and then runs the engine to generate an answer.
2. The `inference.ipynb` Jupyter Notebook which includes a passage and various example questions and allows you to interactively make modifications and see the outcome.


## Performance

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in inference modes.

#### TensorRT inference benchmark

The inference benchmark is performed on a single GPU by the `inference_benchmark.sh` script, which takes the following steps for each set of model parameters:

1. Downloads checkpoints and builds a TensorRT engine if it does not already exist.

2. Runs 1 warm-up iteration then runs inference for 100 iterations for each batch size specified in the script, selecting the profile best for each size.

**Note:** The time measurements do not include the time required to copy inputs to the device and copy outputs to the host.

To run the inference benchmark script, run:
```bash
bash scripts/inference_benchmark.sh
```

Note: Some of the configurations in the benchmark script require 16GB of GPU memory. On GPUs with smaller amounts of memory, parts of the benchmark may fail to run.

Also note that BERT Large engines, especially using mixed precision with large batch sizes and sequence lengths may take a couple hours to build.

### Results

The following sections provide details on how we achieved our performance and inference.

#### Inference performance: NVIDIA T4 (16GB)

Our results were obtained by running the `scripts/inference_benchmark.sh` training script in the container generated by the included Dockerfile on NVIDIA T4 with (1x T4 16G) GPUs.


##### BERT Base

| Sequence Length | Batch Size | TRT Mixed Precision Latency (ms) ||         | TRT FP32 Latency (ms) |           |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 3.88 | 3.89 | 2.34 | 11.18 | 11.19 | 6.48 |
| 128 | 2 | 5.46 | 5.46 | 3.38 | 11.60 | 19.95 | 11.29 |
| 128 | 4 | 8.87 | 8.88 | 5.20 | 22.20 | 22.87 | 21.73 |
| 128 | 8 | 10.35 | 10.47 | 9.75 | 43.59 | 46.57 | 41.98 |
| 128 | 12 | 14.79 | 15.00 | 14.42 | 68.66 | 71.90 | 66.10 |
| 128 | 16 | 21.70 | 24.09 | 19.94 | 87.53 | 88.73 | 86.62 |
| 128 | 24 | 31.80 | 34.47 | 29.42 | 133.56 | 136.35 | 130.90 |
| 128 | 32 | 42.64 | 46.04 | 40.36 | 177.37 | 180.55 | 175.32 |
| 128 | 64 | 82.14 | 87.86 | 78.34 | 346.39 | 347.56 | 343.62 |
| 128 | 128 | 158.23 | 162.59 | 155.07 | 686.51 | 687.55 | 683.74 |
| 384 | 1 | 5.21 | 9.33 | 5.13 | 18.79 | 19.24 | 18.42 |
| 384 | 2 | 9.92 | 16.64 | 9.54 | 37.45 | 41.24 | 36.25 |
| 384 | 4 | 19.46 | 21.00 | 18.74 | 74.32 | 74.79 | 73.72 |
| 384 | 8 | 37.44 | 38.38 | 36.39 | 148.10 | 148.59 | 146.12 |
| 384 | 12 | 55.99 | 61.07 | 54.49 | 222.44 | 226.04 | 219.89 |
| 384 | 16 | 74.84 | 75.38 | 73.98 | 292.29 | 292.93 | 289.43 |
| 384 | 24 | 115.54 | 120.15 | 111.98 | 437.29 | 438.23 | 433.30 |
| 384 | 32 | 148.99 | 149.40 | 146.76 | 583.23 | 584.24 | 578.89 |
| 384 | 64 | 298.82 | 299.71 | 295.27 | 1178.03 | 1178.89 | 1171.15 |
| 384 | 128 | 595.50 | 597.08 | 591.78 | 2337.13 | 2337.88 | 2329.52 |


##### BERT Large

| Sequence Length | Batch Size | TRT Mixed Precision Latency (ms) ||         | TRT FP32 Latency (ms) |           |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 9.87 | 9.88 | 5.61 | 20.82 | 21.40 | 20.58 |
| 128 | 2 | 9.02 | 15.91 | 8.89 | 39.83 | 40.03 | 38.69 |
| 128 | 4 | 16.41 | 17.85 | 15.87 | 76.83 | 76.89 | 75.98 |
| 128 | 8 | 35.97 | 40.19 | 33.43 | 146.27 | 147.10 | 144.46 |
| 128 | 12 | 48.21 | 51.45 | 46.57 | 221.23 | 221.55 | 219.72 |
| 128 | 16 | 65.57 | 69.09 | 63.35 | 283.81 | 285.88 | 280.20 |
| 128 | 24 | 93.58 | 101.40 | 90.73 | 442.93 | 444.13 | 437.57 |
| 128 | 32 | 125.16 | 129.91 | 122.96 | 577.84 | 578.97 | 572.42 |
| 128 | 64 | 247.99 | 249.92 | 242.97 | 1173.16 | 1174.65 | 1164.48 |
| 128 | 128 | 484.90 | 485.28 | 483.70 | 2323.21 | 2323.93 | 2315.76 |
| 384 | 1 | 15.81 | 15.99 | 15.51 | 63.05 | 63.37 | 61.68 |
| 384 | 2 | 30.01 | 31.54 | 29.13 | 121.94 | 122.36 | 120.50 |
| 384 | 4 | 58.08 | 60.09 | 57.09 | 242.38 | 243.52 | 237.58 |
| 384 | 8 | 113.03 | 113.76 | 111.87 | 485.08 | 487.53 | 483.27 |
| 384 | 12 | 167.74 | 168.14 | 165.37 | 722.36 | 723.69 | 715.54 |
| 384 | 16 | 222.91 | 225.62 | 219.78 | 971.95 | 974.37 | 966.16 |
| 384 | 24 | 336.69 | 337.63 | 333.05 | 1457.11 | 1457.96 | 1437.43 |
| 384 | 32 | 452.62 | 455.51 | 444.70 | 1936.90 | 1939.63 | 1923.82 |
| 384 | 64 | 903.01 | 904.37 | 894.41 | 3898.04 | 3900.07 | 3891.59 |
| 384 | 128 | 1804.65 | 1806.82 | 1798.03 | Not measured | Not measured | Not measured |



#### Inference performance: NVIDIA V100 (32GB)

Our results were obtained by running the `scripts/inference_benchmark.sh` training script in the container generated by the included Dockerfile on NVIDIA V100 with (1x V100 32G) GPUs.


##### BERT Base

| Sequence Length | Batch Size | TRT Mixed Precision Latency (ms) ||         | TRT FP32 Latency (ms) |           |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.58 | 1.59 | 1.53 | 3.39 | 3.41 | 3.02 |
| 128 | 2 | 1.86 | 1.86 | 1.74 | 5.25 | 5.26 | 4.68 |
| 128 | 4 | 2.63 | 2.64 | 2.37 | 8.58 | 9.35 | 8.11 |
| 128 | 8 | 4.03 | 4.06 | 3.59 | 15.73 | 17.25 | 15.13 |
| 128 | 12 | 5.59 | 5.61 | 5.08 | 22.82 | 22.95 | 22.59 |
| 128 | 16 | 6.81 | 7.33 | 6.53 | 29.11 | 29.15 | 28.96 |
| 128 | 24 | 9.25 | 10.00 | 8.96 | 43.37 | 43.49 | 43.10 |
| 128 | 32 | 12.68 | 13.54 | 12.32 | 56.16 | 56.48 | 55.83 |
| 128 | 64 | 23.55 | 23.85 | 23.45 | 111.32 | 111.56 | 110.58 |
| 128 | 128 | 45.33 | 45.42 | 45.08 | 221.67 | 221.98 | 220.29 |
| 384 | 1 | 2.46 | 2.47 | 2.24 | 7.76 | 7.78 | 6.82 |
| 384 | 2 | 3.84 | 3.85 | 3.49 | 13.33 | 14.60 | 12.76 |
| 384 | 4 | 6.64 | 6.66 | 6.01 | 24.95 | 25.13 | 24.72 |
| 384 | 8 | 10.95 | 10.97 | 10.78 | 47.67 | 47.99 | 47.41 |
| 384 | 12 | 17.02 | 17.25 | 16.73 | 70.72 | 70.85 | 70.27 |
| 384 | 16 | 21.47 | 21.56 | 21.36 | 93.56 | 93.72 | 93.32 |
| 384 | 24 | 31.61 | 31.72 | 31.45 | 137.73 | 138.19 | 137.12 |
| 384 | 32 | 41.77 | 41.83 | 41.57 | 184.57 | 184.79 | 184.13 |
| 384 | 64 | 84.13 | 84.18 | 83.51 | 369.91 | 370.36 | 368.25 |
| 384 | 128 | 166.11 | 166.77 | 165.32 | 745.16 | 745.91 | 742.54 |




##### BERT Large

| Sequence Length | Batch Size | TRT Mixed Precision Latency (ms) ||         | TRT FP32 Latency (ms) |           |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 3.91 | 3.92 | 3.50 | 9.18 | 9.22 | 8.75 |
| 128 | 2 | 4.83 | 4.83 | 4.29 | 15.34 | 16.73 | 14.61 |
| 128 | 4 | 7.11 | 7.14 | 6.24 | 27.08 | 28.13 | 26.90 | 
| 128 | 8 | 11.38 | 11.49 | 10.97 | 51.95 | 52.08 | 51.60 |
| 128 | 12 | 15.48 | 16.00 | 15.40 | 75.34 | 75.45 | 74.88 |
| 128 | 16 | 20.53 | 21.08 | 20.37 | 99.40 | 99.86 | 98.80 |
| 128 | 24 | 29.37 | 29.41 | 29.18 | 149.51 | 149.86 | 148.57 |
| 128 | 32 | 38.19 | 38.38 | 37.98 | 196.67 | 196.73 | 195.70 |
| 128 | 64 | 74.23 | 74.52 | 73.84 | 386.47 | 386.93 | 383.98 |
| 128 | 128 | 147.13 | 147.85 | 146.24 | 777.01 | 777.75 | 772.92 |
| 384 | 1 | 6.95 | 6.98 | 6.20 | 21.96 | 23.31 | 21.91 |
| 384 | 2 | 10.29 | 10.37 | 9.89 | 42.35 | 42.52 | 42.07 |
| 384 | 4 | 18.00 | 18.53 | 17.86 | 81.00 | 81.06 | 80.41 |
| 384 | 8 | 34.16 | 34.21 | 33.94 | 160.45 | 160.57 | 159.61 |
| 384 | 12 | 50.02 | 50.10 | 49.71 | 238.80 | 239.11 | 237.67 |
| 384 | 16 | 66.45 | 66.58 | 66.07 | 320.78 | 322.03 | 318.12 |
| 384 | 24 | 97.96 | 98.16 | 97.34 | 471.71 | 472.00 | 468.58 |
| 384 | 32 | 128.80 | 129.15 | 128.12 | 635.27 | 635.83 | 631.92 |
| 384 | 64 | 258.09 | 258.34 | 256.21 | 1256.38 | 1257.17 | 1252.49 |
| 384 | 128 | 516.19 | 516.82 | 514.23 | 2525.86 | 2528.41 | 2520.71 |

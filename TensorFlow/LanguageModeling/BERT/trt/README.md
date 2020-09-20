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
|TensorRT|7.0.0.1|
|CUDA|10.2.89|


## Setup

The following section lists the requirements that you need to meet in order to run the BERT model.

### Requirements

This repository contains a `Dockerfile` which extends the TensorRT NGC container and installs some dependencies. Ensure you have the following components:

* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [TensorRT 20.02-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt)
* [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) based GPU with NVIDIA Driver 440.33.01 or later.

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
    This will download checkpoints for a BERT Large FP16 SQuAD v2 model with a sequence length of 128 by default.

**Note:** Since the checkpoints are stored in the directory mounted from the host, they do *not* need to be downloaded each time the container is launched. 

3. Build a TensorRT engine. To build an engine, run the `builder.py` script. For example:
    ```bash
    mkdir -p /workspace/bert/engines && python3 builder.py -m /workspace/bert/models/fine-tuned/bert_tf_v2_large_fp16_128_v2/model.ckpt-8144 -o /workspace/bert/engines/bert_large_128.engine -b 1 -s 128 --fp16 -c /workspace/bert/models/fine-tuned/bert_tf_v2_large_fp16_128_v2
    ```

    This will build an engine with a maximum batch size of 1 (`-b 1`), and sequence length of 128 (`-s 128`) using mixed precision (`--fp16`) using the BERT Large V2 FP16 Sequence Length 128 checkpoint (`-c /workspace/bert/models/fine-tuned/bert_tf_v2_large_fp16_128_v2`).

4. Run inference. Two options are provided for running the model.

    a. `inference.py` script
    This script accepts a passage and question and then runs the engine to generate an answer. The vocabulary file used to train the source model is also specified (`-v /workspace/bert/models/fine-tuned/bert_tf_v2_large_fp16_128_v2/vocab.txt`).
    For example:
    ```bash
    python3 inference.py -e /workspace/bert/engines/bert_large_128.engine -p "TensorRT is a high performance deep learning inference platform that delivers low latency and high throughput for apps such as recommenders, speech and image/video on NVIDIA GPUs. It includes parsers to import models, and plugins to support novel ops and layers before applying optimizations for inference. Today NVIDIA is open-sourcing parsers and plugins in TensorRT so that the deep learning community can customize and extend these components to take advantage of powerful TensorRT optimizations for your apps." -q "What is TensorRT?" -v /workspace/bert/models/fine-tuned/bert_tf_v2_large_fp16_128_v2/vocab.txt
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
to download a BERT Base model instead of the default BERT Large model.

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
1. The `inference.py` script which accepts a passage and a question and then runs the engine to generate an answer. Alternatively, this script can be used to run inference on the Squad dataset.
2. The `inference.ipynb` Jupyter Notebook which includes a passage and various example questions and allows you to interactively make modifications and see the outcome.

## Accuracy

### Evaluating Int8 Accuracy Using The SQuAD Dataset
1.  Download checkpoints for a BERT Large FP32 SQuAD v1.1 model with a sequence length of 128 and 384:
    ```bash
    bash scripts/download_model.sh large fp32 128 v1_1
    bash scripts/download_model.sh large fp32 384 v1_1
    ```

2. Build an engine:
    ```bash
    mkdir -p /workspace/bert/engines && python3 builder.py -m /workspace/bert/models/fine-tuned/bert_tf_v1_1_large_fp32_384_v2/model.ckpt-5474 -o /workspace/bert/engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c /workspace/bert/models/fine-tuned/bert_tf_v2_large_fp16_128_v2 --squad-json ./squad/dev-v1.1.json -v /workspace/bert/models/fine-tuned/bert_tf_v1_1_large_fp32_384_v2/vocab.txt --calib-num 100
    ```

    This will build and engine with a maximum batch size of 1 (`-b 1`), calibration dataset squad (`--squad-json ./squad/dev-v1.1.json`), calibration sentences number 100 (`--calib-num 100`), and sequence length of 128 (`-s 128`) using INT8 mixed precision computation where possible (`--int8 --fp16 --strict`).

3. Run inference using the squad dataset, and evaluate the F1 score and exact match score:
    ```bash
    python3 inference.py -e /workspace/bert/engines/bert_large_384_int8mix.engine -s 384 -sq ./squad/dev-v1.1.json -v /workspace/bert/models/fine-tuned/bert_tf_v1_1_large_fp32_384_v2/vocab.txt -o ./predictions.json
    python3 squad/evaluate-v1.1.py  squad/dev-v1.1.json  ./predictions.json 90
    ```

## Performance

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in inference modes.

#### TensorRT inference benchmark

The inference benchmark is performed on a single GPU by the `inference_benchmark.sh` script, which takes the following steps for each set of model parameters:

1. Downloads checkpoints and builds a TensorRT engine if it does not already exist.

2. Run the inference benchmark, which performs a sweep across batch sizes (1-128) and sequence lengths (128, 384). In each configuration, 1 warm-up iteration is followed by 200 runs to measure and report the BERT inference latencies.

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

| Sequence Length | Batch Size | TensorRT Mixed Precision Latency (ms) ||         | TensorRT FP32 Latency (ms) |           |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.97 | 1.97 | 1.93 | 6.47 | 6.51 | 6.12 |
| 128 | 2 | 2.94 | 2.99 | 2.86 | 11.55 | 11.84 | 11.25 |
| 128 | 4 | 5.00 | 8.44 | 4.88 | 22.08 | 22.63 | 21.90 |
| 128 | 8 | 10.57 | 11.55 | 9.78 | 43.74 | 43.97 | 42.83 |
| 128 | 12 | 15.01 | 15.27 | 14.56 | 68.42 | 69.71 | 67.47 |
| 128 | 16 | 21.64 | 22.92 | 19.12 | 90.90 | 97.17 | 88.47 |
| 128 | 24 | 31 | 31.65 | 29.71 | 131.11 | 133.5 | 129.43 |
| 128 | 32 | 41.27 | 43.65 | 38.54 | 178.45 | 182.65 | 176.77 |
| 128 | 64 | 76.73 | 81.31 | 73.89 | 364.31 | 364.68 | 362.05 |
| 128 | 128 | 151.95 | 152.35 | 150.54 | 672.25 | 673.02 | 669.60 |
| 384 | 1 | 5.18 | 5.19 | 4.97 | 19.11 | 19.13 | 18.44 |
| 384 | 2 | 9.82 | 9.92 | 9.51 | 37.5 | 38.31 | 36.93 |
| 384 | 4 | 18.08 | 19.46 | 17.56 | 77.01 | 81.02 | 74.98 |
| 384 | 8 | 37.32 | 37.94 | 36.77 | 147.05 | 148.85 | 145.27 |
| 384 | 12 | 56.91 | 57.52 | 55.43 | 218.76 | 219.32 | 217.04 |
| 384 | 16 | 73.35 | 76.45 | 71.76 | 302.05 | 303.38 | 299.29 |
| 384 | 24 | 110.14 | 110.78 | 109.03 | 430.22 | 430.91 | 428.49 |
| 384 | 32 | 140.05 | 140.92 | 138.61 | 618.31 | 619.78 | 613.26 |
| 384 | 64 | 284.99 | 285.86 | 282.54 | 1218.55 | 1227.73 | 1215.81 |
| 384 | 128 | 579.86 | 580.91 | 577.25 | 2325.91 | 2327.81 | 2319.26 |



##### BERT Large

| Sequence Length | Batch Size | TensorRT Mixed Precision Latency (ms) ||         | TensorRT FP32 Latency (ms) |           |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 5.63 | 5.66 | 5.39 | 21.53 | 22.16 | 20.74 |
| 128 | 2 | 9.11 | 9.83 | 8.89 | 40.31 | 40.45 | 39.24 |
| 128 | 4 | 16.03 | 17.45 | 15.34 | 81.66 | 85.56 | 78.35 |
| 128 | 8 | 33.2 | 33.98 | 32.59 | 145.86 | 146.2 | 144.46 |
| 128 | 12 | 48.87 | 49.58 | 48.16 | 223.69 | 225.05 | 222.22 |
| 128 | 16 | 64.48 | 68.01 | 62.60 | 289.42 | 292.36 | 286.33 |
| 128 | 24 | 92.63 | 94.4 | 90.90 | 434.81 | 435.49 | 433.37 |
| 128 | 32 | 121.63 | 125.25 | 118.14 | 611.33 | 612.58 | 604.69 |
| 128 | 64 | 237.01 | 239.95 | 233.15 | 1231.35 | 1232.71 | 1220.68 |
| 128 | 128 | 484.48 | 485.39 | 483.37 | 2338.03 | 2341.99 | 2316.32 |
| 384 | 1 | 15.89 | 16.01 | 15.49 | 63.13 | 63.54 | 61.96 |
| 384 | 2 | 30.1 | 30.2 | 29.56 | 121.37 | 122 | 120.19 |
| 384 | 4 | 56.64 | 60.46 | 55.17 | 247.53 | 248.09 | 243.16 |
| 384 | 8 | 114.53 | 115.74 | 112.91 | 485.92 | 486.85 | 484.55 |
| 384 | 12 | 168.8 | 170.65 | 164.88 | 709.33 | 709.88 | 707.13 |
| 384 | 16 | 217.53 | 218.89 | 214.36 | 1005.50 | 1007.29 | 992.56 |
| 384 | 24 | 330.84 | 332.89 | 327.96 | 1489.48 | 1490.96 | 1480.36 |
| 384 | 32 | 454.32 | 461.05 | 443.58 | 1986.66 | 1988.94 | 1976.53 |
| 384 | 64 | 865.36 | 866.96 | 860.22 | 4029.11 | 4031.18 | 4015.06 |
| 384 | 128 | 1762.72 | 1764.65 | 1756.79 | 7736.41 | 7739.45 | 7718.88 |



#### Inference performance: NVIDIA V100 (32GB)

Our results were obtained by running the `scripts/inference_benchmark.sh` training script in the container generated by the included Dockerfile on NVIDIA V100 with (1x V100 32G) GPUs.


##### BERT Base

| Sequence Length | Batch Size | TensorRT Mixed Precision Latency (ms) ||         | TensorRT FP32 Latency (ms) |           |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.39 | 1.45 | 1.37 | 2.93 | 2.95 | 2.91 |
| 128 | 2 | 1.63 | 1.63 | 1.62 | 4.65 | 4.68 | 4.62 |
| 128 | 4 | 2.75 | 2.76 | 2.56 | 8.68 | 9.50 | 8.27 |
| 128 | 8 | 3.58 | 3.59 | 3.55 | 15.56 | 15.63 | 15.42 |
| 128 | 12 | 4.94 | 4.96 | 4.90 | 23.48 | 23.52 | 23.23 |
| 128 | 16 | 7.86 | 7.90 | 7.01 | 30.23 | 30.29 | 29.87 |
| 128 | 24 | 8.94 | 8.94 | 8.89 | 43.52 | 43.59 | 43.24 |
| 128 | 32 | 13.25 | 13.59 | 13.11 | 56.45 | 56.79 | 56.10 |
| 128 | 64 | 25.05 | 25.38 | 24.90 | 111.98 | 112.19 | 111.42 |
| 128 | 128 | 46.31 | 46.38 | 46.01 | 219.6 | 220.3 | 219.22 |
| 384 | 1 | 2.17 | 2.21 | 2.16 | 6.77 | 6.79 | 6.73 |
| 384 | 2 | 3.39 | 3.46 | 3.38 | 13.12 | 13.16 | 13.04 |
| 384 | 4 | 6.79 | 7.09 | 6.29 | 25.33 | 25.45 | 25.16 |
| 384 | 8 | 10.84 | 10.86 | 10.78 | 47.94 | 48.16 | 47.65 |
| 384 | 12 | 16.75 | 16.78 | 16.68 | 72.34 | 72.44 | 72.10 |
| 384 | 16 | 22.66 | 23.28 | 22.56 | 94.65 | 94.93 | 94.08 |
| 384 | 24 | 32.41 | 32.44 | 32.23 | 137.46 | 137.59 | 137.11 |
| 384 | 32 | 44.29 | 44.34 | 44.02 | 186.96 | 187.06 | 185.85 |
| 384 | 64 | 88.56 | 88.72 | 88.15 | 373.48 | 374.26 | 372.37 |
| 384 | 128 | 165.93 | 166.14 | 165.34 | 739.52 | 740.65 | 737.33 |



##### BERT Large

| Sequence Length | Batch Size | TensorRT Mixed Precision Latency (ms) ||         | TensorRT FP32 Latency (ms) |           |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 3.4 | 3.46 | 3.38 | 8.83 | 8.85 | 8.76 |
| 128 | 2 | 4.15 | 4.17 | 4.13 | 14.53 | 14.58 | 14.42 |
| 128 | 4 | 6.76 | 7.41 | 6.45 | 27.40 | 27.52 | 27.22 |
| 128 | 8 | 11.34 | 11.35 | 11.25 | 53.22 | 53.35 | 53.11 |
| 128 | 12 | 15.8 | 15.84 | 15.73 | 75.1 | 75.42 | 74.81 |
| 128 | 16 | 21.64 | 22.27 | 21.50 | 102.64 | 102.71 | 101.92 |
| 128 | 24 | 30.11 | 30.16 | 29.88 | 148.52 | 148.76 | 147.72 |
| 128 | 32 | 40.42 | 40.54 | 40.05 | 203.56 | 203.65 | 202.22 |
| 128 | 64 | 78.77 | 79.01 | 78.04 | 392.26 | 393.11 | 389.84 |
| 128 | 128 | 149.32 | 149.69 | 148.55 | 793.46 | 795.62 | 789.83 |
| 384 | 1 | 6.1 | 6.12 | 6.06 | 21.92 | 21.98 | 21.88 |
| 384 | 2 | 10.16 | 10.18 | 10.08 | 42.47 | 42.52 | 42.35 |
| 384 | 4 | 18.91 | 19.54 | 18.76 | 82.64 | 83.03 | 82.25 |
| 384 | 8 | 35.15 | 35.18 | 34.97 | 164.88 | 164.98 | 164.07 |
| 384 | 12 | 50.31 | 50.36 | 50.04 | 245.53 | 245.85 | 244.50 |
| 384 | 16 | 69.46 | 69.89 | 69.04 | 321.36 | 321.71 | 318.98 |
| 384 | 24 | 97.63 | 97.91 | 97.26 | 485.11 | 485.37 | 482.41 |
| 384 | 32 | 135.16 | 135.70 | 134.39 | 636.32 | 637.40 | 632.66 |
| 384 | 64 | 269.98 | 271.40 | 268.63 | 1264.41 | 1265.69 | 1261.08 |
| 384 | 128 | 513.71 | 514.38 | 511.80 | 2503.02 | 2505.81 | 2499.51 |


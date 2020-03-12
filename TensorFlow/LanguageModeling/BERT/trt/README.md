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
      * [Inference performance: NVIDIA T4 (16GB)](#inference-performance-nvidia-t4-16gb)
        * [BERT Base](#bert-base)
        * [BERT Large](#bert-large)
     * [Inference performance: NVIDIA V100 (32GB)](#inference-performance-nvidia-v100-32gb)
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

This repository contains a `Dockerfile` which extends the TensorRT 19.12-py3 NGC container and installs some dependencies. Ensure you have the following components:

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
    This script accepts a passage and question and then runs the engine to generate an answer.
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
| 128 | 1 | 3.75 | 3.76 | 2.29 | 6.87 | 11.47 | 6.36 |
| 128 | 2 | 5.30 | 5.31 | 3.24 | 11.93 | 12.61 | 11.42 |
| 128 | 4 | 8.44 | 8.45 | 5.07 | 23.06 | 25.00 | 21.95 |
| 128 | 8 | 11.26 | 15.04 | 9.62 | 44.32 | 48.48 | 42.72 |
| 128 | 12 | 14.48 | 15.58 | 13.49 | 71.68 | 74.13 | 68.77 |
| 128 | 16 | 20.59 | 22.37 | 18.85 | 91.68 | 94.78 | 89.14 |
| 128 | 24 | 30.08 | 33.42 | 27.96 | 136.18 | 139.39 | 133.95 |
| 128 | 32 | 40.77 | 42.43 | 37.87 | 185.59 | 190.09 | 183.69 |
| 128 | 64 | 81.62 | 87.71 | 78.75 | 364.41 | 365.23 | 362.20 |
| 128 | 128 | 152.05 | 155.97 | 148.78 | 731.00 | 737.32 | 728.84 |
| 384 | 1 | 8.98 | 8.98 | 5.08 | 18.89 | 20.49 | 18.15 |
| 384 | 2 | 9.44 | 12.65 | 8.99 | 37.40 | 41.15 | 36.33 |
| 384 | 4 | 18.20 | 19.27 | 17.63 | 78.48 | 81.73 | 76.67 |
| 384 | 8 | 36.47 | 38.60 | 35.18 | 146.82 | 147.81 | 145.99 |
| 384 | 12 | 53.24 | 58.13 | 52.20 | 232.26 | 236.12 | 229.47 |
| 384 | 16 | 72.78 | 78.15 | 71.29 | 295.20 | 295.65 | 293.14 |
| 384 | 24 | 109.28 | 111.77 | 106.89 | 443.91 | 444.41 | 440.61 |
| 384 | 32 | 141.79 | 144.52 | 139.52 | 600.35 | 602.22 | 595.49 |
| 384 | 64 | 285.84 | 289.40 | 283.20 | 1221.33 | 1230.01 | 1216.94 |
| 384 | 128 | 575.51 | 576.93 | 572.32 | 2486.46 | 2488.09 | 2476.57 |



##### BERT Large

| Sequence Length | Batch Size | TRT Mixed Precision Latency (ms) ||         | TRT FP32 Latency (ms) |           |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 9.47 | 9.49 | 5.42 | 21.53 | 23.49 | 20.57 |
| 128 | 2 | 9.03 | 9.04 | 8.40 | 41.09 | 41.36 | 40.56 |
| 128 | 4 | 15.80 | 16.86 | 15.36 | 81.76 | 84.27 | 78.16 |
| 128 | 8 | 33.54 | 35.46 | 31.56 | 152.07 | 155.82 | 150.01 |
| 128 | 12 | 47.63 | 51.00 | 45.40 | 225.47 | 226.04 | 222.29 |
| 128 | 16 | 64.19 | 70.93 | 61.87 | 294.67 | 297.90 | 291.08 |
| 128 | 24 | 92.37 | 98.63 | 89.95 | 484.29 | 485.25 | 476.99 |
| 128 | 32 | 120.58 | 124.23 | 117.07 | 598.45 | 600.89 | 593.77 |
| 128 | 64 | 243.68 | 246.62 | 241.75 | 1243.12 | 1244.07 | 1233.66 |
| 128 | 128 | 485.04 | 486.12 | 483.30 | 2436.11 | 2436.93 | 2431.58 |
| 384 | 1 | 15.57 | 17.38 | 14.89 | 63.26 | 68.00 | 62.07 |
| 384 | 2 | 30.27 | 30.89 | 28.72 | 120.27 | 121.33 | 118.62 |
| 384 | 4 | 59.39 | 64.94 | 56.24 | 246.08 | 250.72 | 242.73 |
| 384 | 8 | 109.48 | 112.27 | 107.89 | 495.19 | 497.76 | 487.22 |
| 384 | 12 | 163.00 | 163.99 | 161.43 | 729.38 | 730.70 | 725.92 |
| 384 | 16 | 216.60 | 217.65 | 214.09 | 1011.43 | 1012.51 | 999.24 |
| 384 | 24 | 330.63 | 332.60 | 325.75 | 1489.48 | 1490.96 | 1480.36 |
| 384 | 32 | 458.64 | 461.83 | 445.82 | 2025.18 | 2026.12 | 2012.51 |
| 384 | 64 | 895.19 | 898.09 | 888.67 | 4061.51 | 4066.80 | 4049.73 |
| 384 | 128 | 1782.36 | 1782.99 | 1773.92 | 7736.41 | 7739.45 | 7718.88 |



#### Inference performance: NVIDIA V100 (32GB)

Our results were obtained by running the `scripts/inference_benchmark.sh` training script in the container generated by the included Dockerfile on NVIDIA V100 with (1x V100 32G) GPUs.


##### BERT Base

| Sequence Length | Batch Size | TRT Mixed Precision Latency (ms) ||         | TRT FP32 Latency (ms) |           |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.59 | 1.59 | 1.49 | 3.01 | 3.02 | 2.98 |
| 128 | 2 | 1.97 | 1.97 | 1.86 | 4.86 | 4.89 | 4.82 |
| 128 | 4 | 2.76 | 2.76 | 2.51 | 8.46 | 8.52 | 8.34 |
| 128 | 8 | 4.36 | 4.37 | 3.95 | 15.98 | 16.10 | 15.80 |
| 128 | 12 | 5.99 | 6.00 | 5.37 | 24.01 | 24.10 | 23.83 |
| 128 | 16 | 7.87 | 7.90 | 7.00 | 29.97 | 30.08 | 29.70 |
| 128 | 24 | 9.97 | 10.69 | 9.83 | 45.62 | 45.77 | 45.23 |
| 128 | 32 | 13.55 | 13.74 | 13.45 | 58.93 | 59.47 | 58.65 |
| 128 | 64 | 26.02 | 26.12 | 25.81 | 117.78 | 118.45 | 117.50 |
| 128 | 128 | 50.67 | 50.72 | 50.26 | 236.74 | 237.46 | 235.52 |
| 384 | 1 | 2.59 | 2.59 | 2.42 | 6.77 | 6.78 | 6.73 |
| 384 | 2 | 4.08 | 4.09 | 3.68 | 13.23 | 13.29 | 13.15 |
| 384 | 4 | 6.51 | 7.02 | 6.38 | 26.16 | 26.29 | 25.91 |
| 384 | 8 | 12.00 | 12.77 | 11.72 | 50.08 | 50.18 | 49.68 |
| 384 | 12 | 18.55 | 18.73 | 18.18 | 70.64 | 70.95 | 70.24 |
| 384 | 16 | 22.66 | 22.85 | 22.55 | 99.42 | 99.49 | 98.75 |
| 384 | 24 | 33.23 | 33.36 | 33.07 | 146.38 | 147.06 | 146.08 |
| 384 | 32 | 45.78 | 45.85 | 45.52 | 196.81 | 196.98 | 195.62 |
| 384 | 64 | 88.07 | 88.23 | 87.67 | 396.41 | 396.85 | 394.91 |
| 384 | 128 | 185.00 | 185.56 | 184.23 | 752.42 | 752.86 | 749.65 |



##### BERT Large

| Sequence Length | Batch Size | TRT Mixed Precision Latency (ms) ||         | TRT FP32 Latency (ms) |           |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 4.07 | 4.22 | 3.71 | 9.33 | 10.10 | 9.00 |
| 128 | 2 | 5.02 | 5.05 | 4.45 | 14.70 | 14.73 | 14.64 |
| 128 | 4 | 7.39 | 7.41 | 6.55 | 27.36 | 27.41 | 27.14 |
| 128 | 8 | 12.30 | 12.76 | 11.99 | 56.19 | 56.43 | 55.77 |
| 128 | 12 | 16.80 | 16.88 | 16.64 | 80.41 | 80.50 | 79.72 |
| 128 | 16 | 22.46 | 22.74 | 22.26 | 106.09 | 106.91 | 106.00 |
| 128 | 24 | 31.76 | 31.90 | 31.57 | 150.86 | 150.90 | 149.75 |
| 128 | 32 | 40.38 | 40.45 | 39.97 | 202.57 | 203.10 | 201.35 |
| 128 | 64 | 78.30 | 78.46 | 77.73 | 408.30 | 409.42 | 405.45 |
| 128 | 128 | 160.86 | 161.11 | 159.87 | 807.23 | 808.80 | 801.68 |
| 384 | 1 | 7.21 | 7.23 | 6.45 | 22.54 | 23.82 | 22.45 |
| 384 | 2 | 10.89 | 10.93 | 10.42 | 45.31 | 45.46 | 45.03 |
| 384 | 4 | 18.85 | 19.47 | 18.68 | 86.29 | 86.76 | 85.68 |
| 384 | 8 | 35.60 | 35.66 | 35.37 | 169.35 | 169.91 | 168.20 |
| 384 | 12 | 54.11 | 54.27 | 53.88 | 245.31 | 245.90 | 243.65 |
| 384 | 16 | 72.39 | 72.79 | 71.97 | 319.32 | 321.17 | 317.74 |
| 384 | 24 | 106.13 | 106.29 | 105.59 | 495.00 | 496.08 | 491.71 |
| 384 | 32 | 134.57 | 134.81 | 133.82 | 629.86 | 630.36 | 627.66 |
| 384 | 64 | 269.01 | 269.49 | 267.73 | 1322.65 | 1323.59 | 1314.37 |
| 384 | 128 | 541.98 | 543.76 | 537.95 | 2551.77 | 2552.49 | 2545.75 |

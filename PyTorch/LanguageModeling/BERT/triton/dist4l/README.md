# Deploying the BERT model on Triton Inference Server

This folder contains instructions for deployment to run inference
on Triton Inference Server as well as a detailed performance analysis.
The purpose of this document is to help you with achieving
the best inference performance.

## Table of contents
  - [Solution overview](#solution-overview)
    - [Introduction](#introduction)
    - [Deployment process](#deployment-process)
  - [Setup](#setup)
  - [Quick Start Guide](#quick-start-guide)
  - [Performance](#performance)
    - [Offline scenario](#offline-scenario)
        - [Offline: NVIDIA A30, ONNX Runtime with FP16](#offline-nvidia-a30-onnx-runtime-with-fp16)
        - [Offline: NVIDIA A30, ONNX Runtime with FP16, Backend accelerator TensorRT](#offline-nvidia-a30-onnx-runtime-with-fp16-backend-accelerator-tensorrt)
        - [Offline: NVIDIA A30, NVIDIA TensorRT with FP16](#offline-nvidia-a30-nvidia-tensorrt-with-fp16)
        - [Offline: NVIDIA A30, NVIDIA PyTorch with FP16](#offline-nvidia-a30-pytorch-with-fp16)
        - [Offline: NVIDIA DGX-1 (1x V100 32GB), ONNX Runtime with FP16](#offline-nvidia-dgx-1-1x-v100-32gb-onnx-runtime-with-fp16)
        - [Offline: NVIDIA DGX-1 (1x V100 32GB), ONNX Runtime with FP16, Backend accelerator TensorRT](#offline-nvidia-dgx-1-1x-v100-32gb-onnx-runtime-with-fp16-backend-accelerator-tensorrt)
        - [Offline: NVIDIA DGX-1 (1x V100 32GB), NVIDIA TensorRT with FP16](#offline-nvidia-dgx-1-1x-v100-32gb-nvidia-tensorrt-with-fp16)
        - [Offline: NVIDIA DGX-1 (1x V100 32GB), PyTorch with FP16](#offline-nvidia-dgx-1-1x-v100-32gb-pytorch-with-fp16)
        - [Offline: NVIDIA DGX A100 (1x A100 80GB), ONNX Runtime with FP16](#offline-nvidia-dgx-a100-1x-a100-80gb-onnx-runtime-with-fp16)
        - [Offline: NVIDIA DGX A100 (1x A100 80GB), ONNX Runtime with FP16, Backend accelerator TensorRT](#offline-nvidia-dgx-a100-1x-a100-80gb-onnx-runtime-with-fp16-backend-accelerator-tensorrt)
        - [Offline: NVIDIA DGX A100 (1x A100 80GB), NVIDIA TensorRT with FP16](#offline-nvidia-dgx-a100-1x-a100-80gb-nvidia-tensorrt-with-fp16)
        - [Offline: NVIDIA DGX A100 (1x A100 80GB), PyTorch with FP16](#offline-nvidia-dgx-a100-1x-a100-80gb-pytorch-with-fp16)
        - [Offline: NVIDIA T4, ONNX Runtime with FP16](#offline-nvidia-t4-onnx-runtime-with-fp16)
        - [Offline: NVIDIA T4, ONNX Runtime with FP16, Backend accelerator TensorRT](#offline-nvidia-t4-onnx-runtime-with-fp16-backend-accelerator-tensorrt)
        - [Offline: NVIDIA T4, NVIDIA TensorRT with FP16](#offline-nvidia-t4-nvidia-tensorrt-with-fp16)
        - [Offline: NVIDIA T4, PyTorch with FP16](#offline-nvidia-t4-pytorch-with-fp16)
  - [Advanced](#advanced)
    - [Prepare configuration](#prepare-configuration)
    - [Step by step deployment process](#step-by-step-deployment-process)
    - [Latency explanation](#latency-explanation)
  - [Release notes](#release-notes)
    - [Changelog](#changelog)
    - [Known issues](#known-issues)


## Solution overview
### Introduction
The [NVIDIA Triton Inference Server](https://github.com/NVIDIA/triton-inference-server)
provides a datacenter and cloud inferencing solution optimized for NVIDIA GPUs.
The server provides an inference service via an HTTP or gRPC endpoint,
allowing remote clients to request inferencing for any number of GPU
or CPU models being managed by the server.

This README provides step-by-step deployment instructions for models generated
during training (as described in the [model README](../readme.md)).
Additionally, this README provides the corresponding deployment scripts that
ensure optimal GPU utilization during inferencing on Triton Inference Server.

### Deployment process

The deployment process consists of two steps:

1. Conversion.

   The purpose of conversion is to find the best performing model
   format supported by Triton Inference Server.
   Triton Inference Server uses a number of runtime backends such as
   [TensorRT](https://developer.nvidia.com/tensorrt),
   [LibTorch](https://github.com/triton-inference-server/pytorch_backend) and 
   [ONNX Runtime](https://github.com/triton-inference-server/onnxruntime_backend)
   to support various model types. Refer to the
   [Triton documentation](https://github.com/triton-inference-server/backend#where-can-i-find-all-the-backends-that-are-available-for-triton)
   for a list of available backends.

2. Configuration.

   Model configuration on Triton Inference Server, which generates
   necessary [configuration files](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md).

After deployment Triton inference server is used for evaluation of converted model in two steps:

1. Accuracy tests.

   Produce results which are tested against given accuracy thresholds.

2. Performance tests.

   Produce latency and throughput results for offline (static batching)
   and online (dynamic batching) scenarios.


All steps are executed by provided runner script. Refer to [Quick Start Guide](#quick-start-guide)


## Setup
Ensure you have the following components:
* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch NGC container 21.10](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
* [Triton Inference Server NGC container 21.10](https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver)
* [NVIDIA CUDA](https://docs.nvidia.com/cuda/archive//index.html)
* [NVIDIA Ampere](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/), [Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) based GPU



## Quick Start Guide
Running the following scripts will build and launch the container with all required dependencies for native PyTorch as well as Triton Inference Server. This is necessary for running inference and can also be used for data download, processing, and training of the model.

1. Clone the repository.

```
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/PyTorch/LanguageModeling/BERT/
```

2. Build and run a container that extends NGC PyTorch with the Triton client libraries and necessary dependencies.

```
./triton/dist4l/scripts/docker/build.sh
./triton/dist4l/scripts/docker/interactive.sh
```

3. Prepare dataset.
Runner requires script downloading and preparing publicly available datasets to run the process.
Script will download necessary data to DeepLearningExamples/PyTorch/LanguageModeling/BERT/datasets catalog.

```
./triton/dist4l/runner/prepare_datasets.sh
```

4. Execute runner script (please mind, the run scripts are prepared per NVIDIA GPU).

```
NVIDIA A30: ./triton/dist4l/runner/start_NVIDIA-A30.sh

NVIDIA DGX-1 (1x V100 32GB): ./triton/dist4l/runner/start_NVIDIA-DGX-1-\(1x-V100-32GB\).sh

NVIDIA DGX A100 (1x A100 80GB): ./triton/dist4l/runner/start_NVIDIA-DGX-A100-\(1x-A100-80GB\).sh

NVIDIA T4: ./triton/dist4l/runner/start_NVIDIA-T4.sh
```

## Performance
The performance measurements in this document were conducted at the time of publication and may not reflect
the performance achieved from NVIDIA’s latest software release. For the most up-to-date performance measurements, go to
[NVIDIA Data Center Deep Learning Product Performance](https://developer.nvidia.com/deep-learning-performance-training-inference).
### Offline scenario

The offline scenario assumes the client and server are located on the same host. The tests uses:
- tensors are passed through shared memory between client and server, the Perf Analyzer flag `shared-memory=system` is used
- single request is send from client to server with static size of batch


#### Offline: NVIDIA A30, ONNX Runtime with FP16

Our results were obtained using the following configuration:

| Parameter Name               | Parameter Value |
|:-----------------------------|:----------------|
| GPU                          | NVIDIA A30      |
| Backend                      | ONNX Runtime    |
| Backend accelerator          | -               |
| Precision                    | FP16            |
| Model format                 | ONNX            |
| Max batch size               | 16              |
| Number of model instances    | 1               |
| Accelerator Precision | -               |
| Max Seq Length | 384             |
| SQuAD v1.1 F1 Score       | 83.37           |

<summary>Results Table</summary>

|   Batch |   Concurrency |   Inferences/Second |   Client Send (ms) |   Network+Server Send/Recv (ms) |   Server Queue (ms) |   Server Compute Input (ms) |   Server Compute Infer (ms) |   Server Compute Output (ms) |   Client Recv (ms) |   p50 latency (ms) |   p90 latency (ms) |   p95 latency (ms) |   p99 latency (ms) |   avg latency (ms) |
|--------:|--------------:|--------------------:|-------------------:|--------------------------------:|--------------------:|----------------------------:|----------------------------:|-----------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
|       1 |             1 |               830.0 |                0.0 |                             0.1 |                 0.0 |                         0.0 |                         1.0 |                          0.0 |                0.0 |                1.2 |                1.2 |                1.2 |                1.3 |                1.2 |
|       8 |             1 |              2053.9 |                0.0 |                             0.2 |                 0.0 |                         0.1 |                         3.6 |                          0.0 |                0.0 |                3.9 |                3.9 |                3.9 |                3.9 |                3.9 |
|      16 |             1 |              2128.0 |                0.0 |                             0.4 |                 0.1 |                         0.1 |                         6.9 |                          0.0 |                0.0 |                7.5 |                7.5 |                7.6 |                7.7 |                7.5 |


#### Offline: NVIDIA A30, ONNX Runtime with FP16, Backend accelerator TensorRT

Our results were obtained using the following configuration:

| Parameter Name               | Parameter Value              |
|:-----------------------------|:-----------------------------|
| GPU                          |NVIDIA A30            |
| Backend                      |ONNX Runtime        |
| Backend accelerator          |NVIDIA TensorRT|
| Precision                    |FP16      |
| Model format                 |ONNX   |
| Max batch size               |16 |
| Number of model instances    |1|
| Accelerator Precision | FP16                 |
| Max Seq Length | 384                 |
| SQuAD v1.1 F1 Score       | 83.34               |

<summary>Results Table</summary>

|   Batch |   Concurrency |   Inferences/Second |   Client Send (ms) |   Network+Server Send/Recv (ms) |   Server Queue (ms) |   Server Compute Input (ms) |   Server Compute Infer (ms) |   Server Compute Output (ms) |   Client Recv (ms) |   p50 latency (ms) |   p90 latency (ms) |   p95 latency (ms) |   p99 latency (ms) |   avg latency (ms) |
|--------:|--------------:|--------------------:|-------------------:|--------------------------------:|--------------------:|----------------------------:|----------------------------:|-----------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
|       1 |             1 |              1219.0 |                0.0 |                             0.2 |                 0.0 |                         0.1 |                         0.5 |                          0.0 |                0.0 |                0.8 |                0.8 |                0.8 |                0.8 |                0.8 |
|       8 |             1 |              3616.0 |                0.0 |                             0.1 |                 0.0 |                         0.1 |                         1.9 |                          0.0 |                0.0 |                2.2 |                2.2 |                2.2 |                2.4 |                2.2 |
|      16 |             1 |              4256.0 |                0.0 |                             0.2 |                 0.0 |                         0.1 |                         3.5 |                          0.0 |                0.0 |                3.7 |                3.8 |                3.8 |                3.8 |                3.7 |


#### Offline: NVIDIA A30, NVIDIA TensorRT with FP16

Our results were obtained using the following configuration:

| Parameter Name               | Parameter Value |
|:-----------------------------|:----------------|
| GPU                          | NVIDIA A30      |
| Backend                      | NVIDIA TensorRT |
| Backend accelerator          | -               |
| Precision                    | FP16            |
| Model format                 | NVIDIA TensorRT |
| Max batch size               | 16              |
| Number of model instances    | 1               |
| NVIDIA TensorRT Capture CUDA Graph | Disabled        |
| Accelerator Precision | -               |
| Max Seq Length | 384             |
| SQuAD v1.1 F1 Score       | 83.34           |

<summary>Results Table</summary>

|   Batch |   Concurrency |   Inferences/Second |   Client Send (ms) |   Network+Server Send/Recv (ms) |   Server Queue (ms) |   Server Compute Input (ms) |   Server Compute Infer (ms) |   Server Compute Output (ms) |   Client Recv (ms) |   p50 latency (ms) |   p90 latency (ms) |   p95 latency (ms) |   p99 latency (ms) |   avg latency (ms) |
|--------:|--------------:|--------------------:|-------------------:|--------------------------------:|--------------------:|----------------------------:|----------------------------:|-----------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
|       1 |             1 |              1214.0 |                0.0 |                             0.2 |                 0.0 |                         0.1 |                         0.5 |                          0.0 |                0.0 |                0.8 |                0.8 |                0.8 |                0.8 |                0.8 |
|       8 |             1 |              3456.0 |                0.0 |                             0.2 |                 0.0 |                         0.1 |                         2.0 |                          0.0 |                0.0 |                2.3 |                2.3 |                2.4 |                2.5 |                2.3 |
|      16 |             1 |              3968.0 |                0.0 |                             0.4 |                 0.0 |                         0.1 |                         3.5 |                          0.0 |                0.0 |                4.0 |                4.0 |                4.0 |                4.2 |                4.0 |

#### Offline: NVIDIA A30, PyTorch with FP16

Our results were obtained using the following configuration:

| Parameter Name               | Parameter Value   |
|:-----------------------------|:------------------|
| GPU                          | NVIDIA A30        |
| Backend                      | PyTorch           |
| Backend accelerator          | -                 |
| Precision                    | FP16              |
| Model format                 | TorchScript Trace |
| Max batch size               | 16                |
| Number of model instances    | 1                 |
| Accelerator Precision | -                 |
| Max Seq Length | 384               |
| SQuAD v1.1 F1 Score       | 83.35             |

<summary>Results Table</summary>

|   Batch |   Concurrency |   Inferences/Second |   Client Send (ms) |   Network+Server Send/Recv (ms) |   Server Queue (ms) |   Server Compute Input (ms) |   Server Compute Infer (ms) |   Server Compute Output (ms) |   Client Recv (ms) |   p50 latency (ms) |   p90 latency (ms) |   p95 latency (ms) |   p99 latency (ms) |   avg latency (ms) |
|--------:|--------------:|--------------------:|-------------------:|--------------------------------:|--------------------:|----------------------------:|----------------------------:|-----------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
|       1 |             1 |               558.0 |                0.0 |                             0.1 |                 0.0 |                         0.0 |                         1.6 |                          0.0 |                0.0 |                1.8 |                1.8 |                1.8 |                1.8 |                1.8 |
|       8 |             1 |              2061.9 |                0.0 |                             0.3 |                 0.0 |                         0.1 |                         1.6 |                          1.9 |                0.0 |                3.9 |                3.9 |                3.9 |                3.9 |                3.9 |
|      16 |             1 |              2429.6 |                0.0 |                             0.2 |                 0.0 |                         0.1 |                         1.6 |                          4.7 |                0.0 |                6.6 |                6.6 |                6.6 |                6.6 |                6.6 |



#### Offline: NVIDIA DGX-1 (1x V100 32GB), ONNX Runtime with FP16

Our results were obtained using the following configuration:

| Parameter Name               | Parameter Value             |
|:-----------------------------|:----------------------------|
| GPU                          | NVIDIA DGX-1 (1x V100 32GB) |
| Backend                      | ONNX Runtime                |
| Backend accelerator          | -                           |
| Precision                    | FP16                        |
| Model format                 | ONNX                        |
| Max batch size               | 16                          |
| Number of model instances    | 1                           |
| Accelerator Precision | -                           |
| Max Seq Length | 384                         |
| SQuAD v1.1 F1 Score       | 83.37                       |

<summary>Results Table</summary>

|   Batch |   Concurrency |   Inferences/Second |   Client Send (ms) |   Network+Server Send/Recv (ms) |   Server Queue (ms) |   Server Compute Input (ms) |   Server Compute Infer (ms) |   Server Compute Output (ms) |   Client Recv (ms) |   p50 latency (ms) |   p90 latency (ms) |   p95 latency (ms) |   p99 latency (ms) |   avg latency (ms) |
|--------:|--------------:|--------------------:|-------------------:|--------------------------------:|--------------------:|----------------------------:|----------------------------:|-----------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
|       1 |             1 |               521.0 |                0.0 |                             0.3 |                 0.1 |                         0.1 |                         1.4 |                          0.0 |                0.0 |                1.9 |                2.1 |                2.1 |                2.2 |                1.9 |
|       8 |             1 |              2048.0 |                0.0 |                             0.3 |                 0.0 |                         0.1 |                         3.5 |                          0.0 |                0.0 |                3.9 |                3.9 |                3.9 |                4.0 |                3.9 |
|      16 |             1 |              2304.0 |                0.0 |                             0.3 |                 0.1 |                         0.1 |                         6.4 |                          0.0 |                0.0 |                6.9 |                7.1 |                7.2 |                7.3 |                6.9 |


#### Offline: NVIDIA DGX-1 (1x V100 32GB), ONNX Runtime with FP16, Backend accelerator TensorRT

Our results were obtained using the following configuration:

| Parameter Name               | Parameter Value             |
|:-----------------------------|:----------------------------|
| GPU                          | NVIDIA DGX-1 (1x V100 32GB) |
| Backend                      | ONNX Runtime                |
| Backend accelerator          | NVIDIA TensorRT             |
| Precision                    | FP16                        |
| Model format                 | ONNX                        |
| Max batch size               | 16                          |
| Number of model instances    | 1                           |
| Accelerator Precision | FP16                        |
| Max Seq Length | 384                         |
| SQuAD v1.1 F1 Score       | 83.34                       |

<summary>Results Table</summary>

|   Batch |   Concurrency |   Inferences/Second |   Client Send (ms) |   Network+Server Send/Recv (ms) |   Server Queue (ms) |   Server Compute Input (ms) |   Server Compute Infer (ms) |   Server Compute Output (ms) |   Client Recv (ms) |   p50 latency (ms) |   p90 latency (ms) |   p95 latency (ms) |   p99 latency (ms) |   avg latency (ms) |
|--------:|--------------:|--------------------:|-------------------:|--------------------------------:|--------------------:|----------------------------:|----------------------------:|-----------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
|       1 |             1 |               789.0 |                0.0 |                             0.3 |                 0.1 |                         0.1 |                         0.8 |                          0.0 |                0.0 |                1.2 |                1.4 |                1.5 |                1.6 |                1.2 |
|       8 |             1 |              2536.0 |                0.0 |                             0.3 |                 0.1 |                         0.1 |                         2.6 |                          0.0 |                0.0 |                3.1 |                3.3 |                3.4 |                3.5 |                3.1 |
|      16 |             1 |              2992.0 |                0.0 |                             0.3 |                 0.0 |                         0.1 |                         4.9 |                          0.0 |                0.0 |                5.3 |                5.5 |                5.6 |                5.6 |                5.3 |


#### Offline: NVIDIA DGX-1 (1x V100 32GB), NVIDIA TensorRT with FP16

Our results were obtained using the following configuration:

| Parameter Name               | Parameter Value             |
|:-----------------------------|:----------------------------|
| GPU                          | NVIDIA DGX-1 (1x V100 32GB) |
| Backend                      | NVIDIA TensorRT             |
| Backend accelerator          | -                           |
| Precision                    | FP16                        |
| Model format                 | NVIDIA TensorRT             |
| Max batch size               | 16                          |
| Number of model instances    | 1                           |
| NVIDIA TensorRT Capture CUDA Graph | Disabled                    |
| Accelerator Precision | -                           |
| Max Seq Length | 384                         |
| SQuAD v1.1 F1 Score       | 83.34                       |

<summary>Results Table</summary>

|   Batch |   Concurrency |   Inferences/Second |   Client Send (ms) |   Network+Server Send/Recv (ms) |   Server Queue (ms) |   Server Compute Input (ms) |   Server Compute Infer (ms) |   Server Compute Output (ms) |   Client Recv (ms) |   p50 latency (ms) |   p90 latency (ms) |   p95 latency (ms) |   p99 latency (ms) |   avg latency (ms) |
|--------:|--------------:|--------------------:|-------------------:|--------------------------------:|--------------------:|----------------------------:|----------------------------:|-----------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
|       1 |             1 |               803.0 |                0.0 |                             0.3 |                 0.1 |                         0.2 |                         0.7 |                          0.0 |                0.0 |                1.2 |                1.3 |                1.3 |                1.4 |                1.2 |
|       8 |             1 |              2576.0 |                0.0 |                             0.2 |                 0.1 |                         0.2 |                         2.6 |                          0.0 |                0.0 |                3.1 |                3.1 |                3.1 |                3.2 |                3.1 |
|      16 |             1 |              2928.0 |                0.0 |                             0.3 |                 0.1 |                         0.2 |                         4.8 |                          0.0 |                0.0 |                5.4 |                5.5 |                5.5 |                5.6 |                5.4 |


#### Offline: NVIDIA DGX-1 (1x V100 32GB), PyTorch with FP16

Our results were obtained using the following configuration:

| Parameter Name               | Parameter Value             |
|:-----------------------------|:----------------------------|
| GPU                          | NVIDIA DGX-1 (1x V100 32GB) |
| Backend                      | PyTorch                     |
| Backend accelerator          | -                           |
| Precision                    | FP16                        |
| Model format                 | TorchScript Trace           |
| Max batch size               | 16                          |
| Number of model instances    | 1                           |
| Accelerator Precision | -                           |
| Max Seq Length | 384                         |
| SQuAD v1.1 F1 Score       | 83.35                       |

<summary>Results Table</summary>

|   Batch |   Concurrency |   Inferences/Second |   Client Send (ms) |   Network+Server Send/Recv (ms) |   Server Queue (ms) |   Server Compute Input (ms) |   Server Compute Infer (ms) |   Server Compute Output (ms) |   Client Recv (ms) |   p50 latency (ms) |   p90 latency (ms) |   p95 latency (ms) |   p99 latency (ms) |   avg latency (ms) |
|--------:|--------------:|--------------------:|-------------------:|--------------------------------:|--------------------:|----------------------------:|----------------------------:|-----------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
|       1 |             1 |               325.0 |                0.0 |                             0.3 |                 0.0 |                         0.1 |                         2.6 |                          0.0 |                0.0 |                3.0 |                3.3 |                3.4 |                3.5 |                3.1 |
|       8 |             1 |              2200.0 |                0.0 |                             0.3 |                 0.0 |                         0.1 |                         2.6 |                          0.7 |                0.0 |                3.6 |                3.6 |                3.6 |                3.7 |                3.6 |
|      16 |             1 |              2784.0 |                0.0 |                             0.2 |                 0.0 |                         0.1 |                         2.5 |                          2.8 |                0.0 |                5.7 |                5.7 |                5.8 |                5.8 |                5.7 |


#### Offline: NVIDIA DGX A100 (1x A100 80GB), ONNX Runtime with FP16

Our results were obtained using the following configuration:

| Parameter Name               | Parameter Value                |
|:-----------------------------|:-------------------------------|
| GPU                          | NVIDIA DGX A100 (1x A100 80GB) |
| Backend                      | ONNX Runtime                   |
| Backend accelerator          | -                              |
| Precision                    | FP16                           |
| Model format                 | ONNX                           |
| Max batch size               | 16                             |
| Number of model instances    | 1                              |
| Accelerator Precision | -                              |
| Max Seq Length | 384                            |
| SQuAD v1.1 F1 Score       | 83.37                          |

<summary>Results Table</summary>

|   Batch |   Concurrency |   Inferences/Second |   Client Send (ms) |   Network+Server Send/Recv (ms) |   Server Queue (ms) |   Server Compute Input (ms) |   Server Compute Infer (ms) |   Server Compute Output (ms) |   Client Recv (ms) |   p50 latency (ms) |   p90 latency (ms) |   p95 latency (ms) |   p99 latency (ms) |   avg latency (ms) |
|--------:|--------------:|--------------------:|-------------------:|--------------------------------:|--------------------:|----------------------------:|----------------------------:|-----------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
|       1 |             1 |               679.0 |                0.0 |                             0.1 |                 0.0 |                         0.1 |                         1.3 |                          0.0 |                0.0 |                1.5 |                1.5 |                1.5 |                1.5 |                1.5 |
|       8 |             1 |              3360.0 |                0.0 |                             0.1 |                 0.0 |                         0.1 |                         2.2 |                          0.0 |                0.0 |                2.4 |                2.4 |                2.4 |                2.5 |                2.4 |
|      16 |             1 |              4032.0 |                0.0 |                             0.1 |                 0.0 |                         0.1 |                         3.7 |                          0.0 |                0.0 |                4.0 |                4.0 |                4.0 |                4.1 |                4.0 |


#### Offline: NVIDIA DGX A100 (1x A100 80GB), ONNX Runtime with FP16, Backend accelerator TensorRT

Our results were obtained using the following configuration:

| Parameter Name               | Parameter Value                |
|:-----------------------------|:-------------------------------|
| GPU                          | NVIDIA DGX A100 (1x A100 80GB) |
| Backend                      | ONNX Runtime                   |
| Backend accelerator          | NVIDIA TensorRT                |
| Precision                    | FP16                           |
| Model format                 | ONNX                           |
| Max batch size               | 16                             |
| Number of model instances    | 1                              |
| Accelerator Precision | FP16                           |
| Max Seq Length | 384                            |
| SQuAD v1.1 F1 Score       | 83.35                          |

<summary>Results Table</summary>

|   Batch |   Concurrency |   Inferences/Second |   Client Send (ms) |   Network+Server Send/Recv (ms) |   Server Queue (ms) |   Server Compute Input (ms) |   Server Compute Infer (ms) |   Server Compute Output (ms) |   Client Recv (ms) |   p50 latency (ms) |   p90 latency (ms) |   p95 latency (ms) |   p99 latency (ms) |   avg latency (ms) |
|--------:|--------------:|--------------------:|-------------------:|--------------------------------:|--------------------:|----------------------------:|----------------------------:|-----------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
|       1 |             1 |              1471.0 |                0.0 |                             0.1 |                 0.0 |                         0.1 |                         0.5 |                          0.0 |                0.0 |                0.7 |                0.7 |                0.7 |                0.7 |                0.7 |
|       8 |             1 |              5648.0 |                0.0 |                             0.1 |                 0.0 |                         0.1 |                         1.2 |                          0.0 |                0.0 |                1.4 |                1.4 |                1.4 |                1.4 |                1.4 |
|      16 |             1 |              6960.0 |                0.0 |                             0.1 |                 0.0 |                         0.1 |                         2.1 |                          0.0 |                0.0 |                2.3 |                2.3 |                2.3 |                2.4 |                2.3 |


#### Offline: NVIDIA DGX A100 (1x A100 80GB), NVIDIA TensorRT with FP16

Our results were obtained using the following configuration:

| Parameter Name               | Parameter Value                |
|:-----------------------------|:-------------------------------|
| GPU                          | NVIDIA DGX A100 (1x A100 80GB) |
| Backend                      | NVIDIA TensorRT                |
| Backend accelerator          | -                              |
| Precision                    | FP16                           |
| Model format                 | NVIDIA TensorRT                |
| Max batch size               | 16                             |
| Number of model instances    | 1                              |
| NVIDIA TensorRT Capture CUDA Graph | Disabled                       |
| Accelerator Precision | -                              |
| Max Seq Length | 384                            |
| SQuAD v1.1 F1 Score       | 83.35                          |

<summary>Results Table</summary>

|   Batch |   Concurrency |   Inferences/Second |   Client Send (ms) |   Network+Server Send/Recv (ms) |   Server Queue (ms) |   Server Compute Input (ms) |   Server Compute Infer (ms) |   Server Compute Output (ms) |   Client Recv (ms) |   p50 latency (ms) |   p90 latency (ms) |   p95 latency (ms) |   p99 latency (ms) |   avg latency (ms) |
|--------:|--------------:|--------------------:|-------------------:|--------------------------------:|--------------------:|----------------------------:|----------------------------:|-----------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
|       1 |             1 |              1561.0 |                0.0 |                             0.1 |                 0.0 |                         0.1 |                         0.5 |                          0.0 |                0.0 |                0.6 |                0.6 |                0.7 |                0.7 |                0.6 |
|       8 |             1 |              5096.0 |                0.0 |                             0.1 |                 0.0 |                         0.1 |                         1.4 |                          0.0 |                0.0 |                1.5 |                1.5 |                1.5 |                5.2 |                1.6 |
|      16 |             1 |              6544.0 |                0.0 |                             0.1 |                 0.0 |                         0.1 |                         2.3 |                          0.0 |                0.0 |                2.3 |                2.3 |                2.4 |                6.5 |                2.4 |


#### Offline: NVIDIA DGX A100 (1x A100 80GB), PyTorch with FP16

Our results were obtained using the following configuration:

| Parameter Name               | Parameter Value                |
|:-----------------------------|:-------------------------------|
| GPU                          | NVIDIA DGX A100 (1x A100 80GB) |
| Backend                      | PyTorch                        |
| Backend accelerator          | -                              |
| Precision                    | FP16                           |
| Model format                 | TorchScript Trace              |
| Max batch size               | 16                             |
| Number of model instances    | 1                              |
| Accelerator Precision | -                              |
| Max Seq Length | 384                            |
| SQuAD v1.1 F1 Score       | 83.35                          |

<summary>Results Table</summary>

|   Batch |   Concurrency |   Inferences/Second |   Client Send (ms) |   Network+Server Send/Recv (ms) |   Server Queue (ms) |   Server Compute Input (ms) |   Server Compute Infer (ms) |   Server Compute Output (ms) |   Client Recv (ms) |   p50 latency (ms) |   p90 latency (ms) |   p95 latency (ms) |   p99 latency (ms) |   avg latency (ms) |
|--------:|--------------:|--------------------:|-------------------:|--------------------------------:|--------------------:|----------------------------:|----------------------------:|-----------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
|       1 |             1 |               482.0 |                0.0 |                             0.1 |                 0.0 |                         0.0 |                         1.9 |                          0.0 |                0.0 |                2.1 |                2.1 |                2.1 |                2.1 |                2.1 |
|       8 |             1 |              3368.0 |                0.0 |                             0.1 |                 0.0 |                         0.0 |                         1.7 |                          0.5 |                0.0 |                2.4 |                2.4 |                2.4 |                2.5 |                2.4 |
|      16 |             1 |              4272.0 |                0.0 |                             0.1 |                 0.0 |                         0.1 |                         1.7 |                          1.8 |                0.0 |                3.7 |                3.8 |                3.8 |                3.8 |                3.7 |


#### Offline: NVIDIA T4, ONNX Runtime with FP16

Our results were obtained using the following configuration:

| Parameter Name               | Parameter Value |
|:-----------------------------|:----------------|
| GPU                          | NVIDIA T4       |
| Backend                      | ONNX Runtime    |
| Backend accelerator          | -               |
| Precision                    | FP16            |
| Model format                 | ONNX            |
| Max batch size               | 16              |
| Number of model instances    | 1               |
| Accelerator Precision | -               |
| Max Seq Length | 384             |
| SQuAD v1.1 F1 Score       | 83.37           |

<summary>Results Table</summary>

|   Batch |   Concurrency |   Inferences/Second |   Client Send (ms) |   Network+Server Send/Recv (ms) |   Server Queue (ms) |   Server Compute Input (ms) |   Server Compute Infer (ms) |   Server Compute Output (ms) |   Client Recv (ms) |   p50 latency (ms) |   p90 latency (ms) |   p95 latency (ms) |   p99 latency (ms) |   avg latency (ms) |
|--------:|--------------:|--------------------:|-------------------:|--------------------------------:|--------------------:|----------------------------:|----------------------------:|-----------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
|       1 |             1 |               488.5 |                0.0 |                             0.5 |                 0.1 |                         0.1 |                         1.4 |                          0.0 |                0.0 |                2.0 |                2.1 |                2.2 |                2.3 |                2.0 |
|       8 |             1 |               792.0 |                0.0 |                             0.5 |                 0.1 |                         0.1 |                         9.4 |                          0.0 |                0.0 |               10.1 |               10.2 |               10.2 |               10.2 |               10.1 |
|      16 |             1 |               816.0 |                0.0 |                             0.5 |                 0.1 |                         0.1 |                        18.8 |                          0.0 |                0.0 |               19.5 |               19.6 |               19.7 |               19.7 |               19.5 |


#### Offline: NVIDIA T4, ONNX Runtime with FP16, Backend accelerator TensorRT

Our results were obtained using the following configuration:

| Parameter Name               | Parameter Value              |
|:-----------------------------|:-----------------------------|
| GPU                          |NVIDIA T4            |
| Backend                      |ONNX Runtime        |
| Backend accelerator          |NVIDIA TensorRT|
| Precision                    |FP16      |
| Model format                 |ONNX   |
| Max batch size               |16 |
| Number of model instances    |1|
| Accelerator Precision | FP16                 |
| Max Seq Length | 384                 |
| SQuAD v1.1 F1 Score       | 83.34                    |

<summary>Results Table</summary>

|   Batch |   Concurrency |   Inferences/Second |   Client Send (ms) |   Network+Server Send/Recv (ms) |   Server Queue (ms) |   Server Compute Input (ms) |   Server Compute Infer (ms) |   Server Compute Output (ms) |   Client Recv (ms) |   p50 latency (ms) |   p90 latency (ms) |   p95 latency (ms) |   p99 latency (ms) |   avg latency (ms) |
|--------:|--------------:|--------------------:|-------------------:|--------------------------------:|--------------------:|----------------------------:|----------------------------:|-----------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
|       1 |             1 |               749.0 |                0.0 |                             0.4 |                 0.0 |                         0.1 |                         0.8 |                          0.0 |                0.0 |                1.3 |                1.4 |                1.5 |                1.5 |                1.3 |
|       8 |             1 |              1536.0 |                0.0 |                             0.6 |                 0.1 |                         0.1 |                         4.5 |                          0.0 |                0.0 |                5.2 |                5.3 |                5.4 |                5.4 |                5.2 |
|      16 |             1 |              1648.0 |                0.0 |                             0.5 |                 0.1 |                         0.1 |                         9.0 |                          0.0 |                0.0 |                9.6 |                9.9 |                9.9 |               10.0 |                9.7 |


#### Offline: NVIDIA T4, NVIDIA TensorRT with FP16

Our results were obtained using the following configuration:

| Parameter Name               | Parameter Value |
|:-----------------------------|:----------------|
| GPU                          | NVIDIA T4       |
| Backend                      | NVIDIA TensorRT |
| Backend accelerator          | -               |
| Precision                    | FP16            |
| Model format                 | NVIDIA TensorRT |
| Max batch size               | 16              |
| Number of model instances    | 1               |
| NVIDIA TensorRT Capture CUDA Graph | Disabled        |
| Accelerator Precision | -               |
| Max Seq Length | 384             |
| SQuAD v1.1 F1 Score       | 83.34           |

<summary>Results Table</summary>

|   Batch |   Concurrency |   Inferences/Second |   Client Send (ms) |   Network+Server Send/Recv (ms) |   Server Queue (ms) |   Server Compute Input (ms) |   Server Compute Infer (ms) |   Server Compute Output (ms) |   Client Recv (ms) |   p50 latency (ms) |   p90 latency (ms) |   p95 latency (ms) |   p99 latency (ms) |   avg latency (ms) |
|--------:|--------------:|--------------------:|-------------------:|--------------------------------:|--------------------:|----------------------------:|----------------------------:|-----------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
|       1 |             1 |               798.2 |                0.0 |                             0.4 |                 0.0 |                         0.1 |                         0.7 |                          0.0 |                0.0 |                1.2 |                1.4 |                1.5 |                1.5 |                1.2 |
|       8 |             1 |              1544.0 |                0.0 |                             0.5 |                 0.0 |                         0.1 |                         4.5 |                          0.0 |                0.0 |                5.1 |                5.2 |                5.2 |                5.3 |                5.1 |
|      16 |             1 |              1632.0 |                0.0 |                             0.5 |                 0.0 |                         0.1 |                         9.0 |                          0.0 |                0.0 |                9.7 |               10.0 |               10.0 |               10.1 |                9.7 |


#### Offline: NVIDIA T4, PyTorch with FP16

Our results were obtained using the following configuration:

| Parameter Name               | Parameter Value   |
|:-----------------------------|:------------------|
| GPU                          | NVIDIA T4         |
| Backend                      | PyTorch           |
| Backend accelerator          | -                 |
| Precision                    | FP16              |
| Model format                 | TorchScript Trace |
| Max batch size               | 16                |
| Number of model instances    | 1                 |
| Accelerator Precision | -                 |
| Max Seq Length | 384               |
| SQuAD v1.1 F1 Score       | 83.35             |

<summary>Results Table</summary>

|   Batch |   Concurrency |   Inferences/Second |   Client Send (ms) |   Network+Server Send/Recv (ms) |   Server Queue (ms) |   Server Compute Input (ms) |   Server Compute Infer (ms) |   Server Compute Output (ms) |   Client Recv (ms) |   p50 latency (ms) |   p90 latency (ms) |   p95 latency (ms) |   p99 latency (ms) |   avg latency (ms) |
|--------:|--------------:|--------------------:|-------------------:|--------------------------------:|--------------------:|----------------------------:|----------------------------:|-----------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
|       1 |             1 |               447.0 |                0.0 |                             0.5 |                 0.1 |                         0.0 |                         1.6 |                          0.0 |                0.0 |                2.2 |                2.2 |                2.3 |                2.8 |                2.2 |
|       8 |             1 |               928.0 |                0.0 |                             0.5 |                 0.1 |                         0.1 |                         1.6 |                          6.3 |                0.0 |                8.6 |                8.7 |                8.7 |                8.8 |                8.6 |
|      16 |             1 |               944.0 |                0.0 |                             0.4 |                 0.1 |                         0.1 |                         1.5 |                         14.6 |                0.0 |               16.6 |               17.0 |               17.1 |               17.2 |               16.7 |


## Advanced
### Prepare configuration
You can use the environment variables to set the parameters of your inference
configuration.

Triton deployment scripts support several inference runtimes listed in the table below:


Example values of some key variables in one configuration:
```
FORMAT="onnx"
PRECISION="fp16"
EXPORT_FORMAT="onnx"
EXPORT_PRECISION="fp16"
ACCELERATOR="trt"
ACCELERATOR_PRECISION="fp16"
CAPTURE_CUDA_GRAPH="0"
BATCH_SIZE="16"
MAX_BATCH_SIZE="16"
MAX_SEQ_LENGTH="384"
CHECKPOINT_VARIANT="dist-4l-qa"
CHECKPOINT_DIR=${CHECKPOINTS_DIR}/${CHECKPOINT_VARIANT}
TRITON_MAX_QUEUE_DELAY="1"
TRITON_GPU_ENGINE_COUNT="1"
TRITON_PREFERRED_BATCH_SIZES="1"
```

| Inference runtime | Mnemonic used in scripts |
|-------------------|--------------------------|
| [TorchScript Tracing](https://pytorch.org/docs/stable/jit.html) | `ts-trace` |
| [TorchScript Scripting](https://pytorch.org/docs/stable/jit.html) | `ts-script` |
| [ONNX](https://onnx.ai) | `onnx` |
| [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) | `trt` |

The deployment process consists of the following steps. 

1. Export step. We export the model into the format set by `${EXPORT_FORMAT}`, with precision set by `${EXPORT_PRECISION}`. 

2. Convert step. We convert the exported model from `${EXPORT_FORMAT}` into `${FORMAT}`. The precision of the model in `${FORMAT}` is set by `${PRECISION}`. 

3. Deploy step. We create the triton model repository. 

The most common use-case scenario is to export the model into ONNX format, and then convert it into TensorRT. 
`${ACCELERATOR}` here refers to the accelerator of the ONNX format, which can be either `trt` or `none`. 

All the above values are set in the `triton/scripts/setup_parameters.sh` file. 

### Step by step deployment process
Commands described below can be used for exporting, converting and profiling the model.

#### Clone Repository
IMPORTANT: This step is executed on the host computer.
<details>
<summary>Clone Repository Command</summary>

```shell
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/PyTorch/LanguageModeling/BERT/
```
</details>

#### Setup Environment
Setup the environment in the host computer and start Triton Inference Server.
<details>
<summary>Setup Environment Command</summary>

```shell
source ./triton/dist4l/scripts/setup_environment.sh
./triton/dist4l/scripts/docker/triton_inference_server.sh
```
</details>

#### Setup Container
Build and run a container that extends the NGC PyTorch container with the Triton Inference Server client libraries and dependencies.
<details>
<summary>Setup Container Command</summary>

```shell
./triton/dist4l/scripts/docker/build.sh
./triton/dist4l/scripts/docker/interactive.sh
```
</details>

#### Setup Parameters and Environment
Setup the environment and deployment parameters inside interactive container.

<details>
<summary>Setup Environment Command</summary>

```shell
source ./triton/dist4l/scripts/setup_environment.sh
```
</details>

<details>
<summary>Setup Parameters Command</summary>

```shell
source ./triton/dist4l/scripts/setup_parameters.sh
```
</details>

#### Prepare Dataset and Checkpoint
Prepare datasets and checkpoint if not run automatic evaluation scripts.

<details>
<summary>Prepare Datasets Command</summary>

```shell
./triton/dist4l/runner/prepare_datasets.sh
```
</details>

<details>
<summary>Prepare Checkpoint Command</summary>

Download checkpoint from
```
https://catalog.ngc.nvidia.com/orgs/nvidia/dle/models/bert_pyt_ckpt_distilled_4l_288d_qa_squad11_amp 
```

Create the directory for checkpoint and copy the downloaded checkpoint content:

```shell
mkdir -p ${CHECKPOINTS_DIR}/dist-4l-qa
```
</details>


#### Export Model
Export model from Python source to desired format (e.g. Savedmodel or TorchScript)
<details>
<summary>Export Model Command</summary>

```shell
python3 triton/export_model.py \
    --input-path triton/model.py \
    --input-type pyt \
    --output-path ${SHARED_DIR}/exported_model.${FORMAT_SUFFIX} \
    --output-type ${EXPORT_FORMAT} \
    --dataloader triton/dataloader.py \
    --ignore-unknown-parameters \
    --onnx-opset 13 \
    ${FLAG} \
    \
    --config-file ${CHECKPOINT_DIR}/config.json \
    --checkpoint ${CHECKPOINT_DIR}/pytorch_model.bin \
    --precision ${EXPORT_PRECISION} \
    \
    --vocab-file ${CHECKPOINT_DIR}/vocab.txt \
    --max-seq-length ${MAX_SEQ_LENGTH} \
    --predict-file ${DATASETS_DIR}/data/squad/v1.1/dev-v1.1.json \
    --batch-size ${MAX_BATCH_SIZE}
```
</details>

#### Convert Model
Convert the model from training to inference format (e.g. TensorRT).
<details>
<summary>Convert Model Command</summary>

```shell
model-navigator convert \
        --model-name ${MODEL_NAME} \
        --model-path ${SHARED_DIR}/exported_model.${FORMAT_SUFFIX} \
        --output-path ${SHARED_DIR}/converted_model \
        --target-formats ${FORMAT} \
        --target-precisions ${PRECISION} \
        --launch-mode local \
        --override-workspace \
        --verbose \
        \
        --onnx-opsets 13 \
        --inputs input__0:${MAX_BATCH_SIZE},${MAX_SEQ_LENGTH}:int32 \
        --inputs input__1:${MAX_BATCH_SIZE},${MAX_SEQ_LENGTH}:int32 \
        --inputs input__2:${MAX_BATCH_SIZE},${MAX_SEQ_LENGTH}:int32 \
        --min-shapes input__0=${MAX_BATCH_SIZE},${MAX_SEQ_LENGTH} \
                     input__1=${MAX_BATCH_SIZE},${MAX_SEQ_LENGTH} \
                     input__2=${MAX_BATCH_SIZE},${MAX_SEQ_LENGTH} \
        --max-shapes input__0=${MAX_BATCH_SIZE},${MAX_SEQ_LENGTH} \
                     input__1=${MAX_BATCH_SIZE},${MAX_SEQ_LENGTH} \
                     input__2=${MAX_BATCH_SIZE},${MAX_SEQ_LENGTH} \
        --opt-shapes input__0=${MAX_BATCH_SIZE},${MAX_SEQ_LENGTH} \
                     input__1=${MAX_BATCH_SIZE},${MAX_SEQ_LENGTH} \
                     input__2=${MAX_BATCH_SIZE},${MAX_SEQ_LENGTH} \
        --max-batch-size ${MAX_BATCH_SIZE} \
        --tensorrt-max-workspace-size 8589934592 \
        --atol 2 output__0=5.0 \
                 output__1=5.0 \
        --rtol 1 output__0=5.0 \
                 output__1=5.0
```
</details>


#### Deploy Model
Configure the model on Triton Inference Server.
Generate the configuration from your model repository.
<details>

<summary>Deploy Model Command</summary>

```shell
model-navigator triton-config-model \
    --model-repository ${MODEL_REPOSITORY_PATH} \
    --model-name ${MODEL_NAME} \
    --model-version 1 \
    --model-path ${SHARED_DIR}/converted_model \
    --model-format ${CONFIG_FORMAT} \
    --model-control-mode ${TRITON_LOAD_MODEL_METHOD} \
    --verbose \
    --load-model \
    --load-model-timeout-s 100 \
    \
    --backend-accelerator ${ACCELERATOR} \
    --tensorrt-precision ${ACCELERATOR_PRECISION}  \
    --max-batch-size ${MBS} \
    --preferred-batch-sizes ${TRITON_PREFERRED_BATCH_SIZES} \
    --max-queue-delay-us ${TRITON_MAX_QUEUE_DELAY} \
    --engine-count-per-device gpu=${TRITON_GPU_ENGINE_COUNT}
```
</details>


#### Prepare Triton Profiling Data
Prepare data used for profiling on Triton server.
<details>
<summary>Prepare Triton Profiling Data Command</summary>

```shell
mkdir -p ${SHARED_DIR}/input_data

python triton/prepare_input_data.py \
    --dataloader triton/dataloader.py \
    --input-data-dir ${SHARED_DIR}/input_data \
    \
    --batch-size ${MAX_BATCH_SIZE} \
    --max-seq-length ${MAX_SEQ_LENGTH} \
    --predict-file ${DATASETS_DIR}/data/squad/v1.1/dev-v1.1.json \
    --vocab-file ${CHECKPOINT_DIR}/vocab.txt
```
</details>



#### Triton Performance Offline Test
We want to maximize throughput. It assumes you have your data available
for inference or that your data saturate to maximum batch size quickly.
Triton Inference Server supports offline scenarios with static batching.
Static batching allows inference requests to be served
as they are received. The largest improvements to throughput come
from increasing the batch size due to efficiency gains in the GPU with larger
batches.
<details>
<summary>Triton Performance Offline Test Command</summary>

```shell
python triton/run_performance_on_triton.py \
    --model-repository ${MODEL_REPOSITORY_PATH} \
    --model-name ${MODEL_NAME} \
    --input-data ${SHARED_DIR}/input_data/data.json \
    --input-shapes input__0:${MAX_SEQ_LENGTH} \
    --input-shapes input__1:${MAX_SEQ_LENGTH} \
    --input-shapes input__2:${MAX_SEQ_LENGTH} \
    --batch-sizes ${BATCH_SIZE} \
    --number-of-triton-instances ${TRITON_INSTANCES} \
    --number-of-model-instances ${TRITON_GPU_ENGINE_COUNT} \
    --batching-mode static \
    --evaluation-mode offline \
    --performance-tool perf_analyzer \
    --result-path ${SHARED_DIR}/triton_performance_offline.csv
```

</details>


### Latency explanation
A typical Triton Inference Server pipeline can be broken down into the following steps:

1. The client serializes the inference request into a message and sends it to
the server (Client Send).
2. The message travels over the network from the client to the server (Network).
3. The message arrives at the server and is deserialized (Server Receive).
4. The request is placed on the queue (Server Queue).
5. The request is removed from the queue and computed (Server Compute).
6. The completed request is serialized in a message and sent back to
the client (Server Send).
7. The completed message then travels over the network from the server
to the client (Network).
8. The completed message is deserialized by the client and processed as
a completed inference request (Client Receive).

Generally, for local clients, steps 1-4 and 6-8 will only occupy
a small fraction of time, compared to step 5. As backend deep learning
systems like Jasper are rarely exposed directly to end users, but instead
only interfacing with local front-end servers, for the sake of Jasper,
we can consider that all clients are local.



## Release Notes
We’re constantly refining and improving our performance on AI
and HPC workloads even on the same hardware with frequent updates
to our software stack. For our latest performance data refer
to these pages for
[AI](https://developer.nvidia.com/deep-learning-performance-training-inference)
and [HPC](https://developer.nvidia.com/hpc-application-performance) benchmarks.

### Changelog
### Known issues

- There are no known issues with this model.

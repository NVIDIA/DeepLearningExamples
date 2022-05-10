# Deploying the GPUNet model on Triton Inference Server

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
        - [Offline: NVIDIA DGX-1 (1x V100 32GB), ONNX Runtime with FP16](#offline-nvidia-dgx-1-1x-v100-32gb-onnx-runtime-with-fp16)
        - [Offline: NVIDIA DGX A100 (1x A100 80GB), ONNX Runtime with FP16](#offline-nvidia-dgx-a100-1x-a100-80gb-onnx-runtime-with-fp16)
    - [Online scenario](#online-scenario)
        - [Online: NVIDIA DGX-1 (1x V100 32GB), ONNX Runtime with FP16](#online-nvidia-dgx-1-1x-v100-32gb-onnx-runtime-with-fp16)
        - [Online: NVIDIA DGX A100 (1x A100 80GB), ONNX Runtime with FP16](#online-nvidia-dgx-a100-1x-a100-80gb-onnx-runtime-with-fp16)
  - [Advanced](#advanced)
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

1. Correctness tests.

   Produce results which are tested against given correctness thresholds.

2. Performance tests.

   Produce latency and throughput results for offline (static batching)
   and online (dynamic batching) scenarios.


All steps are executed by provided runner script. Refer to [Quick Start Guide](#quick-start-guide)


## Setup
Ensure you have the following components:
* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [NVIDIA PyTorch NGC container 21.12](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
* [NVIDIA Triton Inference Server NGC container 21.12](https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver)
* [NVIDIA CUDA](https://docs.nvidia.com/cuda/archive//index.html)
* [NVIDIA Ampere](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/), [Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) based GPU



## Quick Start Guide
Running the following scripts will build and launch the container with all required dependencies for native PyTorch as well as Triton Inference Server. This is necessary for running inference and can also be used for data download, processing, and training of the model.

1. Clone the repository.

```
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd PyTorch/Classification/GPUNet
```

2. Prepare dataset.
See the [Quick Start Guide](../../README.md#prepare-the-dataset)

3. Build and run a container that extends NGC PyTorch with the Triton client libraries and necessary dependencies.

```
./triton/scripts/docker/build.sh
./triton/scripts/docker/interactive.sh /path/to/imagenet/val/
```

4. Execute runner script (please mind, the run scripts are prepared per NVIDIA GPU).

```
NVIDIA DGX-1 (1x V100 32GB): ./triton/05ms-D/runner/start_NVIDIA-DGX-1-\(1x-V100-32GB\).sh

NVIDIA DGX A100 (1x A100 80GB): ./triton/05ms-D/runner/start_NVIDIA-DGX-A100-\(1x-A100-80GB\).sh
```

## Performance
The performance measurements in this document were conducted at the time of publication and may not reflect
the performance achieved from NVIDIA’s latest software release. For the most up-to-date performance measurements, go to
[NVIDIA Data Center Deep Learning Product Performance](https://developer.nvidia.com/deep-learning-performance-training-inference).
### Offline scenario

The offline scenario assumes the client and server are located on the same host. The tests uses:
- tensors are passed through shared memory between client and server, the Perf Analyzer flag `shared-memory=system` is used
- single request is send from client to server with static size of batch


#### Offline: NVIDIA DGX-1 (1x V100 32GB), ONNX Runtime with FP16

Our results were obtained using the following configuration:

| Parameter Name               | Parameter Value              |
|:-----------------------------|:-----------------------------|
| GPU                          |NVIDIA DGX-1 (1x V100 32GB)            |
| Backend                      |ONNX Runtime        |
| Backend accelerator          |NVIDIA TensorRT|
| Precision                    |FP16      |
| Model format                 |ONNX   |
| Max batch size               |64 |
| Number of model instances    |2|
| Export Format | ONNX    |
| Device Kind | gpu                 |
| Torch Jit | none                 |


<table>
<tbody>
  <tr>
    <td><img src="./reports/nvidia_dgx-1_(1x_v100_32gb)_experiment_2_triton_performance_offline_2/plots/throughput_vs_batch.png"></td>
    <td><img src="./reports/nvidia_dgx-1_(1x_v100_32gb)_experiment_2_triton_performance_offline_2/plots/throughput_vs_latency.png"></td>
  </tr>
  <tr>
    <td><img src="./reports/nvidia_dgx-1_(1x_v100_32gb)_experiment_2_triton_performance_offline_2/plots/latency_vs_batch.png"></td>
  </tr>
</tbody>
</table>

<details>
<summary>Results Table</summary>

|   Batch |   Concurrency |   Inferences/Second |   Client Send (ms) |   Network+Server Send/Recv (ms) |   Server Queue (ms) |   Server Compute Input (ms) |   Server Compute Infer (ms) |   Server Compute Output (ms) |   Client Recv (ms) |   p50 latency (ms) |   p90 latency (ms) |   p95 latency (ms) |   p99 latency (ms) |   avg latency (ms) |
|--------:|--------------:|--------------------:|-------------------:|--------------------------------:|--------------------:|----------------------------:|----------------------------:|-----------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
|       1 |             1 |              957.00 |               0.05 |                            0.20 |                0.07 |                        0.09 |                        0.63 |                         0.01 |               0.00 |               1.04 |               1.08 |               1.09 |               1.12 |               1.04 |
|       2 |             1 |             1628.00 |               0.05 |                            0.21 |                0.07 |                        0.14 |                        0.75 |                         0.01 |               0.00 |               1.22 |               1.26 |               1.27 |               1.29 |               1.22 |
|       4 |             1 |             2508.00 |               0.04 |                            0.21 |                0.08 |                        0.23 |                        1.02 |                         0.01 |               0.00 |               1.59 |               1.62 |               1.62 |               1.68 |               1.59 |
|       8 |             1 |             3712.00 |               0.04 |                            0.19 |                0.07 |                        0.35 |                        1.49 |                         0.01 |               0.00 |               2.14 |               2.19 |               2.23 |               2.28 |               2.15 |
|      16 |             1 |             4912.00 |               0.04 |                            0.22 |                0.08 |                        0.57 |                        2.33 |                         0.01 |               0.00 |               3.25 |               3.28 |               3.29 |               3.31 |               3.25 |
|      32 |             1 |             5856.00 |               0.05 |                            0.23 |                0.08 |                        1.02 |                        4.03 |                         0.02 |               0.00 |               5.43 |               5.48 |               5.50 |               5.55 |               5.44 |
|      64 |             1 |             6656.00 |               0.05 |                            0.22 |                0.08 |                        1.91 |                        7.28 |                         0.03 |               0.00 |               9.58 |               9.63 |               9.63 |               9.64 |               9.57 |

</details>



#### Offline: NVIDIA DGX A100 (1x A100 80GB), ONNX Runtime with FP16

Our results were obtained using the following configuration:

| Parameter Name               | Parameter Value              |
|:-----------------------------|:-----------------------------|
| GPU                          |NVIDIA DGX A100 (1x A100 80GB)            |
| Backend                      |ONNX Runtime        |
| Backend accelerator          |NVIDIA TensorRT|
| Precision                    |FP16      |
| Model format                 |ONNX   |
| Max batch size               |64 |
| Number of model instances    |2|
| Export Format | ONNX    |
| Device Kind | gpu                 |
| Torch Jit | none                 |


<table>
<tbody>
  <tr>
    <td><img src="./reports/nvidia_dgx_a100_(1x_a100_80gb)_experiment_2_triton_performance_offline_2/plots/throughput_vs_batch.png"></td>
    <td><img src="./reports/nvidia_dgx_a100_(1x_a100_80gb)_experiment_2_triton_performance_offline_2/plots/throughput_vs_latency.png"></td>
  </tr>
  <tr>
    <td><img src="./reports/nvidia_dgx_a100_(1x_a100_80gb)_experiment_2_triton_performance_offline_2/plots/latency_vs_batch.png"></td>
  </tr>
</tbody>
</table>

<details>
<summary>Results Table</summary>

|   Batch |   Concurrency |   Inferences/Second |   Client Send (ms) |   Network+Server Send/Recv (ms) |   Server Queue (ms) |   Server Compute Input (ms) |   Server Compute Infer (ms) |   Server Compute Output (ms) |   Client Recv (ms) |   p50 latency (ms) |   p90 latency (ms) |   p95 latency (ms) |   p99 latency (ms) |   avg latency (ms) |
|--------:|--------------:|--------------------:|-------------------:|--------------------------------:|--------------------:|----------------------------:|----------------------------:|-----------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
|       1 |             1 |             1559.00 |               0.02 |                            0.08 |                0.02 |                        0.06 |                        0.45 |                         0.00 |               0.00 |               0.64 |               0.65 |               0.65 |               0.66 |               0.64 |
|       2 |             1 |             2796.00 |               0.02 |                            0.07 |                0.02 |                        0.10 |                        0.50 |                         0.00 |               0.00 |               0.71 |               0.72 |               0.72 |               0.73 |               0.71 |
|       4 |             1 |             4640.00 |               0.02 |                            0.07 |                0.02 |                        0.15 |                        0.60 |                         0.00 |               0.00 |               0.86 |               0.89 |               0.89 |               1.03 |               0.86 |
|       8 |             1 |             6984.00 |               0.02 |                            0.07 |                0.02 |                        0.21 |                        0.82 |                         0.00 |               0.00 |               1.14 |               1.15 |               1.18 |               1.33 |               1.14 |
|      16 |             1 |             9136.00 |               0.02 |                            0.08 |                0.03 |                        0.36 |                        1.26 |                         0.01 |               0.00 |               1.75 |               1.76 |               1.77 |               1.78 |               1.75 |
|      32 |             1 |             9664.00 |               0.02 |                            0.10 |                0.03 |                        0.93 |                        2.21 |                         0.01 |               0.00 |               3.30 |               3.33 |               3.37 |               3.48 |               3.30 |
|      64 |             1 |             9728.00 |               0.03 |                            0.18 |                0.03 |                        2.15 |                        4.12 |                         0.02 |               0.00 |               6.50 |               6.62 |               6.96 |               7.11 |               6.54 |

</details>




### Online scenario

The online scenario assumes the client and server are located on different hosts. The tests uses:
- tensors are passed through HTTP from client to server
- concurrent requests are send from client to server, the final batch is created on server side


#### Online: NVIDIA DGX-1 (1x V100 32GB), ONNX Runtime with FP16

Our results were obtained using the following configuration:

| Parameter Name               | Parameter Value              |
|:-----------------------------|:-----------------------------|
| GPU                          |NVIDIA DGX-1 (1x V100 32GB)            |
| Backend                      |ONNX Runtime        |
| Backend accelerator          |NVIDIA TensorRT|
| Precision                    |FP16      |
| Model format                 |ONNX   |
| Max batch size               |64 |
| Number of model instances    |2|
| Export Format | ONNX    |
| Device Kind | gpu                 |
| Torch Jit | none                 |


<table>
<tbody>
  <tr>
    <td colspan="2" align="center"><img src="./reports/nvidia_dgx-1_(1x_v100_32gb)_experiment_2_triton_performance_online_2/plots/latency_vs_concurrency.png"></td>
  </tr>
</tbody>
</table>

<details>
<summary>Results Table</summary>

|   Batch |   Concurrency |   Inferences/Second |   Client Send (ms) |   Network+Server Send/Recv (ms) |   Server Queue (ms) |   Server Compute Input (ms) |   Server Compute Infer (ms) |   Server Compute Output (ms) |   Client Recv (ms) |   p50 latency (ms) |   p90 latency (ms) |   p95 latency (ms) |   p99 latency (ms) |   avg latency (ms) |
|--------:|--------------:|--------------------:|-------------------:|--------------------------------:|--------------------:|----------------------------:|----------------------------:|-----------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
|       1 |             8 |              984.00 |               0.07 |                            0.55 |                5.14 |                        0.11 |                        2.21 |                         0.01 |               0.00 |               8.29 |               9.23 |               9.29 |               9.42 |               8.10 |
|       1 |            16 |             1553.00 |               0.08 |                            1.23 |                5.87 |                        0.29 |                        2.76 |                         0.01 |               0.00 |              10.84 |              11.38 |              11.96 |              12.93 |              10.24 |
|       1 |            24 |             2024.00 |               0.08 |                            1.96 |                6.28 |                        0.53 |                        2.91 |                         0.02 |               0.00 |              11.65 |              14.92 |              15.66 |              16.24 |              11.80 |
|       1 |            32 |             2559.00 |               0.09 |                            1.80 |                6.81 |                        0.60 |                        3.10 |                         0.02 |               0.00 |              12.55 |              13.47 |              13.75 |              15.05 |              12.41 |
|       1 |            40 |             2714.29 |               0.09 |                            2.90 |                7.18 |                        0.88 |                        3.58 |                         0.03 |               0.00 |              14.48 |              17.66 |              18.68 |              19.73 |              14.65 |
|       1 |            48 |             2841.00 |               0.10 |                            3.81 |                7.64 |                        1.27 |                        3.86 |                         0.03 |               0.00 |              16.43 |              21.62 |              22.97 |              23.61 |              16.72 |
|       1 |            56 |             3109.00 |               0.11 |                            4.22 |                8.16 |                        1.45 |                        3.90 |                         0.04 |               0.00 |              18.17 |              22.14 |              23.56 |              25.39 |              17.87 |
|       1 |            64 |             3243.00 |               0.10 |                            4.71 |                8.91 |                        1.62 |                        4.18 |                         0.04 |               0.00 |              20.09 |              23.77 |              25.26 |              27.47 |              19.56 |
|       1 |            72 |             3576.00 |               0.11 |                            3.76 |                9.98 |                        1.74 |                        4.33 |                         0.04 |               0.00 |              19.97 |              23.78 |              24.74 |              26.48 |              19.95 |
|       1 |            80 |             3687.00 |               0.11 |                            4.50 |               10.20 |                        2.03 |                        4.67 |                         0.05 |               0.00 |              21.69 |              27.08 |              28.44 |              29.97 |              21.55 |
|       1 |            88 |             3670.00 |               0.12 |                            6.86 |                9.65 |                        2.27 |                        4.78 |                         0.05 |               0.00 |              24.56 |              29.08 |              29.97 |              32.79 |              23.72 |
|       1 |            96 |             3811.00 |               0.12 |                            6.57 |               10.81 |                        2.40 |                        4.98 |                         0.05 |               0.00 |              25.55 |              30.15 |              31.15 |              33.31 |              24.93 |
|       1 |           104 |             3999.00 |               0.13 |                            6.21 |               11.44 |                        2.77 |                        5.20 |                         0.06 |               0.00 |              26.24 |              31.58 |              32.66 |              36.68 |              25.80 |
|       1 |           112 |             4207.00 |               0.14 |                            6.20 |               11.88 |                        2.78 |                        5.22 |                         0.06 |               0.00 |              26.54 |              32.07 |              33.48 |              35.21 |              26.27 |
|       1 |           120 |             4105.00 |               0.15 |                            7.46 |               12.06 |                        3.35 |                        5.81 |                         0.07 |               0.00 |              29.28 |              37.15 |              39.06 |              40.72 |              28.90 |
|       1 |           128 |             4316.00 |               0.16 |                            6.62 |               13.26 |                        3.23 |                        5.83 |                         0.08 |               0.00 |              29.48 |              35.96 |              37.67 |              40.26 |              29.17 |
|       1 |           136 |             4406.00 |               0.17 |                            5.64 |               14.81 |                        3.43 |                        6.14 |                         0.07 |               0.00 |              30.14 |              38.87 |              40.51 |              42.31 |              30.27 |
|       1 |           144 |             4339.00 |               0.16 |                            8.84 |               13.59 |                        3.71 |                        6.15 |                         0.08 |               0.00 |              33.02 |              40.36 |              43.51 |              46.66 |              32.53 |
|       1 |           152 |             4478.00 |               0.19 |                            7.40 |               15.32 |                        3.97 |                        6.44 |                         0.09 |               0.00 |              33.97 |              41.65 |              43.14 |              47.27 |              33.42 |
|       1 |           160 |             4520.00 |               0.18 |                            8.69 |               14.84 |                        4.11 |                        6.78 |                         0.10 |               0.00 |              34.65 |              43.75 |              46.05 |              48.88 |              34.69 |
|       1 |           168 |             4487.00 |               0.18 |                            8.69 |               15.98 |                        4.68 |                        6.99 |                         0.10 |               0.00 |              37.31 |              47.19 |              49.26 |              53.46 |              36.62 |
|       1 |           176 |             4608.39 |               0.18 |                            9.66 |               16.28 |                        4.42 |                        6.82 |                         0.10 |               0.00 |              38.30 |              46.18 |              48.55 |              52.57 |              37.47 |
|       1 |           184 |             4646.00 |               0.22 |                            8.82 |               17.11 |                        4.96 |                        7.28 |                         0.11 |               0.00 |              39.26 |              48.00 |              49.24 |              51.92 |              38.51 |
|       1 |           192 |             4646.00 |               0.21 |                            9.83 |               17.98 |                        4.81 |                        7.38 |                         0.12 |               0.00 |              40.34 |              51.41 |              53.30 |              57.10 |              40.33 |
|       1 |           200 |             4809.00 |               0.26 |                            8.54 |               19.52 |                        4.86 |                        7.26 |                         0.11 |               0.00 |              40.81 |              50.18 |              51.57 |              56.27 |              40.54 |
|       1 |           208 |             4866.00 |               0.33 |                            8.25 |               20.32 |                        5.10 |                        7.85 |                         0.12 |               0.00 |              42.63 |              51.31 |              52.64 |              54.30 |              41.96 |
|       1 |           216 |             4912.00 |               0.40 |                            7.34 |               22.29 |                        5.12 |                        7.78 |                         0.12 |               0.00 |              42.34 |              53.43 |              55.42 |              58.20 |              43.04 |
|       1 |           224 |             4927.00 |               0.30 |                            9.04 |               21.42 |                        5.29 |                        7.87 |                         0.12 |               0.00 |              43.46 |              55.32 |              57.61 |              61.31 |              44.04 |
|       1 |           232 |             4840.00 |               0.26 |                           12.65 |               20.39 |                        5.44 |                        7.89 |                         0.12 |               0.00 |              47.21 |              58.24 |              62.56 |              76.19 |              46.76 |
|       1 |           240 |             5044.00 |               0.35 |                           10.44 |               22.00 |                        5.46 |                        7.97 |                         0.12 |               0.00 |              46.40 |              55.91 |              58.63 |              62.81 |              46.35 |
|       1 |           248 |             4955.00 |               0.32 |                           12.14 |               22.27 |                        5.39 |                        8.04 |                         0.13 |               0.00 |              47.10 |              62.52 |              65.31 |              69.14 |              48.29 |
|       1 |           256 |             5236.00 |               0.52 |                            7.19 |               26.54 |                        5.02 |                        8.37 |                         0.14 |               0.00 |              48.18 |              55.77 |              57.99 |              63.11 |              47.79 |

</details>




#### Online: NVIDIA DGX A100 (1x A100 80GB), ONNX Runtime with FP16

Our results were obtained using the following configuration:

| Parameter Name               | Parameter Value              |
|:-----------------------------|:-----------------------------|
| GPU                          |NVIDIA DGX A100 (1x A100 80GB)            |
| Backend                      |ONNX Runtime        |
| Backend accelerator          |NVIDIA TensorRT|
| Precision                    |FP16      |
| Model format                 |ONNX   |
| Max batch size               |64 |
| Number of model instances    |2|
| Export Format | ONNX    |
| Device Kind | gpu                 |
| Torch Jit | none                 |


<table>
<tbody>
  <tr>
    <td colspan="2" align="center"><img src="./reports/nvidia_dgx_a100_(1x_a100_80gb)_experiment_2_triton_performance_online_2/plots/latency_vs_concurrency.png"></td>
  </tr>
</tbody>
</table>

<details>
<summary>Results Table</summary>

|   Batch |   Concurrency |   Inferences/Second |   Client Send (ms) |   Network+Server Send/Recv (ms) |   Server Queue (ms) |   Server Compute Input (ms) |   Server Compute Infer (ms) |   Server Compute Output (ms) |   Client Recv (ms) |   p50 latency (ms) |   p90 latency (ms) |   p95 latency (ms) |   p99 latency (ms) |   avg latency (ms) |
|--------:|--------------:|--------------------:|-------------------:|--------------------------------:|--------------------:|----------------------------:|----------------------------:|-----------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
|       1 |             8 |             1646.00 |               0.04 |                            0.37 |                3.01 |                        0.13 |                        1.29 |                         0.01 |               0.00 |               4.78 |               5.51 |               5.60 |               5.77 |               4.85 |
|       1 |            16 |             2536.00 |               0.07 |                            0.89 |                3.46 |                        0.29 |                        1.56 |                         0.01 |               0.00 |               6.38 |               7.26 |               7.64 |               8.40 |               6.29 |
|       1 |            24 |             3223.00 |               0.06 |                            1.29 |                3.85 |                        0.50 |                        1.70 |                         0.02 |               0.00 |               7.43 |               9.07 |               9.41 |              10.36 |               7.42 |
|       1 |            32 |             3705.00 |               0.06 |                            1.88 |                4.12 |                        0.70 |                        1.82 |                         0.02 |               0.00 |               8.60 |              10.84 |              11.17 |              11.94 |               8.61 |
|       1 |            40 |             4120.00 |               0.06 |                            2.18 |                4.64 |                        0.83 |                        1.92 |                         0.03 |               0.00 |               9.84 |              11.98 |              12.41 |              13.11 |               9.66 |
|       1 |            48 |             4495.00 |               0.06 |                            2.78 |                4.79 |                        0.98 |                        1.99 |                         0.03 |               0.00 |              10.92 |              12.96 |              13.42 |              14.64 |              10.64 |
|       1 |            56 |             4858.14 |               0.07 |                            2.80 |                5.19 |                        1.20 |                        2.18 |                         0.04 |               0.00 |              11.51 |              14.78 |              15.48 |              16.59 |              11.49 |
|       1 |            64 |             5222.78 |               0.07 |                            3.20 |                5.40 |                        1.24 |                        2.26 |                         0.04 |               0.00 |              12.51 |              15.17 |              16.02 |              17.54 |              12.21 |
|       1 |            72 |             5323.00 |               0.07 |                            3.92 |                5.55 |                        1.43 |                        2.42 |                         0.05 |               0.00 |              13.82 |              17.24 |              17.90 |              19.73 |              13.44 |
|       1 |            80 |             5826.00 |               0.06 |                            3.55 |                6.06 |                        1.41 |                        2.52 |                         0.06 |               0.00 |              13.85 |              17.40 |              18.55 |              19.70 |              13.66 |
|       1 |            88 |             5747.25 |               0.06 |                            4.85 |                5.96 |                        1.62 |                        2.61 |                         0.06 |               0.00 |              15.63 |              19.59 |              20.38 |              21.87 |              15.17 |
|       1 |            96 |             5883.00 |               0.08 |                            4.42 |                6.99 |                        1.96 |                        2.68 |                         0.07 |               0.00 |              16.41 |              20.70 |              21.62 |              25.46 |              16.20 |
|       1 |           104 |             6167.00 |               0.07 |                            4.41 |                7.05 |                        2.24 |                        2.91 |                         0.08 |               0.00 |              16.78 |              21.72 |              22.90 |              24.28 |              16.76 |
|       1 |           112 |             6117.00 |               0.07 |                            4.89 |                7.27 |                        2.52 |                        3.22 |                         0.09 |               0.00 |              18.58 |              22.70 |              23.52 |              25.27 |              18.07 |
|       1 |           120 |             6635.00 |               0.08 |                            4.06 |                8.29 |                        2.36 |                        3.07 |                         0.08 |               0.00 |              18.16 |              22.76 |              24.16 |              26.61 |              17.94 |
|       1 |           128 |             6457.00 |               0.08 |                            5.64 |                7.93 |                        2.63 |                        3.24 |                         0.10 |               0.00 |              19.73 |              26.09 |              26.80 |              27.30 |              19.62 |
|       1 |           136 |             6808.19 |               0.08 |                            4.58 |                9.03 |                        2.72 |                        3.33 |                         0.10 |               0.00 |              20.04 |              25.08 |              26.65 |              28.96 |              19.84 |
|       1 |           144 |             6703.00 |               0.07 |                            6.09 |                8.24 |                        3.12 |                        3.60 |                         0.12 |               0.00 |              21.88 |              26.14 |              27.44 |              28.78 |              21.24 |
|       1 |           152 |             7450.00 |               0.09 |                            3.81 |               10.14 |                        2.45 |                        3.56 |                         0.12 |               0.00 |              20.27 |              25.02 |              26.31 |              28.84 |              20.17 |
|       1 |           160 |             7214.78 |               0.08 |                            5.87 |                9.28 |                        2.75 |                        3.80 |                         0.12 |               0.00 |              21.97 |              27.62 |              29.16 |              30.83 |              21.89 |
|       1 |           168 |             7368.00 |               0.08 |                            6.10 |                9.50 |                        2.79 |                        3.85 |                         0.13 |               0.00 |              22.92 |              27.76 |              29.00 |              30.60 |              22.45 |
|       1 |           176 |             7483.00 |               0.08 |                            5.84 |               10.45 |                        2.96 |                        3.74 |                         0.13 |               0.00 |              23.57 |              28.50 |              30.22 |              33.26 |              23.19 |
|       1 |           184 |             7559.00 |               0.08 |                            6.50 |               10.21 |                        3.18 |                        4.00 |                         0.13 |               0.00 |              24.17 |              29.87 |              30.93 |              33.18 |              24.10 |
|       1 |           192 |             7587.00 |               0.08 |                            6.60 |               10.78 |                        3.27 |                        4.01 |                         0.14 |               0.00 |              25.20 |              30.48 |              31.67 |              34.83 |              24.88 |
|       1 |           200 |             7490.00 |               0.08 |                            7.83 |               10.70 |                        3.39 |                        4.11 |                         0.14 |               0.00 |              26.94 |              31.98 |              33.71 |              35.97 |              26.24 |
|       1 |           208 |             7731.00 |               0.09 |                            6.91 |               11.96 |                        3.45 |                        4.03 |                         0.14 |               0.00 |              26.95 |              32.35 |              33.63 |              36.61 |              26.57 |
|       1 |           216 |             7735.00 |               0.09 |                            7.30 |               11.76 |                        3.62 |                        4.57 |                         0.16 |               0.00 |              27.36 |              34.09 |              35.66 |              37.99 |              27.51 |
|       1 |           224 |             8244.00 |               0.09 |                            6.21 |               12.52 |                        3.24 |                        4.47 |                         0.15 |               0.00 |              26.44 |              32.87 |              34.35 |              37.15 |              26.69 |
|       1 |           232 |             8148.00 |               0.12 |                            6.22 |               13.63 |                        3.41 |                        4.48 |                         0.16 |               0.00 |              28.24 |              34.21 |              35.99 |              39.36 |              28.03 |
|       1 |           240 |             7768.23 |               0.09 |                           10.38 |               12.38 |                        3.26 |                        4.12 |                         0.14 |               0.00 |              29.42 |              40.59 |              42.10 |              44.21 |              30.37 |
|       1 |           248 |             8296.00 |               0.12 |                            6.08 |               14.71 |                        3.78 |                        4.54 |                         0.16 |               0.00 |              29.78 |              34.53 |              35.91 |              37.65 |              29.40 |
|       1 |           256 |             8153.00 |               0.09 |                            7.73 |               14.47 |                        3.82 |                        4.72 |                         0.16 |               0.00 |              31.49 |              37.20 |              38.42 |              41.19 |              30.99 |

</details>




## Advanced

| Inference runtime | Mnemonic used in scripts |
|-------------------|--------------------------|
| [TorchScript Tracing](https://pytorch.org/docs/stable/jit.html) | `ts-trace` |
| [TorchScript Scripting](https://pytorch.org/docs/stable/jit.html) | `ts-script` |
| [ONNX](https://onnx.ai) | `onnx` |
| [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) | `trt` |

### Step by step deployment process
Commands described below can be used for exporting, converting and profiling the model.

#### Clone Repository
IMPORTANT: This step is executed on the host computer.
<details>
<summary>Clone Repository Command</summary>

```shell
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd PyTorch/Classification/GPUNet
```
</details>

#### Start Triton Inference Server
Setup the environment in the host computer and start Triton Inference Server.
<details>
<summary>Setup Environment and Start Triton Inference Server Command</summary>

```shell
source ./triton/scripts/setup_environment.sh
./triton/scripts/docker/triton_inference_server.sh
```
</details>

#### Prepare Dataset.
Please use the data download from the [Main QSG](../../README.md#prepare-the-dataset)

#### Prepare Checkpoint
Please download a checkpoint from [here](https://api.ngc.nvidia.com/v2/models/nvidia/dle/gpunet_p0_pyt_ckpt/versions/21.12.0_amp/zip) 
and place it in `runner_workspace/checkpoints/0.5ms-D/`.  Note that the `0.5ms-D` subdirectory may not be created yet.

#### Setup Container
Build and run a container that extends the NGC PyTorch container with the Triton Inference Server client libraries and dependencies.
<details>
<summary>Setup Container Command</summary>

Build container:

```shell
./triton/scripts/docker/build.sh
```

Run container in interactive mode:

```shell
./triton/scripts/docker/interactive.sh /path/to/imagenet/val/
```

Setup environment in order to share artifacts in steps and with Triton Inference Server:

```shell
source ./triton/scripts/setup_environment.sh
```

</details>

#### Prepare configuration
You can use the environment variables to set the parameters of your inference configuration.

Example values of some key variables in one configuration:
<details>
<summary>Export Variables</summary>

```shell
export FORMAT="onnx"
export PRECISION="fp16"
export EXPORT_FORMAT="onnx"
export EXPORT_PRECISION="fp16"
export BACKEND_ACCELERATOR="trt"
export NUMBER_OF_MODEL_INSTANCES="2"
export TENSORRT_CAPTURE_CUDA_GRAPH="0"
export CHECKPOINT="0.5ms-D"
export CHECKPOINT_DIR=${CHECKPOINTS_DIR}/${CHECKPOINT}
```

</details>


#### Export Model
Export model from Python source to desired format (e.g. Savedmodel or TorchScript)
<details>
<summary>Export Model Command</summary>

```shell
if [[ "${EXPORT_FORMAT}" == "torchscript" ]]; then
    export FORMAT_SUFFIX="pt"
else
    export FORMAT_SUFFIX="${EXPORT_FORMAT}"
fi
python3 triton/export_model.py \
    --input-path triton/model.py \
    --input-type pyt \
    --output-path ${SHARED_DIR}/exported_model.${FORMAT_SUFFIX} \
    --output-type ${EXPORT_FORMAT} \
    --ignore-unknown-parameters \
    --onnx-opset 13 \
    --torch-jit none \
    \
    --config /workspace/gpunet/configs/batch1/GV100/0.5ms-D.json \
    --checkpoint ${CHECKPOINT_DIR}/0.5ms-D.pth.tar \
    --precision ${EXPORT_PRECISION} \
    \
    --dataloader triton/dataloader.py \
    --val-path ${DATASETS_DIR}/ \
    --is-prunet True \
    --batch-size 1
```

</details>



#### Convert Model
Convert the model from training to inference format (e.g. TensorRT).
<details>
<summary>Convert Model Command</summary>

```shell
if [[ "${EXPORT_FORMAT}" == "torchscript" ]]; then
    export FORMAT_SUFFIX="pt"
else
    export FORMAT_SUFFIX="${EXPORT_FORMAT}"
fi
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
    --max-batch-size 64 \
    --container-version 21.12 \
    --max-workspace-size 10000000000 \
    --atol OUTPUT__0=100 \
    --rtol OUTPUT__0=100
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
    --model-format ${FORMAT} \
    --model-control-mode explicit \
    --load-model \
    --load-model-timeout-s 100 \
    --verbose \
    \
    --backend-accelerator ${BACKEND_ACCELERATOR} \
    --tensorrt-precision ${PRECISION} \
    --tensorrt-capture-cuda-graph \
    --tensorrt-max-workspace-size 10000000000 \
    --max-batch-size 64 \
    --batching dynamic \
    --preferred-batch-sizes 64 \
    --engine-count-per-device gpu=${NUMBER_OF_MODEL_INSTANCES}
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
    --input-data random \
    --batch-sizes 1 2 4 8 16 32 64 \
    --concurrency 1 \
    --evaluation-mode offline \
    --measurement-request-count 10 \
    --warmup \
    --performance-tool perf_analyzer \
    --result-path ${SHARED_DIR}/triton_performance_offline.csv
```

 </details>



#### Triton Performance Online Test
We want to maximize throughput within latency budget constraints.
Dynamic batching is a feature of Triton Inference Server that allows
inference requests to be combined by the server, so that a batch is
created dynamically, resulting in a reduced average latency.
<details>
<summary>Triton Performance Online Test</summary>

```shell
python triton/run_performance_on_triton.py \
    --model-repository ${MODEL_REPOSITORY_PATH} \
    --model-name ${MODEL_NAME} \
    --input-data random \
    --batch-sizes 1 \
    --concurrency 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256 \
    --evaluation-mode online \
    --measurement-request-count 500 \
    --warmup \
    --performance-tool perf_analyzer \
    --result-path ${SHARED_DIR}/triton_performance_online.csv
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
a small fraction of time, compared to step 5. In distributed systems and online processing
where client and server side are connect through network, the send and receive steps might have impact
on overall processing performance. In order to analyze the possible bottlenecks the detailed
charts are presented in online scenario cases.



## Release Notes
We’re constantly refining and improving our performance on AI
and HPC workloads even on the same hardware with frequent updates
to our software stack. For our latest performance data refer
to these pages for
[AI](https://developer.nvidia.com/deep-learning-performance-training-inference)
and [HPC](https://developer.nvidia.com/hpc-application-performance) benchmarks.

### Changelog

May 2022
- Initial release

### Known issues

- There are no known issues with this model.
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
NVIDIA DGX-1 (1x V100 32GB): ./triton/065ms/runner/start_NVIDIA-DGX-1-\(1x-V100-32GB\).sh

NVIDIA DGX A100 (1x A100 80GB): ./triton/065ms/runner/start_NVIDIA-DGX-A100-\(1x-A100-80GB\).sh
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
|       1 |             1 |              842.00 |               0.05 |                            0.25 |                0.09 |                        0.15 |                        0.63 |                         0.01 |               0.00 |               1.15 |               1.35 |               1.40 |               1.46 |               1.18 |
|       2 |             1 |             1340.00 |               0.06 |                            0.26 |                0.09 |                        0.25 |                        0.81 |                         0.01 |               0.00 |               1.47 |               1.65 |               1.70 |               1.77 |               1.49 |
|       4 |             1 |             2076.00 |               0.05 |                            0.27 |                0.08 |                        0.37 |                        1.14 |                         0.01 |               0.00 |               1.89 |               2.09 |               2.15 |               2.19 |               1.92 |
|       8 |             1 |             2800.00 |               0.05 |                            0.26 |                0.09 |                        0.61 |                        1.83 |                         0.01 |               0.00 |               2.84 |               3.00 |               3.04 |               3.08 |               2.85 |
|      16 |             1 |             3504.00 |               0.05 |                            0.26 |                0.09 |                        1.06 |                        3.07 |                         0.01 |               0.00 |               4.51 |               4.70 |               4.73 |               4.83 |               4.54 |
|      32 |             1 |             4096.00 |               0.05 |                            0.26 |                0.08 |                        1.97 |                        5.43 |                         0.02 |               0.00 |               7.76 |               7.99 |               8.02 |               8.08 |               7.81 |
|      64 |             1 |             4480.00 |               0.05 |                            0.27 |                0.09 |                        3.82 |                       10.03 |                         0.02 |               0.00 |              14.25 |              14.46 |              14.51 |              14.56 |              14.28 |

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
|       1 |             1 |             1478.00 |               0.02 |                            0.08 |                0.02 |                        0.10 |                        0.46 |                         0.00 |               0.00 |               0.66 |               0.68 |               0.70 |               0.85 |               0.67 |
|       2 |             1 |             2668.00 |               0.02 |                            0.06 |                0.02 |                        0.14 |                        0.51 |                         0.00 |               0.00 |               0.75 |               0.76 |               0.77 |               0.77 |               0.74 |
|       4 |             1 |             4092.00 |               0.02 |                            0.07 |                0.02 |                        0.20 |                        0.66 |                         0.00 |               0.00 |               0.97 |               0.98 |               0.99 |               1.13 |               0.97 |
|       8 |             1 |             5936.00 |               0.02 |                            0.06 |                0.02 |                        0.34 |                        0.91 |                         0.00 |               0.00 |               1.33 |               1.36 |               1.41 |               1.57 |               1.34 |
|      16 |             1 |             7008.00 |               0.02 |                            0.07 |                0.02 |                        0.64 |                        1.52 |                         0.01 |               0.00 |               2.27 |               2.33 |               2.38 |               2.54 |               2.28 |
|      32 |             1 |             7072.00 |               0.02 |                            0.12 |                0.02 |                        1.47 |                        2.84 |                         0.03 |               0.00 |               4.49 |               4.59 |               4.66 |               4.89 |               4.51 |
|      64 |             1 |             7680.00 |               0.02 |                            0.13 |                0.02 |                        2.95 |                        5.12 |                         0.04 |               0.00 |               8.27 |               8.42 |               8.53 |               8.74 |               8.29 |

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
|       1 |             8 |             1225.00 |               0.10 |                            1.04 |                3.52 |                        0.24 |                        1.60 |                         0.01 |               0.00 |               6.72 |               7.36 |               7.59 |               7.78 |               6.52 |
|       1 |            16 |             1658.00 |               0.11 |                            2.18 |                4.45 |                        0.69 |                        2.16 |                         0.02 |               0.00 |               9.34 |              13.55 |              13.97 |              14.62 |               9.61 |
|       1 |            24 |             1987.00 |               0.12 |                            3.02 |                5.29 |                        0.99 |                        2.53 |                         0.02 |               0.00 |              11.90 |              15.62 |              16.94 |              19.39 |              11.96 |
|       1 |            32 |             2208.00 |               0.12 |                            3.73 |                6.15 |                        1.39 |                        2.93 |                         0.02 |               0.00 |              14.02 |              20.91 |              21.83 |              22.93 |              14.34 |
|       1 |            40 |             2368.00 |               0.14 |                            5.38 |                6.05 |                        1.88 |                        3.28 |                         0.03 |               0.00 |              17.98 |              22.21 |              22.74 |              23.55 |              16.75 |
|       1 |            48 |             2368.00 |               0.18 |                            8.29 |                6.44 |                        1.85 |                        3.25 |                         0.03 |               0.00 |              21.42 |              27.00 |              28.50 |              30.02 |              20.03 |
|       1 |            56 |             2509.00 |               0.18 |                            7.99 |                7.28 |                        2.62 |                        3.76 |                         0.04 |               0.00 |              23.58 |              29.22 |              30.09 |              31.43 |              21.86 |
|       1 |            64 |             2674.00 |               0.20 |                            8.42 |                8.53 |                        2.72 |                        3.82 |                         0.04 |               0.00 |              25.18 |              31.21 |              33.56 |              43.09 |              23.73 |
|       1 |            72 |             2688.00 |               0.20 |                           10.05 |                8.97 |                        3.09 |                        4.07 |                         0.04 |               0.00 |              27.59 |              35.28 |              37.35 |              40.03 |              26.44 |
|       1 |            80 |             2610.00 |               0.22 |                           11.76 |                9.11 |                        4.50 |                        4.65 |                         0.06 |               0.00 |              31.73 |              42.18 |              44.10 |              45.90 |              30.30 |
|       1 |            88 |             2573.00 |               0.22 |                           10.54 |                9.94 |                        7.01 |                        5.78 |                         0.08 |               0.00 |              41.51 |              44.72 |              45.70 |              49.03 |              33.58 |
|       1 |            96 |             2815.00 |               0.23 |                           12.18 |               11.07 |                        4.94 |                        5.09 |                         0.06 |               0.00 |              34.73 |              44.71 |              48.38 |              56.71 |              33.58 |
|       1 |           104 |             2732.00 |               0.25 |                           11.90 |               12.11 |                        7.01 |                        6.00 |                         0.08 |               0.00 |              38.49 |              51.49 |              54.39 |              58.54 |              37.36 |
|       1 |           112 |             2869.00 |               0.26 |                           11.69 |               13.93 |                        6.49 |                        5.68 |                         0.08 |               0.00 |              37.86 |              50.94 |              55.40 |              64.80 |              38.11 |
|       1 |           120 |             2958.00 |               0.26 |                           12.24 |               13.02 |                        7.48 |                        6.78 |                         0.10 |               0.00 |              42.24 |              54.54 |              57.09 |              59.88 |              39.87 |
|       1 |           128 |             2990.00 |               0.24 |                           14.14 |               14.39 |                        6.59 |                        6.35 |                         0.09 |               0.00 |              43.49 |              54.44 |              58.77 |              70.31 |              41.80 |
|       1 |           136 |             2989.00 |               0.28 |                           15.34 |               15.02 |                        7.03 |                        6.80 |                         0.10 |               0.00 |              45.64 |              59.21 |              62.02 |              65.34 |              44.59 |
|       1 |           144 |             2989.00 |               0.27 |                           16.48 |               15.56 |                        8.41 |                        6.72 |                         0.10 |               0.00 |              48.12 |              65.24 |              67.14 |              70.71 |              47.54 |
|       1 |           152 |             2964.00 |               0.27 |                           16.89 |               17.22 |                        8.32 |                        7.00 |                         0.10 |               0.00 |              50.68 |              68.96 |              73.71 |              80.31 |              49.80 |
|       1 |           160 |             3026.00 |               0.27 |                           16.01 |               18.03 |                        9.76 |                        7.50 |                         0.13 |               0.00 |              52.67 |              66.81 |              68.20 |              74.10 |              51.69 |
|       1 |           168 |             3113.89 |               0.29 |                           16.40 |               17.93 |                        9.34 |                        8.39 |                         0.13 |               0.00 |              53.71 |              68.57 |              70.61 |              73.08 |              52.48 |
|       1 |           176 |             3194.00 |               0.35 |                           15.40 |               19.42 |                        9.78 |                        8.63 |                         0.13 |               0.00 |              54.05 |              70.48 |              73.20 |              77.89 |              53.72 |
|       1 |           184 |             3246.00 |               0.31 |                           17.21 |               19.43 |                        9.39 |                        8.46 |                         0.13 |               0.00 |              56.10 |              70.56 |              75.07 |              79.31 |              54.94 |
|       1 |           192 |             3165.00 |               0.32 |                           18.71 |               19.74 |                       10.00 |                        9.01 |                         0.15 |               0.00 |              59.04 |              71.28 |              73.31 |              77.61 |              57.92 |
|       1 |           200 |             3230.00 |               0.28 |                           21.48 |               18.28 |                       10.50 |                        9.30 |                         0.16 |               0.00 |              61.72 |              74.04 |              75.61 |              81.67 |              59.99 |
|       1 |           208 |             3268.00 |               0.32 |                           18.42 |               23.43 |                        9.82 |                        8.61 |                         0.14 |               0.00 |              61.70 |              75.20 |              79.59 |              84.76 |              60.73 |
|       1 |           216 |             3263.00 |               0.32 |                           19.63 |               23.60 |                       11.11 |                        9.59 |                         0.15 |               0.00 |              65.28 |              80.60 |              85.08 |              91.09 |              64.41 |
|       1 |           224 |             3145.00 |               0.36 |                           21.09 |               23.86 |                       13.06 |                       10.67 |                         0.16 |               0.00 |              72.96 |              83.93 |              86.35 |              92.58 |              69.20 |
|       1 |           232 |             3148.00 |               0.36 |                           22.02 |               24.26 |                       12.64 |                       11.42 |                         0.17 |               0.00 |              75.53 |              84.75 |              87.35 |              94.60 |              70.87 |
|       1 |           240 |             3342.00 |               0.49 |                           16.67 |               29.95 |                       11.96 |                       10.46 |                         0.17 |               0.00 |              70.85 |              87.04 |              90.95 |              95.84 |              69.70 |
|       1 |           248 |             3357.00 |               0.32 |                           27.51 |               22.90 |                       10.18 |                        9.23 |                         0.15 |               0.00 |              71.50 |              86.61 |              94.11 |             103.46 |              70.30 |
|       1 |           256 |             3361.00 |               0.42 |                           22.20 |               28.57 |                       11.95 |                       10.67 |                         0.16 |               0.00 |              76.68 |              87.06 |              89.44 |              96.00 |              73.98 |

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
|       1 |             8 |             1864.00 |               0.07 |                            0.71 |                2.34 |                        0.16 |                        0.99 |                         0.01 |               0.00 |               4.25 |               4.77 |               4.94 |               5.20 |               4.28 |
|       1 |            16 |             2607.39 |               0.08 |                            1.51 |                2.90 |                        0.40 |                        1.22 |                         0.01 |               0.00 |               6.08 |               7.60 |               8.03 |               8.95 |               6.12 |
|       1 |            24 |             2997.00 |               0.09 |                            2.54 |                3.25 |                        0.72 |                        1.37 |                         0.02 |               0.00 |               7.97 |              10.57 |              11.18 |              13.21 |               7.99 |
|       1 |            32 |             3276.00 |               0.11 |                            3.52 |                3.40 |                        1.14 |                        1.52 |                         0.03 |               0.00 |              10.15 |              13.06 |              13.59 |              14.38 |               9.72 |
|       1 |            40 |             3445.00 |               0.11 |                            4.50 |                3.63 |                        1.51 |                        1.73 |                         0.04 |               0.00 |              12.63 |              15.10 |              15.51 |              16.41 |              11.52 |
|       1 |            48 |             3608.00 |               0.11 |                            5.86 |                3.97 |                        1.51 |                        1.71 |                         0.04 |               0.00 |              13.68 |              16.60 |              17.32 |              18.51 |              13.19 |
|       1 |            56 |             3821.00 |               0.11 |                            6.07 |                4.35 |                        1.99 |                        1.98 |                         0.05 |               0.00 |              15.77 |              19.19 |              19.82 |              20.99 |              14.55 |
|       1 |            64 |             4070.00 |               0.10 |                            6.20 |                4.78 |                        2.33 |                        2.16 |                         0.05 |               0.00 |              16.65 |              20.56 |              21.78 |              23.89 |              15.62 |
|       1 |            72 |             4187.00 |               0.12 |                            5.98 |                5.92 |                        2.62 |                        2.33 |                         0.06 |               0.00 |              18.07 |              22.57 |              23.69 |              25.99 |              17.02 |
|       1 |            80 |             4329.00 |               0.10 |                            6.75 |                5.95 |                        2.85 |                        2.47 |                         0.07 |               0.00 |              19.08 |              23.93 |              25.12 |              26.30 |              18.19 |
|       1 |            88 |             4474.00 |               0.12 |                            6.88 |                6.68 |                        3.18 |                        2.57 |                         0.07 |               0.00 |              20.14 |              25.30 |              26.72 |              30.18 |              19.49 |
|       1 |            96 |             4590.00 |               0.12 |                            8.08 |                6.42 |                        3.23 |                        2.70 |                         0.08 |               0.00 |              21.43 |              26.92 |              28.13 |              32.91 |              20.63 |
|       1 |           104 |             4632.00 |               0.11 |                            7.98 |                7.31 |                        3.73 |                        2.97 |                         0.09 |               0.00 |              22.79 |              28.68 |              31.35 |              36.71 |              22.18 |
|       1 |           112 |             4654.00 |               0.10 |                           10.48 |                6.84 |                        3.39 |                        2.81 |                         0.08 |               0.00 |              24.41 |              31.16 |              33.22 |              37.03 |              23.70 |
|       1 |           120 |             4929.00 |               0.11 |                            9.14 |                8.01 |                        3.66 |                        3.06 |                         0.09 |               0.00 |              24.90 |              31.41 |              32.98 |              39.08 |              24.07 |
|       1 |           128 |             4842.00 |               0.10 |                            9.54 |                8.50 |                        4.48 |                        3.30 |                         0.10 |               0.00 |              26.78 |              34.51 |              36.68 |              38.60 |              26.02 |
|       1 |           136 |             4869.00 |               0.10 |                            9.87 |                9.05 |                        4.83 |                        3.48 |                         0.11 |               0.00 |              27.87 |              34.78 |              36.79 |              40.60 |              27.45 |
|       1 |           144 |             5155.00 |               0.11 |                            9.83 |                9.41 |                        4.60 |                        3.58 |                         0.11 |               0.00 |              28.51 |              36.00 |              37.76 |              41.36 |              27.64 |
|       1 |           152 |             5113.00 |               0.12 |                            9.96 |                9.53 |                        5.55 |                        3.88 |                         0.13 |               0.00 |              30.28 |              37.23 |              38.74 |              41.21 |              29.17 |
|       1 |           160 |             5053.00 |               0.11 |                           11.25 |               10.37 |                        5.44 |                        3.82 |                         0.13 |               0.00 |              32.03 |              40.37 |              43.19 |              45.75 |              31.12 |
|       1 |           168 |             5018.00 |               0.12 |                           11.42 |               11.14 |                        6.20 |                        4.00 |                         0.14 |               0.00 |              33.98 |              42.41 |              45.32 |              48.52 |              33.01 |
|       1 |           176 |             5146.00 |               0.12 |                           11.42 |               11.63 |                        6.05 |                        4.10 |                         0.14 |               0.00 |              34.48 |              43.39 |              45.25 |              50.67 |              33.46 |
|       1 |           184 |             4805.00 |               0.12 |                           18.49 |               10.25 |                        4.99 |                        3.40 |                         0.11 |               0.00 |              32.61 |              58.79 |              62.32 |              67.53 |              37.36 |
|       1 |           192 |             5458.00 |               0.13 |                           10.60 |               11.73 |                        6.86 |                        4.87 |                         0.16 |               0.00 |              36.11 |              42.32 |              43.57 |              45.46 |              34.36 |
|       1 |           200 |             5095.00 |               0.15 |                           11.19 |               14.90 |                        7.52 |                        4.58 |                         0.15 |               0.00 |              38.94 |              48.22 |              50.25 |              54.12 |              38.49 |
|       1 |           208 |             5470.00 |               0.10 |                           12.16 |               12.25 |                        7.59 |                        4.97 |                         0.16 |               0.00 |              38.11 |              45.97 |              46.42 |              48.32 |              37.23 |
|       1 |           216 |             5382.00 |               0.11 |                           13.92 |               13.65 |                        6.74 |                        4.49 |                         0.14 |               0.00 |              39.30 |              50.41 |              53.34 |              58.88 |              39.06 |
|       1 |           224 |             5478.00 |               0.11 |                           13.06 |               15.09 |                        6.65 |                        4.43 |                         0.15 |               0.00 |              39.40 |              50.39 |              53.51 |              57.37 |              39.49 |
|       1 |           232 |             5385.00 |               0.11 |                           13.58 |               13.64 |                        8.54 |                        6.00 |                         0.18 |               0.00 |              43.78 |              50.20 |              51.78 |              55.14 |              42.04 |
|       1 |           240 |             5519.00 |               0.12 |                           11.83 |               17.19 |                        7.90 |                        5.36 |                         0.17 |               0.00 |              43.49 |              51.74 |              54.30 |              59.48 |              42.57 |
|       1 |           248 |             5422.00 |               0.12 |                           14.23 |               16.04 |                        8.82 |                        5.56 |                         0.18 |               0.00 |              46.15 |              53.49 |              56.08 |              59.57 |              44.95 |
|       1 |           256 |             5215.00 |               0.10 |                           22.93 |               12.82 |                        7.06 |                        4.52 |                         0.15 |               0.00 |              41.19 |              76.05 |              83.77 |              91.88 |              47.58 |

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
Please download a checkpoint from [here](https://api.ngc.nvidia.com/v2/models/nvidia/dle/gpunet_0_pyt_ckpt/versions/21.12.0_amp/zip) 
and place it in `runner_workspace/checkpoints/0.65ms/`.  Note that the `0.65ms` subdirectory may not be created yet.

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
export CHECKPOINT="0.65ms"
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
    --config /workspace/gpunet/configs/batch1/GV100/0.65ms.json \
    --checkpoint ${CHECKPOINT_DIR}/0.65ms.pth.tar \
    --precision ${EXPORT_PRECISION} \
    \
    --dataloader triton/dataloader.py \
    --val-path ${DATASETS_DIR}/ \
    --is-prunet False \
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
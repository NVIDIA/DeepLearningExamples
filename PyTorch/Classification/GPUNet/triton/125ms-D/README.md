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
NVIDIA DGX-1 (1x V100 32GB): ./triton/125ms-D/runner/start_NVIDIA-DGX-1-\(1x-V100-32GB\).sh

NVIDIA DGX A100 (1x A100 80GB): ./triton/125ms-D/runner/start_NVIDIA-DGX-A100-\(1x-A100-80GB\).sh
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
|       1 |             1 |              548.00 |               0.05 |                            0.22 |                0.08 |                        0.23 |                        1.24 |                         0.01 |               0.00 |               1.82 |               1.86 |               1.88 |               2.04 |               1.82 |
|       2 |             1 |              763.24 |               0.05 |                            0.22 |                0.08 |                        0.35 |                        1.91 |                         0.01 |               0.00 |               2.61 |               2.65 |               2.66 |               2.71 |               2.61 |
|       4 |             1 |              983.02 |               0.05 |                            0.22 |                0.08 |                        0.58 |                        3.13 |                         0.01 |               0.00 |               4.05 |               4.10 |               4.17 |               4.29 |               4.06 |
|       8 |             1 |             1144.00 |               0.04 |                            0.20 |                0.06 |                        1.06 |                        5.58 |                         0.01 |               0.00 |               6.99 |               7.03 |               7.05 |               7.07 |               6.96 |
|      16 |             1 |             1248.00 |               0.05 |                            0.22 |                0.08 |                        1.97 |                       10.40 |                         0.01 |               0.00 |              12.73 |              12.78 |              12.79 |              12.81 |              12.73 |
|      32 |             1 |             1312.00 |               0.05 |                            0.23 |                0.09 |                        3.89 |                       19.59 |                         0.02 |               0.00 |              23.84 |              23.89 |              24.00 |              24.45 |              23.86 |
|      64 |             1 |             1344.00 |               0.06 |                            0.34 |                0.11 |                        9.01 |                       37.96 |                         0.04 |               0.00 |              47.52 |              47.60 |              47.61 |              47.68 |              47.52 |

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
|       1 |             1 |              831.00 |               0.02 |                            0.07 |                0.02 |                        0.15 |                        0.94 |                         0.00 |               0.00 |               1.20 |               1.21 |               1.22 |               1.26 |               1.20 |
|       2 |             1 |             1272.00 |               0.02 |                            0.06 |                0.02 |                        0.21 |                        1.26 |                         0.00 |               0.00 |               1.56 |               1.59 |               1.60 |               1.65 |               1.57 |
|       4 |             1 |             1652.00 |               0.02 |                            0.07 |                0.02 |                        0.36 |                        1.94 |                         0.00 |               0.00 |               2.40 |               2.46 |               2.47 |               2.55 |               2.41 |
|       8 |             1 |             1904.00 |               0.02 |                            0.08 |                0.02 |                        0.79 |                        3.28 |                         0.01 |               0.00 |               4.19 |               4.25 |               4.26 |               4.29 |               4.20 |
|      16 |             1 |             1936.00 |               0.02 |                            0.09 |                0.02 |                        1.95 |                        6.12 |                         0.01 |               0.00 |               8.22 |               8.30 |               8.31 |               8.39 |               8.22 |
|      32 |             1 |             2016.00 |               0.02 |                            0.11 |                0.02 |                        3.96 |                       11.64 |                         0.02 |               0.00 |              15.78 |              15.84 |              15.89 |              15.95 |              15.79 |
|      64 |             1 |             1984.00 |               0.02 |                            0.20 |                0.03 |                        9.10 |                       22.49 |                         0.03 |               0.00 |              30.71 |              36.15 |              37.38 |              38.14 |              31.87 |

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
|       1 |             8 |              637.00 |               0.15 |                            1.72 |                7.00 |                        0.40 |                        3.21 |                         0.01 |               0.00 |              13.08 |              14.31 |              14.55 |              15.36 |              12.50 |
|       1 |            16 |              758.00 |               0.25 |                            4.63 |                9.69 |                        1.27 |                        5.06 |                         0.02 |               0.00 |              21.28 |              25.24 |              26.39 |              29.37 |              20.92 |
|       1 |            24 |              838.00 |               0.32 |                            5.98 |               12.62 |                        2.42 |                        6.93 |                         0.02 |               0.00 |              27.95 |              39.18 |              40.38 |              42.79 |              28.29 |
|       1 |            32 |              881.00 |               0.31 |                           11.62 |               11.16 |                        4.04 |                        8.38 |                         0.03 |               0.00 |              44.87 |              46.22 |              46.50 |              47.24 |              35.54 |
|       1 |            40 |              930.00 |               0.37 |                           10.12 |               16.95 |                        4.91 |                        9.88 |                         0.04 |               0.00 |              45.47 |              56.34 |              57.76 |              60.10 |              42.28 |
|       1 |            48 |              949.00 |               0.38 |                           12.38 |               18.52 |                        6.35 |                       11.88 |                         0.05 |               0.00 |              52.57 |              66.30 |              68.82 |              72.02 |              49.54 |
|       1 |            56 |              937.00 |               0.35 |                           20.63 |               17.27 |                        6.55 |                       12.56 |                         0.05 |               0.00 |              63.79 |              78.07 |              78.73 |              83.10 |              57.42 |
|       1 |            64 |              955.00 |               0.35 |                           20.90 |               18.75 |                        9.16 |                       15.68 |                         0.07 |               0.00 |              76.95 |              85.39 |              89.50 |              95.02 |              64.91 |
|       1 |            72 |             1002.00 |               0.35 |                           22.84 |               20.77 |                        9.20 |                       16.18 |                         0.06 |               0.00 |              75.24 |              95.72 |              99.89 |             101.37 |              69.40 |
|       1 |            80 |             1002.00 |               0.37 |                           26.88 |               23.61 |                        9.65 |                       16.64 |                         0.07 |               0.00 |              80.54 |             110.17 |             111.28 |             133.01 |              77.21 |
|       1 |            88 |             1039.96 |               0.39 |                           24.71 |               25.96 |                       11.72 |                       18.41 |                         0.08 |               0.00 |              86.60 |             107.94 |             113.54 |             117.13 |              81.27 |
|       1 |            96 |              971.00 |               0.42 |                           31.77 |               27.39 |                       13.33 |                       21.02 |                         0.09 |               0.00 |              99.73 |             133.61 |             137.27 |             158.50 |              94.02 |
|       1 |           104 |             1054.00 |               0.46 |                           25.57 |               31.63 |                       13.73 |                       22.58 |                         0.09 |               0.00 |              97.34 |             130.93 |             142.62 |             145.20 |              94.05 |
|       1 |           112 |             1037.00 |               0.60 |                           21.31 |               37.85 |                       14.70 |                       25.27 |                         0.10 |               0.00 |             100.48 |             134.15 |             134.53 |             151.25 |              99.83 |
|       1 |           120 |             1089.00 |               0.50 |                           24.03 |               43.92 |                       13.48 |                       22.94 |                         0.10 |               0.00 |             107.66 |             132.97 |             140.20 |             161.82 |             104.96 |
|       1 |           128 |             1012.00 |               0.49 |                           29.51 |               42.22 |                       17.59 |                       26.89 |                         0.12 |               0.00 |             128.34 |             165.16 |             165.62 |             176.59 |             116.81 |
|       1 |           136 |              981.02 |               0.44 |                           40.73 |               38.62 |                       19.27 |                       29.14 |                         0.13 |               0.00 |             156.52 |             174.82 |             176.12 |             179.08 |             128.34 |
|       1 |           144 |             1056.00 |               0.50 |                           34.14 |               42.77 |                       20.76 |                       29.69 |                         0.16 |               0.00 |             136.02 |             165.59 |             176.76 |             178.82 |             128.00 |
|       1 |           152 |             1032.00 |               0.67 |                           24.96 |               55.14 |                       20.47 |                       34.28 |                         0.14 |               0.00 |             151.94 |             173.47 |             176.05 |             177.26 |             135.66 |
|       1 |           160 |             1093.00 |               0.68 |                           24.44 |               56.78 |                       20.70 |                       34.85 |                         0.14 |               0.00 |             132.11 |             190.39 |             191.01 |             193.26 |             137.59 |
|       1 |           168 |             1097.00 |               0.50 |                           40.32 |               46.87 |                       22.27 |                       34.31 |                         0.16 |               0.00 |             138.97 |             179.77 |             181.53 |             191.99 |             144.43 |
|       1 |           176 |             1086.00 |               0.54 |                           33.23 |               57.92 |                       19.28 |                       35.83 |                         0.15 |               0.00 |             146.82 |             201.92 |             204.05 |             204.53 |             146.95 |
|       1 |           184 |             1011.00 |               0.62 |                           36.99 |               57.91 |                       25.30 |                       40.37 |                         0.16 |               0.00 |             164.68 |             220.22 |             222.48 |             225.76 |             161.35 |
|       1 |           192 |             1122.00 |               0.69 |                           33.35 |               69.55 |                       21.85 |                       36.47 |                         0.15 |               0.00 |             164.32 |             203.35 |             209.71 |             217.30 |             162.05 |
|       1 |           200 |             1012.00 |               0.79 |                           34.55 |               71.59 |                       29.42 |                       43.90 |                         0.18 |               0.00 |             192.63 |             211.87 |             221.22 |             235.79 |             180.42 |
|       1 |           208 |             1032.97 |               0.59 |                           35.84 |               76.16 |                       28.08 |                       39.68 |                         0.18 |               0.00 |             180.63 |             227.94 |             234.15 |             245.10 |             180.54 |
|       1 |           216 |             1089.00 |               0.60 |                           46.65 |               71.92 |                       25.31 |                       38.54 |                         0.17 |               0.00 |             200.69 |             223.36 |             225.54 |             228.87 |             183.20 |
|       1 |           224 |             1007.00 |               0.92 |                           33.44 |               92.25 |                       28.68 |                       41.06 |                         0.17 |               0.00 |             194.80 |             244.81 |             245.48 |             273.77 |             196.52 |
|       1 |           232 |             1071.00 |               0.70 |                           33.80 |               88.38 |                       22.34 |                       39.10 |                         0.17 |               0.00 |             188.88 |             225.51 |             228.99 |             242.20 |             184.50 |
|       1 |           240 |             1018.00 |               0.78 |                           32.81 |              102.52 |                       24.00 |                       37.39 |                         0.14 |               0.00 |             206.90 |             238.94 |             243.45 |             253.24 |             197.65 |
|       1 |           248 |             1137.00 |               0.67 |                           36.93 |              101.88 |                       24.24 |                       34.72 |                         0.14 |               0.00 |             200.73 |             246.81 |             248.41 |             264.99 |             198.58 |
|       1 |           256 |             1091.00 |               0.67 |                           40.93 |              103.90 |                       26.41 |                       36.21 |                         0.16 |               0.00 |             206.12 |             257.96 |             259.36 |             265.76 |             208.27 |

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
|       1 |             8 |             1001.00 |               0.16 |                            1.43 |                4.08 |                        0.28 |                        2.02 |                         0.01 |               0.00 |               7.92 |               9.69 |               9.92 |              10.26 |               7.96 |
|       1 |            16 |             1187.00 |               0.21 |                            2.97 |                6.34 |                        0.82 |                        3.03 |                         0.02 |               0.00 |              13.52 |              16.50 |              17.48 |              19.44 |              13.38 |
|       1 |            24 |             1259.00 |               0.33 |                            5.53 |                6.83 |                        1.64 |                        4.44 |                         0.03 |               0.00 |              18.25 |              28.73 |              29.85 |              31.06 |              18.80 |
|       1 |            32 |             1405.00 |               0.28 |                            6.77 |                7.96 |                        2.16 |                        5.41 |                         0.04 |               0.00 |              23.38 |              30.94 |              31.47 |              37.44 |              22.61 |
|       1 |            40 |             1530.47 |               0.29 |                            7.02 |               10.11 |                        2.29 |                        6.11 |                         0.04 |               0.00 |              26.08 |              35.46 |              38.88 |              42.02 |              25.86 |
|       1 |            48 |             1542.00 |               0.25 |                           10.69 |               10.39 |                        2.74 |                        6.64 |                         0.05 |               0.00 |              33.68 |              39.42 |              40.20 |              46.24 |              30.76 |
|       1 |            56 |             1556.00 |               0.24 |                           11.63 |               11.33 |                        4.16 |                        7.96 |                         0.06 |               0.00 |              38.70 |              48.09 |              51.18 |              54.74 |              35.39 |
|       1 |            64 |             1603.40 |               0.25 |                           14.20 |               12.32 |                        4.52 |                        8.25 |                         0.07 |               0.00 |              41.96 |              52.62 |              54.76 |              60.10 |              39.61 |
|       1 |            72 |             1593.00 |               0.30 |                           15.61 |               12.70 |                        5.03 |                        9.79 |                         0.08 |               0.00 |              49.58 |              58.78 |              59.42 |              62.12 |              43.50 |
|       1 |            80 |             1733.00 |               0.30 |                           13.26 |               15.87 |                        4.89 |                       10.13 |                         0.08 |               0.00 |              45.74 |              57.32 |              63.43 |              72.32 |              44.53 |
|       1 |            88 |             1787.00 |               0.27 |                           16.85 |               14.66 |                        5.08 |                       11.24 |                         0.08 |               0.00 |              50.32 |              63.56 |              65.50 |              69.74 |              48.18 |
|       1 |            96 |             1699.00 |               0.25 |                           21.14 |               16.76 |                        5.76 |                       11.35 |                         0.10 |               0.00 |              57.12 |              76.26 |              79.36 |              99.10 |              55.37 |
|       1 |           104 |             1763.00 |               0.24 |                           16.36 |               19.64 |                        8.03 |                       13.96 |                         0.11 |               0.00 |              58.91 |              78.63 |              81.58 |              84.28 |              58.33 |
|       1 |           112 |             1773.23 |               0.22 |                           18.51 |               19.47 |                        8.37 |                       14.41 |                         0.11 |               0.00 |              62.71 |              81.65 |              85.27 |              90.86 |              61.09 |
|       1 |           120 |             1702.00 |               0.24 |                           21.71 |               20.10 |                        9.05 |                       15.14 |                         0.12 |               0.00 |              69.00 |              87.50 |              90.05 |              97.77 |              66.35 |
|       1 |           128 |             1757.00 |               0.33 |                           17.17 |               25.20 |                       10.29 |                       16.76 |                         0.12 |               0.00 |              68.94 |              87.07 |             107.74 |             115.25 |              69.88 |
|       1 |           136 |             1770.00 |               0.24 |                           18.59 |               26.00 |                       10.81 |                       17.69 |                         0.13 |               0.00 |              78.22 |              92.84 |              99.19 |             108.84 |              73.46 |
|       1 |           144 |             1775.00 |               0.23 |                           24.96 |               23.41 |                       11.04 |                       17.80 |                         0.12 |               0.00 |              83.74 |             103.76 |             105.57 |             116.22 |              77.56 |
|       1 |           152 |             1751.00 |               0.26 |                           28.09 |               21.81 |                       12.90 |                       18.88 |                         0.13 |               0.00 |              93.29 |             109.52 |             110.68 |             112.37 |              82.07 |
|       1 |           160 |             1854.00 |               0.24 |                           23.43 |               26.88 |                       11.90 |                       20.52 |                         0.13 |               0.00 |              84.35 |             104.49 |             110.54 |             120.01 |              83.10 |
|       1 |           168 |             1761.00 |               0.48 |                           16.45 |               33.65 |                       15.19 |                       25.00 |                         0.16 |               0.00 |             111.80 |             117.00 |             117.60 |             117.91 |              90.94 |
|       1 |           176 |             1923.00 |               0.37 |                           19.76 |               38.24 |                       11.11 |                       19.04 |                         0.13 |               0.00 |              86.64 |             119.33 |             122.06 |             125.40 |              88.64 |
|       1 |           184 |             1982.00 |               0.24 |                           23.18 |               33.14 |                       10.45 |                       21.24 |                         0.12 |               0.00 |              88.08 |             108.97 |             112.51 |             126.41 |              88.37 |
|       1 |           192 |             2002.00 |               0.26 |                           23.93 |               35.62 |                       10.06 |                       21.49 |                         0.12 |               0.00 |              94.42 |             112.48 |             114.59 |             122.92 |              91.48 |
|       1 |           200 |             1935.00 |               0.20 |                           31.17 |               32.02 |                       10.73 |                       23.75 |                         0.14 |               0.00 |             100.00 |             124.33 |             125.97 |             134.48 |              98.01 |
|       1 |           208 |             1961.00 |               0.25 |                           26.61 |               37.94 |                       11.92 |                       23.77 |                         0.14 |               0.00 |             107.09 |             121.03 |             124.07 |             138.64 |             100.62 |
|       1 |           216 |             1969.00 |               0.28 |                           26.74 |               39.63 |                       12.17 |                       25.96 |                         0.15 |               0.00 |             112.16 |             126.24 |             129.48 |             133.03 |             104.94 |
|       1 |           224 |             2057.00 |               0.36 |                           17.45 |               49.87 |                       12.00 |                       25.09 |                         0.14 |               0.00 |             102.93 |             130.17 |             132.46 |             140.88 |             104.92 |
|       1 |           232 |             1858.00 |               0.23 |                           47.00 |               28.61 |                       13.97 |                       28.59 |                         0.16 |               0.00 |             123.72 |             135.24 |             145.10 |             149.58 |             118.56 |
|       1 |           240 |             1943.00 |               0.36 |                           22.44 |               53.78 |                       12.46 |                       26.07 |                         0.15 |               0.00 |             116.49 |             140.15 |             147.14 |             154.82 |             115.26 |
|       1 |           248 |             1979.00 |               0.43 |                           21.27 |               58.61 |                       12.43 |                       25.00 |                         0.15 |               0.00 |             117.69 |             142.88 |             145.82 |             157.04 |             117.89 |
|       1 |           256 |             1939.00 |               0.22 |                           33.66 |               51.01 |                       13.29 |                       25.70 |                         0.16 |               0.00 |             127.11 |             150.69 |             157.93 |             169.28 |             124.04 |

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
Please download a checkpoint from [here](https://api.ngc.nvidia.com/v2/models/nvidia/dle/gpunet_d1_pyt_ckpt/versions/21.12.0_amp/zip) 
and place it in `runner_workspace/checkpoints/1.25ms-D/`.  Note that the `1.25ms-D` subdirectory may not be created yet.

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
export CHECKPOINT="1.25ms-D"
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
    --config /workspace/gpunet/configs/batch1/GV100/1.25ms-D.json \
    --checkpoint ${CHECKPOINT_DIR}/1.25ms-D.pth.tar \
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
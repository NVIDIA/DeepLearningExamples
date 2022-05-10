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
NVIDIA DGX-1 (1x V100 32GB): ./triton/085ms/runner/start_NVIDIA-DGX-1-\(1x-V100-32GB\).sh

NVIDIA DGX A100 (1x A100 80GB): ./triton/085ms/runner/start_NVIDIA-DGX-A100-\(1x-A100-80GB\).sh
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
|       1 |             1 |              783.00 |               0.05 |                            0.21 |                0.07 |                        0.12 |                        0.82 |                         0.01 |               0.00 |               1.28 |               1.31 |               1.34 |               1.38 |               1.27 |
|       2 |             1 |             1330.00 |               0.04 |                            0.19 |                0.07 |                        0.21 |                        0.98 |                         0.01 |               0.00 |               1.50 |               1.52 |               1.53 |               1.58 |               1.50 |
|       4 |             1 |             2069.93 |               0.05 |                            0.22 |                0.08 |                        0.30 |                        1.26 |                         0.01 |               0.00 |               1.93 |               1.97 |               1.99 |               2.05 |               1.92 |
|       8 |             1 |             2824.00 |               0.05 |                            0.23 |                0.08 |                        0.49 |                        1.97 |                         0.01 |               0.00 |               2.82 |               2.85 |               2.86 |               2.90 |               2.82 |
|      16 |             1 |             3680.00 |               0.05 |                            0.23 |                0.08 |                        0.86 |                        3.11 |                         0.01 |               0.00 |               4.34 |               4.38 |               4.39 |               4.43 |               4.34 |
|      32 |             1 |             4256.00 |               0.05 |                            0.20 |                0.05 |                        1.62 |                        5.56 |                         0.01 |               0.00 |               7.50 |               7.58 |               7.60 |               7.62 |               7.50 |
|      64 |             1 |             4672.00 |               0.05 |                            0.23 |                0.08 |                        3.09 |                       10.22 |                         0.02 |               0.00 |              13.69 |              13.74 |              13.76 |              13.80 |              13.69 |

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
|       1 |             1 |             1240.00 |               0.02 |                            0.08 |                0.02 |                        0.09 |                        0.59 |                         0.00 |               0.00 |               0.80 |               0.81 |               0.81 |               0.82 |               0.80 |
|       2 |             1 |             2154.00 |               0.02 |                            0.07 |                0.02 |                        0.15 |                        0.66 |                         0.00 |               0.00 |               0.93 |               0.94 |               0.94 |               0.96 |               0.92 |
|       4 |             1 |             3704.00 |               0.02 |                            0.07 |                0.02 |                        0.18 |                        0.78 |                         0.00 |               0.00 |               1.07 |               1.09 |               1.12 |               1.26 |               1.08 |
|       8 |             1 |             5512.00 |               0.02 |                            0.07 |                0.02 |                        0.30 |                        1.03 |                         0.00 |               0.00 |               1.45 |               1.46 |               1.46 |               1.53 |               1.45 |
|      16 |             1 |             6896.00 |               0.02 |                            0.07 |                0.02 |                        0.60 |                        1.60 |                         0.01 |               0.00 |               2.30 |               2.36 |               2.41 |               2.54 |               2.32 |
|      32 |             1 |             7040.00 |               0.02 |                            0.09 |                0.03 |                        1.60 |                        2.79 |                         0.02 |               0.00 |               4.51 |               4.62 |               4.84 |               5.07 |               4.53 |
|      64 |             1 |             7296.00 |               0.02 |                            0.09 |                0.03 |                        3.44 |                        5.14 |                         0.03 |               0.00 |               8.74 |               8.82 |               8.83 |               8.88 |               8.75 |

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
|       1 |             8 |              896.00 |               0.10 |                            0.86 |                5.35 |                        0.19 |                        2.38 |                         0.01 |               0.00 |               8.78 |              10.04 |              10.18 |              10.36 |               8.88 |
|       1 |            16 |             1411.00 |               0.10 |                            1.53 |                6.27 |                        0.50 |                        2.88 |                         0.02 |               0.00 |              11.49 |              12.83 |              13.28 |              14.12 |              11.30 |
|       1 |            24 |             1755.00 |               0.11 |                            2.55 |                6.82 |                        0.82 |                        3.27 |                         0.02 |               0.00 |              13.70 |              16.42 |              16.90 |              17.96 |              13.59 |
|       1 |            32 |             1970.00 |               0.11 |                            3.53 |                7.59 |                        1.15 |                        3.74 |                         0.03 |               0.00 |              16.04 |              20.75 |              21.35 |              22.24 |              16.15 |
|       1 |            40 |             2137.86 |               0.12 |                            4.78 |                7.88 |                        1.51 |                        4.14 |                         0.03 |               0.00 |              19.01 |              23.00 |              23.40 |              24.30 |              18.46 |
|       1 |            48 |             2392.00 |               0.14 |                            4.49 |                9.02 |                        1.84 |                        4.33 |                         0.04 |               0.00 |              20.45 |              23.80 |              25.32 |              26.69 |              19.85 |
|       1 |            56 |             2437.00 |               0.14 |                            5.97 |                9.70 |                        2.20 |                        4.66 |                         0.04 |               0.00 |              22.75 |              29.53 |              31.37 |              34.17 |              22.70 |
|       1 |            64 |             2645.00 |               0.15 |                            5.06 |               11.37 |                        2.48 |                        4.81 |                         0.04 |               0.00 |              24.24 |              28.32 |              29.55 |              35.58 |              23.91 |
|       1 |            72 |             2722.00 |               0.15 |                            6.80 |               11.37 |                        2.78 |                        5.08 |                         0.05 |               0.00 |              26.70 |              33.58 |              34.66 |              36.85 |              26.24 |
|       1 |            80 |             2762.00 |               0.16 |                            7.52 |               12.04 |                        3.35 |                        5.67 |                         0.06 |               0.00 |              29.54 |              35.28 |              36.80 |              40.63 |              28.80 |
|       1 |            88 |             2844.16 |               0.16 |                            8.53 |               12.05 |                        3.76 |                        5.91 |                         0.06 |               0.00 |              30.37 |              40.36 |              42.01 |              43.28 |              30.48 |
|       1 |            96 |             2877.00 |               0.19 |                           10.35 |               12.48 |                        3.91 |                        5.96 |                         0.07 |               0.00 |              33.47 |              43.26 |              45.26 |              47.61 |              32.95 |
|       1 |           104 |             2918.00 |               0.20 |                           11.17 |               12.95 |                        4.27 |                        6.48 |                         0.07 |               0.00 |              36.44 |              43.56 |              45.20 |              50.37 |              35.14 |
|       1 |           112 |             2977.02 |               0.18 |                           10.92 |               14.21 |                        4.77 |                        6.76 |                         0.08 |               0.00 |              37.34 |              46.95 |              49.44 |              51.85 |              36.92 |
|       1 |           120 |             3196.00 |               0.20 |                            8.79 |               16.46 |                        4.82 |                        6.85 |                         0.08 |               0.00 |              38.52 |              45.54 |              48.42 |              50.26 |              37.20 |
|       1 |           128 |             3118.00 |               0.21 |                           11.73 |               15.55 |                        5.31 |                        7.13 |                         0.09 |               0.00 |              40.85 |              51.22 |              53.00 |              55.95 |              40.02 |
|       1 |           136 |             3167.00 |               0.22 |                           12.38 |               16.21 |                        5.43 |                        7.62 |                         0.09 |               0.00 |              43.41 |              54.40 |              56.76 |              60.66 |              41.95 |
|       1 |           144 |             3273.00 |               0.24 |                           10.17 |               19.19 |                        5.83 |                        7.79 |                         0.10 |               0.00 |              42.21 |              57.11 |              61.41 |              68.18 |              43.32 |
|       1 |           152 |             3283.00 |               0.21 |                           14.04 |               17.10 |                        6.13 |                        7.93 |                         0.10 |               0.00 |              47.00 |              56.20 |              58.60 |              62.74 |              45.52 |
|       1 |           160 |             3269.00 |               0.22 |                           13.18 |               19.16 |                        6.89 |                        8.30 |                         0.12 |               0.00 |              48.07 |              59.74 |              64.15 |              70.49 |              47.87 |
|       1 |           168 |             3247.00 |               0.22 |                           15.60 |               18.64 |                        7.14 |                        8.53 |                         0.12 |               0.00 |              52.14 |              63.73 |              67.51 |              71.19 |              50.25 |
|       1 |           176 |             3468.00 |               0.26 |                           11.81 |               21.98 |                        7.03 |                        8.75 |                         0.12 |               0.00 |              50.83 |              64.12 |              66.47 |              68.26 |              49.95 |
|       1 |           184 |             3297.00 |               0.26 |                           13.90 |               21.98 |                        8.16 |                        9.74 |                         0.14 |               0.00 |              55.11 |              68.09 |              70.53 |              76.24 |              54.18 |
|       1 |           192 |             3376.00 |               0.21 |                           18.13 |               21.16 |                        7.15 |                        8.54 |                         0.12 |               0.00 |              56.58 |              70.31 |              72.58 |              76.07 |              55.31 |
|       1 |           200 |             3307.00 |               0.23 |                           20.58 |               20.21 |                        8.05 |                        9.56 |                         0.14 |               0.00 |              59.98 |              70.56 |              72.69 |              83.93 |              58.77 |
|       1 |           208 |             3489.00 |               0.34 |                           12.51 |               26.31 |                        8.35 |                        9.85 |                         0.13 |               0.00 |              57.90 |              71.05 |              73.68 |              82.01 |              57.50 |
|       1 |           216 |             3384.00 |               0.27 |                           19.95 |               23.24 |                        8.31 |                        9.95 |                         0.13 |               0.00 |              63.55 |              76.54 |              81.43 |              85.78 |              61.85 |
|       1 |           224 |             3627.00 |               0.40 |                           11.76 |               29.08 |                        8.62 |                       10.10 |                         0.14 |               0.00 |              59.27 |              74.63 |              77.66 |              84.30 |              60.10 |
|       1 |           232 |             3539.00 |               0.29 |                           17.64 |               27.73 |                        8.11 |                        9.53 |                         0.13 |               0.00 |              65.13 |              77.27 |              79.50 |              84.71 |              63.43 |
|       1 |           240 |             3654.35 |               0.41 |                           13.24 |               30.22 |                        8.77 |                       10.27 |                         0.15 |               0.00 |              63.00 |              76.36 |              77.15 |              85.25 |              63.04 |
|       1 |           248 |             3528.00 |               0.39 |                           17.15 |               29.76 |                        8.99 |                       10.86 |                         0.14 |               0.00 |              69.04 |              84.26 |              89.40 |              93.28 |              67.29 |
|       1 |           256 |             3670.00 |               0.41 |                           15.51 |               31.64 |                        9.13 |                       10.71 |                         0.15 |               0.00 |              68.76 |              81.96 |              85.84 |              96.11 |              67.54 |

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
|       1 |             8 |             1390.00 |               0.06 |                            0.53 |                3.46 |                        0.18 |                        1.50 |                         0.01 |               0.00 |               5.62 |               6.44 |               6.51 |               6.76 |               5.73 |
|       1 |            16 |             2053.00 |               0.07 |                            1.36 |                4.03 |                        0.52 |                        1.77 |                         0.01 |               0.00 |               7.68 |               9.83 |              10.48 |              10.94 |               7.76 |
|       1 |            24 |             2427.00 |               0.08 |                            2.72 |                4.25 |                        0.83 |                        1.96 |                         0.02 |               0.00 |              10.19 |              13.06 |              14.62 |              15.84 |               9.86 |
|       1 |            32 |             2756.00 |               0.07 |                            3.85 |                4.44 |                        1.04 |                        2.14 |                         0.03 |               0.00 |              12.83 |              15.00 |              15.61 |              16.31 |              11.57 |
|       1 |            40 |             3260.00 |               0.10 |                            3.33 |                5.11 |                        1.31 |                        2.32 |                         0.04 |               0.00 |              12.20 |              15.97 |              16.93 |              18.58 |              12.20 |
|       1 |            48 |             3225.00 |               0.08 |                            5.10 |                5.37 |                        1.80 |                        2.38 |                         0.04 |               0.00 |              15.72 |              19.04 |              19.68 |              20.52 |              14.78 |
|       1 |            56 |             3621.00 |               0.09 |                            5.13 |                5.68 |                        1.79 |                        2.58 |                         0.05 |               0.00 |              16.25 |              19.91 |              20.70 |              22.67 |              15.33 |
|       1 |            64 |             3773.00 |               0.09 |                            6.06 |                5.97 |                        1.96 |                        2.69 |                         0.06 |               0.00 |              17.54 |              21.53 |              22.66 |              24.17 |              16.83 |
|       1 |            72 |             3882.00 |               0.08 |                            6.52 |                6.56 |                        2.24 |                        2.89 |                         0.06 |               0.00 |              19.67 |              23.71 |              24.35 |              25.93 |              18.37 |
|       1 |            80 |             3915.08 |               0.10 |                            6.86 |                7.28 |                        2.99 |                        2.98 |                         0.08 |               0.00 |              20.97 |              26.24 |              27.34 |              29.06 |              20.27 |
|       1 |            88 |             4185.00 |               0.08 |                            7.07 |                7.44 |                        3.03 |                        3.19 |                         0.08 |               0.00 |              21.56 |              27.19 |              28.56 |              30.53 |              20.89 |
|       1 |            96 |             4272.00 |               0.09 |                            8.49 |                7.49 |                        2.90 |                        3.17 |                         0.08 |               0.00 |              23.48 |              28.97 |              30.30 |              33.49 |              22.22 |
|       1 |           104 |             4458.54 |               0.08 |                            8.21 |                8.26 |                        3.12 |                        3.26 |                         0.09 |               0.00 |              23.85 |              29.61 |              31.51 |              33.47 |              23.03 |
|       1 |           112 |             4509.00 |               0.08 |                            9.67 |                8.36 |                        3.11 |                        3.20 |                         0.08 |               0.00 |              25.14 |              31.42 |              33.62 |              36.90 |              24.51 |
|       1 |           120 |             4820.00 |               0.10 |                            7.31 |                9.57 |                        3.66 |                        3.75 |                         0.11 |               0.00 |              25.25 |              30.72 |              32.59 |              35.84 |              24.49 |
|       1 |           128 |             4757.00 |               0.08 |                            8.82 |                9.27 |                        4.11 |                        4.00 |                         0.12 |               0.00 |              26.85 |              35.22 |              36.51 |              37.96 |              26.40 |
|       1 |           136 |             5263.00 |               0.10 |                            7.77 |                9.78 |                        3.24 |                        4.18 |                         0.12 |               0.00 |              26.05 |              31.36 |              33.10 |              34.59 |              25.19 |
|       1 |           144 |             5287.00 |               0.08 |                            8.92 |                9.49 |                        3.86 |                        4.32 |                         0.13 |               0.00 |              27.34 |              33.56 |              34.74 |              36.51 |              26.80 |
|       1 |           152 |             5420.00 |               0.08 |                            8.56 |               10.56 |                        3.93 |                        4.45 |                         0.14 |               0.00 |              28.50 |              34.48 |              36.21 |              40.40 |              27.71 |
|       1 |           160 |             5507.00 |               0.09 |                            8.38 |               11.39 |                        4.04 |                        4.58 |                         0.14 |               0.00 |              29.60 |              35.95 |              37.01 |              42.02 |              28.61 |
|       1 |           168 |             5471.00 |               0.10 |                            9.22 |               11.55 |                        4.52 |                        4.63 |                         0.14 |               0.00 |              30.51 |              38.58 |              41.25 |              43.11 |              30.16 |
|       1 |           176 |             5693.00 |               0.09 |                            9.92 |               11.22 |                        4.38 |                        4.68 |                         0.14 |               0.00 |              31.22 |              38.24 |              39.42 |              43.07 |              30.44 |
|       1 |           184 |             5698.00 |               0.10 |                            8.64 |               13.26 |                        4.63 |                        4.90 |                         0.15 |               0.00 |              32.38 |              39.84 |              41.38 |              43.17 |              31.68 |
|       1 |           192 |             5591.00 |               0.09 |                           11.84 |               12.04 |                        4.66 |                        4.95 |                         0.15 |               0.00 |              35.15 |              42.57 |              44.21 |              59.20 |              33.74 |
|       1 |           200 |             5973.00 |               0.12 |                            7.94 |               14.59 |                        4.95 |                        5.19 |                         0.16 |               0.00 |              33.52 |              40.18 |              42.13 |              44.74 |              32.94 |
|       1 |           208 |             5981.00 |               0.09 |                            9.48 |               14.28 |                        4.98 |                        5.00 |                         0.16 |               0.00 |              34.69 |              40.97 |              42.65 |              46.89 |              33.99 |
|       1 |           216 |             5901.00 |               0.10 |                           12.20 |               12.71 |                        5.48 |                        5.40 |                         0.17 |               0.00 |              37.42 |              44.25 |              46.57 |              49.53 |              36.07 |
|       1 |           224 |             6061.00 |               0.11 |                           10.02 |               15.27 |                        5.35 |                        5.28 |                         0.17 |               0.00 |              36.23 |              44.87 |              46.34 |              50.26 |              36.19 |
|       1 |           232 |             6030.00 |               0.09 |                           11.08 |               15.65 |                        5.25 |                        5.52 |                         0.17 |               0.00 |              38.51 |              44.80 |              48.30 |              51.88 |              37.76 |
|       1 |           240 |             6253.00 |               0.12 |                            9.52 |               17.03 |                        5.03 |                        5.54 |                         0.17 |               0.00 |              37.63 |              45.81 |              48.29 |              52.75 |              37.41 |
|       1 |           248 |             6363.00 |               0.09 |                           11.33 |               15.20 |                        5.29 |                        6.02 |                         0.17 |               0.00 |              38.93 |              46.29 |              48.27 |              51.93 |              38.11 |
|       1 |           256 |             6614.00 |               0.10 |                           11.09 |               16.08 |                        4.89 |                        5.68 |                         0.16 |               0.00 |              38.74 |              46.94 |              48.30 |              50.82 |              38.00 |

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
Please download a checkpoint from [here](https://api.ngc.nvidia.com/v2/models/nvidia/dle/gpunet_1_pyt_ckpt/versions/21.12.0_amp/zip) 
and place it in `runner_workspace/checkpoints/0.85ms/`.  Note that the `0.85ms` subdirectory may not be created yet.

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
export CHECKPOINT="0.85ms"
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
    --config /workspace/gpunet/configs/batch1/GV100/0.85ms.json \
    --checkpoint ${CHECKPOINT_DIR}/0.85ms.pth.tar \
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
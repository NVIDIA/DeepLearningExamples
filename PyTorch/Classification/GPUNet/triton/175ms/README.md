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
NVIDIA DGX-1 (1x V100 32GB): ./triton/175ms/runner/start_NVIDIA-DGX-1-\(1x-V100-32GB\).sh

NVIDIA DGX A100 (1x A100 80GB): ./triton/175ms/runner/start_NVIDIA-DGX-A100-\(1x-A100-80GB\).sh
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
|       1 |             1 |              450.00 |               0.05 |                            0.22 |                0.08 |                        0.18 |                        1.68 |                         0.01 |               0.00 |               2.21 |               2.24 |               2.26 |               2.30 |               2.22 |
|       2 |             1 |              720.00 |               0.05 |                            0.22 |                0.08 |                        0.27 |                        2.14 |                         0.01 |               0.00 |               2.77 |               2.80 |               2.83 |               2.88 |               2.77 |
|       4 |             1 |             1008.00 |               0.05 |                            0.23 |                0.08 |                        0.46 |                        3.14 |                         0.01 |               0.00 |               3.96 |               4.01 |               4.03 |               4.09 |               3.96 |
|       8 |             1 |             1320.00 |               0.05 |                            0.20 |                0.05 |                        0.80 |                        4.94 |                         0.01 |               0.00 |               6.07 |               6.12 |               6.14 |               6.16 |               6.05 |
|      16 |             1 |             1536.00 |               0.05 |                            0.22 |                0.08 |                        1.44 |                        8.54 |                         0.01 |               0.00 |              10.33 |              10.38 |              10.38 |              10.41 |              10.33 |
|      32 |             1 |             1664.00 |               0.05 |                            0.22 |                0.08 |                        2.76 |                       15.88 |                         0.02 |               0.00 |              19.01 |              19.05 |              19.07 |              19.14 |              19.01 |
|      64 |             1 |             1728.00 |               0.05 |                            0.28 |                0.08 |                        5.79 |                       29.59 |                         0.03 |               0.00 |              35.84 |              35.94 |              36.02 |              36.11 |              35.83 |

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
|       1 |             1 |              679.00 |               0.02 |                            0.06 |                0.02 |                        0.12 |                        1.25 |                         0.00 |               0.00 |               1.48 |               1.49 |               1.49 |               1.53 |               1.47 |
|       2 |             1 |             1164.00 |               0.02 |                            0.06 |                0.02 |                        0.17 |                        1.44 |                         0.00 |               0.00 |               1.72 |               1.74 |               1.75 |               1.79 |               1.72 |
|       4 |             1 |             1736.00 |               0.03 |                            0.07 |                0.02 |                        0.28 |                        1.89 |                         0.00 |               0.00 |               2.29 |               2.38 |               2.40 |               2.43 |               2.30 |
|       8 |             1 |             2320.00 |               0.03 |                            0.08 |                0.02 |                        0.52 |                        2.79 |                         0.00 |               0.00 |               3.45 |               3.49 |               3.50 |               3.51 |               3.44 |
|      16 |             1 |             2640.00 |               0.03 |                            0.11 |                0.02 |                        1.25 |                        4.59 |                         0.01 |               0.00 |               6.01 |               6.11 |               6.13 |               6.24 |               6.02 |
|      32 |             1 |             2880.00 |               0.03 |                            0.19 |                0.03 |                        2.77 |                        8.04 |                         0.02 |               0.00 |              11.08 |              11.14 |              11.19 |              11.20 |              11.09 |
|      64 |             1 |             3072.00 |               0.03 |                            0.20 |                0.03 |                        5.49 |                       14.76 |                         0.03 |               0.00 |              20.50 |              20.59 |              20.64 |              21.48 |              20.54 |

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
|       1 |             8 |              463.00 |               0.12 |                            1.16 |               10.98 |                        0.26 |                        4.68 |                         0.01 |               0.00 |              17.12 |              18.64 |              19.30 |              20.15 |              17.21 |
|       1 |            16 |              670.00 |               0.16 |                            2.76 |               13.21 |                        0.76 |                        6.62 |                         0.02 |               0.00 |              25.05 |              26.14 |              26.40 |              27.35 |              23.53 |
|       1 |            24 |              838.00 |               0.19 |                            4.46 |               15.00 |                        1.26 |                        7.33 |                         0.02 |               0.00 |              30.21 |              30.91 |              31.23 |              33.42 |              28.26 |
|       1 |            32 |              906.00 |               0.22 |                            6.16 |               16.01 |                        2.55 |                        9.59 |                         0.03 |               0.00 |              36.47 |              39.98 |              40.78 |              46.44 |              34.57 |
|       1 |            40 |              949.00 |               0.21 |                            9.12 |               18.35 |                        3.13 |                       10.57 |                         0.04 |               0.00 |              43.66 |              50.86 |              51.64 |              56.49 |              41.41 |
|       1 |            48 |             1053.00 |               0.22 |                            8.29 |               21.50 |                        3.56 |                       10.74 |                         0.04 |               0.00 |              46.24 |              51.10 |              53.02 |              54.33 |              44.37 |
|       1 |            56 |             1037.00 |               0.26 |                           13.87 |               21.57 |                        4.19 |                       12.51 |                         0.05 |               0.00 |              54.20 |              68.42 |              71.18 |              76.82 |              52.45 |
|       1 |            64 |             1146.00 |               0.24 |                           10.20 |               25.70 |                        4.64 |                       13.37 |                         0.06 |               0.00 |              55.59 |              65.23 |              66.42 |              74.42 |              54.21 |
|       1 |            72 |             1117.00 |               0.26 |                           14.73 |               24.72 |                        6.95 |                       15.89 |                         0.08 |               0.00 |              70.17 |              78.56 |              79.46 |              80.07 |              62.63 |
|       1 |            80 |             1068.00 |               0.27 |                           20.57 |               24.80 |                        9.11 |                       17.19 |                         0.08 |               0.00 |              83.16 |              94.21 |              95.05 |             104.18 |              72.03 |
|       1 |            88 |             1202.00 |               0.27 |                           17.69 |               28.71 |                        7.16 |                       17.24 |                         0.08 |               0.00 |              72.00 |              89.29 |              97.32 |             112.86 |              71.15 |
|       1 |            96 |             1222.00 |               0.27 |                           19.24 |               29.13 |                        8.20 |                       18.05 |                         0.09 |               0.00 |              79.70 |              94.74 |              99.06 |             112.32 |              74.97 |
|       1 |           104 |             1223.00 |               0.31 |                           17.00 |               33.40 |                        9.15 |                       20.36 |                         0.09 |               0.00 |              85.34 |             100.78 |             111.91 |             116.65 |              80.32 |
|       1 |           112 |             1284.72 |               0.30 |                           17.92 |               35.11 |                       10.01 |                       21.42 |                         0.10 |               0.00 |              84.71 |             110.02 |             114.83 |             120.93 |              84.86 |
|       1 |           120 |             1205.00 |               0.32 |                           20.18 |               36.48 |                       12.68 |                       24.79 |                         0.12 |               0.00 |             101.85 |             120.79 |             122.94 |             124.10 |              94.58 |
|       1 |           128 |             1358.00 |               0.36 |                           19.33 |               40.48 |                        9.30 |                       21.78 |                         0.12 |               0.00 |              91.34 |             111.79 |             117.33 |             119.85 |              91.37 |
|       1 |           136 |             1311.00 |               0.30 |                           19.90 |               40.81 |                       10.97 |                       24.24 |                         0.13 |               0.00 |              97.12 |             121.21 |             122.16 |             138.63 |              96.36 |
|       1 |           144 |             1316.00 |               0.33 |                           21.60 |               40.88 |                       13.16 |                       28.39 |                         0.16 |               0.00 |             113.62 |             131.17 |             136.02 |             138.50 |             104.53 |
|       1 |           152 |             1344.00 |               0.32 |                           21.58 |               46.75 |                       12.27 |                       25.42 |                         0.13 |               0.00 |             107.65 |             128.42 |             130.97 |             157.26 |             106.47 |
|       1 |           160 |             1346.00 |               0.32 |                           27.56 |               40.14 |                       14.34 |                       31.42 |                         0.16 |               0.00 |             131.24 |             145.92 |             146.20 |             146.55 |             113.94 |
|       1 |           168 |             1394.00 |               0.33 |                           22.48 |               52.94 |                       11.70 |                       26.70 |                         0.14 |               0.00 |             116.36 |             139.53 |             144.13 |             147.28 |             114.27 |
|       1 |           176 |             1283.00 |               0.46 |                           19.78 |               52.26 |                       16.86 |                       32.78 |                         0.17 |               0.00 |             120.30 |             162.20 |             162.90 |             164.82 |             122.30 |
|       1 |           184 |             1282.00 |               0.35 |                           33.05 |               47.57 |                       15.81 |                       30.73 |                         0.16 |               0.00 |             132.25 |             162.44 |             166.19 |             177.57 |             127.67 |
|       1 |           192 |             1384.62 |               0.42 |                           18.01 |               62.42 |                       14.52 |                       29.11 |                         0.15 |               0.00 |             127.38 |             153.04 |             158.47 |             175.99 |             124.64 |
|       1 |           200 |             1363.00 |               0.43 |                           23.70 |               59.96 |                       16.24 |                       33.37 |                         0.16 |               0.00 |             131.84 |             166.32 |             180.64 |             200.00 |             133.87 |
|       1 |           208 |             1401.00 |               0.49 |                           23.61 |               64.76 |                       15.95 |                       31.32 |                         0.17 |               0.00 |             138.53 |             157.89 |             160.36 |             199.73 |             136.30 |
|       1 |           216 |             1412.00 |               0.33 |                           33.12 |               61.57 |                       17.23 |                       32.44 |                         0.17 |               0.00 |             147.33 |             167.41 |             174.59 |             188.53 |             144.86 |
|       1 |           224 |             1386.00 |               0.39 |                           30.63 |               67.18 |                       15.08 |                       31.89 |                         0.16 |               0.00 |             145.52 |             176.12 |             183.03 |             212.18 |             145.33 |
|       1 |           232 |             1410.00 |               0.44 |                           27.54 |               73.21 |                       17.18 |                       32.10 |                         0.17 |               0.00 |             150.59 |             183.86 |             194.89 |             214.27 |             150.65 |
|       1 |           240 |             1428.00 |               0.47 |                           32.68 |               71.93 |                       16.27 |                       30.38 |                         0.17 |               0.00 |             152.66 |             178.22 |             181.79 |             182.79 |             151.91 |
|       1 |           248 |             1404.00 |               0.46 |                           35.46 |               75.09 |                       16.28 |                       32.17 |                         0.16 |               0.00 |             159.62 |             206.98 |             216.56 |             218.05 |             159.63 |
|       1 |           256 |             1356.00 |               0.72 |                           20.03 |               99.56 |                       20.05 |                       33.35 |                         0.17 |               0.00 |             193.38 |             201.67 |             202.06 |             202.46 |             173.88 |

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
|       1 |             8 |              814.00 |               0.11 |                            0.74 |                6.07 |                        0.23 |                        2.63 |                         0.01 |               0.00 |               9.63 |              11.29 |              11.77 |              12.12 |               9.78 |
|       1 |            16 |             1119.00 |               0.15 |                            2.98 |                6.25 |                        0.97 |                        3.88 |                         0.02 |               0.00 |              13.15 |              19.89 |              20.16 |              20.80 |              14.26 |
|       1 |            24 |             1272.00 |               0.17 |                            5.14 |                7.02 |                        1.44 |                        4.74 |                         0.04 |               0.00 |              17.71 |              28.49 |              28.75 |              30.49 |              18.55 |
|       1 |            32 |             1561.00 |               0.18 |                            4.72 |                9.26 |                        1.40 |                        4.82 |                         0.03 |               0.00 |              21.40 |              24.38 |              24.97 |              27.41 |              20.41 |
|       1 |            40 |             1595.00 |               0.13 |                            7.56 |                9.58 |                        1.90 |                        5.50 |                         0.05 |               0.00 |              27.03 |              32.66 |              33.99 |              37.28 |              24.72 |
|       1 |            48 |             1790.00 |               0.15 |                            7.37 |               10.12 |                        2.40 |                        6.31 |                         0.05 |               0.00 |              27.54 |              34.28 |              37.34 |              39.77 |              26.40 |
|       1 |            56 |             1904.00 |               0.17 |                            8.47 |               11.34 |                        2.50 |                        6.59 |                         0.05 |               0.00 |              30.16 |              36.48 |              38.73 |              46.14 |              29.12 |
|       1 |            64 |             1948.00 |               0.16 |                           10.50 |               10.93 |                        3.09 |                        7.57 |                         0.06 |               0.00 |              34.87 |              39.38 |              41.02 |              43.31 |              32.30 |
|       1 |            72 |             1921.00 |               0.22 |                           10.51 |               13.95 |                        3.97 |                        7.84 |                         0.07 |               0.00 |              38.30 |              47.12 |              47.94 |              52.26 |              36.56 |
|       1 |            80 |             1992.00 |               0.15 |                           12.90 |               13.38 |                        4.49 |                        8.48 |                         0.08 |               0.00 |              41.92 |              48.74 |              50.64 |              53.21 |              39.48 |
|       1 |            88 |             2015.00 |               0.20 |                           13.38 |               14.43 |                        5.49 |                        9.14 |                         0.09 |               0.00 |              44.11 |              61.05 |              64.80 |              72.96 |              42.72 |
|       1 |            96 |             2155.00 |               0.26 |                           13.70 |               14.67 |                        5.15 |                        9.82 |                         0.09 |               0.00 |              44.64 |              57.22 |              61.18 |              63.20 |              43.69 |
|       1 |           104 |             2222.78 |               0.22 |                           12.69 |               16.85 |                        5.58 |                       10.42 |                         0.10 |               0.00 |              47.78 |              58.21 |              62.93 |              70.73 |              45.85 |
|       1 |           112 |             2229.00 |               0.20 |                           17.51 |               14.78 |                        5.73 |                       10.63 |                         0.10 |               0.00 |              51.75 |              59.36 |              62.31 |              69.00 |              48.95 |
|       1 |           120 |             2323.68 |               0.21 |                           16.15 |               17.07 |                        5.93 |                       10.73 |                         0.10 |               0.00 |              52.25 |              61.82 |              64.02 |              67.84 |              50.20 |
|       1 |           128 |             2302.00 |               0.18 |                           16.02 |               18.20 |                        7.42 |                       12.04 |                         0.13 |               0.00 |              58.55 |              71.09 |              72.66 |              75.42 |              54.00 |
|       1 |           136 |             2403.00 |               0.21 |                           16.19 |               20.46 |                        6.30 |                       11.58 |                         0.10 |               0.00 |              56.70 |              68.51 |              70.63 |              76.49 |              54.85 |
|       1 |           144 |             2340.00 |               0.16 |                           21.65 |               18.10 |                        7.29 |                       12.24 |                         0.12 |               0.00 |              63.71 |              76.17 |              77.64 |              81.72 |              59.57 |
|       1 |           152 |             2365.00 |               0.36 |                           12.02 |               25.57 |                        9.20 |                       16.00 |                         0.17 |               0.00 |              72.96 |              74.10 |              74.33 |              75.89 |              63.32 |
|       1 |           160 |             2389.00 |               0.16 |                           20.48 |               21.98 |                        8.46 |                       13.62 |                         0.14 |               0.00 |              67.97 |              84.11 |              86.19 |              87.81 |              64.83 |
|       1 |           168 |             2508.49 |               0.18 |                           14.93 |               27.00 |                        8.85 |                       14.51 |                         0.14 |               0.00 |              66.48 |              81.76 |              86.21 |              90.01 |              65.61 |
|       1 |           176 |             2448.00 |               0.20 |                           16.68 |               28.23 |                        8.67 |                       14.30 |                         0.14 |               0.00 |              71.33 |              84.01 |              84.99 |              92.62 |              68.22 |
|       1 |           184 |             2502.00 |               0.20 |                           14.14 |               31.41 |                        8.82 |                       15.39 |                         0.15 |               0.00 |              72.39 |              93.16 |              97.28 |             101.08 |              70.11 |
|       1 |           192 |             2494.00 |               0.15 |                           18.25 |               30.75 |                        9.50 |                       15.34 |                         0.18 |               0.00 |              76.37 |              90.09 |             100.47 |             121.06 |              74.17 |
|       1 |           200 |             2516.00 |               0.14 |                           20.15 |               28.80 |                        9.70 |                       16.65 |                         0.16 |               0.00 |              78.82 |              92.92 |              96.43 |             106.64 |              75.60 |
|       1 |           208 |             2665.00 |               0.18 |                           16.11 |               34.07 |                        8.78 |                       14.85 |                         0.15 |               0.00 |              74.73 |              89.39 |              96.32 |             102.39 |              74.14 |
|       1 |           216 |             2624.00 |               0.13 |                           23.95 |               29.77 |                        8.81 |                       16.23 |                         0.16 |               0.00 |              80.98 |              94.58 |             100.95 |             107.04 |              79.05 |
|       1 |           224 |             2671.33 |               0.15 |                           22.46 |               33.23 |                        9.14 |                       16.35 |                         0.16 |               0.00 |              83.92 |              98.92 |             103.15 |             110.30 |              81.49 |
|       1 |           232 |             2675.00 |               0.24 |                           17.74 |               38.86 |                       10.33 |                       16.62 |                         0.16 |               0.00 |              86.56 |             103.55 |             106.26 |             109.90 |              83.94 |
|       1 |           240 |             2725.00 |               0.16 |                           19.73 |               38.43 |                        9.74 |                       16.69 |                         0.17 |               0.00 |              85.56 |             102.57 |             105.98 |             110.88 |              84.92 |
|       1 |           248 |             2822.00 |               0.29 |                           14.35 |               44.63 |                        8.61 |                       16.73 |                         0.14 |               0.00 |              84.43 |             104.72 |             105.87 |             122.66 |              84.77 |
|       1 |           256 |             2852.00 |               0.28 |                           16.62 |               44.71 |                        8.40 |                       16.07 |                         0.14 |               0.00 |              89.04 |             102.50 |             106.28 |             113.40 |              86.23 |

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
Please download a checkpoint from [here](https://api.ngc.nvidia.com/v2/models/nvidia/dle/gpunet_2_pyt_ckpt/versions/21.12.0_amp/zip) 
and place it in `runner_workspace/checkpoints/1.75ms/`.  Note that the `1.75ms` subdirectory may not be created yet.

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
export CHECKPOINT="1.75ms"
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
    --config /workspace/gpunet/configs/batch1/GV100/1.75ms.json \
    --checkpoint ${CHECKPOINT_DIR}/1.75ms.pth.tar \
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
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
NVIDIA DGX-1 (1x V100 32GB): ./triton/08ms-D/runner/start_NVIDIA-DGX-1-\(1x-V100-32GB\).sh

NVIDIA DGX A100 (1x A100 80GB): ./triton/08ms-D/runner/start_NVIDIA-DGX-A100-\(1x-A100-80GB\).sh
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
|       1 |             1 |              733.00 |               0.04 |                            0.19 |                0.07 |                        0.09 |                        0.96 |                         0.01 |               0.00 |               1.36 |               1.39 |               1.40 |               1.43 |               1.36 |
|       2 |             1 |             1294.00 |               0.04 |                            0.18 |                0.03 |                        0.13 |                        1.16 |                         0.01 |               0.00 |               1.54 |               1.57 |               1.58 |               1.61 |               1.54 |
|       4 |             1 |             1764.00 |               0.05 |                            0.22 |                0.08 |                        0.22 |                        1.68 |                         0.01 |               0.00 |               2.26 |               2.31 |               2.32 |               2.37 |               2.26 |
|       8 |             1 |             2408.00 |               0.05 |                            0.22 |                0.08 |                        0.34 |                        2.62 |                         0.01 |               0.00 |               3.31 |               3.35 |               3.37 |               3.42 |               3.31 |
|      16 |             1 |             3056.00 |               0.04 |                            0.20 |                0.06 |                        0.58 |                        4.32 |                         0.01 |               0.00 |               5.24 |               5.30 |               5.31 |               5.35 |               5.21 |
|      32 |             1 |             3520.00 |               0.05 |                            0.20 |                0.06 |                        1.04 |                        7.67 |                         0.02 |               0.00 |               9.04 |               9.11 |               9.13 |               9.16 |               9.03 |
|      64 |             1 |             3840.00 |               0.05 |                            0.22 |                0.08 |                        1.91 |                       14.34 |                         0.02 |               0.00 |              16.62 |              16.67 |              16.67 |              16.69 |              16.62 |

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
|       1 |             1 |             1292.00 |               0.02 |                            0.07 |                0.02 |                        0.05 |                        0.61 |                         0.00 |               0.00 |               0.77 |               0.78 |               0.78 |               0.78 |               0.77 |
|       2 |             1 |             2212.00 |               0.02 |                            0.06 |                0.02 |                        0.09 |                        0.72 |                         0.00 |               0.00 |               0.90 |               0.91 |               0.91 |               0.92 |               0.90 |
|       4 |             1 |             3256.00 |               0.03 |                            0.09 |                0.02 |                        0.14 |                        0.94 |                         0.00 |               0.00 |               1.23 |               1.24 |               1.24 |               1.29 |               1.22 |
|       8 |             1 |             4480.00 |               0.03 |                            0.08 |                0.02 |                        0.21 |                        1.43 |                         0.01 |               0.00 |               1.77 |               1.84 |               1.88 |               2.09 |               1.78 |
|      16 |             1 |             5626.37 |               0.03 |                            0.08 |                0.03 |                        0.36 |                        2.34 |                         0.01 |               0.00 |               2.83 |               2.90 |               2.92 |               3.05 |               2.84 |
|      32 |             1 |             6112.00 |               0.02 |                            0.08 |                0.02 |                        0.76 |                        4.31 |                         0.02 |               0.00 |               5.21 |               5.25 |               5.28 |               5.31 |               5.21 |
|      64 |             1 |             6208.00 |               0.02 |                            0.10 |                0.03 |                        1.88 |                        8.20 |                         0.03 |               0.00 |              10.26 |              10.31 |              10.32 |              10.34 |              10.26 |

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
|       1 |             8 |              743.00 |               0.08 |                            0.57 |                7.01 |                        0.14 |                        2.91 |                         0.01 |               0.00 |              10.70 |              10.87 |              10.92 |              11.21 |              10.72 |
|       1 |            16 |             1094.00 |               0.09 |                            1.13 |                8.66 |                        0.42 |                        4.26 |                         0.02 |               0.00 |              14.81 |              16.51 |              16.73 |              18.27 |              14.58 |
|       1 |            24 |             1413.00 |               0.09 |                            2.12 |                8.76 |                        0.68 |                        5.18 |                         0.02 |               0.00 |              17.36 |              18.89 |              19.21 |              20.27 |              16.85 |
|       1 |            32 |             1632.00 |               0.10 |                            2.79 |               10.27 |                        0.81 |                        5.52 |                         0.02 |               0.00 |              19.84 |              21.90 |              22.86 |              24.68 |              19.50 |
|       1 |            40 |             1778.00 |               0.10 |                            3.24 |               11.72 |                        1.09 |                        6.14 |                         0.03 |               0.00 |              22.68 |              26.46 |              27.43 |              28.98 |              22.31 |
|       1 |            48 |             1995.00 |               0.10 |                            2.81 |               13.28 |                        1.18 |                        6.36 |                         0.03 |               0.00 |              24.32 |              25.87 |              26.44 |              27.47 |              23.75 |
|       1 |            56 |             2141.00 |               0.12 |                            3.01 |               14.99 |                        1.22 |                        6.56 |                         0.03 |               0.00 |              26.14 |              28.29 |              28.92 |              30.32 |              25.92 |
|       1 |            64 |             2230.00 |               0.12 |                            4.16 |               15.10 |                        1.56 |                        7.32 |                         0.04 |               0.00 |              28.69 |              32.04 |              32.57 |              33.79 |              28.29 |
|       1 |            72 |             2274.00 |               0.12 |                            4.71 |               15.76 |                        2.08 |                        8.42 |                         0.05 |               0.00 |              31.37 |              35.62 |              36.51 |              43.12 |              31.13 |
|       1 |            80 |             2387.00 |               0.13 |                            5.12 |               15.77 |                        2.39 |                        9.44 |                         0.06 |               0.00 |              33.39 |              38.13 |              39.92 |              42.30 |              32.91 |
|       1 |            88 |             2353.00 |               0.14 |                            6.48 |               17.23 |                        2.76 |                        9.67 |                         0.06 |               0.00 |              36.79 |              44.02 |              45.41 |              53.34 |              36.34 |
|       1 |            96 |             2537.00 |               0.14 |                            6.26 |               18.43 |                        2.64 |                        9.73 |                         0.06 |               0.00 |              37.48 |              43.40 |              44.76 |              45.97 |              37.26 |
|       1 |           104 |             2568.00 |               0.14 |                            5.83 |               20.78 |                        2.77 |                       10.07 |                         0.06 |               0.00 |              40.65 |              45.18 |              46.23 |              48.59 |              39.65 |
|       1 |           112 |             2585.00 |               0.14 |                            5.98 |               20.90 |                        3.27 |                       11.57 |                         0.07 |               0.00 |              42.58 |              48.80 |              51.80 |              54.90 |              41.93 |
|       1 |           120 |             2631.00 |               0.14 |                            6.71 |               21.88 |                        3.86 |                       11.77 |                         0.08 |               0.00 |              45.05 |              52.94 |              60.38 |              69.36 |              44.43 |
|       1 |           128 |             2724.00 |               0.16 |                            5.65 |               24.63 |                        3.36 |                       11.91 |                         0.07 |               0.00 |              45.78 |              52.47 |              53.57 |              61.43 |              45.79 |
|       1 |           136 |             2693.00 |               0.14 |                            9.29 |               22.06 |                        3.95 |                       13.60 |                         0.09 |               0.00 |              49.17 |              60.32 |              62.49 |              66.88 |              49.14 |
|       1 |           144 |             2892.00 |               0.17 |                            5.93 |               26.43 |                        3.48 |                       12.66 |                         0.08 |               0.00 |              49.18 |              56.22 |              57.22 |              61.44 |              48.75 |
|       1 |           152 |             2839.00 |               0.16 |                            8.29 |               26.26 |                        4.08 |                       13.13 |                         0.09 |               0.00 |              52.98 |              60.49 |              62.68 |              72.28 |              52.00 |
|       1 |           160 |             2887.00 |               0.18 |                            7.89 |               28.05 |                        4.34 |                       14.11 |                         0.10 |               0.00 |              55.34 |              63.33 |              66.00 |              70.22 |              54.67 |
|       1 |           168 |             2872.00 |               0.19 |                            9.51 |               27.67 |                        4.82 |                       14.32 |                         0.11 |               0.00 |              56.78 |              67.60 |              71.42 |              75.32 |              56.61 |
|       1 |           176 |             2755.24 |               0.16 |                           14.71 |               25.67 |                        5.44 |                       16.07 |                         0.12 |               0.00 |              65.88 |              76.09 |              77.14 |              84.01 |              62.18 |
|       1 |           184 |             2965.00 |               0.18 |                           10.89 |               28.68 |                        5.11 |                       15.20 |                         0.11 |               0.00 |              61.29 |              70.30 |              74.27 |              86.00 |              60.18 |
|       1 |           192 |             2923.00 |               0.23 |                            9.41 |               31.15 |                        5.15 |                       16.42 |                         0.13 |               0.00 |              61.35 |              72.51 |              74.64 |              77.50 |              62.50 |
|       1 |           200 |             2993.00 |               0.24 |                            8.40 |               35.80 |                        4.46 |                       15.25 |                         0.11 |               0.00 |              64.70 |              75.36 |              77.55 |              80.57 |              64.25 |
|       1 |           208 |             2962.00 |               0.19 |                           12.76 |               31.13 |                        5.99 |                       17.94 |                         0.14 |               0.00 |              70.46 |              78.12 |              78.63 |              80.08 |              68.15 |
|       1 |           216 |             2928.00 |               0.21 |                           11.81 |               35.19 |                        5.42 |                       16.50 |                         0.12 |               0.00 |              68.29 |              81.61 |              89.14 |              90.65 |              69.25 |
|       1 |           224 |             3216.00 |               0.30 |                            8.31 |               38.26 |                        4.69 |                       15.83 |                         0.11 |               0.00 |              67.84 |              80.80 |              83.85 |              87.46 |              67.50 |
|       1 |           232 |             3143.00 |               0.26 |                           10.96 |               37.60 |                        5.24 |                       16.55 |                         0.12 |               0.00 |              70.91 |              81.65 |              83.04 |              90.01 |              70.74 |
|       1 |           240 |             3106.00 |               0.40 |                            8.01 |               43.70 |                        5.29 |                       16.70 |                         0.13 |               0.00 |              75.05 |              84.97 |              88.22 |              91.28 |              74.23 |
|       1 |           248 |             3149.00 |               0.39 |                            8.78 |               42.48 |                        5.44 |                       17.21 |                         0.13 |               0.00 |              74.10 |              84.51 |              87.22 |              94.99 |              74.42 |
|       1 |           256 |             3113.00 |               0.43 |                            8.75 |               44.77 |                        5.95 |                       18.12 |                         0.14 |               0.00 |              77.60 |              93.04 |              95.93 |              97.40 |              78.15 |

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
|       1 |             8 |             1105.00 |               0.05 |                            0.32 |                4.74 |                        0.10 |                        2.00 |                         0.01 |               0.00 |               7.22 |               7.49 |               7.87 |               8.51 |               7.21 |
|       1 |            16 |             1655.00 |               0.07 |                            0.84 |                5.52 |                        0.29 |                        2.89 |                         0.02 |               0.00 |               9.71 |              10.96 |              11.41 |              12.10 |               9.62 |
|       1 |            24 |             2154.00 |               0.06 |                            1.27 |                6.09 |                        0.39 |                        3.25 |                         0.02 |               0.00 |              11.13 |              13.06 |              13.52 |              14.53 |              11.08 |
|       1 |            32 |             2557.00 |               0.07 |                            1.72 |                6.47 |                        0.55 |                        3.59 |                         0.02 |               0.00 |              12.79 |              14.18 |              14.43 |              14.98 |              12.41 |
|       1 |            40 |             2788.00 |               0.07 |                            2.49 |                7.07 |                        0.72 |                        3.84 |                         0.03 |               0.00 |              14.68 |              16.54 |              16.97 |              17.74 |              14.22 |
|       1 |            48 |             3068.00 |               0.08 |                            2.35 |                8.07 |                        0.84 |                        4.11 |                         0.04 |               0.00 |              15.57 |              18.06 |              18.95 |              20.37 |              15.48 |
|       1 |            56 |             3187.00 |               0.06 |                            3.36 |                8.51 |                        1.02 |                        4.39 |                         0.05 |               0.00 |              17.44 |              21.31 |              22.22 |              25.83 |              17.40 |
|       1 |            64 |             3475.00 |               0.07 |                            3.27 |                9.14 |                        1.07 |                        4.66 |                         0.05 |               0.00 |              18.50 |              21.85 |              23.86 |              26.03 |              18.28 |
|       1 |            72 |             3716.00 |               0.06 |                            3.54 |                9.64 |                        1.11 |                        4.84 |                         0.06 |               0.00 |              19.36 |              22.56 |              24.05 |              25.80 |              19.24 |
|       1 |            80 |             3823.18 |               0.07 |                            4.12 |                9.68 |                        1.33 |                        5.51 |                         0.08 |               0.00 |              20.93 |              25.36 |              27.17 |              30.00 |              20.78 |
|       1 |            88 |             3959.00 |               0.07 |                            3.68 |               11.03 |                        1.42 |                        5.60 |                         0.08 |               0.00 |              22.31 |              26.14 |              27.82 |              28.50 |              21.87 |
|       1 |            96 |             4113.00 |               0.07 |                            4.09 |               11.56 |                        1.49 |                        5.84 |                         0.08 |               0.00 |              23.33 |              26.97 |              28.59 |              30.97 |              23.12 |
|       1 |           104 |             4167.00 |               0.07 |                            4.57 |               12.06 |                        1.59 |                        6.17 |                         0.09 |               0.00 |              24.90 |              29.70 |              30.43 |              33.02 |              24.55 |
|       1 |           112 |             4359.00 |               0.07 |                            4.27 |               12.98 |                        1.74 |                        6.34 |                         0.09 |               0.00 |              25.99 |              29.66 |              30.45 |              31.36 |              25.49 |
|       1 |           120 |             4480.00 |               0.07 |                            4.69 |               13.08 |                        1.82 |                        6.78 |                         0.10 |               0.00 |              26.93 |              31.56 |              32.23 |              34.98 |              26.54 |
|       1 |           128 |             4484.00 |               0.07 |                            4.96 |               13.70 |                        2.06 |                        7.17 |                         0.11 |               0.00 |              28.22 |              33.04 |              34.23 |              37.49 |              28.06 |
|       1 |           136 |             4582.00 |               0.07 |                            5.99 |               14.22 |                        1.89 |                        7.04 |                         0.10 |               0.00 |              29.75 |              34.61 |              36.70 |              41.05 |              29.31 |
|       1 |           144 |             4695.00 |               0.07 |                            4.94 |               15.61 |                        2.12 |                        7.39 |                         0.11 |               0.00 |              30.24 |              36.14 |              37.78 |              38.85 |              30.24 |
|       1 |           152 |             4793.00 |               0.08 |                            4.87 |               16.24 |                        2.23 |                        7.71 |                         0.12 |               0.00 |              31.38 |              37.41 |              39.80 |              40.71 |              31.25 |
|       1 |           160 |             4842.00 |               0.08 |                            5.23 |               16.85 |                        2.39 |                        8.04 |                         0.13 |               0.00 |              33.29 |              37.84 |              39.45 |              41.92 |              32.72 |
|       1 |           168 |             4830.00 |               0.07 |                            6.62 |               16.36 |                        2.52 |                        8.58 |                         0.13 |               0.00 |              34.68 |              41.57 |              43.61 |              46.62 |              34.28 |
|       1 |           176 |             4891.00 |               0.08 |                            5.94 |               18.06 |                        2.55 |                        8.44 |                         0.13 |               0.00 |              34.99 |              41.99 |              44.81 |              48.38 |              35.20 |
|       1 |           184 |             5038.96 |               0.09 |                            5.24 |               19.14 |                        2.53 |                        8.47 |                         0.13 |               0.00 |              35.55 |              41.20 |              43.34 |              45.97 |              35.60 |
|       1 |           192 |             5171.00 |               0.08 |                            5.39 |               19.81 |                        2.62 |                        8.52 |                         0.14 |               0.00 |              37.01 |              42.56 |              43.95 |              46.62 |              36.56 |
|       1 |           200 |             5183.82 |               0.09 |                            5.70 |               20.24 |                        2.61 |                        9.03 |                         0.13 |               0.00 |              37.80 |              43.87 |              44.99 |              46.87 |              37.79 |
|       1 |           208 |             5180.00 |               0.12 |                            5.08 |               22.00 |                        3.02 |                        9.12 |                         0.14 |               0.00 |              39.19 |              47.07 |              47.97 |              49.04 |              39.49 |
|       1 |           216 |             5252.00 |               0.10 |                            6.00 |               21.56 |                        2.91 |                        9.52 |                         0.14 |               0.00 |              40.81 |              45.83 |              47.27 |              49.08 |              40.23 |
|       1 |           224 |             5237.00 |               0.10 |                            6.86 |               22.36 |                        3.00 |                        9.38 |                         0.14 |               0.00 |              42.03 |              48.28 |              50.59 |              56.46 |              41.84 |
|       1 |           232 |             5340.00 |               0.14 |                            5.23 |               23.84 |                        3.24 |                        9.97 |                         0.17 |               0.00 |              41.78 |              48.97 |              50.71 |              55.47 |              42.59 |
|       1 |           240 |             5341.00 |               0.13 |                            6.01 |               24.37 |                        3.20 |                       10.10 |                         0.18 |               0.00 |              44.02 |              51.39 |              52.77 |              54.41 |              43.97 |
|       1 |           248 |             5411.00 |               0.15 |                            5.96 |               25.29 |                        3.13 |                       10.07 |                         0.16 |               0.00 |              44.98 |              50.74 |              51.41 |              53.44 |              44.76 |
|       1 |           256 |             5467.00 |               0.17 |                            5.04 |               26.70 |                        3.62 |                       10.12 |                         0.18 |               0.00 |              45.04 |              53.05 |              54.13 |              58.90 |              45.83 |

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
Please download a checkpoint from [here](https://api.ngc.nvidia.com/v2/models/nvidia/dle/gpunet_p1_pyt_ckpt/versions/21.12.0_amp/zip) 
and place it in `runner_workspace/checkpoints/0.8ms-D/`.  Note that the `0.8ms-D` subdirectory may not be created yet.

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
export CHECKPOINT="0.8ms-D"
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
    --config /workspace/gpunet/configs/batch1/GV100/0.8ms-D.json \
    --checkpoint ${CHECKPOINT_DIR}/0.8ms-D.pth.tar \
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
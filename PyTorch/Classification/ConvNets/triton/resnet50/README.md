# Deploying the ResNet50 v1.5 model on Triton Inference Server

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
  - [Advanced](#advanced)
    - [Prepare configuration](#prepare-configuration)
    - [Latency explanation](#latency-explanation)
  - [Performance](#performance)
    - [Offline scenario](#offline-scenario)
      - [Offline: NVIDIA A40, ONNX Runtime TensorRT with FP16](#offline-nvidia-a40-onnx-runtime-tensorrt-with-fp16)
      - [Offline: NVIDIA DGX A100 (1x A100 80GB), ONNX Runtime TensorRT with FP16](#offline-nvidia-dgx-a100-1x-a100-80gb-onnx-runtime-tensorrt-with-fp16)
      - [Offline: NVIDIA DGX-1 (1x V100 32GB), ONNX Runtime TensorRT with FP16](#offline-nvidia-dgx-1-1x-v100-32gb-onnx-runtime-tensorrt-with-fp16)
      - [Offline: NVIDIA T4, ONNX Runtime TensorRT with FP16](#offline-nvidia-t4-onnx-runtime-tensorrt-with-fp16)
    - [Online scenario](#online-scenario)
      - [Online: NVIDIA A40, ONNX Runtime TensorRT with FP16](#online-nvidia-a40-onnx-runtime-tensorrt-with-fp16)
      - [Online: NVIDIA DGX A100 (1x A100 80GB), ONNX Runtime TensorRT with FP16](#online-nvidia-dgx-a100-1x-a100-80gb-onnx-runtime-tensorrt-with-fp16)
      - [Online: NVIDIA DGX-1 (1x V100 32GB), ONNX Runtime TensorRT with FP16](#online-nvidia-dgx-1-1x-v100-32gb-onnx-runtime-tensorrt-with-fp16)
      - [Online: NVIDIA T4, ONNX Runtime TensorRT with FP16](#online-nvidia-t4-onnx-runtime-tensorrt-with-fp16)
  - [Release Notes](#release-notes)
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
during training (as described in the [model README](../../resnet50v1.5/README.md)).
Additionally, this README provides the corresponding deployment scripts that
ensure optimal GPU utilization during inferencing on Triton Inference Server.

### Deployment process
The deployment process consists of two steps:

1. Conversion. The purpose of conversion is to find the best performing model
   format supported by Triton Inference Server.
   Triton Inference Server uses a number of runtime backends such as
   [TensorRT](https://developer.nvidia.com/tensorrt),
   [LibTorch](https://github.com/triton-inference-server/pytorch_backend) and
   [ONNX Runtime](https://github.com/triton-inference-server/onnxruntime_backend)
   to support various model types. Refer to the
   [Triton documentation](https://github.com/triton-inference-server/backend#where-can-i-find-all-the-backends-that-are-available-for-triton)
   for a list of available backends.
2. Configuration. Model configuration on Triton Inference Server, which generates
   necessary [configuration files](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md).

To run benchmarks measuring the model performance in inference,
perform the following steps:

1. Start the Triton Inference Server.

   The Triton Inference Server container is started
   in one (possibly remote) container and ports for gRPC or REST API are exposed.

2. Run accuracy tests.

   Produce results which are tested against the given accuracy thresholds.
   Refer to step 9 in the [Quick Start Guide](#quick-start-guide).

3. Run performance tests.

   Produce latency and throughput results for offline (static batching)
   and online (dynamic batching) scenarios.
   Refer to step 11 in the [Quick Start Guide](#quick-start-guide).


## Setup



Ensure you have the following components:
* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch NGC container 20.11](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
* [Triton Inference Server NGC container 20.11](https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver)
* [NVIDIA CUDA repository](https://docs.nvidia.com/cuda/archive/11.1.1/index.html)
* [NVIDIA Ampere](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/), [Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) based GPU



## Quick Start Guide
Running the following scripts will build and launch the container with all required dependencies for native PyTorch as well as Triton Inference Server. This is necessary for running inference and can also be used for data download, processing, and training of the model. 
 
1. Clone the repository.
 
   IMPORTANT: This step is executed on the host computer.
 
   ```
    git clone https://github.com/NVIDIA/DeepLearningExamples.git
    cd DeepLearningExamples/PyTorch/Classification/ConvNets
   ```
2. Setup the environment in the host computer and start Triton Inference Server.
 
   ```
    source triton/scripts/setup_environment.sh
    bash triton/scripts/docker/triton_inference_server.sh 
   ```

3. Build and run a container that extends the NGC PyTorch container with the Triton Inference Server client libraries and dependencies.
 
   ```
    bash triton/scripts/docker/build.sh
    bash triton/scripts/docker/interactive.sh
   ```


4. Prepare the deployment configuration and create folders in Docker.
 
   IMPORTANT: These and the following commands must be executed in the PyTorch NGC container.
 
 
   ```
    source triton/scripts/setup_environment.sh
   ```

5. Download and pre-process the dataset.
 
 
   ```
    bash triton/scripts/download_data.sh
    bash triton/scripts/process_dataset.sh
   ```
 
6. Setup the parameters for deployment.
 
   ```
    source triton/scripts/setup_parameters.sh
   ```
 
7. Convert the model from training to inference format (e.g. TensorRT).
 
 
   ```
    python3 triton/convert_model.py \
        --input-path triton/model.py \
        --input-type pyt \
        --output-path ${SHARED_DIR}/model \
        --output-type ${FORMAT} \
        --onnx-opset 11 \
        --onnx-optimized 1 \
        --max-batch-size ${MAX_BATCH_SIZE} \
        --max-workspace-size 1073741824 \
        --ignore-unknown-parameters \
        \
        --checkpoint ${CHECKPOINT_DIR}/nvidia_resnet50_200821.pth.tar \
        --precision ${PRECISION} \
        --config resnet50 \
        --classes 1000 \
        \
        --dataloader triton/dataloader.py \
        --data-dir ${DATASETS_DIR}/imagenet \
        --batch-size ${MAX_BATCH_SIZE}

   ```
 
 
8. Configure the model on Triton Inference Server.
 
   Generate the configuration from your model repository.
 
   ```
    python3 triton/config_model_on_triton.py \
        --model-repository ${MODEL_REPOSITORY_PATH} \
        --model-path ${SHARED_DIR}/model \
        --model-format ${FORMAT} \
        --model-name ${MODEL_NAME} \
        --model-version 1 \
        --max-batch-size ${MAX_BATCH_SIZE} \
        --precision ${PRECISION} \
        --number-of-model-instances ${NUMBER_OF_MODEL_INSTANCES} \
        --max-queue-delay-us 0 \
        --preferred-batch-sizes ${MAX_BATCH_SIZE} \
        --capture-cuda-graph 0 \
        --backend-accelerator ${BACKEND_ACCELERATOR} \
        --load-model ${TRITON_LOAD_MODEL_METHOD}
   ```
 
9. Run the Triton Inference Server accuracy tests.
 
   ```
    python3 triton/run_inference_on_triton.py \
        --server-url localhost:8001 \
        --model-name ${MODEL_NAME} \
        --model-version 1 \
        --output-dir ${SHARED_DIR}/accuracy_dump \
        \
        --precision ${PRECISION} \
        --dataloader triton/dataloader.py \
        --data-dir ${DATASETS_DIR}/imagenet \
        --batch-size ${MAX_BATCH_SIZE} \
        --dump-labels

    python3 triton/calculate_metrics.py \
        --metrics triton/metric.py \
        --dump-dir ${SHARED_DIR}/accuracy_dump \
        --csv ${SHARED_DIR}/accuracy_metrics.csv

    cat ${SHARED_DIR}/accuracy_metrics.csv
   ```
 
 
10. Run the Triton Inference Server performance online tests.
 
   We want to maximize throughput within latency budget constraints.
   Dynamic batching is a feature of Triton Inference Server that allows
   inference requests to be combined by the server, so that a batch is
   created dynamically, resulting in a reduced average latency.
   You can set the Dynamic Batcher parameter `max_queue_delay_microseconds` to
   indicate the maximum amount of time you are willing to wait and
   `preferred_batch_size` to indicate your maximum server batch size
   in the Triton Inference Server model configuration. The measurements
   presented below set the maximum latency to zero to achieve the best latency
   possible with good performance.
 
   ```
    python triton/run_online_performance_test_on_triton.py \
        --model-name ${MODEL_NAME} \
        --input-data random \
        --batch-sizes ${BATCH_SIZE} \
        --triton-instances ${TRITON_INSTANCES} \
        --number-of-model-instances ${NUMBER_OF_MODEL_INSTANCES} \
        --result-path ${SHARED_DIR}/triton_performance_online.csv
 
   ```
 


11. Run the Triton Inference Server performance offline tests.
 
   We want to maximize throughput. It assumes you have your data available
   for inference or that your data saturate to maximum batch size quickly.
   Triton Inference Server supports offline scenarios with static batching.
   Static batching allows inference requests to be served
   as they are received. The largest improvements to throughput come
   from increasing the batch size due to efficiency gains in the GPU with larger
   batches.
 
   ```
    python triton/run_offline_performance_test_on_triton.py \
        --model-name ${MODEL_NAME} \
        --input-data random \
        --batch-sizes ${BATCH_SIZE} \
        --triton-instances ${TRITON_INSTANCES} \
        --result-path ${SHARED_DIR}/triton_performance_offline.csv
   ```
 


## Advanced


### Prepare configuration
You can use the environment variables to set the parameters of your inference
configuration.

Triton deployment scripts support several inference runtimes listed in the table below:
| Inference runtime | Mnemonic used in scripts |
|-------------------|--------------------------|
| [TorchScript Tracing](https://pytorch.org/docs/stable/jit.html) | `ts-trace` |
| [TorchScript Tracing](https://pytorch.org/docs/stable/jit.html) | `ts-script` |
| [ONNX](https://onnx.ai) | `onnx` |
| [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) | `trt` |

The name of the inference runtime should be put into the `FORMAT` variable.


Example values of some key variables in one configuration:
```
PRECISION="fp16"
FORMAT="trt"
BATCH_SIZE="1, 2, 4, 8, 16, 32, 64, 128"
BACKEND_ACCELERATOR="cuda"
MAX_BATCH_SIZE="128"
NUMBER_OF_MODEL_INSTANCES="1"
TRITON_MAX_QUEUE_DELAY="1"
TRITON_PREFERRED_BATCH_SIZES="64 128"

```



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





## Performance

The performance measurements in this document were conducted at the time of publication and may not reflect the performance achieved from NVIDIA’s latest software release. For the most up-to-date performance measurements, go to [NVIDIA Data Center Deep Learning Product Performance](https://developer.nvidia.com/deep-learning-performance-training-inference).


### Offline scenario
This table lists the common variable parameters for all performance measurements:
| Parameter Name               | Parameter Value   |
|:-----------------------------|:------------------|
| Max Batch Size               | 128.0             |
| Number of model instances    | 1.0               |
| Triton Max Queue Delay       | 1.0               |
| Triton Preferred Batch Sizes | 64 128            |



#### Offline: NVIDIA A40, ONNX Runtime TensorRT with FP16

Our results were obtained using the following configuration:
 * **GPU:** NVIDIA A40
 * **Backend:** ONNX Runtime
 * **Backend accelerator:** TensorRT
 * **Precision:** FP16
 * **Model format:** ONNX


<table><tr><td>

![](plots/graph_performance_offline_1l.svg)

</td><td>

![](plots/graph_performance_offline_1r.svg)

</td></tr></table>

<details>

<summary>
Full tabular data
</summary>

| Precision   | Backend Accelerator  |   Client Batch Size |   Inferences/second |   P90 Latency |   P95 Latency |   P99 Latency |   Avg Latency |
|:------------|:---------------------|--------------------:|--------------------:|--------------:|--------------:|--------------:|--------------:|
| FP16        | TensorRT             |                   1 |               491.5 |         2.046 |         2.111 |         2.126 |         2.031 |
| FP16        | TensorRT             |                   2 |               811.8 |         2.509 |         2.568 |         2.594 |         2.459 |
| FP16        | TensorRT             |                   4 |              1094   |         3.814 |         3.833 |         3.877 |         3.652 |
| FP16        | TensorRT             |                   8 |              1573.6 |         5.45  |         5.517 |         5.636 |         5.078 |
| FP16        | TensorRT             |                  16 |              1651.2 |         9.896 |         9.978 |        10.074 |         9.678 |
| FP16        | TensorRT             |                  32 |              2070.4 |        17.49  |        17.837 |        19.228 |        15.451 |
| FP16        | TensorRT             |                  64 |              1766.4 |        37.123 |        37.353 |        37.85  |        36.147 |
| FP16        | TensorRT             |                 128 |              1894.4 |        69.027 |        69.15  |        69.789 |        67.889 |

</details>



#### Offline: NVIDIA DGX A100 (1x A100 80GB), ONNX Runtime TensorRT with FP16

Our results were obtained using the following configuration:
 * **GPU:** NVIDIA DGX A100 (1x A100 80GB)
 * **Backend:** ONNX Runtime
 * **Backend accelerator:** TensorRT
 * **Precision:** FP16
 * **Model format:** ONNX

<table><tr><td>

![](plots/graph_performance_offline_5l.svg)

</td><td>

![](plots/graph_performance_offline_5r.svg)

</td></tr></table>


<details>

<summary>
Full tabular data
</summary>

| Precision   | Backend Accelerator  |   Client Batch Size |   Inferences/second |   P90 Latency |   P95 Latency |   P99 Latency |   Avg Latency |
|:------------|:---------------------|--------------------:|--------------------:|--------------:|--------------:|--------------:|--------------:|
| FP16        | TensorRT             |                   1 |               469.1 |         2.195 |         2.245 |         2.272 |         2.128 |
| FP16        | TensorRT             |                   2 |               910   |         2.222 |         2.229 |         2.357 |         2.194 |
| FP16        | TensorRT             |                   4 |              1447.6 |         3.055 |         3.093 |         3.354 |         2.759 |
| FP16        | TensorRT             |                   8 |              2051.2 |         4.035 |         4.195 |         4.287 |         3.895 |
| FP16        | TensorRT             |                  16 |              2760   |         6.033 |         6.121 |         6.348 |         5.793 |
| FP16        | TensorRT             |                  32 |              2857.6 |        11.47  |        11.573 |        11.962 |        11.193 |
| FP16        | TensorRT             |                  64 |              2534.4 |        26.345 |        26.899 |        29.744 |        25.244 |
| FP16        | TensorRT             |                 128 |              2662.4 |        49.612 |        51.713 |        53.666 |        48.086 |

</details>


#### Offline: NVIDIA DGX-1 (1x V100 32GB), ONNX Runtime TensorRT with FP16

Our results were obtained using the following configuration:
 * **GPU:** NVIDIA DGX-1 (1x V100 32GB)
 * **Backend:** ONNX Runtime
 * **Backend accelerator:** TensorRT
 * **Precision:** FP16
 * **Model format:** ONNX

<table><tr><td>

![](plots/graph_performance_offline_9l.svg)

</td><td>

![](plots/graph_performance_offline_9r.svg)

</td></tr></table>

<details>

<summary>
Full tabular data
</summary>

| Precision   | Backend Accelerator  |   Client Batch Size |   Inferences/second |   P90 Latency |   P95 Latency |   P99 Latency |   Avg Latency |
|:------------|:---------------------|--------------------:|--------------------:|--------------:|--------------:|--------------:|--------------:|
| FP16        | TensorRT             |                   1 |               351.8 |         2.996 |         3.051 |         3.143 |         2.838 |
| FP16        | TensorRT             |                   2 |               596.2 |         3.481 |         3.532 |         3.627 |         3.35  |
| FP16        | TensorRT             |                   4 |               953.6 |         4.314 |         4.351 |         4.45  |         4.191 |
| FP16        | TensorRT             |                   8 |              1337.6 |         6.185 |         6.347 |         6.581 |         5.979 |
| FP16        | TensorRT             |                  16 |              1726.4 |         9.736 |         9.87  |        10.904 |         9.266 |
| FP16        | TensorRT             |                  32 |              2044.8 |        15.833 |        15.977 |        16.438 |        15.664 |
| FP16        | TensorRT             |                  64 |              1670.4 |        38.667 |        38.842 |        40.773 |        38.412 |
| FP16        | TensorRT             |                 128 |              1548.8 |        84.454 |        85.308 |        88.363 |        82.159 |

</details>




#### Offline: NVIDIA T4, ONNX Runtime TensorRT with FP16

Our results were obtained using the following configuration:
 * **GPU:** NVIDIA T4
 * **Backend:** ONNX Runtime
 * **Backend accelerator:** TensorRT
 * **Precision:** FP16
 * **Model format:** ONNX

<table><tr><td>

![](plots/graph_performance_offline_13l.svg)

</td><td>

![](plots/graph_performance_offline_13r.svg)

</td></tr></table>

<details>

<summary>
Full tabular data
</summary>

| Precision   | Backend Accelerator  |   Client Batch Size |   Inferences/second |   P90 Latency |   P95 Latency |   P99 Latency |   Avg Latency |
|:------------|:---------------------|--------------------:|--------------------:|--------------:|--------------:|--------------:|--------------:|
| FP16        | TensorRT             |                   1 |               332.4 |         3.065 |         3.093 |         3.189 |         3.003 |
| FP16        | TensorRT             |                   2 |               499.4 |         4.069 |         4.086 |         4.143 |         3.998 |
| FP16        | TensorRT             |                   4 |               695.2 |         5.779 |         5.786 |         5.802 |         5.747 |
| FP16        | TensorRT             |                   8 |               888   |         9.039 |         9.05  |         9.065 |         8.998 |
| FP16        | TensorRT             |                  16 |              1057.6 |        15.319 |        15.337 |        15.389 |        15.113 |
| FP16        | TensorRT             |                  32 |              1129.6 |        28.77  |        28.878 |        29.082 |        28.353 |
| FP16        | TensorRT             |                  64 |              1203.2 |        54.194 |        54.417 |        55.331 |        53.187 |
| FP16        | TensorRT             |                 128 |              1280   |       102.466 |       102.825 |       103.177 |       100.155 |

</details>





### Online scenario

This table lists the common variable parameters for all performance measurements:
| Parameter Name               | Parameter Value   |
|:-----------------------------|:------------------|
| Max Batch Size               | 128.0             |
| Number of model instances    | 1.0               |
| Triton Max Queue Delay       | 1.0               |
| Triton Preferred Batch Sizes | 64 128            |





#### Online: NVIDIA A40, ONNX Runtime TensorRT with FP16

Our results were obtained using the following configuration:
 * **GPU:** NVIDIA A40
 * **Backend:** ONNX Runtime
 * **Backend accelerator:** TensorRT
 * **Precision:** FP16
 * **Model format:** ONNX


![](plots/graph_performance_online_2.svg)
 
<details>

<summary>
Full tabular data
</summary>

|   Concurrent client requests |   Inferences/second |   Client Send |   Network+server Send/recv |   Server Queue |   Server Compute Input |   Server Compute Infer |   Server Compute Output |   Client Recv |   P50 Latency |   P90 Latency |   P95 Latency |   P99 Latency |   Avg Latency |
|-----------------------------:|--------------------:|--------------:|---------------------------:|---------------:|-----------------------:|-----------------------:|------------------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
|                           16 |              2543.7 |         0.078 |                      1.912 |          1.286 |                  0.288 |                  2.697 |                   0.024 |             0 |         6.624 |         7.039 |         7.414 |         9.188 |         6.285 |
|                           32 |              3166.7 |         0.085 |                      3.478 |          1.81  |                  0.582 |                  4.098 |                   0.047 |             0 |         9.924 |        11.001 |        12.217 |        14.717 |        10.1   |
|                           48 |              3563.9 |         0.085 |                      5.169 |          1.935 |                  0.99  |                  5.204 |                   0.08  |             0 |        13.199 |        14.813 |        16.421 |        19.793 |        13.463 |
|                           64 |              3514.9 |         0.091 |                      5.729 |          3.847 |                  1.553 |                  6.842 |                   0.138 |             0 |        17.986 |        18.85  |        19.916 |        25.825 |        18.2   |
|                           80 |              3703.5 |         0.097 |                      7.244 |          4.414 |                  2     |                  7.675 |                   0.169 |             0 |        21.313 |        23.838 |        28.664 |        32.631 |        21.599 |
|                           96 |              3636.9 |         0.101 |                      8.459 |          5.679 |                  3.157 |                  8.771 |                   0.215 |             0 |        26.131 |        27.751 |        31.269 |        38.695 |        26.382 |
|                          112 |              3701.7 |         0.099 |                      9.332 |          6.711 |                  3.588 |                 10.276 |                   0.241 |             0 |        30.319 |        31.282 |        31.554 |        32.151 |        30.247 |
|                          128 |              3795.8 |         0.106 |                     10.581 |          7.309 |                  4.067 |                 11.386 |                   0.268 |             0 |        33.893 |        34.793 |        35.448 |        43.182 |        33.717 |
|                          144 |              3892.4 |         0.106 |                     11.488 |          8.144 |                  4.713 |                 12.212 |                   0.32  |             0 |        37.184 |        38.277 |        38.597 |        39.393 |        36.983 |
|                          160 |              3950   |         0.106 |                     13.5   |          7.999 |                  5.083 |                 13.481 |                   0.343 |             0 |        40.656 |        42.157 |        44.756 |        53.426 |        40.512 |
|                          176 |              3992.5 |         0.118 |                     13.6   |          9.809 |                  5.596 |                 14.611 |                   0.379 |             0 |        44.324 |        45.689 |        46.331 |        52.155 |        44.113 |
|                          192 |              4058.3 |         0.116 |                     14.902 |         10.223 |                  6.054 |                 15.564 |                   0.416 |             0 |        47.537 |        48.91  |        49.752 |        55.973 |        47.275 |
|                          208 |              4121.8 |         0.117 |                     16.777 |          9.991 |                  6.347 |                 16.827 |                   0.441 |             0 |        50.652 |        52.241 |        53.4   |        62.688 |        50.5   |
|                          224 |              4116.1 |         0.124 |                     17.048 |         11.743 |                  7.065 |                 17.91  |                   0.504 |             0 |        54.571 |        56.204 |        56.877 |        62.169 |        54.394 |
|                          240 |              4100   |         0.157 |                     17.54  |         13.611 |                  7.532 |                 19.185 |                   0.538 |             0 |        58.683 |        60.034 |        60.64  |        64.791 |        58.563 |
|                          256 |              4310.5 |         0.277 |                     18.282 |         13.5   |                  7.714 |                 19.136 |                   0.539 |             0 |        59.244 |        60.686 |        61.349 |        66.84  |        59.448 |

</details>


#### Online: NVIDIA DGX A100 (1x A100 80GB), ONNX Runtime TensorRT with FP16

Our results were obtained using the following configuration:
 * **GPU:** NVIDIA DGX A100 (1x A100 80GB)
 * **Backend:** ONNX Runtime
 * **Backend accelerator:** TensorRT
 * **Precision:** FP16
 * **Model format:** ONNX

![](plots/graph_performance_online_10.svg)
 
<details>

<summary>
Full tabular data
</summary>

|   Concurrent client requests |   Inferences/second |   Client Send |   Network+server Send/recv |   Server Queue |   Server Compute Input |   Server Compute Infer |   Server Compute Output |   Client Recv |   P50 Latency |   P90 Latency |   P95 Latency |   P99 Latency |   Avg Latency |
|-----------------------------:|--------------------:|--------------:|---------------------------:|---------------:|-----------------------:|-----------------------:|------------------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
|                           16 |              2571.2 |         0.067 |                      1.201 |          1.894 |                  0.351 |                  2.678 |                   0.027 |             0 |         6.205 |         6.279 |         6.31  |         6.418 |         6.218 |
|                           32 |              3600.2 |         0.058 |                      2.641 |          2.004 |                  0.716 |                  3.41  |                   0.057 |             0 |         8.852 |         9.233 |         9.353 |        12.253 |         8.886 |
|                           48 |              4274.2 |         0.062 |                      3.102 |          2.738 |                  1.121 |                  4.113 |                   0.089 |             0 |        11.03  |        11.989 |        12.1   |        15.115 |        11.225 |
|                           64 |              4387.7 |         0.07  |                      3.767 |          3.438 |                  2.016 |                  5.164 |                   0.122 |             0 |        14.628 |        15.067 |        15.211 |        15.504 |        14.577 |
|                           80 |              4630.1 |         0.064 |                      4.23  |          5.049 |                  2.316 |                  5.463 |                   0.151 |             0 |        17.205 |        17.726 |        17.9   |        18.31  |        17.273 |
|                           96 |              4893.9 |         0.068 |                      4.811 |          5.764 |                  2.741 |                  6.044 |                   0.179 |             0 |        19.44  |        20.23  |        20.411 |        22.781 |        19.607 |
|                          112 |              4887.6 |         0.069 |                      6.232 |          5.202 |                  3.597 |                  7.586 |                   0.236 |             0 |        23.099 |        23.665 |        23.902 |        24.192 |        22.922 |
|                          128 |              5411.5 |         0.081 |                      5.921 |          7     |                  3.387 |                  7.016 |                   0.255 |             0 |        23.852 |        24.349 |        24.557 |        26.433 |        23.66  |
|                          144 |              5322.9 |         0.08  |                      7.066 |          7.55  |                  3.996 |                  8.059 |                   0.299 |             0 |        27.024 |        28.487 |        29.725 |        33.7   |        27.05  |
|                          160 |              5310.5 |         0.079 |                      6.98  |          9.157 |                  4.61  |                  8.98  |                   0.331 |             0 |        30.446 |        31.497 |        31.91  |        34.269 |        30.137 |
|                          176 |              5458.7 |         0.081 |                      7.857 |          9.272 |                  5.047 |                  9.634 |                   0.345 |             0 |        32.588 |        33.271 |        33.478 |        35.47  |        32.236 |
|                          192 |              5654.1 |         0.081 |                      9.355 |          8.898 |                  5.294 |                  9.923 |                   0.388 |             0 |        34.35  |        35.895 |        36.302 |        39.288 |        33.939 |
|                          208 |              5643.7 |         0.093 |                      9.407 |         10.488 |                  5.953 |                 10.54  |                   0.383 |             0 |        36.994 |        38.14  |        38.766 |        41.616 |        36.864 |
|                          224 |              5795.5 |         0.101 |                      9.862 |         10.852 |                  6.331 |                 11.081 |                   0.415 |             0 |        38.536 |        39.741 |        40.563 |        43.227 |        38.642 |
|                          240 |              5855.8 |         0.131 |                      9.994 |         12.391 |                  6.589 |                 11.419 |                   0.447 |             0 |        40.721 |        43.344 |        44.449 |        46.902 |        40.971 |
|                          256 |              6127.3 |         0.131 |                     10.495 |         12.342 |                  6.979 |                 11.344 |                   0.473 |             0 |        41.606 |        43.106 |        43.694 |        46.457 |        41.764 |

</details>




#### Online: NVIDIA DGX-1 (1x V100 32GB), ONNX Runtime TensorRT with FP16

Our results were obtained using the following configuration:
 * **GPU:** NVIDIA DGX-1 (1x V100 32GB)
 * **Backend:** ONNX Runtime
 * **Backend accelerator:** TensorRT
 * **Precision:** FP16
 * **Model format:** ONNX

![](plots/graph_performance_online_18.svg)
 
<details>

<summary>
Full tabular data
</summary>

|   Concurrent client requests |   Inferences/second |   Client Send |   Network+server Send/recv |   Server Queue |   Server Compute Input |   Server Compute Infer |   Server Compute Output |   Client Recv |   P50 Latency |   P90 Latency |   P95 Latency |   P99 Latency |   Avg Latency |
|-----------------------------:|--------------------:|--------------:|---------------------------:|---------------:|-----------------------:|-----------------------:|------------------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
|                           16 |              1679.6 |         0.096 |                      3.312 |          1.854 |                  0.523 |                  3.713 |                   0.026 |             0 |         8.072 |        12.416 |        12.541 |        12.729 |         9.524 |
|                           32 |              2760.1 |         0.095 |                      3.933 |          1.978 |                  0.949 |                  4.597 |                   0.035 |             0 |        11.569 |        11.728 |        11.785 |        12.39  |        11.587 |
|                           48 |              3127.1 |         0.099 |                      4.919 |          3.105 |                  1.358 |                  5.816 |                   0.051 |             0 |        15.471 |        15.86  |        18.206 |        20.198 |        15.348 |
|                           64 |              3287.4 |         0.101 |                      5.874 |          4.346 |                  1.789 |                  7.293 |                   0.069 |             0 |        19.44  |        19.727 |        19.838 |        20.584 |        19.472 |
|                           80 |              3209   |         0.131 |                      7.032 |          6.014 |                  3.227 |                  8.418 |                   0.111 |             0 |        25.362 |        25.889 |        26.095 |        29.005 |        24.933 |
|                           96 |              3273.6 |         0.14  |                      8.539 |          6.74  |                  4.371 |                  9.369 |                   0.153 |             0 |        29.217 |        29.641 |        29.895 |        31.002 |        29.312 |
|                          112 |              3343.3 |         0.149 |                      9.683 |          7.802 |                  4.214 |                 11.484 |                   0.159 |             0 |        30.933 |        37.027 |        37.121 |        37.358 |        33.491 |
|                          128 |              3335.1 |         0.152 |                      9.865 |         10.127 |                  5.519 |                 12.534 |                   0.195 |             0 |        38.762 |        40.022 |        40.336 |        42.943 |        38.392 |
|                          144 |              3304.2 |         0.185 |                     11.017 |         11.901 |                  6.877 |                 13.35  |                   0.209 |             0 |        43.372 |        43.812 |        44.042 |        46.708 |        43.539 |
|                          160 |              3319.9 |         0.206 |                     12.701 |         12.625 |                  7.49  |                 14.907 |                   0.238 |             0 |        48.31  |        49.135 |        49.343 |        50.441 |        48.167 |
|                          176 |              3335   |         0.271 |                     13.013 |         14.788 |                  8.564 |                 15.789 |                   0.263 |             0 |        52.352 |        53.653 |        54.385 |        57.332 |        52.688 |
|                          192 |              3380   |         0.243 |                     13.894 |         15.719 |                  9.865 |                 16.841 |                   0.283 |             0 |        56.872 |        58.64  |        58.944 |        62.097 |        56.845 |
|                          208 |              3387.6 |         0.273 |                     16.221 |         15.73  |                 10.334 |                 18.448 |                   0.326 |             0 |        61.402 |        63.099 |        63.948 |        68.63  |        61.332 |
|                          224 |              3477.2 |         0.613 |                     14.167 |         18.902 |                 10.896 |                 19.605 |                   0.34  |             0 |        64.495 |        65.69  |        66.101 |        67.522 |        64.523 |
|                          240 |              3528   |         0.878 |                     14.713 |         20.894 |                 10.259 |                 20.859 |                   0.436 |             0 |        66.404 |        71.807 |        72.857 |        75.076 |        68.039 |
|                          256 |              3558.4 |         1.035 |                     15.534 |         22.837 |                 11     |                 21.062 |                   0.435 |             0 |        71.657 |        77.271 |        78.269 |        80.804 |        71.903 |

</details>




#### Online: NVIDIA T4, ONNX Runtime TensorRT with FP16

Our results were obtained using the following configuration:
 * **GPU:** NVIDIA T4
 * **Backend:** ONNX Runtime
 * **Backend accelerator:** TensorRT
 * **Precision:** FP16
 * **Model format:** ONNX

![](plots/graph_performance_online_26.svg)
 
<details>

<summary>
Full tabular data
</summary>

|   Concurrent client requests |   Inferences/second |   Client Send |   Network+server Send/recv |   Server Queue |   Server Compute Input |   Server Compute Infer |   Server Compute Output |   Client Recv |   P50 Latency |   P90 Latency |   P95 Latency |   P99 Latency |   Avg Latency |
|-----------------------------:|--------------------:|--------------:|---------------------------:|---------------:|-----------------------:|-----------------------:|------------------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
|                           16 |              1078.4 |         0.169 |                      6.163 |          2.009 |                  0.495 |                  5.963 |                   0.022 |         0     |        15.75  |        16.219 |        16.376 |        16.597 |        14.821 |
|                           32 |              2049.6 |         0.195 |                      4.342 |          3.384 |                  0.849 |                  6.804 |                   0.032 |         0     |        15.606 |        15.792 |        15.853 |        15.975 |        15.606 |
|                           48 |              2133.1 |         0.189 |                      6.365 |          4.926 |                  1.379 |                  9.573 |                   0.063 |         0     |        22.304 |        23.432 |        23.73  |        27.241 |        22.495 |
|                           64 |              2114.3 |         0.206 |                      9.038 |          6.258 |                  1.863 |                 12.812 |                   0.086 |         0     |        30.074 |        31.063 |        31.535 |        42.845 |        30.263 |
|                           80 |              2089.3 |         0.204 |                     11.943 |          7.841 |                  2.676 |                 15.556 |                   0.108 |         0     |        38.289 |        40.895 |        52.977 |        58.393 |        38.328 |
|                           96 |              2145.3 |         0.23  |                     12.987 |          9.63  |                  3.597 |                 18.132 |                   0.134 |         0     |        44.511 |        47.352 |        47.809 |        48.429 |        44.71  |
|                          112 |              2062.3 |         0.28  |                     13.253 |         14.112 |                  5.088 |                 21.398 |                   0.154 |         0     |        54.289 |        55.441 |        55.69  |        56.205 |        54.285 |
|                          128 |              2042.6 |         0.485 |                     14.377 |         16.957 |                  6.279 |                 24.487 |                   0.169 |         0     |        62.718 |        63.902 |        64.178 |        64.671 |        62.754 |
|                          144 |              2066.6 |         0.726 |                     16.363 |         18.601 |                  7.085 |                 26.801 |                   0.193 |         0.001 |        69.67  |        71.418 |        71.765 |        73.255 |        69.77  |
|                          160 |              2073.1 |         0.557 |                     17.787 |         20.809 |                  7.378 |                 30.43  |                   0.215 |         0     |        77.212 |        79.089 |        79.815 |        83.434 |        77.176 |
|                          176 |              2076.8 |         1.209 |                     18.446 |         23.075 |                  8.689 |                 32.894 |                   0.253 |         0     |        84.13  |        86.732 |        87.404 |        95.286 |        84.566 |
|                          192 |              2073.9 |         1.462 |                     19.845 |         25.653 |                  9.088 |                 36.153 |                   0.272 |         0     |        92.32  |        94.276 |        94.805 |        96.765 |        92.473 |
|                          208 |              2053.2 |         1.071 |                     22.995 |         26.411 |                 10.123 |                 40.415 |                   0.322 |         0     |       101.178 |       103.725 |       105.498 |       110.695 |       101.337 |
|                          224 |              1994.1 |         0.968 |                     24.931 |         31.14  |                 14.276 |                 40.804 |                   0.389 |         0     |       114.177 |       116.977 |       118.248 |       121.879 |       112.508 |
|                          240 |              1952.6 |         1.028 |                     27.957 |         34.546 |                 16.535 |                 42.685 |                   0.38  |         0     |       122.846 |       126.022 |       128.074 |       136.541 |       123.131 |
|                          256 |              2017.8 |         0.85  |                     27.437 |         38.553 |                 15.224 |                 44.637 |                   0.401 |         0     |       129.052 |       132.762 |       134.337 |       138.108 |       127.102 |

</details>



## Release Notes
We’re constantly refining and improving our performance on AI
and HPC workloads even on the same hardware with frequent updates
to our software stack. For our latest performance data refer
to these pages for
[AI](https://developer.nvidia.com/deep-learning-performance-training-inference)
and [HPC](https://developer.nvidia.com/hpc-application-performance) benchmarks.

### Changelog

April 2021
- NVIDIA Ampere results added

September 2020
- Initial release

### Known issues

- There are no known issues with this model.



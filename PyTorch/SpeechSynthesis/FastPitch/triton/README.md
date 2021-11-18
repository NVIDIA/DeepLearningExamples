# Deploying the FastPitch model on Triton Inference Server

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
        - [Offline: NVIDIA A40 with FP16](#offline-nvidia-a40-with-fp16)
        - [Offline: NVIDIA A40 with FP32](#offline-nvidia-a40-with-fp32)
        - [Offline: NVIDIA DGX A100 (1x A100 80GB) with FP16](#offline-nvidia-dgx-a100-1x-a100-80gb-with-fp16)
        - [Offline: NVIDIA DGX A100 (1x A100 80GB) with FP32](#offline-nvidia-dgx-a100-1x-a100-80gb-with-fp32)
        - [Offline: NVIDIA DGX-1 (1x V100 32GB) with FP16](#offline-nvidia-dgx-1-1x-v100-32gb-with-fp16)
        - [Offline: NVIDIA DGX-1 (1x V100 32GB) with FP32](#offline-nvidia-dgx-1-1x-v100-32gb-with-fp32)
        - [Offline: NVIDIA T4 with FP16](#offline-nvidia-t4-with-fp16)
        - [Offline: NVIDIA T4 with FP32](#offline-nvidia-t4-with-fp32)
    - [Online scenario](#online-scenario)
        - [Online: NVIDIA A40 with FP16](#online-nvidia-a40-with-fp16)
        - [Online: NVIDIA A40 with FP32](#online-nvidia-a40-with-fp32)
        - [Online: NVIDIA DGX A100 (1x A100 80GB) with FP16](#online-nvidia-dgx-a100-1x-a100-80gb-with-fp16)
        - [Online: NVIDIA DGX A100 (1x A100 80GB) with FP32](#online-nvidia-dgx-a100-1x-a100-80gb-with-fp32)
        - [Online: NVIDIA DGX-1 (1x V100 32GB) with FP16](#online-nvidia-dgx-1-1x-v100-32gb-with-fp16)
        - [Online: NVIDIA DGX-1 (1x V100 32GB) with FP32](#online-nvidia-dgx-1-1x-v100-32gb-with-fp32)
        - [Online: NVIDIA T4 with FP16](#online-nvidia-t4-with-fp16)
        - [Online: NVIDIA T4 with FP32](#online-nvidia-t4-with-fp32)
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
during training (as described in the [model README](../README.md)).
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
   to support various model types. Refer to
   [Triton documentation](https://github.com/triton-inference-server/backend#where-can-i-find-all-the-backends-that-are-available-for-triton)
   for the list of available backends.
2. Configuration. Model configuration on Triton Inference Server, which generates
   necessary [configuration files](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md).

To run benchmarks measuring the model performance in inference,
perform the following steps:

1. Start the Triton Inference Server.

   The Triton Inference Server container is started
   in one (possibly remote) container and ports for gRPC or REST API are exposed.

2. Run accuracy tests.

   Produce results which are tested against given accuracy thresholds.
   Refer to step 8 in the [Quick Start Guide](#quick-start-guide).

3. Run performance tests.

   Produce latency and throughput results for offline (static batching)
   and online (dynamic batching) scenarios.
   Refer to step 10 in the [Quick Start Guide](#quick-start-guide).


## Setup



Ensure you have the following components:
* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch NGC container 21.02](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
* [Triton Inference Server NGC container 21.02](https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver)
* [NVIDIA CUDA repository](https://docs.nvidia.com/cuda/archive/11.2.0/index.html) (use CUDA 11.2 or newer)
* [NVIDIA Ampere](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/), [Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) based GPU



## Quick Start Guide

Running the following scripts will build and launch the container with all required dependencies for native PyTorch as well as Triton Inference Server. This is necessary for running inference and can also be used for data download, processing, and training of the model.

1. Clone the repository.

   IMPORTANT: This step is executed on the host computer.

   ```
    git clone https://github.com/NVIDIA/DeepLearningExamples.git
    cd DeepLearningExamples/PyTorch/SpeechSynthesis/FastPitch
   ```
1. Setup environment in host PC and start Triton Inference Server.

   ```
    source triton/scripts/setup_environment.sh
    bash triton/scripts/docker/triton_inference_server.sh
   ```

1. Build and run a container that extends the NGC PyTorch container with the Triton Inference Server client libraries and dependencies.

   ```
    bash triton/scripts/docker/build.sh
    bash triton/scripts/docker/interactive.sh
   ```


1. Prepare the deployment configuration and create folders in Docker.

   IMPORTANT: These and the following commands must be executed in the PyTorch NGC container.

   ```
    source triton/scripts/setup_environment.sh
   ```

1. Download and pre-process the dataset.


   ```
    bash triton/scripts/download_data.sh
    bash triton/scripts/process_dataset.sh
   ```

1. Setup parameters for deployment.

   ```
    source triton/scripts/setup_parameters.sh
   ```

1. Convert the model from training to inference format (e.g. TensorRT).


   ```
    python3 triton/convert_model.py \
        --input-path ./triton/model.py \
        --input-type pyt \
        --output-path ${SHARED_DIR}/model \
        --output-type ${FORMAT} \
        --checkpoint ${CHECKPOINT_DIR}/nvidia_fastpitch_200518.pt \
        --onnx-opset 12 \
        --model-path triton/model.py \
        --output-format ${FORMAT} \
        --dataloader triton/dataloader.py \
        --dataset-path ${DATASETS_DIR}/LJSpeech-1.1/LJSpeech-1.1_fastpitch \
        --batch-size 1 \
        --max-batch-size ${MAX_BATCH_SIZE} \
        --max-workspace-size 512 \
        --precision ${PRECISION} \
        --ignore-unknown-parameters
   ```


1. Configure the model on Triton Inference Server.

   Generate the configuration from your model repository.

   ```
   model-navigator triton-config-model \
	   --model-repository ${MODEL_REPOSITORY_PATH} \
	   --model-name ${MODEL_NAME} \
	   --model-version 1 \
	   --model-path ${SHARED_DIR}/model \
	   --model-format ${CONFIG_FORMAT} \
	   --model-control-mode ${TRITON_LOAD_MODEL_METHOD} \
	   --load-model \
	   --load-model-timeout-s 100 \
	   --verbose \
	   \
	   --backend-accelerator ${BACKEND_ACCELERATOR} \
	   --tensorrt-precision ${PRECISION} \
	   --max-batch-size ${MAX_BATCH_SIZE} \
	   --preferred-batch-sizes ${TRITON_PREFERRED_BATCH_SIZES} \
	   --max-queue-delay-us ${TRITON_MAX_QUEUE_DELAY} \
	   --engine-count-per-device gpu=${NUMBER_OF_MODEL_INSTANCES}
   ```

1. Run the Triton Inference Server accuracy tests.

   ```
    python3 triton/run_inference_on_triton.py \
        --server-url localhost:8001 \
        --model-name ${MODEL_NAME} \
        --model-version 1 \
        --dataloader triton/dataloader.py \
        --dataset-path ${DATASETS_DIR}/LJSpeech-1.1/LJSpeech-1.1_fastpitch \
        --batch-size ${MAX_BATCH_SIZE} \
        --output-dir ${SHARED_DIR}/accuracy_dump

    ls ${SHARED_DIR}/accuracy_dump

    python3 triton/calculate_metrics.py \
        --dump-dir ${SHARED_DIR}/accuracy_dump \
        --metrics triton/metrics.py \
        --csv ${SHARED_DIR}/accuracy_metrics.csv \
        --output-used-for-metrics OUTPUT__0

    cat ${SHARED_DIR}/accuracy_metrics.csv

   ```

1. Prepare performance input.

   ```
    mkdir -p ${SHARED_DIR}/input_data

    python triton/prepare_input_data.py \
        --dataloader triton/dataloader.py \
        --input-data-dir ${SHARED_DIR}/input_data \
        --dataset-path ${DATASETS_DIR}/LJSpeech-1.1/LJSpeech-1.1_fastpitch \
        --precision ${PRECISION} \
        --length ${SEQUENCE_LENGTH}
   ```


1. Run the Triton Inference Server performance online tests.

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
        --input-data ${SHARED_DIR}/input_data \
        --input-shape INPUT__0:${SEQUENCE_LENGTH} \
        --batch-sizes ${BATCH_SIZE} \
        --triton-instances ${TRITON_INSTANCES} \
        --number-of-model-instances ${NUMBER_OF_MODEL_INSTANCES} \
        --result-path ${SHARED_DIR}/triton_performance_online.csv
   ```


1. Run the Triton Inference Server performance offline tests.

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
        --input-data ${SHARED_DIR}/input_data \
        --input-shape INPUT__0:${SEQUENCE_LENGTH} \
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





Example values of some key variables in one configuration:
```
PRECISION="fp16"
FORMAT="ts-trace"
BATCH_SIZE="1, 2, 4, 8"
BACKEND_ACCELERATOR="cuda"
MAX_BATCH_SIZE="8"
NUMBER_OF_MODEL_INSTANCES="2"
TRITON_MAX_QUEUE_DELAY="1"
TRITON_PREFERRED_BATCH_SIZES="4 8"
SEQUENCE_LENGTH="128"

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
a small fraction of time, compared to steps 5-6. As backend deep learning
systems like Jasper are rarely exposed directly to end users, but instead
only interfacing with local front-end servers, for the sake of Jasper,
we can consider that all clients are local.





## Performance



### Offline scenario
This table lists the common variable parameters for all performance measurements:
| Parameter Name               | Parameter Value     |
|:-----------------------------|:--------------------|
| Model Format                 | TorchScript, Trace  |
| Backend Accelerator          | CUDA                |
| Max Batch Size               | 8                   |
| Number of model instances    | 2                   |
| Triton Max Queue Delay       | 1                   |
| Triton Preferred Batch Sizes | 4 8                 |



#### Offline: NVIDIA A40 with FP16

Our results were obtained using the following configuration:
 * **GPU:** NVIDIA A40
 * **Backend:** PyTorch
 * **Precision:** FP16
 * **Model format:** TorchScript
 * **Conversion variant:** Trace



|![](plots/graph_performance_offline_1l.svg)|![](plots/graph_performance_offline_1r.svg)|
|-----|-----|

<details>

<summary>
Full tabular data
</summary>

| Precision   |   Sequence Length |   Client Batch Size |   Inferences/second |   P90 Latency |   P95 Latency |   P99 Latency |   Avg Latency |
|:------------|------------------:|--------------------:|--------------------:|--------------:|--------------:|--------------:|--------------:|
| FP16        |               128 |                   1 |                81.8 |        12.828 |        13.384 |        13.493 |        12.22  |
| FP16        |               128 |                   2 |               164   |        12.906 |        13.222 |        13.635 |        12.199 |
| FP16        |               128 |                   4 |               315.6 |        13.565 |        13.635 |        13.875 |        12.674 |
| FP16        |               128 |                   8 |               592.8 |        13.534 |        15.352 |        15.801 |        13.491 |

</details>


#### Offline: NVIDIA A40 with FP32

Our results were obtained using the following configuration:
 * **GPU:** NVIDIA A40
 * **Backend:** PyTorch
 * **Precision:** FP32
 * **Model format:** TorchScript
 * **Conversion variant:** Trace


|![](plots/graph_performance_offline_2l.svg)|![](plots/graph_performance_offline_2r.svg)|
|-----|-----|

<details>

<summary>
Full tabular data
</summary>

| Precision   |   Sequence Length |   Client Batch Size |   Inferences/second |   P90 Latency |   P95 Latency |   P99 Latency |   Avg Latency |
|:------------|------------------:|--------------------:|--------------------:|--------------:|--------------:|--------------:|--------------:|
| FP32        |               128 |                   1 |                83.3 |        12.387 |        12.59  |        12.814 |        11.994 |
| FP32        |               128 |                   2 |               197   |        12.058 |        12.418 |        13.14  |        10.151 |
| FP32        |               128 |                   4 |               320.8 |        12.474 |        12.527 |        14.722 |        12.476 |
| FP32        |               128 |                   8 |               439.2 |        18.546 |        18.578 |        18.63  |        18.204 |

</details>


#### Offline: NVIDIA DGX A100 (1x A100 80GB) with FP16

Our results were obtained using the following configuration:
 * **GPU:** NVIDIA NVIDIA DGX A100 (1x A100 80GB)
 * **Backend:** PyTorch
 * **Precision:** FP16
 * **Model format:** TorchScript
 * **Conversion variant:** Trace

|![](plots/graph_performance_offline_3l.svg)|![](plots/graph_performance_offline_3r.svg)|
|-----|-----|

<details>

<summary>
Full tabular data
</summary>

| Precision   |   Sequence Length |   Client Batch Size |   Inferences/second |   P90 Latency |   P95 Latency |   P99 Latency |   Avg Latency |
|:------------|------------------:|--------------------:|--------------------:|--------------:|--------------:|--------------:|--------------:|
| FP16        |               128 |                   1 |               152.3 |         6.84  |         6.889 |         7.429 |         6.561 |
| FP16        |               128 |                   2 |               298.2 |         6.918 |         7.014 |         7.135 |         6.703 |
| FP16        |               128 |                   4 |               537.6 |         7.649 |         7.76  |         7.913 |         7.435 |
| FP16        |               128 |                   8 |               844   |         9.723 |         9.809 |        10.027 |         9.482 |

</details>




#### Offline: NVIDIA DGX A100 (1x A100 80GB) with FP32

Our results were obtained using the following configuration:
 * **GPU:** NVIDIA NVIDIA DGX A100 (1x A100 80GB)
 * **Backend:** PyTorch
 * **Precision:** FP32
 * **Model format:** TorchScript
 * **Conversion variant:** Trace


|![](plots/graph_performance_offline_4l.svg)|![](plots/graph_performance_offline_4r.svg)|
|-----|-----|

<details>

<summary>
Full tabular data
</summary>

| Precision   |   Sequence Length |   Client Batch Size |   Inferences/second |   P90 Latency |   P95 Latency |   P99 Latency |   Avg Latency |
|:------------|------------------:|--------------------:|--------------------:|--------------:|--------------:|--------------:|--------------:|
| FP32        |               128 |                   1 |               149.8 |         6.873 |         6.935 |         7.061 |         6.668 |
| FP32        |               128 |                   2 |               272.4 |         7.508 |         7.614 |         8.215 |         7.336 |
| FP32        |               128 |                   4 |               465.2 |         8.828 |         8.881 |         9.253 |         8.6   |
| FP32        |               128 |                   8 |               749.6 |        10.86  |        10.968 |        11.154 |        10.669 |

</details>




#### Offline: NVIDIA DGX-1 (1x V100 32GB) with FP16

Our results were obtained using the following configuration:
 * **GPU:**  NVIDIA DGX-1 (1x V100 32GB)
 * **Backend:** PyTorch
 * **Precision:** FP16
 * **Model format:** TorchScript
 * **Conversion variant:** Trace



|![](plots/graph_performance_offline_5l.svg)|![](plots/graph_performance_offline_5r.svg)|
|-----|-----|

<details>

<summary>
Full tabular data
</summary>

| Precision   |   Sequence Length |   Client Batch Size |   Inferences/second |   P90 Latency |   P95 Latency |   P99 Latency |   Avg Latency |
|:------------|------------------:|--------------------:|--------------------:|--------------:|--------------:|--------------:|--------------:|
| FP16        |               128 |                   1 |               101.3 |        10.039 |        10.14  |        10.333 |         9.866 |
| FP16        |               128 |                   2 |               199.2 |        10.191 |        10.359 |        10.911 |        10.034 |
| FP16        |               128 |                   4 |               349.2 |        11.541 |        11.629 |        11.807 |        11.45  |
| FP16        |               128 |                   8 |               567.2 |        14.266 |        14.307 |        14.426 |        14.107 |

</details>


#### Offline: NVIDIA DGX-1 (1x V100 32GB) with FP32

Our results were obtained using the following configuration:
 * **GPU:**  NVIDIA DGX-1 (1x V100 32GB)
 * **Backend:** PyTorch
 * **Precision:** FP32
 * **Model format:** TorchScript
 * **Conversion variant:** Trace


|![](plots/graph_performance_offline_6l.svg)|![](plots/graph_performance_offline_6r.svg)|
|-----|-----|

<details>

<summary>
Full tabular data
</summary>

| Precision   |   Sequence Length |   Client Batch Size |   Inferences/second |   P90 Latency |   P95 Latency |   P99 Latency |   Avg Latency |
|:------------|------------------:|--------------------:|--------------------:|--------------:|--------------:|--------------:|--------------:|
| FP32        |               128 |                   1 |               107.7 |         9.413 |         9.58  |        10.265 |         9.278 |
| FP32        |               128 |                   2 |               159   |        12.71  |        12.889 |        13.228 |        12.565 |
| FP32        |               128 |                   4 |               205.6 |        19.874 |        19.995 |        20.156 |        19.456 |
| FP32        |               128 |                   8 |               248.8 |        32.237 |        32.273 |        32.347 |        32.091 |

</details>



#### Offline: NVIDIA T4 with FP16

Our results were obtained using the following configuration:
 * **GPU:**  NVIDIA T4
 * **Backend:** PyTorch
 * **Precision:** FP16
 * **Model format:** TorchScript
 * **Conversion variant:** Trace



|![](plots/graph_performance_offline_7l.svg)|![](plots/graph_performance_offline_7r.svg)|
|-----|-----|

<details>

<summary>
Full tabular data
</summary>

| Precision   |   Sequence Length |   Client Batch Size |   Inferences/second |   P90 Latency |   P95 Latency |   P99 Latency |   Avg Latency |
|:------------|------------------:|--------------------:|--------------------:|--------------:|--------------:|--------------:|--------------:|
| FP16        |               128 |                   1 |                53.7 |        19.583 |        19.746 |        20.223 |        18.631 |
| FP16        |               128 |                   2 |                99.6 |        20.385 |        20.585 |        20.835 |        20.078 |
| FP16        |               128 |                   4 |               193.6 |        23.293 |        24.649 |        25.708 |        20.656 |
| FP16        |               128 |                   8 |               260   |        31.21  |        31.409 |        33.953 |        30.739 |

</details>



#### Offline: NVIDIA T4 with FP32

Our results were obtained using the following configuration:
 * **GPU:**  NVIDIA T4
 * **Backend:** PyTorch
 * **Precision:** FP32
 * **Model format:** TorchScript
 * **Conversion variant:** Trace


|![](plots/graph_performance_offline_8l.svg)|![](plots/graph_performance_offline_8r.svg)|
|-----|-----|

<details>

<summary>
Full tabular data
</summary>

| Precision   |   Sequence Length |   Client Batch Size |   Inferences/second |   P90 Latency |   P95 Latency |   P99 Latency |   Avg Latency |
|:------------|------------------:|--------------------:|--------------------:|--------------:|--------------:|--------------:|--------------:|
| FP32        |               128 |                   1 |                53.7 |        19.402 |        19.494 |        19.635 |        18.619 |
| FP32        |               128 |                   2 |                86.2 |        25.448 |        25.921 |        26.419 |        23.182 |
| FP32        |               128 |                   4 |                98.8 |        41.163 |        41.562 |        41.865 |        40.549 |
| FP32        |               128 |                   8 |               111.2 |        73.033 |        73.204 |        73.372 |        72.165 |

</details>



### Online scenario

This table lists the common variable parameters for all performance measurements:
| Parameter Name               | Parameter Value     |
|:-----------------------------|:--------------------|
| Model Format                 | TorchScript, Tracing|
| Backend Accelerator          | CUDA                |
| Max Batch Size               | 8                   |
| Number of model instances    | 2                   |
| Triton Max Queue Delay       | 1                   |
| Triton Preferred Batch Sizes | 4 8                 |



#### Online: NVIDIA A40 with FP16

Our results were obtained using the following configuration:
 * **GPU:**  NVIDIA A40
 * **Backend:** PyTorch
 * **Precision:** FP16
 * **Model format:** TorchScript
 * **Conversion variant:** Trace



![](plots/graph_performance_online_1.svg)

<details>

<summary>
Full tabular data
</summary>

|   Sequence Length |   Concurrent client requests |   Inferences/second |   Client Send |   Network+server Send/recv |   Server Queue |   Server Compute Input |   Server Compute Infer |   Server Compute Output |   Client Recv |   P50 Latency |   P90 Latency |   P95 Latency |   P99 Latency |   Avg Latency |
|------------------:|-----------------------------:|--------------------:|--------------:|---------------------------:|---------------:|-----------------------:|-----------------------:|------------------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
|               128 |                            1 |                82.1 |         0.061 |                      0.38  |          0.033 |                  0.036 |                 11.501 |                   0.122 |         0.032 |        12.166 |        12.884 |        13.175 |        13.541 |        12.165 |
|               128 |                            2 |               127.6 |         0.062 |                      0.361 |          0.02  |                  0.066 |                 14.944 |                   0.195 |         0.033 |        15.246 |        17.223 |        17.546 |        18.699 |        15.681 |
|               128 |                            3 |               134.6 |         0.048 |                      0.271 |          7.119 |                  0.05  |                 14.54  |                   0.192 |         0.066 |        22.009 |        28.693 |        29.875 |        31.877 |        22.286 |
|               128 |                            4 |               173   |         0.063 |                      0.336 |          7.278 |                  0.062 |                 15.053 |                   0.258 |         0.072 |        23.099 |        29.053 |        30.21  |        32.361 |        23.122 |
|               128 |                            5 |               212.6 |         0.063 |                      0.393 |          7.327 |                  0.075 |                 15.168 |                   0.341 |         0.122 |        23.398 |        29.099 |        30.253 |        32.099 |        23.489 |
|               128 |                            6 |               246.1 |         0.054 |                      0.353 |          7.716 |                  0.087 |                 15.496 |                   0.436 |         0.247 |        24.086 |        30.768 |        31.833 |        33.181 |        24.389 |
|               128 |                            7 |               290.9 |         0.06  |                      0.437 |          7.405 |                  0.094 |                 15.207 |                   0.566 |         0.293 |        23.754 |        30.664 |        31.577 |        33.009 |        24.062 |
|               128 |                            8 |               320.3 |         0.059 |                      0.455 |          7.344 |                  0.117 |                 15.343 |                   1.219 |         0.442 |        24.579 |        31.313 |        32.409 |        34.271 |        24.979 |
|               128 |                            9 |               344.5 |         0.058 |                      0.396 |          7.703 |                  0.134 |                 16.035 |                   1.34  |         0.467 |        25.812 |        31.951 |        33.019 |        34.873 |        26.133 |
|               128 |                           10 |               378.8 |         0.058 |                      0.517 |          7.795 |                  0.137 |                 16.05  |                   1.343 |         0.465 |        26.106 |        32.899 |        34.166 |        36.33  |        26.365 |
|               128 |                           11 |               413.1 |         0.056 |                      0.342 |          7.871 |                  0.141 |                 16.154 |                   1.569 |         0.488 |        26.077 |        33.343 |        34.532 |        36.262 |        26.621 |
|               128 |                           12 |               427.2 |         0.055 |                      0.857 |          8.059 |                  0.158 |                 16.668 |                   1.785 |         0.523 |        28.44  |        34.58  |        36.211 |        37.894 |        28.105 |
|               128 |                           13 |               465.1 |         0.054 |                      0.558 |          8.185 |                  0.157 |                 16.614 |                   1.835 |         0.55  |        27.839 |        34.834 |        36.023 |        37.601 |        27.953 |
|               128 |                           14 |               537.1 |         0.056 |                      0.395 |          7.547 |                  0.146 |                 15.489 |                   1.913 |         0.525 |        25.232 |        32.118 |        33.33  |        35.574 |        26.071 |
|               128 |                           15 |               536   |         0.054 |                      0.382 |          8.166 |                  0.174 |                 16.504 |                   2.122 |         0.555 |        27.507 |        34.662 |        36.181 |        38.592 |        27.957 |
|               128 |                           16 |               560.8 |         0.055 |                      0.472 |          8.434 |                  0.176 |                 16.377 |                   2.446 |         0.601 |        28.267 |        35.102 |        36.282 |        38.229 |        28.561 |

</details>


#### Online: NVIDIA A40 with FP32

Our results were obtained using the following configuration:
 * **GPU:**  NVIDIA A40
 * **Backend:** PyTorch
 * **Precision:** FP32
 * **Model format:** TorchScript
 * **Conversion variant:** Trace


![](plots/graph_performance_online_2.svg)

<details>

<summary>
Full tabular data
</summary>

|   Sequence Length |   Concurrent client requests |   Inferences/second |   Client Send |   Network+server Send/recv |   Server Queue |   Server Compute Input |   Server Compute Infer |   Server Compute Output |   Client Recv |   P50 Latency |   P90 Latency |   P95 Latency |   P99 Latency |   Avg Latency |
|------------------:|-----------------------------:|--------------------:|--------------:|---------------------------:|---------------:|-----------------------:|-----------------------:|------------------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
|               128 |                            1 |               110.2 |         0.052 |                      0.318 |          0.019 |                  0.041 |                  8.412 |                   0.128 |         0.098 |         9.057 |         9.113 |         9.122 |         9.288 |         9.068 |
|               128 |                            2 |               154.8 |         0.045 |                      0.229 |          0.015 |                  0.063 |                 12.179 |                   0.24  |         0.136 |        12.601 |        14.375 |        14.896 |        15.36  |        12.907 |
|               128 |                            3 |               158.3 |         0.046 |                      0.235 |          5.947 |                  0.058 |                 12.271 |                   0.244 |         0.139 |        18.654 |        23.975 |        24.778 |        26.432 |        18.94  |
|               128 |                            4 |               201.3 |         0.059 |                      0.467 |          5.962 |                  0.066 |                 12.642 |                   0.529 |         0.145 |        19.573 |        24.86  |        25.498 |        27.134 |        19.87  |
|               128 |                            5 |               229.8 |         0.061 |                      0.554 |          6.339 |                  0.078 |                 13.62  |                   0.924 |         0.176 |        21.27  |        26.668 |        27.417 |        29.052 |        21.752 |
|               128 |                            6 |               253.2 |         0.057 |                      0.441 |          6.63  |                  0.095 |                 14.46  |                   1.579 |         0.449 |        24.231 |        28.977 |        29.719 |        31.173 |        23.711 |
|               128 |                            7 |               283.8 |         0.057 |                      0.426 |          6.752 |                  0.102 |                 14.749 |                   2.021 |         0.53  |        24.64  |        29.875 |        30.748 |        32.599 |        24.637 |
|               128 |                            8 |               300.9 |         0.056 |                      0.604 |          7.057 |                  0.113 |                 15.442 |                   2.634 |         0.669 |        26.929 |        32.007 |        32.902 |        34.674 |        26.575 |
|               128 |                            9 |               330.7 |         0.054 |                      0.434 |          7.248 |                  0.121 |                 15.833 |                   2.796 |         0.707 |        27.338 |        32.766 |        33.935 |        36.28  |        27.193 |
|               128 |                           10 |               327.1 |         0.055 |                      0.536 |          8.154 |                  0.153 |                 17.753 |                   3.173 |         0.783 |        30.417 |        37.22  |        38.515 |        40.813 |        30.607 |
|               128 |                           11 |               342.8 |         0.054 |                      0.601 |          8.563 |                  0.16  |                 18.398 |                   3.472 |         0.832 |        32.205 |        38.823 |        40.226 |        42.314 |        32.08  |
|               128 |                           12 |               364.3 |         0.054 |                      0.299 |          9.32  |                  0.164 |                 18.918 |                   3.371 |         0.799 |        32.326 |        40.15  |        41.456 |        43.995 |        32.925 |
|               128 |                           13 |               397.3 |         0.052 |                      0.57  |          8.506 |                  0.167 |                 17.784 |                   4.715 |         0.944 |        33.95  |        39.302 |        40.772 |        44.117 |        32.738 |
|               128 |                           14 |               413.5 |         0.051 |                      0.562 |          9.554 |                  0.174 |                 18.423 |                   4.132 |         0.973 |        34.27  |        40.553 |        42.599 |        45.688 |        33.869 |
|               128 |                           15 |               397.6 |         0.048 |                      0.606 |         10.659 |                  0.212 |                 20.533 |                   4.608 |         1.111 |        38.44  |        45.484 |        47.037 |        51.264 |        37.777 |
|               128 |                           16 |               411.4 |         0.053 |                      0.605 |         11.127 |                  0.222 |                 20.87  |                   4.969 |         1.048 |        40.638 |        47.265 |        48.693 |        51.886 |        38.894 |

</details>


#### Online: NVIDIA DGX A100 (1x A100 80GB) with FP16

Our results were obtained using the following configuration:
 * **GPU:**  NVIDIA DGX A100 (1x A100 80GB)
 * **Backend:** PyTorch
 * **Precision:** FP16
 * **Model format:** TorchScript
 * **Conversion variant:** Trace



![](plots/graph_performance_online_3.svg)

<details>

<summary>
Full tabular data
</summary>

|   Sequence Length |   Concurrent client requests |   Inferences/second |   Client Send |   Network+server Send/recv |   Server Queue |   Server Compute Input |   Server Compute Infer |   Server Compute Output |   Client Recv |   P50 Latency |   P90 Latency |   P95 Latency |   P99 Latency |   Avg Latency |
|------------------:|-----------------------------:|--------------------:|--------------:|---------------------------:|---------------:|-----------------------:|-----------------------:|------------------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
|               128 |                            1 |               152.4 |         0.02  |                      0.109 |          0.014 |                  0.031 |                  6.254 |                   0.09  |         0.042 |         6.471 |         6.832 |         6.881 |         6.983 |         6.56  |
|               128 |                            2 |               209.1 |         0.02  |                      0.11  |          0.011 |                  0.048 |                  9.144 |                   0.194 |         0.038 |         9.532 |         9.747 |         9.839 |        10.955 |         9.565 |
|               128 |                            3 |               209   |         0.021 |                      0.069 |          4.669 |                  0.038 |                  9.316 |                   0.203 |         0.036 |        13.806 |        18.261 |        18.589 |        19.265 |        14.352 |
|               128 |                            4 |               268.8 |         0.022 |                      0.128 |          4.809 |                  0.043 |                  9.503 |                   0.318 |         0.063 |        14.609 |        19.148 |        19.459 |        21.103 |        14.886 |
|               128 |                            5 |               329.3 |         0.024 |                      0.071 |          4.884 |                  0.053 |                  9.631 |                   0.462 |         0.061 |        14.759 |        19.328 |        19.901 |        20.689 |        15.186 |
|               128 |                            6 |               381.2 |         0.027 |                      0.094 |          4.866 |                  0.064 |                  9.793 |                   0.767 |         0.129 |        15.497 |        19.599 |        20.151 |        21.114 |        15.74  |
|               128 |                            7 |               437.7 |         0.025 |                      0.071 |          5.05  |                  0.064 |                  9.87  |                   0.778 |         0.138 |        15.723 |        19.844 |        20.748 |        21.68  |        15.996 |
|               128 |                            8 |               480.5 |         0.025 |                      0.211 |          5.163 |                  0.073 |                 10.019 |                   1.006 |         0.158 |        16.31  |        21.126 |        21.547 |        22.021 |        16.655 |
|               128 |                            9 |               526.9 |         0.024 |                      0.134 |          5.266 |                  0.083 |                 10.145 |                   1.217 |         0.199 |        16.933 |        21.398 |        21.97  |        22.583 |        17.068 |
|               128 |                           10 |               574.2 |         0.027 |                      0.252 |          5.106 |                  0.088 |                 10.453 |                   1.275 |         0.215 |        17.445 |        20.922 |        22.044 |        23.077 |        17.416 |
|               128 |                           11 |               607.3 |         0.026 |                      0.233 |          5.498 |                  0.095 |                 10.596 |                   1.46  |         0.224 |        18.007 |        22.761 |        23.277 |        24.159 |        18.132 |
|               128 |                           12 |               642.4 |         0.029 |                      0.258 |          5.654 |                  0.101 |                 10.808 |                   1.587 |         0.24  |        18.578 |        23.363 |        23.816 |        24.722 |        18.677 |
|               128 |                           13 |               661.1 |         0.028 |                      0.228 |          5.964 |                  0.114 |                 11.415 |                   1.666 |         0.247 |        19.496 |        24.522 |        25.26  |        26.797 |        19.662 |
|               128 |                           14 |               709   |         0.029 |                      0.21  |          6.113 |                  0.116 |                 11.203 |                   1.822 |         0.249 |        19.76  |        24.659 |        25.474 |        27.112 |        19.742 |
|               128 |                           15 |               738.8 |         0.029 |                      0.262 |          6.338 |                  0.121 |                 11.369 |                   1.934 |         0.256 |        20.499 |        25.183 |        25.911 |        26.981 |        20.309 |
|               128 |                           16 |               775.8 |         0.027 |                      0.294 |          6.272 |                  0.128 |                 11.568 |                   2.042 |         0.28  |        20.766 |        25.316 |        25.918 |        27.265 |        20.611 |

</details>


#### Online: NVIDIA DGX A100 (1x A100 80GB) with FP32

Our results were obtained using the following configuration:
 * **GPU:**  NVIDIA DGX A100 (1x A100 80GB)
 * **Backend:** PyTorch
 * **Precision:** FP32
 * **Model format:** TorchScript
 * **Conversion variant:** Trace


![](plots/graph_performance_online_4.svg)

<details>

<summary>
Full tabular data
</summary>

|   Sequence Length |   Concurrent client requests |   Inferences/second |   Client Send |   Network+server Send/recv |   Server Queue |   Server Compute Input |   Server Compute Infer |   Server Compute Output |   Client Recv |   P50 Latency |   P90 Latency |   P95 Latency |   P99 Latency |   Avg Latency |
|------------------:|-----------------------------:|--------------------:|--------------:|---------------------------:|---------------:|-----------------------:|-----------------------:|------------------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
|               128 |                            1 |               148.4 |         0.02  |                      0.098 |          0.014 |                  0.032 |                  6.374 |                   0.125 |         0.07  |         6.68  |         6.951 |         7.019 |         7.139 |         6.733 |
|               128 |                            2 |               196.1 |         0.018 |                      0.082 |          0.011 |                  0.052 |                  9.703 |                   0.26  |         0.074 |        10.196 |        10.462 |        10.602 |        12.079 |        10.2   |
|               128 |                            3 |               203.3 |         0.02  |                      0.059 |          4.775 |                  0.041 |                  9.489 |                   0.297 |         0.079 |        14.285 |        19.316 |        19.563 |        20.723 |        14.76  |
|               128 |                            4 |               249.6 |         0.02  |                      0.16  |          5.045 |                  0.047 |                 10.157 |                   0.476 |         0.111 |        15.581 |        20.396 |        21.039 |        21.506 |        16.016 |
|               128 |                            5 |               305.7 |         0.022 |                      0.109 |          5.011 |                  0.06  |                 10.245 |                   0.729 |         0.178 |        15.9   |        20.525 |        21.236 |        21.943 |        16.354 |
|               128 |                            6 |               351.1 |         0.027 |                      0.172 |          5.15  |                  0.063 |                 10.516 |                   0.933 |         0.228 |        16.755 |        20.641 |        22.263 |        23.198 |        17.089 |
|               128 |                            7 |               390.1 |         0.026 |                      0.187 |          5.398 |                  0.069 |                 10.909 |                   1.089 |         0.271 |        17.749 |        22.145 |        22.984 |        23.545 |        17.949 |
|               128 |                            8 |               434.2 |         0.024 |                      0.24  |          5.23  |                  0.08  |                 11.082 |                   1.414 |         0.337 |        18.15  |        21.854 |        22.955 |        24.232 |        18.407 |
|               128 |                            9 |               459.2 |         0.027 |                      0.236 |          5.765 |                  0.083 |                 11.595 |                   1.533 |         0.349 |        19.471 |        23.521 |        24.357 |        25.754 |        19.588 |
|               128 |                           10 |               494.5 |         0.027 |                      0.282 |          6.032 |                  0.097 |                 11.604 |                   1.768 |         0.409 |        20.057 |        25.18  |        25.611 |        26.491 |        20.219 |
|               128 |                           11 |               542.4 |         0.024 |                      0.237 |          5.399 |                  0.103 |                 11.858 |                   2.153 |         0.495 |        20.149 |        23.651 |        24.332 |        26.042 |        20.269 |
|               128 |                           12 |               563   |         0.027 |                      0.302 |          6.266 |                  0.111 |                 11.918 |                   2.183 |         0.486 |        21.361 |        26.142 |        26.604 |        28.143 |        21.293 |
|               128 |                           13 |               597.9 |         0.028 |                      0.152 |          6.492 |                  0.118 |                 12.156 |                   2.274 |         0.512 |        21.719 |        26.516 |        27.27  |        28.705 |        21.732 |
|               128 |                           14 |               619.4 |         0.026 |                      0.303 |          6.576 |                  0.126 |                 12.524 |                   2.498 |         0.557 |        22.577 |        27.346 |        27.928 |        29.136 |        22.61  |
|               128 |                           15 |               657   |         0.024 |                      0.19  |          6.529 |                  0.132 |                 12.703 |                   2.66  |         0.602 |        22.774 |        27.187 |        28.158 |        29.452 |        22.84  |
|               128 |                           16 |               674.9 |         0.028 |                      0.266 |          7.032 |                  0.14  |                 12.847 |                   2.792 |         0.584 |        23.905 |        29.061 |        29.839 |        31.466 |        23.689 |

</details>


#### Online: NVIDIA DGX-1 (1x V100 32GB) with FP16

Our results were obtained using the following configuration:
 * **GPU:**  NVIDIA DGX-1 (1x V100 32GB)
 * **Backend:** PyTorch
 * **Precision:** FP16
 * **Model format:** TorchScript
 * **Conversion variant:** Trace


![](plots/graph_performance_online_5.svg)

<details>

<summary>
Full tabular data
</summary>

|   Sequence Length |   Concurrent client requests |   Inferences/second |   Client Send |   Network+server Send/recv |   Server Queue |   Server Compute Input |   Server Compute Infer |   Server Compute Output |   Client Recv |   P50 Latency |   P90 Latency |   P95 Latency |   P99 Latency |   Avg Latency |
|------------------:|-----------------------------:|--------------------:|--------------:|---------------------------:|---------------:|-----------------------:|-----------------------:|------------------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
|               128 |                            1 |               100.5 |         0.043 |                      0.271 |          0.043 |                  0.039 |                  9.408 |                   0.108 |         0.03  |         9.879 |        10.247 |        10.329 |        10.592 |         9.942 |
|               128 |                            2 |               151.5 |         0.044 |                      0.3   |          0.048 |                  0.067 |                 12.475 |                   0.238 |         0.034 |        12.972 |        14.525 |        15.161 |        15.692 |        13.206 |
|               128 |                            3 |               158.4 |         0.044 |                      0.227 |          6.028 |                  0.045 |                 12.296 |                   0.25  |         0.037 |        18.563 |        24.091 |        24.562 |        25.234 |        18.927 |
|               128 |                            4 |               205.4 |         0.044 |                      0.249 |          6.129 |                  0.055 |                 12.41  |                   0.516 |         0.067 |        18.767 |        25.126 |        25.524 |        26.199 |        19.47  |
|               128 |                            5 |               242.4 |         0.044 |                      0.308 |          6.384 |                  0.065 |                 12.824 |                   0.888 |         0.11  |        20.052 |        26.303 |        26.858 |        27.476 |        20.623 |
|               128 |                            6 |               279.6 |         0.044 |                      0.301 |          6.585 |                  0.075 |                 13.074 |                   1.237 |         0.14  |        20.76  |        27.575 |        28.037 |        28.974 |        21.456 |
|               128 |                            7 |               314   |         0.046 |                      0.269 |          6.844 |                  0.08  |                 13.385 |                   1.48  |         0.196 |        21.705 |        28.573 |        29.121 |        29.847 |        22.3   |
|               128 |                            8 |               342.8 |         0.047 |                      0.452 |          6.695 |                  0.097 |                 13.94  |                   1.826 |         0.26  |        23.164 |        29.564 |        30.467 |        31.278 |        23.317 |
|               128 |                            9 |               364.6 |         0.047 |                      0.375 |          7.022 |                  0.103 |                 14.39  |                   2.373 |         0.347 |        24.599 |        31.093 |        31.868 |        32.917 |        24.657 |
|               128 |                           10 |               389.3 |         0.048 |                      0.448 |          7.375 |                  0.115 |                 14.873 |                   2.477 |         0.345 |        25.412 |        31.847 |        32.733 |        34.499 |        25.681 |
|               128 |                           11 |               411.3 |         0.047 |                      0.466 |          7.65  |                  0.125 |                 15.464 |                   2.582 |         0.38  |        26.432 |        33.057 |        34.029 |        36.509 |        26.714 |
|               128 |                           12 |               439.7 |         0.047 |                      0.546 |          8.002 |                  0.125 |                 15.342 |                   2.873 |         0.363 |        27.282 |        33.765 |        34.579 |        36.181 |        27.298 |
|               128 |                           13 |               458.6 |         0.049 |                      0.46  |          8.421 |                  0.139 |                 15.689 |                   3.173 |         0.402 |        28.226 |        34.756 |        35.961 |        38.42  |        28.333 |
|               128 |                           14 |               479.8 |         0.048 |                      0.528 |          8.631 |                  0.144 |                 16.278 |                   3.124 |         0.421 |        28.925 |        35.885 |        37.331 |        39.311 |        29.174 |
|               128 |                           15 |               494.2 |         0.048 |                      0.488 |          9.049 |                  0.147 |                 16.642 |                   3.558 |         0.441 |        30.541 |        37.113 |        38.568 |        40.605 |        30.373 |
|               128 |                           16 |               516.9 |         0.049 |                      0.61  |          9.469 |                  0.166 |                 16.669 |                   3.601 |         0.409 |        31.962 |        38.323 |        39.16  |        40.616 |        30.973 |

</details>



#### Online: NVIDIA DGX-1 (1x V100 32GB) with FP32

Our results were obtained using the following configuration:
 * **GPU:**  NVIDIA DGX-1 (1x V100 32GB)
 * **Backend:** PyTorch
 * **Precision:** FP32
 * **Model format:** TorchScript
 * **Conversion variant:** Trace


![](plots/graph_performance_online_6.svg)

<details>

<summary>
Full tabular data
</summary>

|   Sequence Length |   Concurrent client requests |   Inferences/second |   Client Send |   Network+server Send/recv |   Server Queue |   Server Compute Input |   Server Compute Infer |   Server Compute Output |   Client Recv |   P50 Latency |   P90 Latency |   P95 Latency |   P99 Latency |   Avg Latency |
|------------------:|-----------------------------:|--------------------:|--------------:|---------------------------:|---------------:|-----------------------:|-----------------------:|------------------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
|               128 |                            1 |               110.6 |         0.038 |                      0.203 |          0.017 |                  0.033 |                  7.407 |                   1.227 |         0.109 |         8.989 |         9.095 |         9.201 |        10.374 |         9.034 |
|               128 |                            2 |               119.4 |         0.048 |                      0.284 |          0.055 |                  0.204 |                 14.442 |                   1.613 |         0.099 |        16.705 |        17.275 |        17.478 |        17.934 |        16.745 |
|               128 |                            3 |               118.3 |         0.043 |                      0.368 |          8.044 |                  0.065 |                 15.021 |                   1.707 |         0.111 |        26.011 |        31.049 |        31.999 |        33.798 |        25.359 |
|               128 |                            4 |               140   |         0.042 |                      0.278 |          8.922 |                  0.077 |                 15.948 |                   3.114 |         0.17  |        28.949 |        35.762 |        36.454 |        38.914 |        28.551 |
|               128 |                            5 |               159.3 |         0.044 |                      0.303 |          9.009 |                  0.097 |                 17.258 |                   4.412 |         0.254 |        31.81  |        37.571 |        38.675 |        41.042 |        31.377 |
|               128 |                            6 |               165.4 |         0.044 |                      0.378 |          9.866 |                  0.113 |                 20.096 |                   5.443 |         0.345 |        37.16  |        43.107 |        45.435 |        52.102 |        36.285 |
|               128 |                            7 |               180.8 |         0.045 |                      0.308 |         11.011 |                  0.147 |                 20.175 |                   6.605 |         0.388 |        39.446 |        46.791 |        49.684 |        54.777 |        38.679 |
|               128 |                            8 |               192.2 |         0.048 |                      0.36  |         11.298 |                  0.153 |                 21.965 |                   7.467 |         0.414 |        42.309 |        51.787 |        55.15  |        58.38  |        41.705 |
|               128 |                            9 |               200.5 |         0.048 |                      0.357 |         12.823 |                  0.158 |                 23.488 |                   7.594 |         0.474 |        45.72  |        53.947 |        55.908 |        61.154 |        44.942 |
|               128 |                           10 |               208.7 |         0.047 |                      0.421 |         13.27  |                  0.162 |                 24.334 |                   9.03  |         0.6   |        48.705 |        57.995 |        59.473 |        65.057 |        47.864 |
|               128 |                           11 |               214.3 |         0.047 |                      0.395 |         15.778 |                  0.217 |                 24.846 |                   9.588 |         0.483 |        52.653 |        63.823 |        66.897 |        69.067 |        51.354 |
|               128 |                           12 |               215.7 |         0.048 |                      0.616 |         15.895 |                  0.24  |                 25.579 |                  12.456 |         0.648 |        56.333 |        63.09  |        64.429 |        74.218 |        55.482 |
|               128 |                           13 |               222.5 |         0.048 |                      0.397 |         16.294 |                  0.24  |                 28.246 |                  12.469 |         0.645 |        59.08  |        69.552 |        73.32  |        81.029 |        58.339 |
|               128 |                           14 |               228.2 |         0.05  |                      0.496 |         18.186 |                  0.27  |                 29.653 |                  12.178 |         0.562 |        62.211 |        72.935 |        77.152 |        83.805 |        61.395 |
|               128 |                           15 |               234   |         0.05  |                      0.418 |         19.624 |                  0.317 |                 30.497 |                  12.504 |         0.569 |        64.758 |        79.884 |        82.316 |        86.467 |        63.979 |
|               128 |                           16 |               236   |         0.048 |                      0.379 |         21.46  |                  0.352 |                 30.808 |                  14.245 |         0.566 |        69.054 |        82.334 |        87.213 |        94.892 |        67.858 |

</details>



#### Online: NVIDIA T4 with FP16

Our results were obtained using the following configuration:
 * **GPU:**  NVIDIA T4
 * **Backend:** PyTorch
 * **Precision:** FP16
 * **Model format:** TorchScript
 * **Conversion variant:** Trace


![](plots/graph_performance_online_7.svg)

<details>

<summary>
Full tabular data
</summary>

|   Sequence Length |   Concurrent client requests |   Inferences/second |   Client Send |   Network+server Send/recv |   Server Queue |   Server Compute Input |   Server Compute Infer |   Server Compute Output |   Client Recv |   P50 Latency |   P90 Latency |   P95 Latency |   P99 Latency |   Avg Latency |
|------------------:|-----------------------------:|--------------------:|--------------:|---------------------------:|---------------:|-----------------------:|-----------------------:|------------------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
|               128 |                            1 |                53.6 |         0.102 |                      0.56  |          0.087 |                  0.105 |                 17.485 |                   0.261 |         0.052 |        18.882 |        19.597 |        19.712 |        19.948 |        18.652 |
|               128 |                            2 |               129.9 |         0.097 |                      0.494 |          0.017 |                  0.291 |                 12.386 |                   2.059 |         0.054 |        15.273 |        16.187 |        16.906 |        21.99  |        15.398 |
|               128 |                            3 |               122.3 |         0.098 |                      0.506 |          7.577 |                  0.07  |                 14.428 |                   1.796 |         0.049 |        24.851 |        30.177 |        32.726 |        34.667 |        24.524 |
|               128 |                            4 |               141.4 |         0.095 |                      0.533 |          8.459 |                  0.083 |                 16.254 |                   2.798 |         0.064 |        28.512 |        34.407 |        36.983 |        40.366 |        28.286 |
|               128 |                            5 |               153.1 |         0.097 |                      0.613 |          9.277 |                  0.095 |                 18.608 |                   3.878 |         0.114 |        32.559 |        40.931 |        43.966 |        47.479 |        32.682 |
|               128 |                            6 |               168.6 |         0.098 |                      0.587 |          9.407 |                  0.115 |                 20.512 |                   4.603 |         0.222 |        35.182 |        45.268 |        47.867 |        51.381 |        35.544 |
|               128 |                            7 |               184.3 |         0.094 |                      0.697 |          9.432 |                  0.13  |                 21.351 |                   6.037 |         0.259 |        36.83  |        50.213 |        54.732 |        62.848 |        38     |
|               128 |                            8 |               187   |         0.093 |                      0.665 |         11.347 |                  0.155 |                 23.914 |                   6.27  |         0.257 |        41.379 |        57.516 |        62.209 |        66.726 |        42.701 |
|               128 |                            9 |               199.5 |         0.094 |                      0.775 |         11.261 |                  0.163 |                 24.54  |                   7.938 |         0.385 |        44.016 |        58.752 |        65.017 |        71.694 |        45.156 |
|               128 |                           10 |               210.2 |         0.091 |                      0.897 |         11.848 |                  0.183 |                 24.714 |                   9.401 |         0.449 |        44.964 |        65.754 |        73.463 |        79.672 |        47.583 |
|               128 |                           11 |               217.3 |         0.092 |                      0.838 |         12.487 |                  0.202 |                 25.694 |                  10.75  |         0.523 |        47.864 |        69.923 |        77.628 |        85.826 |        50.586 |
|               128 |                           12 |               219.6 |         0.09  |                      0.771 |         14.799 |                  0.206 |                 27.126 |                  11.095 |         0.495 |        52.728 |        73.813 |        79.036 |        95.389 |        54.582 |
|               128 |                           13 |               227.6 |         0.09  |                      0.758 |         14.886 |                  0.247 |                 29.603 |                  10.932 |         0.527 |        54.152 |        80.264 |        86.911 |        97.091 |        57.043 |
|               128 |                           14 |               235   |         0.093 |                      0.64  |         15.942 |                  0.26  |                 29.521 |                  12.755 |         0.519 |        56.969 |        82.85  |        89.545 |       104.486 |        59.73  |
|               128 |                           15 |               236.7 |         0.092 |                      0.686 |         17.532 |                  0.294 |                 31.765 |                  12.432 |         0.557 |        59.681 |        91.908 |       100.856 |       119.919 |        63.358 |
|               128 |                           16 |               242.3 |         0.091 |                      0.693 |         16.804 |                  0.289 |                 32.901 |                  14.663 |         0.559 |        63.006 |        96.607 |        99.376 |       108.381 |        66     |

</details>



#### Online: NVIDIA T4 with FP32

Our results were obtained using the following configuration:
 * **GPU:**  NVIDIA T4
 * **Backend:** PyTorch
 * **Precision:** FP32
 * **Model format:** TorchScript
 * **Conversion variant:** Trace


![](plots/graph_performance_online_8.svg)

<details>

<summary>
Full tabular data
</summary>

|   Sequence Length |   Concurrent client requests |   Inferences/second |   Client Send |   Network+server Send/recv |   Server Queue |   Server Compute Input |   Server Compute Infer |   Server Compute Output |   Client Recv |   P50 Latency |   P90 Latency |   P95 Latency |   P99 Latency |   Avg Latency |
|------------------:|-----------------------------:|--------------------:|--------------:|---------------------------:|---------------:|-----------------------:|-----------------------:|------------------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
|               128 |                            1 |                53.5 |         0.103 |                      0.57  |          0.085 |                  0.108 |                 16.195 |                   1.506 |         0.112 |        18.777 |        19.448 |        19.513 |        19.697 |        18.679 |
|               128 |                            2 |                78.1 |         0.097 |                      0.476 |          0.021 |                  0.37  |                 19.778 |                   4.735 |         0.113 |        19.266 |        48.198 |        50.37  |        51.933 |        25.59  |
|               128 |                            3 |                78.9 |         0.092 |                      0.511 |         12.039 |                  0.126 |                 20.597 |                   4.568 |         0.104 |        34.628 |        55.275 |        62.943 |        69.63  |        38.037 |
|               128 |                            4 |                86.4 |         0.094 |                      0.492 |         14.143 |                  0.163 |                 24.336 |                   6.955 |         0.16  |        42.424 |        69.874 |        73.991 |        81.048 |        46.343 |
|               128 |                            5 |                87.4 |         0.096 |                      0.569 |         16.207 |                  0.174 |                 28.415 |                  11.335 |         0.344 |        52.867 |        85.206 |        92.721 |       106.801 |        57.14  |
|               128 |                            6 |                91.5 |         0.094 |                      0.644 |         16.815 |                  0.207 |                 33.454 |                  13.923 |         0.471 |        62.079 |        96.925 |       100.852 |       115.651 |        65.608 |
|               128 |                            7 |                96.3 |         0.094 |                      0.622 |         18.675 |                  0.219 |                 36.551 |                  16.332 |         0.621 |        69.447 |       103.115 |       108.706 |       130.277 |        73.114 |
|               128 |                            8 |                95.7 |         0.096 |                      0.642 |         18.336 |                  0.24  |                 41.708 |                  21.953 |         0.868 |        79.887 |       113.645 |       117.36  |       145.151 |        83.843 |
|               128 |                            9 |                95.2 |         0.095 |                      1.01  |         18.682 |                  0.249 |                 48.823 |                  24.68  |         1.059 |        90.799 |       126.669 |       129.592 |       167.038 |        94.598 |
|               128 |                           10 |               102.6 |         0.093 |                      0.767 |         19.687 |                  0.26  |                 46.234 |                  29.561 |         1.219 |        95.095 |       121.245 |       128.962 |       170.8   |        97.821 |
|               128 |                           11 |               104.9 |         0.09  |                      0.629 |         23.884 |                  0.317 |                 49.746 |                  29.621 |         1.19  |       101.884 |       133.615 |       141.351 |       186.759 |       105.477 |
|               128 |                           12 |               103.8 |         0.093 |                      0.427 |         29.107 |                  0.375 |                 52.974 |                  32.07  |         1.145 |       113.659 |       154.182 |       172.429 |       204.619 |       116.191 |
|               128 |                           13 |               104   |         0.096 |                      0.458 |         30.526 |                  0.433 |                 58.923 |                  33.204 |         1.247 |       120.19  |       174.267 |       189.165 |       216.331 |       124.887 |
|               128 |                           14 |               106.1 |         0.091 |                      0.401 |         38.587 |                  0.443 |                 60.805 |                  30.968 |         1.081 |       127.547 |       182.202 |       198.122 |       222.625 |       132.376 |
|               128 |                           15 |               106.5 |         0.09  |                      1.093 |         38.282 |                  0.47  |                 63.64  |                  36.439 |         1.256 |       138.848 |       182.504 |       203.954 |       219.243 |       141.27  |
|               128 |                           16 |               104.9 |         0.089 |                      0.365 |         41.181 |                  0.51  |                 68.818 |                  39.515 |         1.402 |       148.399 |       223.069 |       230.082 |       257.301 |       151.88  |

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
- Initial release

### Known issues

There are no known issues with this model.

# Deploying the Jasper Inference model using Triton Inference Server
 
This subfolder of the Jasper for PyTorch repository contains scripts for  deployment of high-performance inference on NVIDIA Triton Inference Server as well as detailed performance analysis. It offers different options for the inference model pipeline.
 
 
## Table Of Contents
- [Solution overview](#solution-overview)
- [Inference Pipeline in Triton Inference Server](#inference-pipeline-in-triton-inference-server)
- [Setup](#setup)
- [Quick Start Guide](#quick-start-guide)
- [Advanced](#advanced)
     * [Scripts and sample code](#scripts-and-sample-code)
- [Performance](#performance)
     * [Inference Benchmarking in Triton Inference Server](#inference-benchmarking-in-triton-inference-server)
     * [Results](#results)
       * [Performance Analysis for Triton Inference Server: NVIDIA T4
](#performance-analysis-for-triton-inference-server-nvidia-t4)
       * [Maximum batch size](#maximum-batch-size)
            * [Batching techniques: Static versus Dynamic Batching](#batching-techniques-static-versus-dynamic)
            * [TensorRT, ONNX, and PyTorch JIT comparisons](#tensorrt-onnx-and-pytorch-jit-comparisons)
- [Release Notes](#release-notes)
	* [Changelog](#change-log)
	* [Known issues](#known-issues)
 
 
## Solution Overview
 
The [NVIDIA Triton Inference Server](https://github.com/NVIDIA/triton-inference-server) provides a datacenter and cloud inferencing solution optimized for NVIDIA GPUs. The server provides an inference service via an HTTP or gRPC endpoint, allowing remote clients to request inferencing for any number of GPU or CPU models being managed by the server.
 
This folder contains detailed performance analysis as well as scripts to run Jasper inference using Triton Inference Server.
 
A typical Triton Inference Server pipeline can be broken down into the following steps:
 
1. The client serializes the inference request into a message and sends it to the server (Client Send).
2. The message travels over the network from the client to the server (Network).
3. The message arrives at the server, and is deserialized (Server Receive).
4. The request is placed on the queue (Server Queue).
5. The request is removed from the queue and computed (Server Compute).
6. The completed request is serialized in a message and sent back to the client (Server Send).
7. The completed message then travels over the network from the server to the client (Network).
8. The completed message is deserialized by the client and processed as a completed inference request (Client Receive).
 
Generally, for local clients, steps 1-4 and 6-8 will only occupy a small fraction of time, compared to step 5. As backend deep learning systems like Jasper are rarely exposed directly to end users, but instead only interfacing with local front-end servers, for the sake of Jasper, we can consider that all clients are local.
 
In this section, we will go over how to launch both the Triton Inference Server and the client and get the best performance solution that fits your specific application needs.
 
More information on how to perform inference using NVIDIA Triton Inference Server can be found in [triton/README.md](https://github.com/triton-inference-server/server/blob/master/README.md).
 
 
## Inference Pipeline in Triton Inference Server
 
The Jasper model pipeline consists of 3 components, where each part can be customized to be a different backend:
 
**Data preprocessor**
 
The data processor transforms an input raw audio file into a spectrogram. By default the pipeline uses mel filter banks as spectrogram features. This part does not have any learnable weights.
 
**Acoustic model**
 
The acoustic model takes in the spectrogram and outputs a probability over a list of characters. This part is the most compute intensive, taking more than 90% of the entire end-to-end pipeline. The acoustic model is the only component with learnable parameters and what differentiates Jasper from other end-to-end neural speech recognition models. In the original paper, the acoustic model contains a masking operation for training (More details in [Jasper PyTorch README](../README.md)). We do not use masking for inference.
 
**Greedy decoder**
 
The decoder takes the probabilities over the list of characters and outputs the final transcription. Greedy decoding is a fast and simple way of doing this by always choosing the character with the maximum probability.
 
To run a model with TensorRT, we first construct the model in PyTorch, which is then exported into a ONNX static graph. Finally, a TensorRT engine is constructed from the ONNX file and can be launched to do inference. The following table shows which backends are supported for each part along the model pipeline.
 
|Backend\Pipeline component|Data preprocessor|Acoustic Model|Decoder|
|---|---|---|---|
|PyTorch JIT|x|x|x|
|ONNX|-|x|-|
|TensorRT|-|x|-|
 
In order to run inference with TensorRT outside of the inference server, refer to the [Jasper TensorRT README](../tensort/README.md).
 
 
## Setup
 
The repository contains a folder `./triton` with a `Dockerfile` which extends the PyTorch 20.10-py3 NGC container and encapsulates some dependencies. Ensure you have the following components:
 
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [PyTorch 20.10-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
- [Triton Inference Server 20.10 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver)
- Access to [NVIDIA machine learning repository](https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb) and [NVIDIA CUDA repository](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb) for NVIDIA TensorRT 6
- Supported GPUs:
    - [NVIDIA Volta architecture](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
    - [NVIDIA Turing architecture](https://www.nvidia.com/en-us/geforce/turing/)
    - [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)
- [Pretrained Jasper Model Checkpoint](https://ngc.nvidia.com/catalog/models/nvidia:jasper_pyt_ckpt_amp)
 
Required Python packages are listed in `requirements.txt`. These packages are automatically installed when the Docker container is built.
 
 
## Quick Start Guide
 
Running the following scripts will build and launch the container containing all required dependencies for native PyTorch as well as Triton. This is necessary for using inference and can also be used for data download, processing, and training of the model. For more information on the scripts and arguments, refer to the [Advanced](#advanced) section.
 
1. Clone the repository.
 
    ```bash
    git clone https://github.com/NVIDIA/DeepLearningExamples
    cd DeepLearningExamples/PyTorch/SpeechRecognition/Jasper
    ```
 
2. Build the Jasper PyTorch container.
 
    Running the following scripts will build the container which contains all the required dependencies for data download and processing as well as converting the model.
 
    ```bash
    bash scripts/docker/build.sh
    ```
 
3. Start an interactive session in the Docker container:
 
    ```bash
    bash scripts/docker/launch.sh <DATA_DIR> <CHECKPOINT_DIR> <RESULT_DIR>
    ```
 
    Where <DATA_DIR>, <CHECKPOINT_DIR> and <RESULT_DIR> can be either empty or absolute directory paths to dataset, existing checkpoints or potential output files. When left empty, they default to `datasets/`, `/checkpoints`, and `results/`, respectively. The `/datasets`, `/checkpoints`, `/results` directories will be mounted as volumes and mapped to the corresponding directories `<DATA_DIR>`, `<CHECKPOINT_DIR>`, `<RESULT_DIR>` on the host.
 
    Note that `<DATA_DIR>`, `<CHECKPOINT_DIR>`, and `<RESULT_DIR>` directly correspond to the same arguments in `scripts/docker/launch.sh` and `trt/scripts/docker/launch.sh` mentioned in the [Jasper PyTorch README](../README.md) and [Jasper TensorRT README](../tensorrt/README.md).
 
    Briefly, `<DATA_DIR>` should contain, or be prepared to contain a `LibriSpeech` sub-directory (created in [Acquiring Dataset](../trt/README.md)), `<CHECKPOINT_DIR>` should contain a PyTorch model checkpoint (`*.pt`) file obtained through training described in [Jasper PyTorch README](../README.md), and `<RESULT_DIR>` should be prepared to contain converted model and logs.
 
4. Downloading the `test-clean` part of `LibriSpeech` is required for model conversion. But it is not required for inference on Triton Inference Server, which can use a single .wav audio file. To download and preprocess LibriSpeech, run the following inside the container:
 
   ```bash
   bash triton/scripts/download_triton_librispeech.sh
   bash triton/scripts/preprocess_triton_librispeech.sh
   ```
 
5. Convert pretrained PyTorch model checkpoint into Triton Inference Server compatible model backends.
 
    Inside the container, run:
 
    ```bash
    export CHECKPOINT_PATH=<CHECKPOINT_PATH>
    export CONVERT_PRECISIONS=<CONVERT_PRECISIONS>
    export CONVERTS=<CONVERTS>
    bash triton/scripts/export_model.sh
    ```
 
    Where `<CHECKPOINT_PATH>` (`"/checkpoints/jasper_fp16.pt"`) is the absolute file path of the pretrained checkpoint, `<CONVERT_PRECISIONS>` (`"fp16" "fp32"`) is the list of precisions used for conversion, and `<CONVERTS>` (`"feature-extractor" "decoder" "ts-trace" "onnx" "tensorrt"`) is the list of conversions to be applied. The feature extractor converts only to TorchScript trace module (`feature-extractor`), the decoder only to TorchScript script module (`decoder`), and the Jasper model can convert to TorchScript trace module (`ts-trace`), ONNX (`onnx`), or TensorRT (`tensorrt`).
 
    A pretrained PyTorch model checkpoint for model conversion can be downloaded from the [NGC model repository](https://ngc.nvidia.com/catalog/models/nvidia:jasperpyt_fp16).
 
    More details can be found in the [Advanced](#advanced) section under [Scripts and sample code](#scripts-and-sample-code).
 
6. Build a container that extends Triton Inference Client:
 
    From outside the container, run:
 
    ```bash
    bash triton/scripts/docker/build_triton_client.sh
    ```
 
Once the above steps are completed you can either run inference benchmarks or perform inference on real data.
 
7. (Option 1) Run all inference benchmarks.
 
    From outside the container, run:
 
    ```bash
    export RESULT_DIR=<RESULT_DIR>
    export PRECISION_TESTS=<PRECISION_TESTS>
    export BATCH_SIZES=<BATCH_SIZES>
    export SEQ_LENS=<SEQ_LENS>
    bash triton/scripts/execute_all_perf_runs.sh
    ```
 
    Where `<RESULT_DIR>` is the absolute path to potential output files (`./results`), `<PRECISION_TESTS>` is a list of precisions to be tested (`"fp16" "fp32"`), `<BATCH_SIZES>` is a list of tested batch sizes (`"1" "2" "4" "8"`), and `<SEQ_LENS>` are tested sequnce lengths (`"32000" "112000" "267200"`).
 
    Note: This can take several hours to complete due to the extensiveness of the benchmark. More details about the benchmark are found in the [Advanced](#advanced) section under [Performance](#performance).
 
8. (Option 2) Run inference on real data using the Client and Triton Inference Server.
 
    8.1 From outside the container, restart the server:
 
    ```bash
    bash triton/scripts/run_server.sh <MODEL_TYPE> <PRECISION>
    ```
 
    8.2 From outside the container, submit the client request using:
    ```bash
    bash triton/scripts/run_client.sh <MODEL_TYPE> <DATA_DIR> <FILE>
    ```
 
    Where `<MODEL_TYPE>` can be either "ts-trace", "tensorrt" or "onnx", `<PRECISION>` is either "fp32" or "fp16". `<DATA_DIR>` is an absolute local path to the directory of files. <FILE> is the relative path to <DATA_DIR> to either an audio file in .wav format or a manifest file in .json format.
 
    Note: If <FILE> is *.json <DATA_DIR> should be the path to the LibriSpeech dataset. In this case this script will do both inference and evaluation on the accoring LibriSpeech dataset.
 
 
## Advanced
 
The following sections provide greater details about the Triton Inference Server pipeline and inference analysis and benchmarking results.
 
 
### Scripts and sample code
 
The `triton/` directory contains the following files:
* `jasper-client.py`: Python client script that takes an audio file and a specific model pipeline type and submits a client request to the server to run inference with the model on the given audio file.
* `speech_utils.py`: helper functions for `jasper-client.py`.
* `converter.py`: Python script for model conversion to different backends.
* `jasper_module.py`: helper functions for `converter.py`.
* `model_repo_configs/`: directory with Triton model config files for different backend and precision configurations.
 
The `triton/scripts/` directory has easy to use scripts to run supported functionalities, such as:
* `./docker/build_triton_client.sh`: builds container
* `execute_all_perf_runs.sh`: runs all benchmarks using Triton Inference Server performance client; calls `generate_perf_results.sh`
* `export_model.sh`: from pretrained PyTorch checkpoint generates backends for every version of the model inference pipeline.
* `prepare_model_repository.sh`: copies model config files from `./model_repo_configs/` to `./deploy/model_repo` and creates links to generated model backends, setting up the model repository for Triton Inference Server
* `generate_perf_results.sh`: runs benchmark with `perf-client` for specific configuration and calls `run_perf_client.sh`
* `run_server.sh`: launches Triton Inference Server
* `run_client.sh`: launches client by using `jasper-client.py` to submit inference requests to server


### Running the Triton Inference Server
 
Launch the Triton Inference Server in detached mode to run in the background by default:
 
```bash
bash triton/scripts/run_server.sh
```
 
To run in the foreground interactively, for debugging purposes, run:
 
```bash
DAEMON="--detach=false" bash trinton/scripts/run_server.sh
```
 
The script mounts and loads models at `$PWD/triton/deploy/model_repo` to the server with all visible GPUs. In order to selectively choose the devices, set `NVIDIA_VISIBLE_DEVICES`.
 
 
### Running the Triton Inference Client
 
*Real data*
In order to run the client with real data, run:
 
```bash
bash triton/scripts/run_client.sh <backend> <data directory> <audio file>
```
 
The script calls `triton/jasper-client.py` which preprocesses data and sends/receives requests to/from the server.
 
*Synthetic data*
In order to run the client with synthetic data for performance measurements, run:
 
```bash
export MODEL_NAME=jasper-tensorrt-ensemble
export MODEL_VERSION=1
export BATCH_SIZE=1
export MAX_LATENCY=500
export MAX_CONCURRENCY=64
export AUDIO_LENGTH=32000
export SERVER_HOSTNAME=localhost
export RESULT_DIR_H=${PWD}/results/perf_client/${MODEL_NAME}/batch_${BATCH_SIZE}_len_${AUDIO_LENGTH}
bash triton/scripts/run_perf_client.sh
```
 
The export values above are default values. The script waits until the server is up and running, sends requests as per the constraints set and writes results to `/results/results_${TIMESTAMP}.csv` where `TIMESTAMP=$(date "+%y%m%d_%H%M")` and `/results/` is the results directory mounted in the docker .
 
For more information about `perf_client`, refer to the [official documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/optimization.html#perf-client).
 
 
## Performance
 
### Inference Benchmarking in Triton Inference Server
 
To benchmark the inference performance on Volta Turing or Ampere GPU, run `bash triton/scripts/execute_all_perf_runs.sh` according to [Quick-Start-Guide](#quick-start-guide) Step 7.
 
By default, this script measures inference performance for all 3 model pipelines: PyTorch JIT  (`ts-trace`) pipeline, ONNX (`onnx`) pipeline, TensorRT(`tensorrt`) pipeline, both with FP32 and FP16 precision. Each of these pipelines is measured for different audio input lengths (2sec, 7sec, 16.7sec) and a range of different server batch sizes (up to 8). This takes place in `triton/scripts/generate_perf_results.sh`. For a specific audio length and batch size, static and dynamic batching comparison is performed. 
 
 
### Results
 
#### Performance Analysis for Triton Inference Server: NVIDIA T4
 
Based on the figure below, we recommend using the Dynamic Batcher with `max_batch_size=8`, `max_queue_delay_microseconds` as large as possible to fit within your latency window (the values used below are extremely large to exaggerate their effect). The largest improvements to both throughput and latency come from increasing the batch size due to efficiency gains in the GPU with larger batches. The Dynamic Batcher combines the best of both worlds by efficiently batching together a large number of concurrent requests, while also keeping latency down for infrequent requests.
 
All results below are obtained using the following configurations:
* Single T4 16GB GPU on a local server
* FP16 precision
 
Latencies are indicated by bar plots using the left axis. Throughput is indicated by the blue line plot using the right axis. X-axis indicates the concurrency - the maximum number of inference requests that can be in the pipeline at any given time. For example, when the concurrency is set to 1, the client waits for an inference request to be completed (Step 8) before it sends another to the server (Step 1). A high number of concurrent requests can reduce the impact of network latency on overall throughput.
 
 
| ![](../images/Lat_vs_Thr_FP16_Len_2sec_Concurrency_1.png) | ![](../images/Lat_vs_Thr_FP16_Len_2sec_Max_thr_at_conc.png) |
|:-------------------------:|:-------------------------:|
<td colspan=2>Figure 1: Latency vs Throughput for Jasper, FP16, Audio Length = 2sec using various configurations and TensorRT model backends  available in Triton Inference Server. Results are shown for concurrency = 1 (left) and max throughput with reached concurrency in brackets (right).
 
 
| ![](../images/Lat_vs_Thr_FP16_Len_7sec_Concurrency_1.png) | ![](../images/Lat_vs_Thr_FP16_Len_7sec_Max_thr_at_conc.png) |
|:-------------------------:|:-------------------------:|
<td colspan=2>Figure 2: Latency vs Throughput for Jasper, FP16, Audio Length = 7sec using various configurations and TensorRT model backends  available in Triton Inference Server. Results are shown for concurrency = 1 (left) and max throughput with reached concurrency in brackets (right).
 
| ![](../images/Lat_vs_Thr_FP16_Len_16x7sec_Concurrency_1.png) | ![](../images/Lat_vs_Thr_FP16_Len_16x7sec_Max_thr_at_conc.png) |
|:-------------------------:|:-------------------------:|
<td colspan=2>Figure 3: Latency vs Throughput for Jasper, FP16, Audio Length = 16.7sec using various configurations and TensorRT model backends  available in Triton Inference Server. Results are shown for concurrency = 1 (left) and max throughput with reached concurrency in brackets (right).
 
 
##### Maximum Batch Size
 
In general, increasing batch size leads to higher throughput at the cost of higher latency. In the following sections, we analyze the results using the example of the Triton pipeline.
 
As we can see in Figure 4, the throughput at Batch Size=1, Client Concurrent Requests = 8 is 47 and in Figure 3, the throughput at Batch Size=8, Client Concurrent Requests = 1 is 109, giving a speedup of ~2.32x.
Note: We compare Batch Size=1, Client Concurrent Requests = 8 to Batch Size=8, Client Concurrent Requests = 1 to keep the Total Number of Outstanding Requests equal between the two different modes (where Total Number of Outstanding Requests = Batch Size * Client Concurrent Requests).
Increasing the batch size by 8-fold from 1 to 8 results in an increase in compute time by only 2.29x (28ms to 64ms) showing that computation is more efficient at higher batch sizes. Hence, an optimal batch size would be the maximum batch size that can both fit in memory and is within the preferred latency threshold.
 
| ![](../images/length_7_batch_size_1.png) | ![](../images/length_7_batch_size_8.png) |
|:-------------------------:|:-------------------------:|
| Figure 4: Latency & Throughput vs Concurrency using Static Batching at Batch size = 1 | Figure 5: Triton pipeline - Latency & Throughput vs Concurrency using Static Batching at Batch size = 8 |
 
 
##### Batching techniques: Static versus Dynamic Batching
 
In the following section, we analyze the results using the example of the Triton pipeline.
 
Static batching is a feature of the inference server that allows inference requests to be served as they are received. It is preferred in scenarios where low latency is desired at the cost of throughput when the GPU is under utilized.
 
Dynamic batching is a feature of the inference server that allows inference requests to be combined by the server, so that a batch is created dynamically, resulting in an increased throughput. It is preferred in scenarios where we would like to maximize throughput and GPU utilization at the cost of higher latencies. You can set the Dynamic Batcher parameter `max_queue_delay_microseconds` to indicate the maximum amount of time you are willing to wait and `preferred_batch_size` to indicate your maximum server batch size in the Triton Inference Server model config.
 
Figure 5 emphasizes the increase in overall throughput with dynamic batching. At low numbers of concurrent requests, the increased throughput comes at the cost of increasing latency as the requests are queued up to max_queue_delay_microseconds. The effect of preferred_batchsize for dynamic batching is visually depicted by the dip in Server Queue time at integer multiples of the preferred batch sizes. At higher numbers of concurrent requests, the throughput approaches a maximum limit as we saturate the GPU utilization.
 
| ![](../images/length_7_batch_size_8_dynamic.png) |
|:-------------------------:|
| Figure 5: Triton pipeline - Latency & Throughput vs Concurrency using dynamic Batching at maximum server batch size=8, max_queue_delay_microseconds = 5000. |
 
 
##### TensorRT, ONNX, and PyTorch JIT comparisons
 
The following tables show inference and latency comparisons across all 3 backends for mixed precision and static batching. The main observations are:
Increasing the batch size leads to higher inference throughput and - latency up to a certain batch size, after which it slowly saturates.
The longer the audio length, the lower the throughput and the higher the latency.
 
 
###### Throughput Comparison
 
The following table shows the throughput benchmark results for all 3 model backends in Triton Inference Server using static batching under optimal concurrency
 
|Audio length in seconds|Batch Size|TensorRT (inf/s)|PyTorch (inf/s)|ONNX (inf/s)|TensorRT/PyTorch Speedup|TensorRT/Onnx Speedup|
|---    |---    |---    |---    |---    |---    |---    |
|  2.0| 1|  49.67|  55.67|  35.00| 0.89| 1.42|
|  2.0| 2|  98.67|  96.00|  70.00| 1.03| 1.41|
|  2.0| 4| 180.00| 141.33| 138.67| 1.27| 1.30|
|  2.0| 8| 285.33| 202.67| 274.67| 1.41| 1.04|
|  7.0| 1|  47.67|  37.00|  35.00| 1.29| 1.36|
|  7.0| 2|  79.33|  47.33|  69.33| 1.68| 1.14|
|  7.0| 4| 100.00|  73.33| 137.33| 1.36| 0.73|
|  7.0| 8| 117.33|  82.67| 269.33| 1.42| 0.44|
| 16.7| 1|  36.33|  21.67|  34.67| 1.68| 1.05|
| 16.7| 2|  40.67|  25.33|  68.67| 1.61| 0.59|
| 16.7| 4|  46.67|  37.33| 133.33| 1.25| 0.35|
| 16.7| 8|  48.00|  40.00| 256.00| 1.20| 0.19|
 
 
###### Latency Comparison
 
The following table shows the throughput benchmark results for all 3 model backends in Triton Inference Server using static batching and a single concurrent request.
 
|Audio length in seconds|Batch Size|TensorRT (ms)|PyTorch (ms)|ONNX (ms)|TensorRT/PyTorch Speedup|TensorRT/Onnx Speedup|
|---    |---    |---    |---    |---    |---    |---    |
|  2.0| 1|  23.61|  25.06| 32.50| 1.06| 1.38|
|  2.0| 2|  24.56|  25.11| 32.29| 1.02| 1.31|
|  2.0| 4|  25.90|  31.00| 33.03| 1.20| 1.28|
|  2.0| 8|  31.57|  41.76| 33.63| 1.32| 1.07|
|  7.0| 1|  24.79|  30.55| 32.52| 1.23| 1.31|
|  7.0| 2|  28.48|  45.05| 32.96| 1.58| 1.16|
|  7.0| 4|  41.71|  57.71| 33.67| 1.38| 0.81|
|  7.0| 8|  72.19|  98.84| 35.25| 1.37| 0.49|
| 16.7| 1|  30.66|  48.42| 33.41| 1.58| 1.09|
| 16.7| 2|  52.79|  81.89| 33.66| 1.55| 0.64|
| 16.7| 4|  92.86| 115.03| 37.94| 1.24| 0.41|
| 16.7| 8| 170.34| 203.52| 43.60| 2.36| 0.26|
 
## Release Notes
 
### Changelog

December 2020
* Updated Triton scripts for compatibility with Triton Inference Server version 2
* Updated performance results
 
### Known issues
There are no known issues in this deployment.

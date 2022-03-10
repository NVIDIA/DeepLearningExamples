# Deploying the BERT model on Triton Inference Server
 
This folder contains instructions for deployment to run inference
on Triton Inference Server, as well as detailed performance analysis.
The purpose of this document is to help you with achieving
the best inference performance.
 
## Table of contents
 - [Solution overview](#solution-overview)
   - [Introduction](#introduction)
   - [Deployment process](#deployment-process)
 - [Setup](#setup)
 - [Quick Start Guide](#quick-start-guide)
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
ensure optimal GPU utilization during inferencing on the Triton Inference Server.
 
### Deployment process
 
The deployment process consists of two steps:
 
1. Conversion.
 
  The purpose of conversion is to find the best performing model
  format supported by the Triton Inference Server.
  Triton Inference Server uses a number of runtime backends such as
  [TensorRT](https://developer.nvidia.com/tensorrt),
  [LibTorch](https://github.com/triton-inference-server/pytorch_backend) and
  [ONNX Runtime](https://github.com/triton-inference-server/onnxruntime_backend)
  to support various model types. Refer to the
  [Triton documentation](https://github.com/triton-inference-server/backend#where-can-i-find-all-the-backends-that-are-available-for-triton)
  for a list of available backends.
 
2. Configuration.
 
  Model configuration on the Triton Inference Server, which generates
  necessary [configuration files](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md).
 
After deployment, the Triton inference server is used for evaluation of the converted model in two steps:
 
1. Accuracy tests.
 
  Produce results that are tested against given accuracy thresholds.
 
2. Performance tests.
 
  Produce latency and throughput results for offline (static batching)
  and online (dynamic batching) scenarios.
 
 
All steps are executed by the provided runner script. Refer to [Quick Start Guide](#quick-start-guide)
 
 
## Setup
Ensure you have the following components:
* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch NGC container 21.10](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
* [Triton Inference Server NGC container 21.10](https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver)
* [NVIDIA CUDA](https://docs.nvidia.com/cuda/archive//index.html)
* [NVIDIA Ampere](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/), [Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) based GPU
 
 
## Quick Start Guide
Deployment is supported for the following architectures. For the deployment steps, refer to the appropriate readme file:
* [BERT-large](./large/README.md)
* [BERT-distilled-4l](./dist4l/README.md)
* [BERT-distilled-6l](./dist6l/README.md)
 
 
## Release Notes
We’re constantly refining and improving our performance on AI
and HPC workloads with frequent updates
to our software stack. For our latest performance data refer
to these pages for
[AI](https://developer.nvidia.com/deep-learning-performance-training-inference)
and [HPC](https://developer.nvidia.com/hpc-application-performance) benchmarks.
 
### Changelog
### Known issues
 
- There are no known issues with this model.

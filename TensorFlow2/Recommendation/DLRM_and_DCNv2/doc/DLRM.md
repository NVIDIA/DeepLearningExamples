# DLRM for TensorFlow 2

This document provides detailed instructions on running DLRM training as well as benchmark results for this model.

## Table Of Contents

  * [Model overview](#model-overview)
     * [Model architecture](#model-architecture)
  * [Quick Start Guide](#quick-start-guide)
  * [Performance](#performance)
     * [Benchmarking](#benchmarking)
        * [Training performance benchmark](#training-performance-benchmark)
        * [Inference performance benchmark](#inference-performance-benchmark)
     * [Training process](#training-process)
     * [Results](#results)
        * [Training accuracy results](#training-accuracy-results)
           * [Training accuracy: NVIDIA DGX A100 (8x A100 80GB)](#training-accuracy-nvidia-dgx-a100-8x-a100-80gb)
           * [Training accuracy: NVIDIA DGX-1 (8x V100 32GB)](#training-accuracy-nvidia-dgx-1-8x-v100-32gb)
           * [Training accuracy: NVIDIA DGX-2 (16x V100 32GB)](#training-accuracy-nvidia-dgx-2-16x-v100-32gb)
           * [Training stability test](#training-stability-test)
        * [Training performance results](#training-performance-results)
           * [Training performance: NVIDIA DGX A100 (8x A100 80GB)](#training-performance-nvidia-dgx-a100-8x-a100-80gb)
           * [Training performance: comparison with CPU for the "extra large" model](#training-performance-comparison-with-cpu-for-the-extra-large-model)
           * [Training performance: NVIDIA DGX-1 (8x V100 32GB)](#training-performance-nvidia-dgx-1-8x-v100-32gb)
           * [Training performance: NVIDIA DGX-2 (16x V100 32GB)](#training-performance-nvidia-dgx-2-16x-v100-32gb)
        * [Inference performance results](#inference-performance-results)
           * [Inference performance: NVIDIA DGX A100 (8x A100 80GB)](#inference-performance-nvidia-dgx-a100-8x-a100-80gb)
           * [Inference performance: NVIDIA DGX1V-32GB (8x V100 32GB)](#inference-performance-nvidia-dgx1v-32gb-8x-v100-32gb)
           * [Inference performance: NVIDIA DGX2 (16x V100 16GB)](#inference-performance-nvidia-dgx2-16x-v100-16gb)


## Model overview

The Deep Learning Recommendation Model (DLRM) is a recommendation model designed to make use of both categorical and numerical inputs.
It was first described in [Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/abs/1906.00091).
This repository provides a reimplementation of the code base provided originally [here](https://github.com/facebookresearch/dlrm).
The scripts enable you to train DLRM on a synthetic dataset or on the [Criteo Terabyte Dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/).

For the Criteo 1TB Dataset, we use a slightly different preprocessing procedure than the one found in the original implementation.
Most importantly, we use a technique called frequency thresholding to demonstrate models of different sizes.
The smallest model can be trained on a single V100-32GB GPU, while the largest one needs 8xA100-80GB GPUs.  

The table below summarizes the model sizes and frequency thresholds used in this repository, for both the synthetic and real datasets supported.

| Dataset   | Frequency Threshold | Final dataset size | Intermediate preprocessing storage required |  Suitable for accuracy tests | Total download & preprocess time |GPU Memory required for training | Total embedding size | Number of model parameters | 
|:-------|:-------|:-------|:-------------|:-------------------|:-------------------|:-------------------|:-------------------|:-------------------|
| Synthetic T15             |15  |  6 GiB   | None       | No  | ~Minutes | 15.6 GiB | 15.6 GiB  | 4.2B  |
| Synthetic T3              |3   |  6 GiB   | None       | No  | ~Minutes | 84.9 GiB | 84.9 GiB  | 22.8B |
| Synthetic T0              |0   |  6 GiB   | None       | No  | ~Minutes | 421 GiB  | 421 GiB   | 113B  |
| Real Criteo T15           |15  |  370 GiB | ~Terabytes | Yes | ~Hours   | 15.6 GiB | 15.6 GiB  | 4.2B  |
| Real Criteo T3            |3   |  370 GiB | ~Terabytes | Yes | ~Hours   | 84.9 GiB | 84.9 GiB  | 22.8B |
| Real Criteo T0            |0   |  370 GiB | ~Terabytes | Yes | ~Hours   | 421 GiB  | 421 GiB   | 113B  |

You can find a detailed description of the Criteo dataset preprocessing the [preprocessing documentation](./criteo_dataset.md#advanced).

### Model architecture

DLRM accepts two types of features: categorical and numerical. For each categorical feature,
an embedding table is used to provide a dense representation of each unique value.
The dense features enter the model and are transformed by a simple neural network referred to as "Bottom MLP."

This part of the network consists of a series
of linear layers with ReLU activations. The output of the bottom MLP and the embedding vectors are then fed into the
"dot interaction" operation. The output of "dot interaction" is then concatenated
with the features resulting from the bottom MLP and fed
into the "top MLP," which is a series of dense layers with activations.
The model outputs a single number which can be interpreted as a likelihood of a certain user clicking an ad.

<p align="center">
  <img width="100%" src="./img/dlrm_singlegpu_architecture.svg" />
  <br>
Figure 1. The architecture of DLRM.
</p>

## Quick Start Guide

To train DLRM perform the following steps.
For the specifics concerning training and inference,
refer to the [Advanced](../README.md#advanced) section.

1. Clone the repository.
```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/TensorFlow2/Recommendation/DLRM
```

2. Build and run a DLRM Docker container.
```bash
docker build -t train_docker_image .
docker run --cap-add SYS_NICE --runtime=nvidia -it --rm --ipc=host  -v ${PWD}/data:/data train_docker_image bash
```

3. Generate a synthetic dataset.

Downloading and preprocessing the Criteo 1TB dataset requires a lot of time and disk space.
Because of this we provide a synthetic dataset generator that roughly matches Criteo 1TB characteristics. 
This will enable you to benchmark quickly.
If you prefer to benchmark on the real data, please follow [these instructions](./criteo_dataset.md#quick-start-guide)
to download and preprocess the dataset.

```bash
python -m dataloading.generate_feature_spec --variant criteo_t15_synthetic  --dst feature_spec.yaml
python -m dataloading.transcribe --src_dataset_type synthetic --src_dataset_path . \
       --dst_dataset_path /data/preprocessed --max_batches_train 1000 --max_batches_test 100 --dst_dataset_type tf_raw
```

4. Verify the input data:

After running `tree /data/preprocessed` you should see the following directory structure:
```bash
$ tree /data/preprocessed
/data/preprocessed
├── feature_spec.yaml
├── test
│   ├── cat_0.bin
│   ├── cat_1.bin
│   ├── ...
│   ├── label.bin
│   └── numerical.bin
└── train
    ├── cat_0.bin
    ├── cat_1.bin
    ├── ...
    ├── label.bin
    └── numerical.bin

2 directories, 57 files
```

5. Start training.

- single-GPU:
```bash
horovodrun -np 1 -H localhost:1 --mpi-args=--oversubscribe numactl --interleave=all -- python -u dlrm.py --dataset_path /data/preprocessed --amp --xla --save_checkpoint_path /data/checkpoint/
```

- multi-GPU:
```bash
horovodrun -np 8 -H localhost:8 --mpi-args=--oversubscribe numactl --interleave=all -- python -u dlrm.py --dataset_path /data/preprocessed --amp --xla --save_checkpoint_path /data/checkpoint/
```

6. Start evaluation.

To evaluate a previously trained checkpoint, append `--restore_checkpoint_path <path> --mode eval` to the command used for training. For example, to test a checkpoint trained on 8xA100 80GB, run:

```bash
horovodrun -np 8 -H localhost:8 --mpi-args=--oversubscribe numactl --interleave=all -- python -u dlrm.py --dataset_path /data/preprocessed --amp --xla --restore_checkpoint_path /data/checkpoint --mode eval
```

## Performance

The performance measurements in this document were conducted at the time of publication and may not reflect the performance achieved from NVIDIA’s latest software release. For the most up-to-date performance measurements, go to [NVIDIA Data Center Deep Learning Product Performance](https://developer.nvidia.com/deep-learning-performance-training-inference).

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in training and inference modes.

#### Training performance benchmark

To benchmark the training performance on a specific batch size, follow the instructions
in the [Quick Start Guide](#quick-start-guide). You can also add the `--max_steps 1000`
if you want to get a reliable throughput measurement without running the entire training.

You can also use synthetic data by running with the `--dataset_type synthetic` option if you haven't downloaded the dataset yet.

#### Inference performance benchmark

To benchmark the inference performance on a specific batch size, run:

```
horovodrun -np 1 -H localhost:1 --mpi-args=--oversubscribe numactl --interleave=all -- python -u dlrm.py --dataset_path /data/preprocessed/ --amp --restore_checkpoint_path <checkpoint_path> --mode inference
```

### Training process

The main training scripts resides in `dlrm.py`. The training speed  is measured by throughput,  i.e.,
the number of samples processed per second.
We use mixed precision training with static loss scaling for the bottom and top MLPs
while embedding tables are stored in FP32 format.

### Results

The following sections provide details on how we achieved our performance and accuracy in training and inference.

We used three model size variants to show memory scalability in a multi-GPU setup
(4.2B params, 22.8B params, and 113B params). Refer to the [Model overview](#model-overview) section for detailed
information about the model variants.

#### Training accuracy results

##### Training accuracy: NVIDIA DGX A100 (8x A100 80GB)

Our results were obtained by running training scripts as described in the Quick Start Guide in the DLRM Docker container.

| GPUs   | Model size   | Batch size / GPU   | Accuracy (AUC) - TF32   | Accuracy (AUC) - mixed precision   | Time to train - TF32 [minutes]   | Time to train - mixed precision [minutes]   | Time to train speedup (TF32 to mixed precision)   |
|:-------|:-------------|:-------------------|:------------------------|:-----------------------------------|:---------------------------------|:--------------------------------------------|:--------------------------------------------------|
| 1      | small        | 64k                | 0.8025                  | 0.8025                             | 26.75                            | 16.27                                       | 1.64                                              |
| 8      | large        | 8k                 | 0.8027                  | 0.8026                             | 8.77                             | 6.57                                        | 1.33                                              |
| 8      | extra large  | 8k                 | 0.8026                  | 0.8026                             | 10.47                            | 9.08                                        | 1.15                                              |

##### Training accuracy: NVIDIA DGX-1 (8x V100 32GB)

Our results were obtained by running training scripts as described in the Quick Start Guide in the DLRM Docker container.

| GPUs   | Model size   | Batch size / GPU   | Accuracy (AUC) - FP32   | Accuracy (AUC) - mixed precision   | Time to train - FP32 [minutes]   | Time to train - mixed precision [minutes]   | Time to train speedup (FP32 to mixed precision)   |
|:-------|:-------------|:-------------------|:------------------------|:-----------------------------------|:---------------------------------|:--------------------------------------------|:--------------------------------------------------|
| 1      | small        | 64k                | 0.8027                  | 0.8025                             | 109.63                           | 34.83                                       |  3.15                                             |
| 8      | large        | 8k                 | 0.8028                  | 0.8026                             | 26.01                            | 13.73                                       |  1.89                                             |
##### Training accuracy: NVIDIA DGX-2 (16x V100 32GB)

Our results were obtained by running training scripts as described in the Quick Start Guide in the DLRM Docker container.

| GPUs   | Model size   | Batch size / GPU   | Accuracy (AUC) - FP32   | Accuracy (AUC) - mixed precision   | Time to train - FP32 [minutes]   | Time to train - mixed precision [minutes]   | Time to train speedup (FP32 to mixed precision)   |
|:-------|:-------------|:-------------------|:------------------------|:-----------------------------------|:---------------------------------|:--------------------------------------------|:--------------------------------------------------|
| 1      | small        | 64k                | 0.8026                  | 0.8026                             | 105.13                           | 33.37                                       | 3.15                                              |
| 8      | large        | 8k                 | 0.8027                  | 0.8027                             | 21.21                            | 11.43                                       | 1.86                                              |
| 16     | large        | 4k                 | 0.8025                  | 0.8026                             | 15.52                            | 10.88                                       | 1.43                                              |

##### Training stability test

The histograms below show the distribution of ROC AUC results achieved at the end of the training for each precision/hardware platform tested. No statistically significant differences exist between precision, number of GPUs, or hardware platform. Using the larger dataset has a modest, positive impact on the final AUC score.   


<p align="center">
  <img width="100%" src="img/dlrm_histograms.svg" />
  <br>
Figure 4. Results of stability tests for DLRM.
</p>


#### Training performance results


We used throughput in items processed per second as the performance metric.


##### Training performance: NVIDIA DGX A100 (8x A100 80GB)

Our results were obtained by following the commands from the Quick Start Guide
in the DLRM Docker container on NVIDIA DGX A100 (8x A100 80GB) GPUs. Performance numbers (in items per second) were averaged over 1000 training steps.

| GPUs   | Model size   | Batch size / GPU   | Throughput - TF32   | Throughput - mixed precision   | Throughput speedup (TF32 to mixed precision)   |
|:-------|:-------------|:-------------------|:--------------------|:-------------------------------|:-----------------------------------------------|
| 1      | small        | 64k                | 2.84M               | 4.55M                          | 1.60                                           |
| 8      | large        | 8k                 | 10.9M               | 13.8M                          | 1.27                                           |
| 8      | extra large  | 8k                 | 9.76M               | 11.5M                          | 1.17

To achieve these results, follow the steps in the [Quick Start Guide](#quick-start-guide).

##### Training performance: comparison with CPU for the "extra large" model 

For the "extra large" model (113B parameters), we also obtained CPU results for comparison using the same source code
(using the `--cpu` command line flag for the CPU-only experiments).  

We compare three hardware setups:
- CPU only,
- a single GPU that uses CPU memory for the largest embedding tables,
- Hybrid-Parallel using the full DGX A100-80GB

| Hardware | Throughput [samples / second]|  Speedup over CPU|
|:---|:---|:---|
2xAMD EPYC 7742 | 17.7k | 1x |
1xA100-80GB + 2xAMD EPYC 7742 (large embeddings on CPU) | 768k |43x |
DGX A100 (8xA100-80GB) (hybrid parallel) | 11.5M | 649x |


##### Training performance: NVIDIA DGX-1 (8x V100 32GB)

| GPUs   | Model size   | Batch size / GPU   | Throughput - FP32   | Throughput - mixed precision   | Throughput speedup (FP32 to mixed precision)   |
|:-------|:-------------|:-------------------|:--------------------|:-------------------------------|:-----------------------------------------------|
| 1      | small        | 64k                | 0.663M               | 2.23M                          | 3.37                                          |
| 8      | large        | 8k                 | 3.13M               | 6.31M                          | 2.02                                          |

To achieve the same results, follow the steps in the [Quick Start Guide](#quick-start-guide).


##### Training performance: NVIDIA DGX-2 (16x V100 32GB)

| GPUs   | Model size   | Batch size / GPU   | Throughput - FP32   | Throughput - mixed precision   | Throughput speedup (FP32 to mixed precision)   |
|:-------|:-------------|:-------------------|:--------------------|:-------------------------------|:-----------------------------------------------|
| 1      | small        | 64k                | 0.698M              | 2.44M                          | 3.49                                           |
| 8      | large        | 8k                 | 3.79M               | 7.82M                          | 2.06                                           |
| 16     | large        | 4k                 | 6.43M               | 10.5M                          | 1.64                                           |


To achieve the same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

#### Inference performance results

##### Inference performance: NVIDIA DGX A100 (8x A100 80GB)

|   GPUs | Model size   |   Batch size / GPU | Throughput - TF32   | Throughput - mixed precision   |   Average latency - TF32 [ms] |   Average latency - mixed precision [ms] |   Throughput speedup (mixed precision to TF32) |
|-------:|:-------------|-------------------:|:--------------------|:-------------------------------|------------------------------:|-----------------------------------------:|-----------------------------------------------:|
|      1 | small        |               2048 | 1.38M               | 1.48M                          |                          1.49 |                                     1.38 |                                           1.07 |


##### Inference performance: NVIDIA DGX1V-32GB (8x V100 32GB)

|   GPUs | Model size   |   Batch size / GPU | Throughput - FP32   | Throughput - mixed precision   |   Average latency - FP32 [ms] |   Average latency - mixed precision [ms] |   Throughput speedup (mixed precision to FP32) |
|-------:|:-------------|-------------------:|:--------------------|:-------------------------------|------------------------------:|-----------------------------------------:|-----------------------------------------------:|
|      1 | small        |               2048 | 0.871M               | 0.951M                          | 2.35                          | 2.15                                   | 1.09                                           |

##### Inference performance: NVIDIA DGX2 (16x V100 16GB)

|   GPUs | Model size   |   Batch size / GPU | Throughput - FP32   | Throughput - mixed precision   |   Average latency - FP32 [ms] |   Average latency - mixed precision [ms] |   Throughput speedup (mixed precision to FP32) |
|-------:|:-------------|-------------------:|:--------------------|:-------------------------------|------------------------------:|-----------------------------------------:|-----------------------------------------------:|
|      1 | small        |               2048 | 1.15M               | 1.37M                          | 1.78                          |   1.50                                   | 1.19                                           |


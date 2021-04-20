# Convolutional Network for Image Classification in PyTorch

In this repository you will find implementations of various image classification models.

Detailed information on each model can be found here:

## Table Of Contents

* [Models](#models)
* [Validation accuracy results](#validation-accuracy-results)
* [Training performance results](#training-performance-results)
  * [Training performance: NVIDIA DGX A100 (8x A100 80GB)](#training-performance-nvidia-dgx-a100-8x-a100-80gb)
  * [Training performance: NVIDIA DGX-1 16GB (8x V100 16GB)](#training-performance-nvidia-dgx-1-16gb-8x-v100-16gb)
  * [Training performance: NVIDIA DGX-2 (16x V100 32GB)](#training-performance-nvidia-dgx-2-16x-v100-32gb)
* [Model comparison](#model-comparison)
  * [Accuracy vs FLOPS](#accuracy-vs-flops)
  * [Latency vs Throughput on different batch sizes](#latency-vs-throughput-on-different-batch-sizes)

## Models

The following table provides links to where you can find additional information on each model:

| **Model** | **Link**|
|:-:|:-:|
| resnet50 | [README](./resnet50v1.5/README.md) |
| resnext101-32x4d | [README](./resnext101-32x4d/README.md) |
| se-resnext101-32x4d | [README](./se-resnext101-32x4d/README.md) |
| EfficientNet | [README](./efficientnet/README.md) |

## Validation accuracy results

Our results were obtained by running the applicable
training scripts in the 20.12 PyTorch NGC container
on NVIDIA DGX-1 with (8x V100 16GB) GPUs.
The specific training script that was run is documented
in the corresponding model's README.


The following table shows the validation accuracy results of the
three classification models side-by-side.

|       **Model**        | **Mixed Precision Top1** | **Mixed Precision Top5** | **32 bit Top1** | **32 bit Top5** |
|:----------------------:|:------------------------:|:------------------------:|:---------------:|:---------------:|
|    efficientnet-b0     |          77.63           |          93.82           |      77.31      |      93.76      |
|    efficientnet-b4     |          82.98           |          96.44           |      82.92      |      96.43      |
| efficientnet-widese-b0 |          77.89           |          94.00           |      77.97      |      94.05      |
| efficientnet-widese-b4 |          83.28           |          96.45           |      83.30      |      96.47      |
|        resnet50        |          78.60           |          94.19           |      78.69      |      94.16      |
|    resnext101-32x4d    |          80.43           |          95.06           |      80.40      |      95.04      |
|  se-resnext101-32x4d   |          81.00           |          95.48           |      81.09      |      95.45      |


## Training performance results

### Training performance: NVIDIA DGX A100 (8x A100 80GB)


Our results were obtained by running the applicable
training scripts in the 21.03 PyTorch NGC container
on NVIDIA DGX A100 with (8x A100 80GB) GPUs.
Performance numbers (in images per second)
were averaged over an entire training epoch.
The specific training script that was run is documented
in the corresponding model's README.

The following table shows the training accuracy results of
all the classification models side-by-side.

|       **Model**        | **Mixed Precision** |  **TF32**  | **Mixed Precision Speedup** |
|:----------------------:|:-------------------:|:----------:|:---------------------------:|
|    efficientnet-b0     |     16652 img/s     | 8193 img/s |           2.03 x            |
|    efficientnet-b4     |     2570 img/s      | 1223 img/s |            2.1 x            |
| efficientnet-widese-b0 |     16368 img/s     | 8244 img/s |           1.98 x            |
| efficientnet-widese-b4 |     2585 img/s      | 1223 img/s |           2.11 x            |
|        resnet50        |     16621 img/s     | 7248 img/s |           2.29 x            |
|    resnext101-32x4d    |     7925 img/s      | 3471 img/s |           2.28 x            |
|  se-resnext101-32x4d   |     5779 img/s      | 2991 img/s |           1.93 x            |

### Training performance: NVIDIA DGX-1 16G (8x V100 16GB)

Our results were obtained by running the applicable
training scripts in the 21.03 PyTorch NGC container
on NVIDIA DGX-1 with (8x V100 16GB) GPUs.
Performance numbers (in images per second)
were averaged over an entire training epoch.
The specific training script that was run is documented
in the corresponding model's README.

The following table shows the training accuracy results of all the
classification models side-by-side.

|       **Model**        | **Mixed Precision** |  **FP32**  | **Mixed Precision Speedup** |
|:----------------------:|:-------------------:|:----------:|:---------------------------:|
|    efficientnet-b0     |     7789 img/s      | 4672 img/s |           1.66 x            |
|    efficientnet-b4     |     1366 img/s      | 616 img/s  |           2.21 x            |
| efficientnet-widese-b0 |     7875 img/s      | 4592 img/s |           1.71 x            |
| efficientnet-widese-b4 |     1356 img/s      | 612 img/s  |           2.21 x            |
|        resnet50        |     8322 img/s      | 2855 img/s |           2.91 x            |
|    resnext101-32x4d    |     4065 img/s      | 1133 img/s |           3.58 x            |
|  se-resnext101-32x4d   |     2971 img/s      | 1004 img/s |           2.95 x            |

## Model Comparison

### Accuracy vs FLOPS
![ACCvsFLOPS](./img/ACCvsFLOPS.png)

Plot describes relationship between floating point operations
needed for computing forward pass on a 224px x 224px image, 
for the implemented models.
Dot size indicates number of trainable parameters.

### Latency vs Throughput on different batch sizes
![LATvsTHR](./img/LATvsTHR.png)

Plot describes relationship between
inference latency, throughput and batch size
for the implemented models.

# Convolutional Networks for Image Classification in PyTorch

In this repository you will find implementations of various image classification models.

Detailed information on each model can be found here:

## Table Of Contents

* [Models](#models)
* [Validation accuracy results](#validation-accuracy-results)
* [Training performance results](#training-performance-results)
  * [Training performance: NVIDIA DGX A100 (8x A100 40GB)](#training-performance-nvidia-dgx-a100-8x-a100-40gb)
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

## Validation accuracy results

Our results were obtained by running the applicable
training scripts in the [framework-container-name] NGC container
on NVIDIA DGX-1 with (8x V100 16GB) GPUs.
The specific training script that was run is documented
in the corresponding model's README.


The following table shows the validation accuracy results of the
three classification models side-by-side.


| **arch** | **AMP Top1** | **AMP Top5** | **FP32 Top1** | **FP32 Top5** |
|:-:|:-:|:-:|:-:|:-:|
| resnet50 | 78.46 | 94.15 | 78.50 | 94.11 |
| resnext101-32x4d | 80.08 | 94.89 | 80.14 | 95.02 |
| se-resnext101-32x4d | 81.01 | 95.52 | 81.12 | 95.54 |


## Training performance results

### Training performance: NVIDIA DGX A100 (8x A100 40GB)


Our results were obtained by running the applicable
training scripts in the pytorch-20.06 NGC container
on NVIDIA DGX A100 with (8x A100 40GB) GPUs.
Performance numbers (in images per second)
were averaged over an entire training epoch.
The specific training script that was run is documented
in the corresponding model's README.

The following table shows the training accuracy results of the
three classification models side-by-side.


|      **arch**       | **Mixed Precision** |   **TF32**    | **Mixed Precision Speedup** |
|:-------------------:|:-------------------:|:-------------:|:---------------------------:|
|      resnet50       |    9488.39 img/s    | 5322.10 img/s |            1.78x            |
|  resnext101-32x4d   |    6758.98 img/s    | 2353.25 img/s |            2.87x            |
| se-resnext101-32x4d |    4670.72 img/s    | 2011.21 img/s |            2.32x            |

ResNeXt and SE-ResNeXt use [NHWC data layout](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html) when training using Mixed Precision,
which improves the model performance. We are currently working on adding it for ResNet.


### Training performance: NVIDIA DGX-1 16G (8x V100 16GB)


Our results were obtained by running the applicable
training scripts in the pytorch-20.06 NGC container
on NVIDIA DGX-1 with (8x V100 16GB) GPUs.
Performance numbers (in images per second)
were averaged over an entire training epoch.
The specific training script that was run is documented
in the corresponding model's README.

The following table shows the training accuracy results of the
three classification models side-by-side.


|      **arch**       | **Mixed Precision** |   **FP32**    | **Mixed Precision Speedup** |
|:-------------------:|:-------------------:|:-------------:|:---------------------------:|
|      resnet50       |    6565.61 img/s    | 2869.19 img/s |            2.29x            |
|  resnext101-32x4d   |    3922.74 img/s    | 1136.30 img/s |            3.45x            |
| se-resnext101-32x4d |    2651.13 img/s    | 982.78 img/s  |            2.70x            |

ResNeXt and SE-ResNeXt use [NHWC data layout](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html) when training using Mixed Precision,
which improves the model performance. We are currently working on adding it for ResNet.


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



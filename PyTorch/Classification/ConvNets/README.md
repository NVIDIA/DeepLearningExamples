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
| EfficientNet-B0 | [README](./efficientnet/README.md) |

## Validation accuracy results

Our results were obtained by running the applicable
training scripts in the [framework-container-name] NGC container
on NVIDIA DGX-1 with (8x V100 16GB) GPUs.
The specific training script that was run is documented
in the corresponding model's README.


The following table shows the validation accuracy results of the
three classification models side-by-side.

|      **Model**      | **Mixed Precision Top1** | **Mixed Precision Top5** | **32 bit Top1** | **32 bit Top5** |
|:-------------------:|:------------------------:|:------------------------:|:---------------:|:---------------:|
|      resnet50       |          78.60           |          94.19           |      78.69      |      94.16      |
|  resnext101-32x4d   |          80.43           |          95.06           |      80.40      |      95.04      |
| se-resnext101-32x4d |          81.00           |          95.48           |      81.09      |      95.45      |

## Training performance results

### Training performance: NVIDIA DGX A100 (8x A100 80GB)


Our results were obtained by running the applicable
training scripts in the pytorch-20.12 NGC container
on NVIDIA DGX A100 with (8x A100 80GB) GPUs.
Performance numbers (in images per second)
were averaged over an entire training epoch.
The specific training script that was run is documented
in the corresponding model's README.

The following table shows the training accuracy results of the
three classification models side-by-side.


|      **Model**      | **Mixed Precision** |  **TF32**  | **Mixed Precision Speedup** |
|:-------------------:|:-------------------:|:----------:|:---------------------------:|
|      resnet50       |     15977 img/s     | 7365 img/s |           2.16 x            |
|  resnext101-32x4d   |     7399 img/s      | 3193 img/s |           2.31 x            |
| se-resnext101-32x4d |     5248 img/s      | 2665 img/s |           1.96 x            |

### Training performance: NVIDIA DGX-1 16G (8x V100 16GB)

Our results were obtained by running the applicable
training scripts in the pytorch-20.12 NGC container
on NVIDIA DGX-1 with (8x V100 16GB) GPUs.
Performance numbers (in images per second)
were averaged over an entire training epoch.
The specific training script that was run is documented
in the corresponding model's README.

The following table shows the training accuracy results of the
three classification models side-by-side.

|      **Model**      | **Mixed Precision** |  **FP32**  | **Mixed Precision Speedup** |
|:-------------------:|:-------------------:|:----------:|:---------------------------:|
|      resnet50       |     7608 img/s      | 2851 img/s |           2.66 x            |
|  resnext101-32x4d   |     3742 img/s      | 1117 img/s |           3.34 x            |
| se-resnext101-32x4d |     2716 img/s      | 994 img/s  |           2.73 x            |

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

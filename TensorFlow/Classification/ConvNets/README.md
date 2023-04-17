# Resnet-family Convolutional Neural Networks for Image Classification in Tensorflow

In this repository you will find implementation of Resnet and its variations for image classification.
Convolutional Network models for TensorFlow1 are no longer maintained and will soon become unavailable, please consider PyTorch or TensorFlow2 models as a substitute for your requirements.

## Table Of Contents

* [Models](#models)
* [Validation accuracy results](#validation-accuracy-results)
* [Training performance results](#training-performance-results)
  * [Training performance: NVIDIA DGX A100 (8x A100 40G)](#training-performance-nvidia-dgx-a100-8x-a100-40g)
  * [Training performance: NVIDIA DGX-1 (8x V100 16G)](#training-performance-nvidia-dgx-1-8x-v100-16g)
* [Release notes](#release-notes)
  * [Changelog](#changelog)


## Models

The following table provides links to where you can find additional information on each model:

| **Model** | **Link**|
|-----------|---------|
| resnet50 | [README](./resnet50v1.5/README.md) |
| resnext101-32x4d | [README](./resnext101-32x4d/README.md) |
| se-resnext101-32x4d | [README](./se-resnext101-32x4d/README.md) |

## Validation accuracy results

Our results were obtained by running the applicable training scripts in the tensorflow-20.06-tf1-py3 NGC container 
on NVIDIA DGX-1 with (8x V100 16G) GPUs. The specific training script that was run is documented in the corresponding model's README.

The following table shows the validation accuracy results of the 
three classification models side-by-side.


| **arch** | **AMP Top1** | **AMP Top5** | **FP32 Top1** | **FP32 Top5** |
|:-:|:-:|:-:|:-:|:-:|
| resnet50            | 78.35 | 94.21 | 78.34 | 94.21 |
| resnext101-32x4d    | 80.21 | 95.00 | 80.21 | 94.99 |
| se-resnext101-32x4d | 80.87 | 95.35 | 80.84 | 95.37 |

## Training performance results

### Training performance: NVIDIA DGX A100 (8x A100 40G)

Our results were obtained by running the applicable 
training scripts in the tensorflow-20.06-tf1-py3 NGC container 
on NVIDIA DGX A100 with (8x A100 40G) GPUs. 
Performance numbers (in images per second) 
were averaged over an entire training epoch.
The specific training script that was run is documented 
in the corresponding model's README.

The following table shows the training performance results of the 
three classification models side-by-side.


| **arch** | **Mixed Precision XLA** | **TF32 XLA** | **Mixed Precision speedup** |
|:-:|:-:|:-:|:-:|
| resnet50            | 16400 img/s | 6300 img/s | 2.60x |
| resnext101-32x4d    | 8000 img/s | 2630 img/s | 3.05x |
| se-resnext101-32x4d | 6930 img/s | 2400 img/s | 2.88x |

### Training performance: NVIDIA DGX-1 (8x V100 16G)

Our results were obtained by running the applicable 
training scripts in the tensorflow-20.06-tf1-py3 NGC container 
on NVIDIA DGX-1 with (8x V100 16G) GPUs. 
Performance numbers (in images per second) 
were averaged over an entire training epoch.
The specific training script that was run is documented 
in the corresponding model's README.

The following table shows the training performance results of the 
three classification models side-by-side.


| **arch** | **Mixed Precision XLA** | **FP32 XLA** | **Mixed Precision speedup** |
|:-:|:-:|:-:|:-:|
| resnet50            | 9510 img/s | 3170 img/s | 3.00x |
| resnext101-32x4d    | 4160 img/s | 1210 img/s | 3.44x |
| se-resnext101-32x4d | 3360 img/s | 1120 img/s | 3.00x |

## Release notes

### Changelog

April 2021
  - Ceased maintenance of ConvNets in TensorFlow1

June 2020
  - ConvNets repo restructurization
  - Initial release of ResNext and SE-Resnext

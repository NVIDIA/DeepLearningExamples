# Resnet-family Convolutional Neural Networks for Image Classification in Tensorflow

In this repository you will find implementation of Resnet and its variations for image
classification

## Table Of Contents

* [Models](#models)
* [Validation accuracy results](#validation-accuracy-results)
* [Training performance results](#training-performance-results)
  * [Training performance: NVIDIA DGX-1 (8x V100 16G)](#training-performance-nvidia-dgx-1-(8x-v100-16G))
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

Our results were obtained by running the applicable training scripts in the tensorflow-20.03-tf1-py3 NGC container 
on NVIDIA DGX-1 with (8x V100 16G) GPUs. The specific training script that was run is documented in the corresponding model's README.

The following table shows the validation accuracy results of the 
three classification models side-by-side.


| **arch** | **AMP Top1** | **AMP Top5** | **FP32 Top1** | **FP32 Top5** |
|:-:|:-:|:-:|:-:|:-:|
| resnet50            | 78.35 | 94.21 | 78.34 | 94.21 |
| resnext101-32x4d    | 80.21 | 95.00 | 80.21 | 94.99 |
| se-resnext101-32x4d | 80.87 | 95.35 | 80.84 | 95.37 |

## Training performance results

### Training performance: NVIDIA DGX-1 (8x V100 16G)

Our results were obtained by running the applicable 
training scripts in the tensorflow-20.03-tf1-py3 NGC container 
on NVIDIA DGX-1 with (8x V100 16G) GPUs. 
Performance numbers (in images per second) 
were averaged over an entire training epoch.
The specific training script that was run is documented 
in the corresponding model's README.

The following table shows the training accuracy results of the 
three classification models side-by-side.


| **arch** | **Mixed Precision** | **Mixed Prcesision XLA** | **FP32** | **Mixed Precision speedup** | **XLA Mixed Precision speedup**|
|:-:|:-:|:-:|:-:|:-:|:-:|
| resnet50            | 8277.91 img/s | 9485.21 img/s | 2785.81 img/s | 2.97x | 1.14x |
| resnext101-32x4d    | 3151.81 img/s | 4231.42 img/s | 1055.82 img/s | 2.98x | 1.34x |
| se-resnext101-32x4d | 2168.40 img/s | 3297.39 img/s | 921.38 img/s  | 2.35x | 1.52x |

## Release notes

### Changelog
June 2020
  - ConvNets repo restructurization
  - Initial release of ResNext and SE-Resnext

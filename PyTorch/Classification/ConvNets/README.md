# Convolutional Networks for Image Classification in PyTorch

In this repository you will find implementations of various image classification  models.

Detailed information on each model can be found here:

| **Model** | **Link**|
|:-:|:-:|
| resnet50 | [README](./resnet50v1.5/README.md) |
| resnext101-32x4d | [README](./resnext101-32x4d/README.md) |
| se-resnext101-32x4d | [README](./se-resnext101-32x4d/README.md) |

## Accuracy


| **Model** | **AMP Top1** | **AMP Top5** | **FP32 Top1** | **FP32 Top1** |
|:-:|:-:|:-:|:-:|:-:|
| resnet50 | 78.46 | 94.15 | 78.50 | 94.11 |
| resnext101-32x4d | 80.08 | 94.89 | 80.14 | 95.02 |
| se-resnext101-32x4d | 81.01 | 95.52 | 81.12 | 95.54 |


## Training Performance


### NVIDIA DGX-1 (8x V100 16G)

| **Model** | **Mixed Precision** | **FP32** | **Mixed Precision speedup** |
|:-:|:-:|:-:|:-:|
| resnet50 | 6888.75 img/s | 2945.37 img/s | 2.34x |
| resnext101-32x4d | 2384.85 img/s | 1116.58 img/s | 2.14x |
| se-resnext101-32x4d | 2031.17 img/s | 977.45 img/s | 2.08x |

### NVIDIA DGX-2 (16x V100 32G)

| **Model** | **Mixed Precision** | **FP32** | **Mixed Precision speedup** |
|:-:|:-:|:-:|:-:|
| resnet50 | 13443.82 img/s | 6263.41 img/s | 2.15x |
| resnext101-32x4d | 4473.37 img/s | 2261.97 img/s | 1.98x |
| se-resnext101-32x4d | 3776.03 img/s | 1953.13 img/s | 1.93x |


## Model Comparison

### Accuracy vs FLOPS
![ACCvsFLOPS](./img/ACCvsFLOPS.png)

Dot size indicates number of trainable parameters

### Latency vs Throughput on different batch sizes
![LATvsTHR](./img/LATvsTHR.png)

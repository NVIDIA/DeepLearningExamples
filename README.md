# NVIDIA Deep Learning Examples for Tensor Cores

## Introduction
This repository provides State-of-the-Art Deep Learning examples that are easy to train and deploy, achieving the best reproducible convergence and performance with NVIDIA CUDA-X software stack running on NVIDIA Volta, Turing and Ampere GPUs.

## NVIDIA GPU Cloud (NGC) Container Registry
These examples, along with our NVIDIA deep learning software stack, are provided in a monthly updated Docker container on the NGC container registry (https://ngc.nvidia.com). These containers include:  

- The latest NVIDIA examples from this repository
- The latest NVIDIA contributions shared upstream to the respective framework
- The latest NVIDIA Deep Learning software libraries, such as cuDNN, NCCL, cuBLAS, etc. which have all been through a rigorous monthly quality assurance process to ensure that they provide the best possible performance
- [Monthly release notes](https://docs.nvidia.com/deeplearning/dgx/index.html#nvidia-optimized-frameworks-release-notes) for each of the NVIDIA optimized containers


## Computer Vision
| Models  | Framework | DALI | AMP | Multi-GPU | Multi-Node  | TensorRT  | ONNX  | Triton | TF-TRT | Notebook |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |------------- |------------- |------------- |------------- |
| Computer Vision |
| [ResNet-50 v1.5](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5)  |PyTorch  | Yes  | Yes  | Yes  | -  | -  | -  | -  | -  | - |
| [ResNeXt101-32x4d](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnext101-32x4d)  |PyTorch  | Yes  | Yes  | Yes  | -  | -  |   -  | -  | -  | - |
| [SE-ResNeXt101-32x4d](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/se-resnext101-32x4d)  |PyTorch  | Yes  | Yes  | Yes  | -  | -  | -  | -  | -  | - |
| [Mask R-CNN](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/MaskRCNN) |PyTorch  | N/A  | Yes  | Yes  | -  | -  |   -  | -  | -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Segmentation/MaskRCNN/pytorch/notebooks/pytorch_MaskRCNN_pyt_train_and_inference.ipynb) |
| [SSD300 v1.1](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD) |PyTorch  | Yes  | Yes  | Yes  | -  | -  |   -  | -  | -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/examples/inference.ipynb) |
| [ResNet-50 v1.5](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5) |TensorFlow  | Yes  | Yes  | Yes  | -  | -  | -  | -  | -  | - |
| [ResNeXt101-32x4d](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnext101-32x4d)  |TensorFlow  | Yes  | Yes  | Yes  | -  | -  |   -  | -  | -  | - |
| [SE-ResNeXt101-32x4d](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TEnsorFlow/Classification/ConvNets/se-resnext101-32x4d)  |TensorFlow  | Yes  | Yes  | Yes  | -  | -  | -  | -  | -  | - |
| [Mask R-CNN](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN) |TensorFlow  | N/A  | Yes  | Yes  | -  | -  |   -  | -  | -  | - |
| [SSD320 v1.2](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Detection/SSD) | TensorFlow  | N/A  | Yes  | Yes  | -  | -  | -  | -  | -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/Detection/SSD/models/research/object_detection/object_detection_tutorial.ipynb) |
| [U-Net Industrial](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_Industrial) |TensorFlow  | N/A  | Yes  | Yes  | -  | Yes  |   -  | -  | Yes  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_Industrial/notebooks) |
| [U-Net Medical](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_Medical) | TensorFlow  | N/A  | Yes  | Yes  | -  |  Yes  |-  |   -  | Yes  | - |
| [V-Net Medical](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/VNet) | TensorFlow  | N/A  | Yes  | Yes  | -  |  Yes  | Yes |   -  | Yes  | - |
| [U-Net Medical](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/UNet_Medical) | TensorFlow-2  | N/A  | Yes  | Yes  | -  |  Yes  |-  |   -  | Yes  | - |
| [Mask R-CNN](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN) |TensorFlow-2  | N/A  | Yes  | Yes  | -  |  -  |-  |   -  | -  | - |
| [ResNet50 v1.5](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5) | MXNet  | Yes  | Yes  | Yes  | -  | -  |   -  | -  | -  | - |

## Natural Language Processing
| Models  | Framework | DALI | AMP | Multi-GPU | Multi-Node  | TensorRT  | ONNX  | Triton | TF-TRT | Notebook |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |------------- |------------- |------------- |------------- |
| [BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT) |PyTorch  | N/A  | Yes  | Yes  | Yes  | -  |   -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT/triton)  | -  | - |
| [Transformer-XL](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/Transformer-XL) |PyTorch  | N/A  | Yes  | Yes  | Yes  | -  |   -  | -  | -  | - |
| [GNMT v2](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/GNMT) |PyTorch  | N/A  | Yes  | Yes  | -  | -  |   -  | -  | -  | - |
| [Transformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer) |PyTorch  | N/A  | Yes  | Yes  | -  | -  |   -  | -  | -  | - |
| [BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) |TensorFlow  | N/A  | Yes  | Yes  | Yes  | Yes  | -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT/triton)  | Yes  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT/notebooks) |
| [BioBert](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT/biobert) | TensorFlow  | N/A  | Yes  | Yes  | -  | -  | -  | -  | -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/notebooks/biobert_ner_tf_inference.ipynb) |
| [Transformer-XL](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/Transformer-XL) |TensorFlow  | N/A  | Yes  | Yes  | -  | -  |   -  | -  | -  | - |
| [GNMT v2](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Translation/GNMT) | TensorFlow  | N/A  | Yes  | Yes  | -  | -  |   -  | -  | -  | - |
| [Faster Transformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer) | Tensorflow  | N/A  | -  | -  | -  | Yes  |   -  | -  | -  | - |
| [Transformer-XL](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/Transformer-XL) |TensorFlow  | N/A  | Yes  | Yes  | -  | -  |   -  | -  | -  | - |

## Recommender Systems
| Models  | Framework | DALI | AMP | Multi-GPU | Multi-Node  | TensorRT  | ONNX  | Triton | TF-TRT | Notebook |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |------------- |------------- |------------- |------------- |
| [DLRM](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/DLRM) |PyTorch  | N/A  | Yes  | Yes  | -  |  -  | Yes  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/DLRM/triton)  | -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/DLRM/notebooks) |
| [Neural Collaborative Filtering](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/NCF) |PyTorch  | N/A  | Yes  | Yes  | -  |  -  |-  | -  | -  | - |
| [Wide and Deep](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/WideAndDeep) | TensorFlow  | N/A  | Yes  | Yes  | -  | -  |   -  | -  | -  | - |
| [Neural Collaborative Filtering](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/NCF) |TensorFlow  | N/A  | Yes  | Yes  | -  | -  | -  | -  | -  | - |
| [Variational Autoencoder Collaborative Filtering](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/VAE-CF) |TensorFlow  | N/A  | Yes  | Yes  | -  | -  |   -  | -  | -  | - |


## Speech to Text
| Models  | Framework | DALI | AMP | Multi-GPU | Multi-Node  | TensorRT  | ONNX  | Triton | TF-TRT | Notebook |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |------------- |------------- |------------- |------------- |
| [Jasper](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper) |PyTorch  | N/A  | Yes  | Yes  | -  | Yes  |   Yes  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper/trtis)  | -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper/notebooks) |
| [HMM](https://github.com/NVIDIA/DeepLearningExamples/tree/master/Kaldi/SpeechRecognition) | Kaldi  | N/A  | -  | Yes  | -  | -  |   -  | Yes  | -  | - |

## Text to Speech
| Models  | Framework | DALI | AMP | Multi-GPU | Multi-Node  | TensorRT  | ONNX  | Triton | TF-TRT | Notebook |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |------------- |------------- |------------- |------------- |
| [Tacotron 2 and WaveGlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2) | PyTorch  | N/A  | Yes  | Yes  | -  | Yes  |   Yes  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2/trtis_cpp)  | -  | - |
| [FastPitch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch) | PyTorch  | N/A  | Yes  | Yes  | - | - | - | - | - | - |


## NVIDIA support
In each of the network READMEs, we indicate the level of support that will be provided. The range is from ongoing updates and improvements to a point-in-time release for thought leadership.

## Feedback / Contributions
We're posting these examples on GitHub to better support the community, facilitate feedback, as well as collect and implement contributions using GitHub Issues and pull requests. We welcome all contributions!

## Known issues
In each of the network READMEs, we indicate any known issues and encourage the community to provide feedback.

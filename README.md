# NVIDIA Deep Learning Examples for Tensor Cores

## Introduction
This repository provides State-of-the-Art Deep Learning examples that are easy to train and deploy, achieving the best reproducible accuracy and performance with NVIDIA CUDA-X software stack running on NVIDIA Volta, Turing and Ampere GPUs.

The Deep Learning Examples repository offers a unique value proposition that it provides end-to-end deep learning solutions from data preprocessing, data loading, training, inference, and deployment, customizable with your own data.


## NVIDIA GPU Cloud (NGC) Container Registry
These examples, along with our NVIDIA deep learning software stack, are provided in a monthly updated Docker container on the NGC container registry (https://ngc.nvidia.com). These containers include:  

- The latest NVIDIA examples from this repository
- The latest NVIDIA contributions shared upstream to the respective framework
- The latest NVIDIA Deep Learning software libraries, such as cuDNN, NCCL, cuBLAS, etc. which have all been through a rigorous monthly quality assurance process to ensure that they provide the best possible performance
- [Monthly release notes](https://docs.nvidia.com/deeplearning/dgx/index.html#nvidia-optimized-frameworks-release-notes) for each of the NVIDIA optimized containers


## Table Of Contents

| [Computer Vision](README.md#Computer-Vision) | [Natural Language Processing](README.md#Natural-Language-Processing) | [Recommender Systems](README.md#Recommender-Systems) | [Speech to Text](README.md#Speech-to-Text) | [Text to Speech](README.md#Text-to-Speech) | [Graph Neural Networks](README.md#Graph-Neural-Networks) |
| ------------ | --------------- | --------------- | ------------ | --------------- | --------------- | 
 
- [Computer Vision](#Computer-Vision)
- [Natural Language Processing](#Natural-Language-Processing)
- [Recommender Systems](#Recommender-Systems)
- [Speech to Text](#Speech-to-Text)
- [Text to Speech](#Text-to-Speech)
- [Graph Neural Networks](#Graph-Neural-Networks)


## <summary>Computer Vision</summary>
| Models  | Framework | Category | A100 | AMP | Multi-GPU | Multi-Node  | TRT  | ONNX  | Triton | DLC | NB |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |------------- |------------- |------------- |------------- |
| [ResNet-50](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5)  |PyTorch | Classification | Yes  | Yes  | Yes  | -  | Yes  | -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/triton/resnet50)  | Yes  | - |
| [ResNeXt-101](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnext101-32x4d)  |PyTorch  | Classification | Yes  | Yes  | Yes  | -  | Yes  |   -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/triton/resnext101-32x4d)  | Yes  | - |
| [SE-ResNeXt-101](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/se-resnext101-32x4d)  |PyTorch  | Classification | Yes  | Yes  | Yes  | -  | Yes  | -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/triton/se-resnext101-32x4d)  | Yes  | - |
| [EfficientNet-B0](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/efficientnet)  |PyTorch  | Classification | Yes  | Yes  | Yes  | -  | - | - | - | Yes  | - |
| [EfficientNet-B4](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/efficientnet)  |PyTorch  | Classification | Yes  | Yes  | Yes  | -  | - | - | - | Yes  | - |
| [EfficientNet-WideSE-B0](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/efficientnet)  |PyTorch  | Classification | Yes  | Yes  | Yes  | -  | - | - | - | Yes  | - |
| [EfficientNet-WideSE-B4](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/efficientnet)  |PyTorch  | Classification | Yes  | Yes  | Yes  | -  | - | - | - | Yes  | - |
| [EfficientDet](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/Efficientdet)  |PyTorch  | Detection | Yes  | Yes  | Yes  | -  | - | - | - | Yes  | - |
| [Mask R-CNN](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/MaskRCNN) |PyTorch  | Segmentation | Yes  | Yes  | Yes  | -  | -  |   -  | -  | -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Segmentation/MaskRCNN/pytorch/notebooks/pytorch_MaskRCNN_pyt_train_and_inference.ipynb) |
| [nnUNet](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet) |PyTorch  | Segmentation | Yes  | Yes  | Yes  | -  | -  |   -  | -  | Yes  | - |
| [SSD](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD) |PyTorch  |  Detection | Yes  | Yes  | Yes  | -  | -  |   -  | -  | -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/examples/inference.ipynb) |
| [ResNet-50](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5) |TensorFlow  | Classification | Yes  | Yes  | Yes  | -  | -  | -  | -  | Yes  | - |
| [ResNeXt101](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnext101-32x4d)  |TensorFlow  | Classification | Yes  | Yes  | Yes  | -  | -  |   -  | -  | Yes  | - |
| [SE-ResNeXt-101](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/se-resnext101-32x4d)  |TensorFlow  | Classification | Yes  | Yes  | Yes  | -  | -  | -  | -  | Yes  | - |
| [Mask R-CNN](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN) |TensorFlow  | Segmentation | Yes  | Yes  | Yes  | -  | -  |   -  | -  | Yes  | - |
| [SSD](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Detection/SSD) | TensorFlow  | Detection | Yes  | Yes  | Yes  | -  | -  | -  | -  | Yes  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/Detection/SSD/models/research/object_detection/object_detection_tutorial.ipynb) |
| [U-Net Ind](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_Industrial) |TensorFlow  | Segmentation | Yes  | Yes  | Yes  | -  | -  |   -  | -  | Yes  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_Industrial/notebooks) |
| [U-Net Med](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_Medical) | TensorFlow  | Segmentation | Yes  | Yes  | Yes  | -  |  -  |-  |   -  | Yes  | - |
| [U-Net 3D](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_3D_Medical) | TensorFlow  | Segmentation | Yes  | Yes  | Yes  | -  |  -  | -  |   -  | Yes | - |
| [V-Net Med](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/VNet) | TensorFlow  | Segmentation | Yes  | Yes  | Yes  | -  |  -  | -  |   -  | Yes | - |
| [U-Net Med](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/UNet_Medical) | TensorFlow2  | Segmentation | Yes  | Yes  | Yes  | -  |  -  |-  |   -  | Yes  | - |
| [Mask R-CNN](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN) |TensorFlow2  | Segmentation | Yes  | Yes  | Yes  | -  |  -  |-  |   -  | Yes  | - |
| [EfficientNet](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Classification/ConvNets/efficientnet) |TensorFlow2  | Classification | Yes  | Yes  | Yes  | Yes  |  -  |-  |   -  | Yes  | - |
| [ResNet-50](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5) | MXNet  | Classification | -  | Yes  | Yes  | -  | -  |   -  | -  | -  | - |

## <summary>Natural Language Processing</summary>
| Models  | Framework | Category | A100 | AMP | Multi-GPU | Multi-Node  | TRT  | ONNX  | Triton | DLC | NB |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |------------- |------------- |------------- |------------- |
| [BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT) |PyTorch  | Language Modeling | Yes  | Yes  | Yes  | Yes  | -  |   -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT/triton)  | Yes  | - |
| [TransformerXL](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/Transformer-XL) |PyTorch  | Language Modeling | Yes  | Yes  | Yes  | Yes  | -  |   -  | -  | Yes  | - |
| [GNMT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/GNMT) |PyTorch | Translation |  Yes  | Yes  | Yes  | -  | -  |   -  | -  | -  | - |
| [Transformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer) |PyTorch  | Translation | Yes  | Yes  | Yes  | -  | -  |   -  | -  | -  | - |
| [ELECTRA](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/LanguageModeling/ELECTRA) | TensorFlow2  | Language Modeling | Yes  | Yes  | Yes  | Yes  | -  |   -  | -  | Yes  | - |
| [BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) |TensorFlow  | Language Modeling | Yes  | Yes  | Yes  | Yes  | Yes  | -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT/triton)  | Yes  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT/notebooks) |
| [BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/LanguageModeling/BERT) |TensorFlow2  | Language Modeling | Yes  | Yes  | Yes  | Yes  | - | -  | - | Yes  | - |
| [BioBert](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT/biobert) | TensorFlow  | Language Modeling | Yes  | Yes  | Yes  | -  | -  | -  | -  | Yes  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/notebooks/biobert_ner_tf_inference.ipynb) |
| [TransformerXL](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/Transformer-XL) |TensorFlow  | Language Modeling | Yes  | Yes  | Yes  | -  | -  |   -  | -  | -  | - |
| [GNMT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Translation/GNMT) | TensorFlow  | Translation | Yes  | Yes  | Yes  | -  | -  |   -  | -  | -  | - |
| [Faster Transformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer) | Tensorflow  | Language Modeling | -  | -  | -  | -  | Yes  |   -  | -  | -  | - |


## <summary>Recommender Systems</summary>
| Models  | Framework | A100 | AMP | Multi-GPU | Multi-Node  | TRT  | ONNX  | Triton | DLC | NB |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |------------- |------------- |------------- |------------- |
| [DLRM](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/DLRM) |PyTorch  | Yes  | Yes  | Yes  | -  |  -  | Yes  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/DLRM/triton)  | Yes | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/DLRM/notebooks) |
| [DLRM](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Recommendation/DLRM) | TensorFlow2  | Yes  | Yes  | Yes  | Yes  |  -  | -  | - | Yes | - |
| [NCF](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/NCF) |PyTorch  | Yes  | Yes  | Yes  | -  |  -  |-  | -  | -  | - |
| [Wide&Deep](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/WideAndDeep) | TensorFlow  | Yes  | Yes  | Yes  | -  | -  |   -  | -  | Yes  | - |
| [Wide&Deep](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Recommendation/WideAndDeep) | TensorFlow2  | Yes  | Yes  | Yes  | -  | -  |   -  | -  | Yes  | - |
| [NCF](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/NCF) |TensorFlow  | Yes  | Yes  | Yes  | -  | -  | -  | - | Yes | - |
| [VAE-CF](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/VAE-CF) |TensorFlow  | Yes  | Yes  | Yes  | -  | -  |   -  | -  | -  | - |


## <summary>Speech to Text</summary>
| Models  | Framework | A100 | AMP | Multi-GPU | Multi-Node  | TRT  | ONNX  | Triton | DLC | NB |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |------------- |------------- |------------- |------------- |
| [Jasper](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper) |PyTorch  | Yes  | Yes  | Yes  | -  | Yes  |   Yes  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper/triton)  | Yes  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper/notebooks) |
| [Hidden Markov Model](https://github.com/NVIDIA/DeepLearningExamples/tree/master/Kaldi/SpeechRecognition) | Kaldi  | -  | -  | Yes  | -  | -  |   -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/Kaldi/SpeechRecognition)  | -  | - |

## <summary>Text to Speech</summary>
| Models  | Framework | A100 | AMP | Multi-GPU | Multi-Node  | TRT  | ONNX  | Triton | DLC | NB | 
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |------------- |------------- |------------- |------------- |
| [FastPitch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch) | PyTorch  | Yes  | Yes  | Yes  | - | - | - | - | Yes | - |
| [FastSpeech](https://github.com/NVIDIA/DeepLearningExamples/tree/master/CUDA-Optimized/FastSpeech) | PyTorch  | -  | Yes  | Yes  | - | Yes | - | - | - | - |
| [Tacotron 2 and WaveGlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2) | PyTorch  | Yes  | Yes  | Yes  | -  | Yes  |   Yes  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2/trtis_cpp)  | Yes  | - |

## <summary>Graph Neural Networks</summary>
| Models  | Framework | A100 | AMP | Multi-GPU | Multi-Node  | TRT  | ONNX  | Triton | DLC | NB | 
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |------------- |------------- |------------- |------------- |
| [SE(3)-Transformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/DGLPyTorch/DrugDiscovery/SE3Transformer) | PyTorch  | Yes  | Yes  | Yes  | - | - | - | - | - | - |


## NVIDIA support
In each of the network READMEs, we indicate the level of support that will be provided. The range is from ongoing updates and improvements to a point-in-time release for thought leadership.

## Glossary
 
**Multinode Training**  
Supported on a pyxis/enroot Slurm cluster.

**Deep Learning Compiler (DLC)**  
TensorFlow XLA and PyTorch JIT and/or TorchScript

**Accelerated Linear Algebra (XLA)**  
XLA is a domain-specific compiler for linear algebra that can accelerate TensorFlow models with potentially no source code changes. The results are improvements in speed and memory usage.

**PyTorch JIT and/or TorchScript**  
TorchScript is a way to create serializable and optimizable models from PyTorch code. TorchScript, an intermediate representation of a PyTorch model (subclass of nn.Module) that can then be run in a high-performance environment such as C++.

**Automatic Mixed Precision (AMP)**  
Automatic Mixed Precision (AMP) enables mixed precision training on Volta, Turing, and NVIDIA Ampere GPU architectures automatically.

**TensorFloat-32 (TF32)**  
TensorFloat-32 (TF32) is the new math mode in [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) GPUs for handling the matrix math also called tensor operations. TF32 running on Tensor Cores in A100 GPUs can provide up to 10x speedups compared to single-precision floating-point math (FP32) on Volta GPUs. TF32 is supported in the NVIDIA Ampere GPU architecture and is enabled by default.

**Jupyter Notebooks (NB)**  
The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text.


## Feedback / Contributions
We're posting these examples on GitHub to better support the community, facilitate feedback, as well as collect and implement contributions using GitHub Issues and pull requests. We welcome all contributions!

## Known issues
In each of the network READMEs, we indicate any known issues and encourage the community to provide feedback.

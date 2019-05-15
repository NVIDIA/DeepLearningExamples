# NVIDIA Deep Learning Examples for Volta Tensor Cores

## Introduction
This repository provides the latest deep learning example networks for training.  These examples focus on achieving the best performance and convergence from NVIDIA Volta Tensor Cores.

## NVIDIA GPU Cloud (NGC) Container Registry
These examples, along with our NVIDIA deep learning software stack, are provided in a monthly updated Docker container on the NGC container registry (https://ngc.nvidia.com). These containers include:  

- The latest NVIDIA examples from this repository
- The latest NVIDIA contributions shared upstream to the respective framework
- The latest NVIDIA Deep Learning software libraries, such as cuDNN, NCCL, cuBLAS, etc. which have all been through a rigorous monthly quality assurance process to ensure that they provide the best possible performance
- [Monthly release notes](https://docs.nvidia.com/deeplearning/dgx/index.html#nvidia-optimized-frameworks-release-notes) for each of the NVIDIA optimized containers

## Directory structure
The examples are organized first by framework, such as TensorFlow, PyTorch, etc. and second by use case, such as computer vision, natural language processing, etc. We hope this structure enables you to quickly locate the example networks that best suit your needs. Here are the currently supported models:

### Computer Vision
- __ResNet-50__ [[MXNet](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5)] [[PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/RN50v1.5)] [[TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/RN50v1.5)]
- __SSD__ [[PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD)] [[TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Detection/SSD)]
- __Mask R-CNN__ [[PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/MaskRCNN)]
- __U-Net__ [[TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_Industrial)]

### Natural Language Processing
- __GNMT__ [[PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/GNMT)] [[TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Translation/GNMT)]
- __Transformer__ [[PyTorch]](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer)]
- __BERT__ [[TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT)]

### Recommender Systems
- __NCF__ [[PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/NCF)] [[TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/NCF)]

### Text to Speech
- __Tacotron & WaveGlow__ [[PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2)]

## NVIDIA support
In each of the network READMEs, we indicate the level of support that will be provided. The range is from ongoing updates and improvements to a point-in-time release for thought leadership.

## Feedback / Contributions
We're posting these examples on GitHub to better support the community, facilitate feedback, as well as collect and implement contributions using GitHub Issues and pull requests. We welcome all contributions!

## Known issues
In each of the network READMEs, we indicate any known issues and encourage the community to provide feedback.

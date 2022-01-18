# NVIDIA Deep Learning Examples for Tensor Cores

## Introduction
This repository provides State-of-the-Art Deep Learning examples that are easy to train and deploy, achieving the best reproducible accuracy and performance with NVIDIA CUDA-X software stack running on NVIDIA Volta, Turing and Ampere GPUs.

## NVIDIA GPU Cloud (NGC) Container Registry
These examples, along with our NVIDIA deep learning software stack, are provided in a monthly updated Docker container on the NGC container registry (https://ngc.nvidia.com). These containers include:  

- The latest NVIDIA examples from this repository
- The latest NVIDIA contributions shared upstream to the respective framework
- The latest NVIDIA Deep Learning software libraries, such as cuDNN, NCCL, cuBLAS, etc. which have all been through a rigorous monthly quality assurance process to ensure that they provide the best possible performance
- [Monthly release notes](https://docs.nvidia.com/deeplearning/dgx/index.html#nvidia-optimized-frameworks-release-notes) for each of the NVIDIA optimized containers


## Natural Language Processing
| Models  | Framework | A100 | AMP | Multi-GPU | Multi-Node  | TRT  | ONNX  | Triton | DLC | NB |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |------------- |------------- |------------- |------------- |
| [BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT) |PyTorch  | Yes  | Yes  | Yes  | Yes  | -  |   -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT/triton)  | Yes  | - |
| [TransformerXL](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/Transformer-XL) |PyTorch  | Yes  | Yes  | Yes  | Yes  | -  |   -  | -  | Yes  | - |
| [GNMT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/GNMT) |PyTorch  | Yes  | Yes  | Yes  | -  | -  |   -  | -  | -  | - |
| [Transformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer) |PyTorch  | Yes  | Yes  | Yes  | -  | -  |   -  | -  | -  | - |
| [ELECTRA](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/LanguageModeling/ELECTRA) | TensorFlow2  | Yes  | Yes  | Yes  | Yes  | -  |   -  | -  | Yes  | - |
| [BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) |TensorFlow  | Yes  | Yes  | Yes  | Yes  | Yes  | -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT/triton)  | Yes  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT/notebooks) |
| [BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/LanguageModeling/BERT) |TensorFlow2  | Yes  | Yes  | Yes  | Yes  | - | -  | - | Yes  | - |
| [BioBert](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT/biobert) | TensorFlow  | Yes  | Yes  | Yes  | -  | -  | -  | -  | Yes  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/notebooks/biobert_ner_tf_inference.ipynb) |
| [TransformerXL](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/Transformer-XL) |TensorFlow  | Yes  | Yes  | Yes  | -  | -  |   -  | -  | -  | - |
| [GNMT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Translation/GNMT) | TensorFlow  | Yes  | Yes  | Yes  | -  | -  |   -  | -  | -  | - |
| [Faster Transformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer) | Tensorflow  | -  | -  | -  | -  | Yes  |   -  | -  | -  | - |


## Speech to Text
| Models  | Framework | A100 | AMP | Multi-GPU | Multi-Node  | TRT  | ONNX  | Triton | DLC | NB |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |------------- |------------- |------------- |------------- |
| [Jasper](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper) |PyTorch  | Yes  | Yes  | Yes  | -  | Yes  |   Yes  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper/trtis)  | Yes  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper/notebooks) |
| [Hidden Markov Model](https://github.com/NVIDIA/DeepLearningExamples/tree/master/Kaldi/SpeechRecognition) | Kaldi  | -  | -  | Yes  | -  | -  |   -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/Kaldi/SpeechRecognition)  | -  | - |

## Text to Speech
| Models  | Framework | A100 | AMP | Multi-GPU | Multi-Node  | TRT  | ONNX  | Triton | DLC | NB | 
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |------------- |------------- |------------- |------------- |
| [FastPitch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch) | PyTorch  | Yes  | Yes  | Yes  | - | - | - | - | Yes | - |
| [FastSpeech](https://github.com/NVIDIA/DeepLearningExamples/tree/master/CUDA-Optimized/FastSpeech) | PyTorch  | -  | Yes  | Yes  | - | Yes | - | - | - | - |
| [Tacotron 2 and WaveGlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2) | PyTorch  | Yes  | Yes  | Yes  | -  | Yes  |   Yes  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2/trtis_cpp)  | Yes  | - |


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

# BENCHMARK ANY NVIDIA GPU CARD

## Quickstart
### General workflow
1) replace the wandb api key by yours
3) define the GPU setup you have
4) set the benchmark you want to explore
5) run the shell

### Before you start
We highly suggest to setup and pipenv isolated environment
```shell
$ pip install --user pipenv
```

then
```
$ git clone git@github.com:theunifai/DeepLearningExamples.git
```

```shell
$ cd DeepLearningExamples

$ pipenv shell

$ pipenv install -r requirements.txt
```

### Setup the wandb key
you can either set it in the benchmark.yml file 
or use the shell 
```shell
$ wandb login
```
if the API Key for wandb is not set in the benchmark.yml file 
the system will look into your environment to fetch your api key

### Define the GPU topology to benchmark
in the Yaml file set the topology using you GPU configuration:
```shell
$ nvidia-smi
```

<img width="577" alt="Capture d’écran 2021-12-23 à 13 13 27" src="https://user-images.githubusercontent.com/690878/147239024-df41cc89-1cee-4a20-aacf-098b84b43e92.png">
nvidia-smi will help you see the ids of the GPU to analyse.


as presented above in the example with nvidia-smi here is the corresponding configuration in the yaml file.
<img width="232" alt="Capture d’écran 2021-12-23 à 13 14 32" src="https://user-images.githubusercontent.com/690878/147239140-54873109-7e63-46eb-aedb-e1dae4722862.png">

you can activate the capabilities to explore for each GPU (for instance V100s doesnt support AMP so it should be set to false).

### setup of the benchmarks to explore

<img width="323" alt="Capture d’écran 2021-12-23 à 13 16 08" src="https://user-images.githubusercontent.com/690878/147239334-38b9e762-a4d8-4ebe-9b0a-e078574d4e67.png">
In the above example we can see that the benchmarks to explore are based on template already structure by UnifAI's team.
all you have to set is (if needed) overwrite the hyperparameters you want to explore.

Everything param value should be an array following this standard:
```yaml
benchmarks
  benchmark-name
    benchmark-template: <template on which you want to base your benchmark on>
    active: <boolean status of the benchmark to explore : false means skip the benchmark>
    params:
      param1: [<custom value1=a>, <custom value2=b>] <- this must be an array
      param2: [<custom value1=c>, <custom value2=d>] <- this must be an array
```

the system will do the cartesian exploration of the benchmark meaning in our example exploring 4 parameters combination:
 - a.c
 - a.d
 - b.c
 - b.d


### Running the benchmarks
You are now ready to run the benchmarks
you have many options that can be set
```shell
# ./benchmark.py --help
```

```shell
# ./benchmark.py --run
```
This command will build and run the benchmarks for AMP (Automatic Mixed Precision), FP32 and TF32.


# Work Benchmark Implementation Status
| Framework   | Domain            | Task              | Model               | Status |
| ----------- | ----------------- | ----------------- | ------------------- | ------ |
| PyTorch     | Image             | Classification    | efficientnet        | Ok     |
| PyTorch     | Image             | Classification    | resnet50v1.5        | Ok     |
| PyTorch     | Image             | Classification    | resnext101-32x4d    | Ok     |
| PyTorch     | Image             | Classification    | se-resnext101-32x4d | Ok     |
| PyTorch     | Image             | Detection         | Efficientdet        | Ok     |
| PyTorch     | Image             | Detection         | SSD                 | Ok     |
| PyTorch     | DrugDiscovery     | SE3Transformer    | SE3Transformer      |
| PyTorch     | Forecasting       | TFT               | TFT                 |        |
| PyTorch     | LanguageModeling  | BART              | BART                | Ok     |
| PyTorch     | LanguageModeling  | BERT              | BERT                |        |
| PyTorch     | LanguageModeling  | Transformer-XL    | Transformer-XL      |
| PyTorch     | Recommendation    | DLRM              | DLRM                |        |
| PyTorch     | Recommendation    | NCF               | NCF                 |        |
| PyTorch     | Segmentation      | MaskRCNN          | MaskRCNN            |        |
| PyTorch     | Segmentation      | nnUNet            | nnUNet              |        |
| PyTorch     | SpeechRecognition | Jasper            | Jasper              |        |
| PyTorch     | SpeechRecognition | QuartzNet         | QuartzNet           |        |
| PyTorch     | SpeechSynthesis   | FastPitch         | FastPitch           |        |
| PyTorch     | SpeechSynthesis   | Tacotron2         | Tacotron2           |        |
| PyTorch     | Translation       | GNMT              | GNMT                |        |
| PyTorch     | Translation       | Transformer       | Transformer         |
| TensorFlow  | Image             | Classification    | resnet50v1.5        |
| TensorFlow  | Image             | Classification    | resnext101-32x4d    |
| TensorFlow  | Image             | Classification    | se-resnext101-32x4d |
| TensorFlow  | Image             | Detection         | SSD                 |        |
| TensorFlow  | LanguageModeling  | BERT              | BERT                |        |
| TensorFlow  | LanguageModeling  | Transformer-XL    | Transformer-XL      |
| TensorFlow  | Recommendation    | VAE-CF            | VAE-CF              |        |
| TensorFlow  | Recommendation    | NCF               | NCF                 |        |
| TensorFlow  | Recommendation    | WideAndDeep       | WideAndDeep         |
| TensorFlow  | Segmentation      | MaskRCNN          | MaskRCNN            |        |
| TensorFlow  | Segmentation      | UNet\_3D\_Medical | UNet\_3D\_Medical   |
| TensorFlow  | Segmentation      | UNet\_Industrial  | UNet\_Industrial    |
| TensorFlow  | Segmentation      | UNet\_Medical     | UNet\_Medical       |
| TensorFlow  | Segmentation      | Vnet              | Vnet                |        |
| TensorFlow  | Translation       | GNMT              | GNMT                |        |
| TensorFlow2 | Image             | Classification    | efficientnet        |        |
| TensorFlow2 | LanguageModeling  | BERT              | BERT                |        |
| TensorFlow2 | LanguageModeling  | ELECTRA           | ELECTRA             |        |
| TensorFlow2 | Recommendation    | DLRM              | DLRM                |        |
| TensorFlow2 | Recommendation    | WideAndDeep       | WideAndDeep         |
| TensorFlow2 | Segmentation      | MaskRCNN          | MaskRCNN            |        |
| TensorFlow2 | Segmentation      | UNet\_Medical     | UNet\_Medical       |
| DGLPyTorch  | DrugDiscovery     | SE3Transformer    | SE3Transformer      |
| MxNet       | Image             | Classification    | resnet50v1.5        |



# ORIGINALLY : NVIDIA Deep Learning Examples for Tensor Cores

## Introduction
This repository provides State-of-the-Art Deep Learning examples that are easy to train and deploy, achieving the best reproducible accuracy and performance with NVIDIA CUDA-X software stack running on NVIDIA Volta, Turing and Ampere GPUs.

## NVIDIA GPU Cloud (NGC) Container Registry
These examples, along with our NVIDIA deep learning software stack, are provided in a monthly updated Docker container on the NGC container registry (https://ngc.nvidia.com). These containers include:  

- The latest NVIDIA examples from this repository
- The latest NVIDIA contributions shared upstream to the respective framework
- The latest NVIDIA Deep Learning software libraries, such as cuDNN, NCCL, cuBLAS, etc. which have all been through a rigorous monthly quality assurance process to ensure that they provide the best possible performance
- [Monthly release notes](https://docs.nvidia.com/deeplearning/dgx/index.html#nvidia-optimized-frameworks-release-notes) for each of the NVIDIA optimized containers


## Computer Vision
| Models  | Framework | A100 | AMP | Multi-GPU | Multi-Node  | TRT  | ONNX  | Triton | DLC | NB |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |------------- |------------- |------------- |------------- |
| [ResNet-50](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5)  |PyTorch  | Yes  | Yes  | Yes  | -  | Yes  | -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/triton/resnet50)  | Yes  | - |
| [ResNeXt-101](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnext101-32x4d)  |PyTorch  | Yes  | Yes  | Yes  | -  | Yes  |   -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/triton/resnext101-32x4d)  | Yes  | - |
| [SE-ResNeXt-101](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/se-resnext101-32x4d)  |PyTorch  | Yes  | Yes  | Yes  | -  | Yes  | -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/triton/se-resnext101-32x4d)  | Yes  | - |
| [EfficientNet-B0](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/efficientnet)  |PyTorch  | Yes  | Yes  | Yes  | -  | - | - | - | Yes  | - |
| [EfficientNet-B4](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/efficientnet)  |PyTorch  | Yes  | Yes  | Yes  | -  | - | - | - | Yes  | - |
| [EfficientNet-WideSE-B0](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/efficientnet)  |PyTorch  | Yes  | Yes  | Yes  | -  | - | - | - | Yes  | - |
| [EfficientNet-WideSE-B4](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/efficientnet)  |PyTorch  | Yes  | Yes  | Yes  | -  | - | - | - | Yes  | - |
| [Mask R-CNN](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/MaskRCNN) |PyTorch  | Yes  | Yes  | Yes  | -  | -  |   -  | -  | -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Segmentation/MaskRCNN/pytorch/notebooks/pytorch_MaskRCNN_pyt_train_and_inference.ipynb) |
| [nnUNet](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet) |PyTorch  | Yes  | Yes  | Yes  | -  | -  |   -  | -  | Yes  | - |
| [SSD](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD) |PyTorch  | Yes  | Yes  | Yes  | -  | -  |   -  | -  | -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/examples/inference.ipynb) |
| [ResNet-50](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5) |TensorFlow  | Yes  | Yes  | Yes  | -  | -  | -  | -  | Yes  | - |
| [ResNeXt101](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnext101-32x4d)  |TensorFlow  | Yes  | Yes  | Yes  | -  | -  |   -  | -  | Yes  | - |
| [SE-ResNeXt-101](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/se-resnext101-32x4d)  |TensorFlow  | Yes  | Yes  | Yes  | -  | -  | -  | -  | Yes  | - |
| [Mask R-CNN](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN) |TensorFlow  | Yes  | Yes  | Yes  | -  | -  |   -  | -  | Yes  | - |
| [SSD](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Detection/SSD) | TensorFlow  | Yes  | Yes  | Yes  | -  | -  | -  | -  | Yes  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/Detection/SSD/models/research/object_detection/object_detection_tutorial.ipynb) |
| [U-Net Ind](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_Industrial) |TensorFlow  | Yes  | Yes  | Yes  | -  | -  |   -  | -  | Yes  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_Industrial/notebooks) |
| [U-Net Med](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_Medical) | TensorFlow  | Yes  | Yes  | Yes  | -  |  -  |-  |   -  | Yes  | - |
| [U-Net 3D](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_3D_Medical) | TensorFlow  | Yes  | Yes  | Yes  | -  |  -  | -  |   -  | Yes | - |
| [V-Net Med](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/VNet) | TensorFlow  | Yes  | Yes  | Yes  | -  |  -  | -  |   -  | Yes | - |
| [U-Net Med](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/UNet_Medical) | TensorFlow2  | Yes  | Yes  | Yes  | -  |  -  |-  |   -  | Yes  | - |
| [Mask R-CNN](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN) |TensorFlow2  | Yes  | Yes  | Yes  | -  |  -  |-  |   -  | Yes  | - |
| [EfficientNet](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Classification/ConvNets/efficientnet) |TensorFlow2  | Yes  | Yes  | Yes  | Yes  |  -  |-  |   -  | Yes  | - |
| [ResNet-50](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5) | MXNet  | -  | Yes  | Yes  | -  | -  |   -  | -  | -  | - |

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


## Recommender Systems
| Models  | Framework | A100 | AMP | Multi-GPU | Multi-Node  | TRT  | ONNX  | Triton | DLC | NB |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |------------- |------------- |------------- |------------- |
| [DLRM](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/DLRM) |PyTorch  | Yes  | Yes  | Yes  | -  |  -  | Yes  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/DLRM/triton)  | Yes | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/DLRM/notebooks) |
| [DLRM](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Recommendation/DLRM) | TensorFlow2  | Yes  | Yes  | Yes  | Yes  |  -  | -  | - | Yes | - |
| [NCF](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/NCF) |PyTorch  | Yes  | Yes  | Yes  | -  |  -  |-  | -  | -  | - |
| [Wide&Deep](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/WideAndDeep) | TensorFlow  | Yes  | Yes  | Yes  | -  | -  |   -  | -  | Yes  | - |
| [Wide&Deep](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Recommendation/WideAndDeep) | TensorFlow2  | Yes  | Yes  | Yes  | -  | -  |   -  | -  | Yes  | - |
| [NCF](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/NCF) |TensorFlow  | Yes  | Yes  | Yes  | -  | -  | -  | - | Yes | - |
| [VAE-CF](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/VAE-CF) |TensorFlow  | Yes  | Yes  | Yes  | -  | -  |   -  | -  | -  | - |


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

## Graph Neural Networks
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

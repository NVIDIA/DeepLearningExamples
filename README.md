# NVIDIA Deep Learning Examples for Tensor Cores

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
- __ResNet-50__ [[MXNet](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5)] [[PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets)] [[TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/RN50v1.5)]
- __ResNext__ [[PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets)]
- __SE-ResNext__ [[PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets)]
- __SSD__ [[PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD)] [[TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Detection/SSD)]
- __Mask R-CNN__ [[PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/MaskRCNN)] [[TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN)] [[TensorFlow 2](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN)] 
- __U-Net(industrial)__ [[TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_Industrial)]
- __U-Net(medical)__ [[TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_Medical)] [[TensorFlow 2](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/UNet_Medical)]
- __VNet__ [[TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/VNet)]

### Natural Language Processing
- __BERT__ [[PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)] [[TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT)]
- __GNMT__ [[PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/GNMT)] [[TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Translation/GNMT)]
- __Transformer__ [[PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer)]
- __Transformer-XL__ [[PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/Transformer-XL)] [[TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/Transformer-XL)]


### Recommender Systems
- __DLRM__ [[PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/DLRM)]
- __NCF__ [[PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/NCF)] [[TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/NCF)]
- __VAE-CF__ [[TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/VAE-CF)]
- __WideAndDeep__ [[TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/WideAndDeep)]


### Text to Speech
- __Tacotron2 & WaveGlow__ [[PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2)]

### Speech Recognition
- __Jasper__ [[PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper)]

### CUDA Accelerated Applications
- __Kaldi__ [[TRTIS](https://github.com/NVIDIA/DeepLearningExamples/tree/master/Kaldi/SpeechRecognition)]

## Jupyter Notebooks

|  Models | TensorFlow | PyTorch | TensorRT | TRTIS |  
| -------------               | ------------- | ------------- | ------------- | ------------- |
|   SSD                       |   [ Inference](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/Detection/SSD/models/research/object_detection/object_detection_tutorial.ipynb)  |   [ Inference](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/examples/inference.ipynb)     |  -     |     -     |
|   MaskRCNN                  |   -  |   [ Training & Inference](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Segmentation/MaskRCNN/pytorch/notebooks/pytorch_MaskRCNN_pyt_train_and_inference.ipynb)     |  -     |     -     |
|   Jasper                    |   -  |   -     |  [ PyTorch Inference TensorRT Colab](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechRecognition/Jasper/notebooks/Colab_Jasper_TRT_inference_demo.ipynb), [ PyTorch Inference TensorRT](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechRecognition/Jasper/notebooks/JasperTRT.ipynb)     |     [ PyTorch Inference TRTIS](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechRecognition/Jasper/notebooks/JasperTRTIS.ipynb)     |
|   Tacotron2 & WaveGlow      |   -  |   [ Training & Inference](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/notebooks/Tacotron2.ipynb)     |  -     |     [ PyTorch Inference TRTIS](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/notebooks/trtis/notebook.ipynb)     |
|   BERT                      |   [ Inference Movie Review Sentiment](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/predicting_movie_reviews_with_bert_on_tf_hub.ipynb), [ Fine-Tuning SQuaD](https://github.com/NVIDIA/DeepLearningExamples/blob/80f9481ef8a2c61958f240618077bf89cfce78f6/TensorFlow/LanguageModeling/BERT/notebooks/bert_squad_tf_finetuning.ipynb), [ Inference Colab](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/notebooks/bert_squad_tf_inference_colab.ipynb), [ Inference](https://github.com/NVIDIA/DeepLearningExamples/blob/80f9481ef8a2c61958f240618077bf89cfce78f6/TensorFlow/LanguageModeling/BERT/notebooks/bert_squad_tf_inference.ipynb)  |   -     |  -     |     -     |
|   BioBERT                   |   [ Inference](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/notebooks/biobert_ner_tf_inference.ipynb)  |   -     |  -     |     -     |
|   UNet Industrial           |   [ Export and Inference Colab](https://github.com/NVIDIA/DeepLearningExamples/blob/80f9481ef8a2c61958f240618077bf89cfce78f6/TensorFlow/Segmentation/UNet_Industrial/notebooks/Colab_UNet_Industrial_TF_TFHub_export.ipynb), [ Inference](https://github.com/NVIDIA/DeepLearningExamples/blob/80f9481ef8a2c61958f240618077bf89cfce78f6/TensorFlow/Segmentation/UNet_Industrial/notebooks/Colab_UNet_Industrial_TF_TFHub_inference_demo.ipynb)  |   -     |  -     |     -     |
|   Automatic Mixed Precision |   [ AMP Training](https://github.com/NVIDIA/DeepLearningExamples/blob/80f9481ef8a2c61958f240618077bf89cfce78f6/TensorFlow/docs/amp/notebook_v1.14/auto_mixed_precision_demo_cifar10.ipynb)  |   -     |  -     |     -     |


## Feature Matrix
| Models  | Framework | DALI | AMP | Multi-GPU | Multi-Node  | TensorRT  | ONNX  | TRTIS | TF-TRT |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |------------- |------------- |------------- |
| [ResNet50 v1.5](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5)  |PyTorch  | Yes  | Yes  | Yes  | -  | -  | -  | -  | -  |
| [ResNeXt101-32x4d](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnext101-32x4d)  |PyTorch  | Yes  | Yes  | Yes  | -  | -  |   -  | -  | -  |
| [SE-ResNeXt101-32x4d](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/se-resnext101-32x4d)  |PyTorch  | Yes  | Yes  | Yes  | -  | -  | -  | -  | -  |
| [SSD300 v1.1](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD) |PyTorch  | Yes  | Yes  | Yes  | -  | -  |   -  | -  | -  |
| [BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT) |PyTorch  | N/A  | Yes  | Yes  | Yes  | -  |   -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT/triton)  | -  |
| [Transformer-XL](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/Transformer-XL) |PyTorch  | N/A  | Yes  | Yes  | Yes  | -  |   -  | -  | -  |
| [Neural Collaborative Filtering](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/NCF) |PyTorch  | N/A  | Yes  | Yes  | -  |  -  |-  | -  | -  |
| [DLRM](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/NCF) |PyTorch  | N/A  | Yes  | -  | -  |  -  |-  | -  | -  |
| [Mask R-CNN](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/MaskRCNN) |PyTorch  | N/A  | Yes  | Yes  | -  | -  |   -  | -  | -  |
| [Jasper](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper) |PyTorch  | N/A  | Yes  | Yes  | -  | Yes  |   Yes  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper/trtis)  | -  |
| [Tacotron 2 And WaveGlow v1.10](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2) | PyTorch  | N/A  | Yes  | Yes  | -  | Yes  |   Yes  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2/notebooks/trtis)  | -  |
| [GNMT v2](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/GNMT) |PyTorch  | N/A  | Yes  | Yes  | -  | -  |   -  | -  | -  |
| [Transformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer) |PyTorch  | N/A  | Yes  | Yes  | -  | -  |   -  | -  | -  |
| [ResNet-50 v1.5](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/RN50v1.5) |TensorFlow  | Yes  | Yes  | Yes  | -  | -  | -  | -  | -  |
| [SSD320 v1.2](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Detection/SSD) | TensorFlow  | N/A  | Yes  | Yes  | -  | -  | -  | -  | -  |
| [BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) |TensorFlow  | N/A  | Yes  | Yes  | Yes  | Yes  | -  | [Yes](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT/trtis)  | Yes  |
| [BioBert](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT/biobert) | TensorFlow  | N/A  | Yes  | Yes  | -  | -  | -  | -  | -  |
| [Transformer-XL](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/Transformer-XL) |TensorFlow  | N/A  | Yes  | Yes  | -  | -  |   -  | -  | -  |
| [Neural Collaborative Filtering](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/NCF) |TensorFlow  | N/A  | Yes  | Yes  | -  | -  | -  | -  | -  |
| [Variational Autoencoder Collaborative Filtering](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/VAE-CF) |TensorFlow  | N/A  | Yes  | Yes  | -  | -  |   -  | -  | -  |
| [WideAndDeep](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/WideAndDeep) | TensorFlow  | N/A  | Yes  | Yes  | -  | -  |   -  | -  | -  |
| [U-Net Industrial](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_Industrial) |TensorFlow  | N/A  | Yes  | Yes  | -  | Yes  |   -  | -  | Yes  |
| [U-Net Medical](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_Medical) | TensorFlow  | N/A  | Yes  | Yes  | -  |  Yes  |-  |   -  | Yes  |
| [V-Net Medical](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/VNet) | TensorFlow  | N/A  | Yes  | Yes  | -  |  Yes  | Yes |   -  | Yes  |
| [Mask R-CNN](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN) |TensorFlow  | N/A  | Yes  | Yes  | -  | -  |   -  | -  | -  |
| [GNMT v2](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Translation/GNMT) | TensorFlow  | N/A  | Yes  | Yes  | -  | -  |   -  | -  | -  |
| [Faster Transformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer) | Tensorflow  | N/A  | -  | -  | -  | Yes  |   -  | -  | -  |
| [U-Net Medical](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/UNet_Medical) | TensorFlow-2  | N/A  | Yes  | Yes  | -  |  Yes  |-  |   -  | Yes  |
| [Mask R-CNN](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN) |TensorFlow-2  | N/A  | Yes  | Yes  | -  |  -  |-  |   -  | -  |
| [ResNet50 v1.5](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5) | MXNet  | Yes  | Yes  | Yes  | -  | -  |   -  | -  | -  |
| [HMM](https://github.com/NVIDIA/DeepLearningExamples/tree/master/Kaldi/SpeechRecognition) | Kaldi  | N/A  | -  | Yes  | -  | -  |   -  | Yes  | -  |

## NVIDIA support
In each of the network READMEs, we indicate the level of support that will be provided. The range is from ongoing updates and improvements to a point-in-time release for thought leadership.

## Feedback / Contributions
We're posting these examples on GitHub to better support the community, facilitate feedback, as well as collect and implement contributions using GitHub Issues and pull requests. We welcome all contributions!

## Known issues
In each of the network READMEs, we indicate any known issues and encourage the community to provide feedback.





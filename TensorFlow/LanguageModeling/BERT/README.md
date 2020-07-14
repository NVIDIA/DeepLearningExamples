# BERT For TensorFlow

This repository provides a script and recipe to train the BERT model for TensorFlow to achieve state-of-the-art accuracy, and is tested and maintained by NVIDIA.

## Table Of Contents

- [Model overview](#model-overview)
  * [Model architecture](#model-architecture)
  * [Default configuration](#default-configuration)
  * [Feature support matrix](#feature-support-matrix)
    * [Features](#features)
  * [Mixed precision training](#mixed-precision-training)
    * [Enabling mixed precision](#enabling-mixed-precision)
    * [Enabling TF32](#enabling-tf32)
    * [Glossary](#glossary)
- [Setup](#setup)
  * [Requirements](#requirements)
- [Quick Start Guide](#quick-start-guide)
- [Advanced](#advanced)
  * [Scripts and sample code](#scripts-and-sample-code)
  * [Parameters](#parameters)
  * [Command-line options](#command-line-options)
  * [Getting the data](#getting-the-data)
    * [Dataset guidelines](#dataset-guidelines)
    * [Multi-dataset](#multi-dataset)
  * [Training process](#training-process)
    * [Pre-training](#pre-training)
    * [Fine tuning](#fine-tuning)
    * [Multi-node](#multi-node)
  * [Inference process](#inference-process)
  * [Inference Process With TensorRT](#inference-process-with-tensorrt)
  * [Deploying the BERT model using Triton Inference Server](#deploying-the-bert-model-using-triton-inference-server)
  * [BioBERT](#biobert)
- [Performance](#performance)
  * [Benchmarking](#benchmarking)
    * [Training performance benchmark](#training-performance-benchmark)
    * [Inference performance benchmark](#inference-performance-benchmark)
  * [Results](#results)
    * [Training accuracy results](#training-accuracy-results)
      * [Pre-training accuracy](#pre-training-accuracy)
      * [Fine-tuning accuracy for SQuAD v1.1: NVIDIA DGX A100 (8x A100 40GB)](#fine-tuning-accuracy-for-squad-v1.1-nvidia-dgx-a100-8x-a100-40GB)
      * [Fine-tuning accuracy for GLUE MRPC: NVIDIA DGX A100 (8x A100 40GB)](#fine-tuning-accuracy-for-glue-mrpc-nvidia-dgx-a100-8x-a100-40GB)
      * [Training stability test](#training-stability-test)
        * [Pre-training SQuAD v1.1 stability test: NVIDIA DGX A100 (256x A100 40GB)](#pre-training-squad-v1.1-stability-test-nvidia-dgx-a100-256x-a100-40GB)
        * [Fine-tuning SQuAD v1.1 stability test: NVIDIA DGX A100 (8x A100 40GB)](#fine-tuning-squad-v1.1-stability-test-nvidia-dgx-a100-8x-a100-40GB)
        * [Fine-tuning GLUE MRPC stability test: NVIDIA DGX A100 (8x A100 40GB)](#fine-tuning-glue-mrpc-stability-test-nvidia-dgx-a100-8x-a100-40GB)
    * [Training performance results](#training-performance-results)
      * [Training performance: NVIDIA DGX-1 (8x V100 16GB)](#training-performance-nvidia-dgx-1-8x-v100-16GB)
        * [Pre-training training performance: single-node on DGX-1 16GB](#pre-training-training-performance-single-node-on-dgx-1-16GB)
        * [Fine-tuning training performance for SQuAD v1.1 on DGX-1 16GB](#fine-tuning-training-performance-for-squad-v1.1-on-dgx-1-16GB)
      * [Training performance: NVIDIA DGX-1 (8x V100 32GB)](#training-performance-nvidia-dgx-1-8x-v100-32GB)
        * [Pre-training training performance: single-node on DGX-1 32GB](#pre-training-training-performance-single-node-on-dgx-1-32GB)
        * [Fine-tuning training performance for SQuAD v1.1 on DGX-1 32GB](#fine-tuning-training-performance-for-squad-v1.1-on-dgx-1-32GB)
      * [Training performance: NVIDIA DGX-2 (16x V100 32GB)](#training-performance-nvidia-dgx-2-16x-v100-32GB)
        * [Pre-training training performance: single-node on DGX-2 32GB](#pre-training-training-performance-single-node-on-dgx-2-32GB)
        * [Pre-training training performance: multi-node on DGX-2 32GB](#pre-training-training-performance-multi-node-on-dgx-2-32GB)
        * [Fine-tuning training performance for SQuAD v1.1 on DGX-2 32GB](#fine-tuning-training-performance-for-squad-v1.1-on-dgx-2-32GB)
      * [Training performance: NVIDIA DGX A100 (8x A100 40GB)](#training-performance-nvidia-dgx-a100-8x-a100-40gb)
        * [Pre-training training performance: single-node on DGX A100 40GB](#pre-training-training-performance-single-node-on-dgx-a100-40gb)
        * [Pre-training training performance: multi-node on DGX A100 40GB](#pre-training-training-performance-multi-node-on-dgx-a100-40gb)
        * [Fine-tuning training performance for SQuAD v1.1 on DGX A100 40GB](#fine-tuning-training-performance-for-squad-v1.1-on-dgx-a100-40gb)
    * [Inference performance results](#inference-performance-results)
      * [Inference performance: NVIDIA DGX-1 (1x V100 16GB)](#inference-performance-nvidia-dgx-1-1x-v100-16GB)
        * [Fine-tuning inference performance for SQuAD v1.1 on 16GB](#fine-tuning-inference-performance-for-squad-v1.1-on-16GB)
      * [Inference performance: NVIDIA DGX-1 (1x V100 32GB)](#inference-performance-nvidia-dgx-1-1x-v100-32GB)
        * [Fine-tuning inference performance for SQuAD v1.1 on 32GB](#fine-tuning-inference-performance-for-squad-v1.1-on-32GB)
      * [Inference performance: NVIDIA DGX-2 (1x V100 32GB)](#inference-performance-nvidia-dgx-2-1x-v100-32GB)
        * [Fine-tuning inference performance for SQuAD v1.1 on DGX-2  32GB](#fine-tuning-inference-performance-for-squad-v1.1-on-dgx-2-32GB)
      * [Inference performance: NVIDIA DGX A100 (1x A100 40GB)](#inference-performance-nvidia-dgx-a100-1x-a100-40gb)
        * [Fine-tuning inference performance for SQuAD v1.1 on DGX A100 (1x A100 40GB)](#fine-tuning-inference-performance-for-squad-v1.1-on-dgx-a100-40gb)
      * [Inference performance: NVIDIA Tesla T4 (1x T4 16GB)](#inference-performance-nvidia-tesla-t4-1x-t4-16GB)
        * [Fine-tuning inference performance for SQuAD v1.1 on Tesla T4 16GB](#fine-tuning-inference-performance-for-squad-v1.1-on-tesla-t4-16GB)
- [Release notes](#release-notes)
  * [Changelog](#changelog)
  * [Known issues](#known-issues)




## Model overview

BERT, or Bidirectional Encoder Representations from Transformers, is a new method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks. This model is based on the [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) paper. NVIDIA's BERT is an optimized version of [Google's official implementation](https://github.com/google-research/bert), leveraging mixed precision arithmetic and Tensor Cores on A100, V100 and T4 GPUs for faster training times while maintaining target accuracy.

Other publicly available implementations of BERT include:
1. [NVIDIA PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)
2. [Hugging Face](https://github.com/huggingface/pytorch-pretrained-BERT)
3. [codertimo](https://github.com/codertimo/BERT-pytorch)
4. [gluon-nlp](https://github.com/dmlc/gluon-nlp/tree/master/scripts/bert)
5. [Google's official implementation](https://github.com/google-research/bert)

This model is trained with mixed precision using Tensor Cores on NVIDIA Volta, Ampere and Turing GPUs. Therefore, researchers can get results up to 4x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.

### Model architecture

BERT's model architecture is a multi-layer bidirectional Transformer encoder. Based on the model size, we have the following two default configurations of BERT:

| **Model** | **Hidden layers** | **Hidden unit size** | **Attention heads** | **Feedforward filter size** | **Max sequence length** | **Parameters** |
|:---------:|:----------:|:----:|:---:|:--------:|:---:|:----:|
|BERTBASE |12 encoder| 768| 12|4 x  768|512|110M|
|BERTLARGE|24 encoder|1024| 16|4 x 1024|512|330M|

BERT training consists of two steps, pre-training the language model in an unsupervised fashion on vast amounts of unannotated datasets, and then using this pre-trained model for fine-tuning for various NLP tasks, such as question and answer, sentence classification, or sentiment analysis. Fine-tuning typically adds an extra layer or two for the specific task and further trains the model using a task-specific annotated dataset, starting from the pre-trained backbone weights. The end-to-end process in depicted in the following image:

![](data/images/bert_pipeline.png?raw=true)

Figure 1: BERT Pipeline

### Default configuration

This repository contains scripts to interactively launch data download, training, benchmarking and inference routines in a Docker container for both pre-training and fine tuning for Question Answering. The major differences between the official implementation of the paper and our version of BERT are as follows:

- Mixed precision support with TensorFlow Automatic Mixed Precision (TF-AMP), which enables mixed precision training without any changes to the code-base by performing automatic graph rewrites and loss scaling controlled by an environmental variable.
- Scripts to download dataset for:
    - Pre-training - [Wikipedia](https://dumps.wikimedia.org/),  [BookCorpus](http://yknzhu.wixsite.com/mbweb)
    - Fine tuning - [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (Stanford Question Answering Dataset)
    - Fine tuning - [GLUE](https://gluebenchmark.com/) (The General Language Understanding Evaluation benchmark)
    - Pretrained weights from Google
- Custom fused CUDA kernels for faster computations
- Multi-GPU/Multi-node support using Horovod

The following performance optimizations were implemented in this model:
- [XLA](https://www.tensorflow.org/xla) support (experimental).

These techniques and optimizations improve model performance and reduce training time, allowing you to perform various NLP tasks with no additional effort.


### Feature support matrix

The following features are supported by this model.

| **Feature**               | **BERT** |
|:-----------------------:|:--------------------------:|
| Horovod Multi-GPU      | Yes |
| Horovod Multi-Node     | Yes |
| Automatic mixed precision (AMP)      | Yes |
| LAMB        | Yes |

#### Features

Multi-GPU training with Horovod - Our model uses Horovod to implement efficient multi-GPU training with NCCL. For details, see example sources in this repository or see the [TensorFlow tutorial](https://github.com/horovod/horovod/#usage)

[LAMB](https://arxiv.org/pdf/1904.00962.pdf) stands for Layerwise Adaptive Moments based optimizer, is a large batch optimization technique that helps accelerate training of deep neural networks using large minibatches. It allows using a global batch size of 65536 and 32768 on sequence lengths 128 and 512 respectively, compared to a batch size of 256 for Adam. The optimized implementation accumulates 1024 gradient batches in phase 1 and 4096 steps in phase 2 before updating weights once. This results in 27% training speedup on a single DGX2 node. On multi-node systems, LAMB allows scaling up to 1024 GPUs resulting in training speedups of up to 17x in comparison to [Adam](https://arxiv.org/pdf/1412.6980.pdf). Adam has limitations on the learning rate that can be used since it is applied globally on all parameters whereas LAMB follows a layerwise learning rate strategy.

NVLAMB adds necessary tweaks to [LAMB version 1](https://arxiv.org/abs/1904.00962v1), to ensure correct convergence. A guide to implementating the LAMB optimizer can be found in our [article](https://medium.com/@NvidiaAI/a-guide-to-optimizer-implementation-for-bert-at-scale-8338cc7f45fd) on Medium.com. The algorithm is as follows:
  ![NVLAMB](data/images/images_nvlamb.png)

### Mixed precision training

Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in Volta, and following with both the Turing and Ampere architectures, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using [mixed precision training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) previously required two steps:
1.  Porting the model to use the FP16 data type where appropriate.    
2.  Adding loss scaling to preserve small gradient values.

This can now be achieved using Automatic Mixed Precision (AMP) for TensorFlow to enable the full [mixed precision methodology](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#tensorflow) in your existing TensorFlow model code.  AMP enables mixed precision training on Volta, Turing, and NVIDIA Ampere GPU architectures automatically. The TensorFlow framework code makes all necessary model changes internally.

In TF-AMP, the computational graph is optimized to use as few casts as necessary and maximize the use of FP16, and the loss scaling is automatically applied inside of supported optimizers. AMP can be configured to work with the existing tf.contrib loss scaling manager by disabling the AMP scaling with a single environment variable to perform only the automatic mixed-precision optimization. It accomplishes this by automatically rewriting all computation graphs with the necessary operations to enable mixed precision training and automatic loss scaling.

For information about:
-   How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) documentation.
-   Techniques used for mixed precision training, see the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
-   How to access and enable AMP for TensorFlow, see [Using TF-AMP](https://docs.nvidia.com/deeplearning/dgx/tensorflow-user-guide/index.html#tfamp) from the TensorFlow User Guide.
-   APEX tools for mixed precision training, see the [NVIDIA Apex: Tools for Easy Mixed-Precision Training in PyTorch](https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/).


#### Enabling mixed precision

Mixed precision is enabled in TensorFlow by using the Automatic Mixed Precision (TF-AMP) extension which casts variables to half-precision upon retrieval, while storing variables in single-precision format. Furthermore, to preserve small gradient magnitudes in backpropagation, a [loss scaling](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#lossscaling) step must be included when applying gradients. In TensorFlow, loss scaling can be applied statically by using simple multiplication of loss by a constant value or automatically, by TF-AMP. Automatic mixed precision makes all the adjustments internally in TensorFlow, providing two benefits over manual operations. First, programmers need not modify network model code, reducing development and maintenance effort. Second, using AMP maintains forward and backward compatibility with all the APIs for defining and running TensorFlow models.

To enable mixed precision, you can simply add the values to the environmental variables inside your training script:
- Enable TF-AMP graph rewrite:
  ```
  os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
  ```
  
- Enable Automated Mixed Precision:
  ```
  os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

#### Enabling TF32

TensorFloat-32 (TF32) is the new math mode in [NVIDIA A100](#https://www.nvidia.com/en-us/data-center/a100/) GPUs for handling the matrix math also called tensor operations. TF32 running on Tensor Cores in A100 GPUs can provide up to 10x speedups compared to single-precision floating-point math (FP32) on Volta GPUs. 

TF32 Tensor Cores can speed up networks using FP32, typically with no loss of accuracy. It is more robust than FP16 for models which require high dynamic range for weights or activations.

For more information, refer to the [TensorFloat-32 in the A100 GPU Accelerates AI Training, HPC up to 20x](#https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) blog post.

TF32 is supported in the NVIDIA Ampere GPU architecture and is enabled by default.

### Glossary

**Fine-tuning**
Training an already pretrained model further using a task specific dataset for subject-specific refinements, by adding task-specific layers on top if required.

**Language Model**
Assigns a probability distribution over a sequence of words. Given a sequence of words, it assigns a probability to the whole sequence.

**Pre-training**
Training a model on vast amounts of data on the same (or different) task to build general understandings.

**Transformer**
The paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) introduces a novel architecture called Transformer that uses an attention mechanism and transforms one sequence into another.


## Setup

The following section lists the requirements in order to start training the BERT model.


### Requirements

This repository contains `Dockerfile` which extends the TensorFlow NGC container and encapsulates some dependencies.  Aside from these dependencies, ensure you have the following components:
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [TensorFlow 20.06-py3+](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) NGC container
-   GPU-based architecture:
    - [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
    - [NVIDIA Turing](https://www.nvidia.com/en-us/geforce/turing/)
    - [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)

For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
- [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
- [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)
- [Running TensorFlow](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html#running)

For those unable to use the TensorFlow NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

For multi-node, the sample provided in this repository requires [Enroot](https://github.com/NVIDIA/enroot) and [Pyxis](https://github.com/NVIDIA/pyxis) set up on a [SLURM](https://slurm.schedmd.com) cluster.

More information on how to set up and launch can be found in the [Multi-node Documentation](https://docs.nvidia.com/ngc/multi-node-bert-user-guide).


## Quick Start Guide

To pretrain or fine tune your model for Question Answering using mixed precision with Tensor Cores or using FP32/TF32, perform the following steps using the default parameters of the BERT model.

1. Clone the repository.

```bash
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/TensorFlow/LanguageModeling/BERT
```

2. Build the BERT TensorFlow NGC container.

```bash
bash scripts/docker/build.sh
```

3. Download and preprocess the dataset.

This repository provides scripts to download, verify and extract the SQuAD dataset, GLUE dataset and pretrained weights for fine tuning as well as Wikipedia and BookCorpus dataset for pre-training.

To download, verify, and extract the required datasets, run:

```bash
bash scripts/data_download.sh
```

The script launches a Docker container with the current directory mounted and downloads the datasets to a `data/` folder on the host.

Note: For fine tuning only, Wikipedia and Bookscorpus dataset download and preprocessing can be skipped by commenting it out.

- Download Wikipedia only for pretraining

The pretraining dataset is 170GB+ and takes 15+ hours to download. The BookCorpus server most of the times get overloaded and also contain broken links resulting in HTTP 403 and 503 errors. Hence, it is recommended to skip downloading BookCorpus data by running:

`bash scripts/data_download.sh wiki_only`

- Download Wikipedia and BookCorpus

Users are welcome to download BookCorpus from other sources to match our accuracy, or repeatedly try our script until the required number of files are downloaded by running the following:
`bash scripts/data_download.sh wiki_books`

Note: Not using BookCorpus can potentially change final accuracy on a few downstream tasks.

4. Download the pretrained models from NGC.

We have uploaded checkpoints that have been [fine tuned](https://ngc.nvidia.com/catalog/models/nvidia:bert_tf_v1_1_large_fp32_384) and [pre-trained](https://ngc.nvidia.com/catalog/models/nvidia:bert_tf_pretraining_lamb_16n) for various configurations on the NGC Model Registry. You can browse and download the relevant checkpoints directly from the [NGC model catalog](https://ngc.nvidia.com/catalog/models). Download them to the `results/models/` to easily access them in your scripts. 

5. Start an interactive session in the NGC container to run training/inference.

After you build the container image and download the data, you can start an interactive CLI session as follows:

```bash
bash scripts/docker/launch.sh
```

The `launch.sh` script assumes that the datasets are in the following locations by default after downloading the data.

- SQuAD v1.1 - `data/download/squad/v1.1`
- SQuAD v2.0 - `data/download/squad/v2.0`
- GLUE The Corpus of Linguistic Acceptability (CoLA) - `data/download/CoLA`
- GLUE Microsoft Research Paraphrase Corpus (MRPC) - `data/download/MRPC`
- GLUE The Multi-Genre NLI Corpus (MNLI) - `data/download/MNLI`
- BERT Large - `data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16`
- BERT Base - `data/download/google_pretrained_weights/uncased_L-12_H-768_A-12`
- BERT - `data/download/google_pretrained_weights/`
- Wikipedia + BookCorpus TFRecords - `data/tfrecords<config>/books_wiki_en_corpus`

6. Start pre-training.

BERT is designed to pre-train deep bidirectional representations for language representations. The following scripts are to replicate pre-training on Wikipedia and BookCorpus from the [LAMB paper](https://arxiv.org/pdf/1904.00962.pdf). These scripts are general and can be used for pre-training language representations on any corpus of choice.

From within the container, you can use the following script to run pre-training using LAMB.
```bash
bash scripts/run_pretraining_lamb.sh <train_batch_size_phase1> <train_batch_size_phase2> <eval_batch_size> <learning_rate_phase1> <learning_rate_phase2> <precision> <use_xla> <num_gpus> <warmup_steps_phase1> <warmup_steps_phase2> <train_steps> <save_checkpoint_steps> <num_accumulation_phase1> <num_accumulation_steps_phase2> <bert_model>
```

For BERT Large FP16 training with XLA using a DGX-1 V100 32GB, run:
```bash
bash scripts/run_pretraining_lamb.sh 64 8 8 7.5e-4 5e-4 fp16 true 8 2000 200 7820 100 128 512 large
```

This repository also contains a number of predefined configurations to run the Lamb pretraining on NVIDIA DGX-1, NVIDIA DGX-2H or NVIDIA DGX A100 nodes in `scripts/configs/pretrain_config.sh`. For example, to use the default DGX A100 8 gpu config, run:

```bash
bash scripts/run_pretraining_lamb.sh $(source scripts/configs/pretrain_config.sh && dgxa100_8gpu_fp16)
```

Alternatively, to run pre-training with Adam as in the original [BERT paper](https://arxiv.org/pdf/1810.04805.pdf) from within the container, run:

```bash
bash scripts/run_pretraining_adam.sh <train_batch_size_per_gpu> <eval_batch_size> <learning_rate_per_gpu> <precision> <use_xla> <num_gpus> <warmup_steps> <train_steps> <save_checkpoint_steps>
```

7. Start fine tuning.

The above pretrained BERT representations can be fine tuned with just one additional output layer for a state-of-the-art Question Answering system. From within the container, you can use the following script to run fine-training for SQuAD.

```bash
bash scripts/run_squad.sh <batch_size_per_gpu> <learning_rate_per_gpu> <precision> <use_xla> <num_gpus> <seq_length> <doc_stride> <bert_model> <squad_version> <checkpoint> <epochs>
```

For SQuAD 1.1 FP16 training with XLA using a DGX A100 40GB, run:
```bash
bash scripts/run_squad.sh 32 5e-6 fp16 true 8 384 128 large 1.1 data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/bert_model.ckpt 2.0
```

This repository contains a number of predefined configurations to run the SQuAD fine tuning on NVIDIA DGX-1, NVIDIA DGX-2H or NVIDIA DGX A100 nodes in `scripts/configs/squad_config.sh`. For example, to use the default DGX A100 8 gpu config, run:

```bash
bash scripts/run_squad.sh $(source scripts/configs/squad_config.sh && dgxa100_8gpu_fp16) 1.1 data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/bert_model.ckpt 2.0
```

Alternatively, to run fine tuning on GLUE benchmark, run:

```bash
bash scripts/run_glue.sh <task_name> <batch_size_per_gpu> <learning_rate_per_gpu> <precision> <use_xla> <num_gpus> <seq_length> <doc_stride> <bert_model> <epochs> <warmup_proportion> <checkpoint>
```
For MRPC FP16 training with XLA using a DGX A100 40GB, run:

```bash
bash scripts/run_glue.sh MRPC 16 3e-6 true 8 128 64 large 3 0.1
```

The GLUE tasks supported include CoLA, MRPC and MNLI.

8. Start validation/evaluation.

The `run_squad_inference.sh` script runs inference on a checkpoint fine tuned for SQuAD and evaluates the validity of predictions on the basis of exact match and F1 score.

```bash
bash scripts/run_squad_inference.sh <init_checkpoint> <batch_size> <precision> <use_xla> <seq_length> <doc_stride> <bert_model> <squad_version>
```

For SQuAD 2.0 FP16 inference with XLA using a DGX-1 V100 32GB using checkpoint at `/results/model.ckpt` , run:
```bash
bash scripts/run_squad_inference.sh /results/model.ckpt 8 fp16 true 384 128 large 2.0
```

For SQuAD 1.1 FP32 inference without XLA using a DGX A100 40GB using checkpoint at `/results/model.ckpt`, run:
```bash
bash scripts/run_squad_inference.sh /results/model.ckpt 8 fp32 false 384 128 large 1.1
```

Alternatively, to run inference on GLUE benchmark, run:
```bash
bash scripts/run_glue_inference.sh <task_name> <init_checkpoint> <batch_size_per_gpu> <precision> <use_xla> <seq_length> <doc_stride> <bert_model>
```

## Advanced

The following sections provide greater details of the dataset, running training and inference, and the training results.

### Scripts and sample code

In the root directory, the most important files are:
* `run_pretraining.py` - Serves as entry point for pre-training
* `run_squad.py` - Serves as entry point for SQuAD training
* `run_classifier.py` - Serves as entry point for GLUE training
* `Dockerfile` - Container with the basic set of dependencies to run BERT

The `scripts/` folder encapsulates all the one-click scripts required for running various functionalities supported such as:
* `run_squad.sh` - Runs SQuAD training and inference using `run_squad.py` file
* `run_glue.sh` - Runs GLUE training and inference using the `run_classifier.py` file
* `run_pretraining_adam.sh` - Runs pre-training with Adam optimizer using the `run_pretraining.py` file
* `run_pretraining_lamb.sh` - Runs pre-training with LAMB optimizer using the `run_pretraining.py` file in two phases. Phase 1 does 90% of training with sequence length = 128. In phase 2, the remaining 10% of the training is done with sequence length = 512.
* `data_download.sh` - Downloads datasets using files in the `data/` folder
* `finetune_train_benchmark.sh` - Captures performance metrics of training for multiple configurations
* `finetune_inference_benchmark.sh` - Captures performance metrics of inference for multiple configurations

Other folders included in the root directory are:
* `data/` - Necessary folders and scripts to download datasets required for fine tuning and pre-training BERT.
* `utils/` - Necessary files for preprocessing data before feeding into BERT and hooks for obtaining performance metrics from BERT.

### Parameters

Aside from the options to set hyperparameters, the relevant options to control the behaviour of the `run_pretraining.py` script are:

```
  --bert_config_file: The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.
  --init_checkpoint: Initial checkpoint (usually from a pre-trained BERT model).
  --[no]do_eval: Whether to run evaluation on the dev set.(default: 'false')
  --[no]do_train: Whether to run training.(evaluation: 'false')
  --eval_batch_size: Total batch size for eval.(default: '8')(an integer)
  --[no]horovod: Whether to use Horovod for multi-gpu runs(default: 'false')
  --[no]amp: Whether to enable AMP ops. When false, uses TF32 on A100 and FP32 on V100 GPUS.(default: 'True')
  --[no]use_xla: Whether to enable XLA JIT compilation.(default: 'True')
  --input_files_dir: Input TF example files (can be a dir or comma separated).
  --output_dir: The output directory where the model checkpoints will be    written.
  --optimizer_type: Optimizer used for training - LAMB or ADAM
  --num_accumulation_steps: Number of accumulation steps before gradient update. Global batch size = num_accumulation_steps * train_batch_size
  --allreduce_post_accumulation: Whether to all reduce after accumulation of N steps or after each step
```

Aside from the options to set hyperparameters, some relevant options to control the behaviour of the `run_squad.py` script are:

```
  --bert_config_file: The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.
  --output_dir: The output directory where the model checkpoints will be written.
  --[no]do_predict: Whether to run evaluation on the dev set. (default: 'false')
  --[no]do_train: Whether to run training. (default: 'false')
  --learning_rate: The initial learning rate for Adam.(default: '5e-06')(a number)
  --max_answer_length: The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.(default: '30')(an integer)
  --max_query_length: The maximum number of tokens for the question. Questions longer than this will be truncated to this length.(default: '64')(an integer)
  --max_seq_length: The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.(default: '384')(an integer)
  --predict_batch_size: Total batch size for predictions.(default: '8')(an integer)
  --train_batch_size: Total batch size for training.(default: '8')(an integer)
  --[no]amp: Whether to enable AMP ops. When false, uses TF32 on A100 and FP32 on V100 GPUS.(default: 'True')
  --[no]use_xla: Whether to enable XLA JIT compilation.(default: 'True')
  --[no]version_2_with_negative: If true, the SQuAD examples contain some that do not have an answer.(default: 'false')
```

Aside from the options to set hyperparameters, some relevant options to control the behaviour of the `run_classifier.py` script are:

```
  --bert_config_file: The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.
  --data_dir: The input data dir. Should contain the .tsv files (or other data files) for the task.
  --[no]do_eval: Whether to run eval on the dev set.
    (default: 'false')
  --[no]do_predict: Whether to run the model in inference mode on the test set.(default: 'false')
  --[no]do_train: Whether to run training.(default: 'false')
  --[no]horovod: Whether to use Horovod for multi-gpu runs(default: 'false')
  --init_checkpoint: Initial checkpoint (usually from a pre-trained BERT model).
  --max_seq_length: The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.(default: '128')(an integer)
  --num_train_epochs: Total number of training epochs to perform.(default: '3.0')(a number)
  --output_dir: The output directory where the model checkpoints will be written.
  --task_name: The name of the task to train.
  --train_batch_size: Total batch size for training.(default: '32')(an integer)
  --[no]amp: Whether to enable AMP ops. When false, uses TF32 on A100 and FP32 on V100 GPUS.(default: 'True')
  --[no]use_xla: Whether to enable XLA JIT compilation.(default: 'True')
  --vocab_file: The vocabulary file that the BERT model was trained on.
  --warmup_proportion: Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.(default: '0.1')(a number)
```

Note: When initializing from a checkpoint using `--init_checkpoint` and a corpus of your choice, keep in mind that `bert_config_file` and `vocab_file` should remain unchanged.

### Command-line options

To see the full list of available options and their descriptions, use the `-h` or `--help` command-line option with the Python file, for example:

```bash
python run_pretraining.py --help
python run_squad.py --help
python run_classifier.py --help
```

### Getting the data

For pre-training BERT, we use the concatenation of Wikipedia (2500M words) as well as BookCorpus (800M words). For Wikipedia, we extract only the text passages from [here](ftp://ftpmirror.your.org/pub/wikimedia/dumps/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2) and ignore headers list and tables. It is structured as a document level corpus rather than a shuffled sentence level corpus because it is critical to extract long contiguous sentences.

The next step is to run `create_pretraining_data.py` with the document level corpus as input, which generates input data and labels for the masked language modeling and next sentence prediction tasks. Pre-training can also be performed on any corpus of your choice. The collection of data generation scripts are intended to be modular to allow modifications for additional preprocessing steps or to use additional data. They can hence easily be modified for an arbitrary corpus.

The preparation of an individual pre-training dataset is described in the `create_datasets_from_start.sh` script found in the `data/` folder. The component steps to prepare the datasets are as follows:

1.  Data download and extract - the dataset is downloaded and extracted.
2.  Clean and format - document tags, etc. are removed from the dataset. The end result of this step is a `{dataset_name_one_article_per_line}.txt` file that contains the entire corpus. Each line in the text file contains an entire document from the corpus. One file per dataset is created in the `formatted_one_article_per_line` folder.
3.  Sharding - the sentence segmented corpus file is split into a number of smaller text documents. The sharding is configured so that a document will not be split between two shards. Sentence segmentation is performed at this time using NLTK.
4.  TFRecord file creation - each text file shard is processed by the `create_pretraining_data.py` script to produce a corresponding TFRecord file. The script generates input data and labels for masked language modeling and sentence prediction tasks for the input text shard.


For fine tuning BERT for the task of Question Answering, we use SQuAD and GLUE. SQuAD v1.1 has 100,000+ question-answer pairs on 500+ articles. SQuAD v2.0 combines v1.1 with an additional 50,000 new unanswerable questions and must not only answer questions but also determine when that is not possible. GLUE consists of single-sentence tasks, similarity and paraphrase tasks and inference tasks. We support one of each: CoLA, MNLI and MRPC.

#### Dataset guidelines

The procedure to prepare a text corpus for pre-training is described in the previous section. This section provides additional insight into how exactly raw text is processed so that it is ready for pre-training.

First, raw text is tokenized using [WordPiece tokenization](https://arxiv.org/pdf/1609.08144.pdf). A [CLS] token is inserted at the start of every sequence, and the two sentences in the sequence are separated by a [SEP] token.

Note: BERT pre-training looks at pairs of sentences at a time. A sentence embedding token [A] is added to the first sentence and token [B] to the next.

BERT pre-training optimizes for two unsupervised classification tasks. The first is Masked Language Modelling (Masked LM). One training instance of Masked LM is a single modified sentence. Each token in the sentence has a 15% chance of being replaced by a [MASK] token. The chosen token is replaced with [MASK] 80% of the time, 10% with another random token and the remaining 10% with the same token. The task is then to predict the original token.

The second task is next sentence prediction. One training instance of BERT pre-training is two sentences (a sentence pair). A sentence pair may be constructed by simply taking two adjacent sentences from a single document, or by pairing up two random sentences with equal probability. The goal of this task is to predict whether or not the second sentence followed the first in the original document.

The `create_pretraining_data.py` script takes in raw text and creates training instances for both pre-training tasks.

#### Multi-dataset

We are able to combine multiple datasets into a single dataset for pre-training on a diverse text corpus. Once TFRecords have been created for each component dataset, you can create a combined dataset by adding the directory to `SOURCES` in `run_pretraining_*.sh`. This will feed all matching files to the input pipeline in `run_pretraining.py`. However, in the training process, only one TFRecord file is consumed at a time, therefore, the training instances of any given training batch will all belong to the same source dataset.

### Training process

The training process consists of two steps: pre-training and fine tuning.

#### Pre-training

Pre-training is performed using the `run_pretraining.py` script along with parameters defined in the `scripts/run_pretraining_lamb.sh`.

The `run_pretraining_lamb.sh` script runs a job on a single node that trains the BERT-large model from scratch using the Wikipedia and BookCorpus datasets as training data. By default, the training script:
- Runs on 8 GPUs.
- Has FP16 precision enabled.
- Is XLA enabled.
- Creates a log file containing all the output.
- Saves a checkpoint every 100 iterations (keeps only the latest checkpoint) and at the end of training. All checkpoints, evaluation results and training logs are saved to the `/results` directory (in the container which can be mounted to a local directory).
- Evaluates the model at the end of each phase.

- Phase 1
    - Runs 7038 steps with 2000 warmup steps
    - Sets Maximum sequence length as 128
    - Sets Global Batch size as 64K

- Phase 2
    - Runs 1564 steps with 200 warm-up steps
    - Sets Maximum sequence length as 512
    - Sets Global Batch size as 32K
    - Starts from Phase1's final checkpoint

These parameters train Wikipedia and BookCorpus with reasonable accuracy on a DGX-1 with 32GB V100 cards.

For example:
```bash
scripts/run_pretraining_lamb.sh <train_batch_size_phase1> <train_batch_size_phase2> <eval_batch_size> <learning_rate_phase1> <learning_rate_phase2> <precision> <use_xla> <num_gpus> <warmup_steps_phase1> <warmup_steps_phase2> <train_steps> <save_checkpoint_steps> <num_accumulation_phase1> <num_accumulation_steps_phase2> <bert_model>
```

Where:
- `<training_batch_size_phase*>` is per-GPU batch size used for training in the respective phase. Batch size varies with precision, larger batch sizes run more efficiently, but require more memory.

- `<eval_batch_size>` is per-GPU batch size used for evaluation after training.

- `<learning_rate_phase1>` is the default rate of 1e-4 is good for global batch size 256.

- `<learning_rate_phase2>` is the default rate of 1e-4 is good for global batch size 256.

- `<precision>` is the type of math in your model, can be either `fp32` or `fp16`. Specifically:

    - `fp32` is 32-bit IEEE single precision floats. Is enabled by default on V100.
    - `fp16` is Automatic rewrite of TensorFlow compute graph to take advantage of 16-bit arithmetic whenever it is safe.
    - `tf32` uses same 10 bit mantissa as fp16 and 8 bit exponent as fp32. Is enabled by default on A100.


- `<num_gpus>` is the number of GPUs to use for training. Must be equal to or smaller than the number of GPUs attached to your node.

- `<warmup_steps_phase*>` is the number of warm-up steps at the start of training in the respective phase.

- `<training_steps>` is the total number of training steps in both phases combined.

- `<save_checkpoint_steps>` controls how often checkpoints are saved. Default is 100 steps.

- `<num_accumulation_phase*>` is used to mimic higher batch sizes in the respective phase by accumulating gradients N times before weight update.

- `<bert_model>` is used to indicate whether to pretrain BERT Large or BERT Base model

The following sample code trains BERT-large from scratch on a single DGX-2 using FP16 arithmetic. This will take around 4.5 days.

```bash
bert_tf/scripts/run_pretraining_lamb.sh 32 8 8 3.75e-4 2.5e-4 fp16 true 16 2000 200 7820 100 128 256 large
```

#### Fine tuning

Fine tuning is performed using the `run_squad.py` script along with parameters defined in `scripts/run_squad.sh`.

The `run_squad.sh` script trains a model and performs evaluation on the SQuAD dataset. By default, the training script:

- Trains for SQuAD v1.1 dataset.
- Trains on BERT Large Model.
- Uses 8 GPUs and batch size of 10 on each GPU.
- Has FP16 precision enabled.
- Is XLA enabled.
- Runs for 2 epochs.
- Saves a checkpoint every 1000 iterations (keeps only the latest checkpoint) and at the end of training. All checkpoints, evaluation results and training logs are saved to the `/results` directory (in the container which can be mounted to a local directory).
- Evaluation is done at the end of training. To skip evaluation, modify `--do_predict` to `False`.

This script outputs checkpoints to the `/results` directory, by default, inside the container. Mount point of `/results` can be changed in the `scripts/docker/launch.sh` file. The training log contains information about:
- Loss for the final step
- Training and evaluation performance
- F1 and exact match score on the Dev Set of SQuAD after evaluation.

The summary after training is printed in the following format:
```bash
I0312 23:10:45.137036 140287431493376 run_squad.py:1332] 0 Total Training Time = 3007.00 Training Time W/O start up overhead = 2855.92 Sentences processed = 175176
I0312 23:10:45.137243 140287431493376 run_squad.py:1333] 0 Training Performance = 61.3378 sentences/sec
I0312 23:14:00.550846 140287431493376 run_squad.py:1396] 0 Total Inference Time = 145.46 Inference Time W/O start up overhead = 131.86 Sentences processed = 10840
I0312 23:14:00.550973 140287431493376 run_squad.py:1397] 0 Inference Performance = 82.2095 sentences/sec
{"exact_match": 83.69914853358561, "f1": 90.8477003317459}
```

Multi-GPU training is enabled with the Horovod TensorFlow module. The following example runs fine tuning on 8 GPUs:

```bash
BERT_DIR=data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16
SQUAD_DIR=data/download/squad/v1.1
mpi_command="mpirun -np 8 -H localhost:8 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH -mca pml ob1 -mca btl ^openib" \
     python run_squad.py --horovod --vocab_file=$BERT_DIR/vocab.txt \
     --bert_config_file=$BERT_DIR/bert_config.json \
     --output_dir=/results --do_train --train_file=$SQUAD_DIR/train-v1.1.json
```

#### Multi-node


Multi-node runs can be launched on a pyxis/enroot Slurm cluster (see [Requirements](#requirements)) with the `run.sub` script with the following command for a 4-node DGX1 example for both phase 1 and phase 2:
```
BATCHSIZE=16 LEARNING_RATE='1.875e-4' NUM_ACCUMULATION_STEPS=128 PHASE=1 sbatch -N4 --ntasks-per-node=8 run.sub
BATCHSIZE=2 LEARNING_RATE='1.25e-4' NUM_ACCUMULATION_STEPS=512 PHASE=1 sbatch -N4 --ntasks-per-node=8 run.sub
```


Checkpoint after phase 1 will be saved in `checkpointdir` specified in `run.sub`. The checkpoint will be automatically picked up to resume training on phase 2. Note that phase 2 should be run after phase 1.

Variables to re-run the [Training performance results](#training-performance-results) are available in the `scripts/configs/configurations.yml` file.

The batch variables `BATCHSIZE`, `LEARNING_RATE`, `NUM_ACCUMULATION_STEPS` refer to the Python arguments `train_batch_size`, `learning_rate`, `num_accumulation_steps` respectively.
The variable `PHASE` refers to phase specific arguments available in `run.sub`.

Note that the `run.sub` script is a starting point that has to be adapted depending on the environment. In particular, variables such as `datadir` handle the location of the files for each phase.

Refer to the files contents to see the full list of variables to adjust for your system.

### Inference process

Inference on a fine tuned Question Answering system is performed using the `run_squad.py` script along with parameters defined in `scripts/run_squad_inference.sh`. Inference is supported on a single GPU.

The `run_squad_inference.sh` script trains a model and performs evaluation on the SQuAD dataset. By default, the inferencing script:

- Uses SQuAD v1.1 dataset
- Has FP16 precision enabled
- Is XLA enabled
- Evaluates the latest checkpoint present in `/results` with a batch size of 8

This script outputs predictions file to `/results/predictions.json` and computes F1 score and exact match score using SQuAD's evaluate file. Mount point of `/results` can be changed in the `scripts/docker/launch.sh` file.

The output log contains information about:
Inference performance
Inference Accuracy (F1 and exact match scores) on the Dev Set of SQuAD after evaluation.

The summary after inference is printed in the following format:
```bash
I0312 23:14:00.550846 140287431493376 run_squad.py:1396] 0 Total Inference Time = 145.46 Inference Time W/O start up overhead = 131.86 Sentences processed = 10840
I0312 23:14:00.550973 140287431493376 run_squad.py:1397] 0 Inference Performance = 82.2095 sentences/sec
{"exact_match": 83.69914853358561, "f1": 90.8477003317459}
```

### Inference Process With TensorRT
NVIDIA TensorRT is a platform for high-performance deep learning inference. It includes a deep learning inference optimizer and runtime that delivers low latency and high-throughput for deep learning inference applications. More information on how to perform inference using TensorRT can be found in the subfolder [./trt/README.md](trt/README.md)

### Deploying the BERT model using Triton Inference Server

The [NVIDIA Triton Inference Server](https://github.com/NVIDIA/triton-inference-server) provides a datacenter and cloud inferencing solution optimized for NVIDIA GPUs. The server provides an inference service via an HTTP or gRPC endpoint, allowing remote clients to request inferencing for any number of GPU or CPU models being managed by the server. More information on how to perform inference using `Triton Inference Server` can be found in the subfolder `./triton/README.md`.

### BioBERT

Many works, including [BioBERT](https://arxiv.org/pdf/1901.08746.pdf), [SciBERT](https://arxiv.org/pdf/1903.10676.pdf), [NCBI-BERT](https://arxiv.org/pdf/1906.05474.pdf), [ClinicalBERT (MIT)](https://arxiv.org/pdf/1904.03323.pdf), [ClinicalBERT (NYU, Princeton)](https://arxiv.org/pdf/1904.05342.pdf), and others at [BioNLP’19 workshop](https://aclweb.org/aclwiki/BioNLP_Workshop), show that pre-training of BERT on large biomedical text corpus such as [PubMed](https://www.ncbi.nlm.nih.gov/pubmed/) results in better performance in biomedical text-mining tasks.

More information on how to download a biomedical corpus and pre-train as well as finetune for biomedical tasks can be found in the subfolder `./biobert/README.md`.

## Performance

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in training and inference modes.

Both of these benchmarking scripts enable you to run a number of epochs, extract performance numbers, and run the BERT model for fine tuning.

#### Training performance benchmark

Training benchmarking can be performed by running the script:
``` bash
scripts/finetune_train_benchmark.sh <bert_model> <use_xla> <num_gpu> squad
```

This script runs 2 epochs by default on the SQuAD v1.1 dataset and extracts performance numbers for various batch sizes and sequence lengths in both FP16 and FP32/TF32. These numbers are saved at `/results/squad_train_benchmark_bert_<bert_model>_gpu_<num_gpu>.log`.

#### Inference performance benchmark

Inference benchmarking can be performed by running the script:

``` bash
scripts/finetune_inference_benchmark.sh squad
```

This script runs 1024 eval iterations by default on the SQuAD v1.1 dataset and extracts performance and latency numbers for various batch sizes and sequence lengths in both FP16 and FP32/TF32, for base and large models. These numbers are saved at `/results/squad_inference_benchmark_bert_<bert_model>.log`.

### Results

The following sections provide details on how we achieved our performance and accuracy in training and inference for pre-training using LAMB optimizer as well as fine tuning for Question Answering. All results are on BERT-large model unless otherwise mentioned. All fine tuning results are on SQuAD v1.1 using a sequence length of 384 unless otherwise mentioned.

#### Training accuracy results

##### Training accuracy

###### Pre-training accuracy

Our results were obtained by running the `scripts/run_pretraining_lamb.sh` training script in the TensorFlow 20.06-py3 NGC container.

| **DGX System** | **Nodes x GPUs** | **Precision** | **Batch Size/GPU: Phase1, Phase2** | **Accumulation Steps: Phase1, Phase2** | **Time to Train (Hrs)** | **Final Loss** |
|----------------|-----------|---------------|------------------------------------|----------------------------------------|----------------|-------------------------|
| DGX2H | 32 x 16 | FP16 | 64, 8 | 2, 8   | 2.63  | 1.59 |
| DGX2H | 32 x 16 | FP32 | 32, 8 | 4, 8   | 8.48  | 1.56 |
| DGXA100 | 32 x 8 | FP16 | 64, 16 | 4, 8   | 3.24  | 1.56 |
| DGXA100 | 32 x 8 | TF32 | 64, 8 | 4, 16   | 4.58  | 1.58 |

Note: Time to train includes upto 16 minutes of start up time for every restart (atleast once for each phase). Experiments were run on clusters with a maximum wall clock time of 8 hours.

###### Fine-tuning accuracy for SQuAD v1.1: NVIDIA DGX A100 (8x A100 40G)

Our results were obtained by running the `scripts/run_squad.sh` training script in the TensorFlow 20.06-py3 NGC container on NVIDIA DGX A100 with 8x A100 40GB GPUs.

| **GPUs** | **Batch size / GPU** | **Accuracy - TF32** | **Accuracy - mixed precision** | **Time to Train - TF32 (Hrs)** | **Time to Train - mixed precision (Hrs)** |
|:---:|:----:|:----:|:---:|:----:|:----:|
| 8 | 24 |91.41 |91.52 |0.26|0.26|

###### Fine-tuning accuracy for GLUE MRPC: NVIDIA DGX A100 (8x A100 40G)

Our results were obtained by running the `scripts/run_glue.sh` training script in the TensorFlow 20.06-py3 NGC container on NVIDIA DGX A100 with 8x A100 40GB GPUs for 10 different seeds and picking the maximum accuracy on MRPC dev set.

| **GPUs** | **Batch size / GPU** | **Accuracy - TF32** | **Accuracy - mixed precision** | **Time to Train - TF32 (Hrs)** | **Time to Train - mixed precision (Hrs)** | **Throughput - TF32** | **Throughput - mixed precision ** |
|:---:|:----:|:----:|:---:|:----:|:----:|:----:|:----:|
| 8 | 16 | 87.99 | 87.09 |0.009 | 0.009 |357.91|230.16|

##### Training stability test

###### Pre-training SQuAD v1.1 stability test: NVIDIA DGX A100 (256x A100 40GB)

The following tables compare `Final Loss` scores across 2 different training runs with different seeds, for both FP16 and TF32.  The runs showcase consistent convergence on all 2 seeds with very little deviation.

| **FP16, 256x GPUs** | **seed 1** | **seed 2** | **mean** | **std** |
|:-----------:|:-----:|:-----:|:-----:|:-----:|
|Final Loss         |1.570  |1.561 |1.565 |0.006 |

| **TF32, 256x GPUs** | **seed 1** | **seed 2** | **mean** | **std** |
|:-----------:|:-----:|:-----:|:-----:|:-----:|
|Final Loss         |1.583  |1.582 |1.582 |0.0007 |

###### Fine-tuning SQuAD v1.1 stability test: NVIDIA DGX A100 (8x A100 40GB)

The following tables compare `F1` scores across 5 different training runs with different seeds, for both FP16 and TF32 respectively using (Nvidia's Pretrained Checkpoint)[https://ngc.nvidia.com/catalog/models/nvidia:bert_tf_pretraining_lamb_16n].  The runs showcase consistent convergence on all 5 seeds with very little deviation.

| **FP16, 8x GPUs** | **seed 1** | **seed 2** | **seed 3** | **seed 4** | **seed 5** | **mean** | **std** |
|:-----------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|F1         |91.61|91.04|91.59|91.32|91.52|91.41|0.24|

| **TF32, 8x GPUs** | **seed 1** | **seed 2** | **seed 3** | **seed 4** | **seed 5** | **mean** | **std** |
|:-----------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|F1         |91.50|91.49|91.64|91.29|91.67|91.52|0.15 |

###### Fine-tuning GLUE MRPC stability test: NVIDIA DGX A100 (8x A100 40GB)

The following tables compare `F1` scores across 10 different training runs with different seeds, for both FP16 and TF32 respectively using (Nvidia's Pretrained Checkpoint)[https://ngc.nvidia.com/catalog/models/nvidia:bert_tf_pretraining_lamb_16n].  The runs showcase consistent convergence on all 10 seeds with very little deviation.

| ** FP16, 8 GPUs ** | ** seed 1 ** | ** seed 2 ** | ** seed 3 ** | ** seed 4 ** | ** seed 5 ** | ** seed 6 ** | ** seed 7 ** | ** seed 8 ** | ** seed 9 ** | ** seed 10 ** | ** Mean **  | ** Std **   |
|--------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|---------------|-------------|-------------|
| Eval Accuracy      |  84.31372643 |  85.78431606 |  86.76471114 |  87.00980544 |  86.27451062 |  86.27451062 |   85.5392158 |  86.51961088 |  86.27451062 |    85.2941215 | 86.00490391 | 0.795887906 |

| ** TF32, 8 GPUs ** | ** seed 1 ** | ** seed 2 ** | ** seed 3 ** | ** seed 4 ** | ** seed 5 ** | ** seed 6 ** | ** seed 7 ** | ** seed 8 ** | ** seed 9 ** | ** seed 10 ** | ** Mean ** | ** Std **    |
|--------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|---------------|------------|--------------|
| Eval Accuracy      |  87.00980544 |  86.27451062 |  87.99020052 |  86.27451062 |  86.02941632 |  87.00980544 |  86.27451062 |  86.51961088 |  87.74510026 |   86.02941632 | 86.7156887 | 0.7009024515 |


#### Training performance results

##### Training performance: NVIDIA DGX-1 (8x V100 16GB)

###### Pre-training training performance: single-node on DGX-1 16GB

Our results were obtained by running the `scripts/run_pretraining_lamb.sh` training script in the TensorFlow 20.06-py3 NGC container on NVIDIA DGX-1 with 8x V100 16GB GPUs. Performance (in sentences per second) is the steady state throughput.


| **GPUs** | **Sequence Length** | **Batch size / GPU: mixed precision, FP32** | **Gradient Accumulation: mixed precision, FP32** | **Global Batch Size** | **Throughput - mixed precision** | **Throughput - FP32** | **Throughput speedup (FP32 - mixed precision)** | **Weak scaling - mixed precision** | **Weak scaling - FP32** |
|:--------:|:-------------------:|:-------------------------------------------:|--------------------------------------------------|:---------------------:|:--------------------------------:|-----------------------|-------------------------------------------------|------------------------------------|-------------------------|
|        1 |                 128 | 16 , 8                                      | 4096, 8192                                       |                 65536 |                           134.34 |                 39.43 |                                            3.41 |                               1.00 |                    1.00 |
|        4 |                 128 | 16 , 8                                      | 1024, 2048                                       |                 65536 |                           449.68 |                152.33 |                                            2.95 |                               3.35 |                    3.86 |
|        8 |                 128 | 16 , 8                                      | 512, 1024                                        |                 65536 |                          1001.39 |                285.79 |                                            3.50 |                               7.45 |                    7.25 |
|        1 |                 512 | 4 , 2                                       | 8192, 16384                                      |                 32768 |                            28.72 |                  9.80 |                                            2.93 |                               1.00 |                    1.00 |
|        4 |                 512 | 4 , 2                                       | 2048, 4096                                       |                 32768 |                           109.96 |                 35.32 |                                            3.11 |                               3.83 |                    3.60 |
|        8 |                 512 | 4 , 2                                       | 1024, 2048                                       |                 32768 |                           190.65 |                 69.53 |                                            2.74 |                               6.64 |                    7.09 |


Note: The respective values for FP32 runs that use a batch size of 16, 4 in sequence lengths 128 and 512 respectively are not available due to out of memory errors that arise.

###### Fine-tuning training performance for SQuAD v1.1 on DGX-1 16GB

Our results were obtained by running the `scripts/run_squad.sh` training script in the TensorFlow 20.06-py3 NGC container on NVIDIA DGX-1 with 8x V100 16GB GPUs. Performance (in sentences per second) is the steady state throughput.

| **GPUs** | **Batch size / GPU: mixed precision, FP32** | **Throughput - mixed precision** | **Throughput - FP32** | **Throughput speedup (FP32 to mixed precision)** | **Weak scaling - FP32** | **Weak scaling - mixed precision** |
|----------|---------------------------------------------|----------------------------------|-----------------------|--------------------------------------------------|-------------------------|------------------------------------|
|        1 | 4,2                                         |                            29.74 |                  7.36 |                                             4.04 |                    1.00 |                               1.00 |
|        4 | 4,2                                         |                            97.28 |                 26.64 |                                             3.65 |                    3.27 |                               3.62 |
|        8 | 4,2                                         |                           189.77 |                 52.39 |                                             3.62 |                    6.38 |                               7.12 |

Note: The respective values for FP32 runs that use a batch size of 4 are not available due to out of memory errors that arise.

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

##### Training performance: NVIDIA DGX-1 (8x V100 32GB)

###### Pre-training training performance: single-node on DGX-1 32GB

Our results were obtained by running the `scripts/run_pretraining_lamb.sh` training script in the TensorFlow 20.06-py3 NGC container on NVIDIA DGX-1 with 8x V100 32GB GPUs. Performance (in sentences per second) is the steady state throughput.

| **GPUs** | **Sequence Length** | **Batch size / GPU: mixed precision, FP32** | **Gradient Accumulation: mixed precision, FP32** | **Global Batch Size** | **Throughput - mixed precision** | **Throughput - FP32** | **Throughput speedup (FP32 - mixed precision)** | **Weak scaling - mixed precision** | **Weak scaling - FP32** |
|:--------:|:-------------------:|:-------------------------------------------:|--------------------------------------------------|:---------------------:|:--------------------------------:|-----------------------|-------------------------------------------------|------------------------------------|-------------------------|
|        1 |                 128 | 64 , 32                                     | 1024, 2048                                       |                 65536 |                           168.63 |                 46.78 |                                            3.60 |                               1.00 |                    1.00 |
|        4 |                 128 | 64 , 32                                     | 256, 512                                         |                 65536 |                           730.25 |                179.73 |                                            4.06 |                               4.33 |                    3.84 |
|        8 |                 128 | 64 , 32                                     | 128, 256                                         |                 65536 |                          1443.05 |                357.00 |                                            4.04 |                               8.56 |                    7.63 |
|        1 |                 512 | 8 , 8                                       | 4096, 4096                                       |                 32768 |                            31.23 |                 10.67 |                                            2.93 |                               1.00 |                    1.00 |
|        4 |                 512 | 8 , 8                                       | 1024, 1024                                       |                 32768 |                           118.84 |                 39.55 |                                            3.00 |                               3.81 |                    3.71 |
|        8 |                 512 | 8 , 8                                       | 512, 512                                         |                 32768 |                           255.64 |                 81.42 |                                            3.14 |                               8.19 |                    7.63 |


Note: The respective values for FP32 runs that use a batch size of 64 in sequence lengths 128 are not available due to out of memory errors that arise.

###### Fine-tuning training performance for SQuAD v1.1 on DGX-1 32GB

Our results were obtained by running the `scripts/run_squad.sh` training script in the TensorFlow 20.06-py3 NGC container on NVIDIA DGX-1 with 8x V100 32GB GPUs. Performance (in sentences per second) is the steady state throughput.

| **GPUs** | **Batch size / GPU: mixed precision, FP32** | **Throughput - mixed precision** | **Throughput - FP32** | **Throughput speedup (FP32 to mixed precision)** | **Weak scaling - FP32** | **Weak scaling - mixed precision** |
|----------|---------------------------------------------|----------------------------------|-----------------------|--------------------------------------------------|-------------------------|------------------------------------|
|        1 | 24, 10                                      |                            51.02 |                 31.33 |                                             1.63 |                    1.00 |                               1.00 |
|        4 | 24, 10                                      |                           181.37 |                 94.19 |                                             1.93 |                    3.55 |                               3.01 |
|        8 | 24, 10                                      |                            314.6 |                155.53 |                                             2.02 |                    6.17 |                               4.96 |

Note: The respective values for FP32 runs that use a batch size of 24 are not available due to out of memory errors that arise.

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

##### Training performance: NVIDIA DGX-2 (16x V100 32GB)

###### Pre-training training performance: single-node on DGX-2 32GB

Our results were obtained by running the `scripts/run_pretraining_lamb.sh` training script in the TensorFlow 20.06-py3 NGC container on NVIDIA DGX-2 with 16x V100 32GB GPUs. Performance (in sentences per second) is the steady state throughput.

| **GPUs** | **Sequence Length** | **Batch size / GPU: mixed precision, FP32** | **Gradient Accumulation: mixed precision, FP32** | **Global Batch Size** | **Throughput - mixed precision** | **Throughput - FP32** | **Throughput speedup (FP32 - mixed precision)** | **Weak scaling - mixed precision** | **Weak scaling - FP32** |
|:--------:|:-------------------:|:-------------------------------------------:|--------------------------------------------------|:---------------------:|:--------------------------------:|:---------------------:|-------------------------------------------------|------------------------------------|-------------------------|
|        1 |                 128 | 64 , 32                                     | 1024 , 8192                                      |                 65536 |                           188.04 |                 35.32 |                                            5.32 |                               1.00 |                    1.00 |
|        4 |                 128 | 64 , 32                                     | 256 , 2048                                       |                 65536 |                           790.89 |                193.08 |                                            4.10 |                               4.21 |                    5.47 |
|        8 |                 128 | 64 , 32                                     | 128 , 1024                                       |                 65536 |                          1556.89 |                386.89 |                                            4.02 |                               8.28 |                   10.95 |
|       16 |                 128 | 64 , 32                                     | 64 , 128                                         |                 65536 |                          3081.69 |                761.92 |                                            4.04 |                              16.39 |                   21.57 |
|        1 |                 512 | 8 , 8                                       | 4096 , 4096                                      |                 32768 |                            35.32 |                 11.67 |                                            3.03 |                               1.00 |                    1.00 |
|        4 |                 512 | 8 , 8                                       | 1024 , 1024                                      |                 32768 |                           128.98 |                 42.84 |                                            3.01 |                               3.65 |                    3.67 |
|        8 |                 512 | 8 , 8                                       | 512 , 512                                        |                 32768 |                           274.04 |                 86.78 |                                            3.16 |                               7.76 |                    7.44 |
|       16 |                 512 | 8 , 8                                       | 256 , 256                                        |                 32768 |                           513.43 |                173.26 |                                            2.96 |                              14.54 |                   14.85 |

Note: The respective values for FP32 runs that use a batch size of 64 in sequence lengths 128 are not available due to out of memory errors that arise.

###### Pre-training training performance: multi-node on DGX-2H 32GB

Our results were obtained by running the `run.sub` training script in the TensorFlow 19.08-py3 NGC container using multiple NVIDIA DGX-2 with 16x V100 32GB GPUs. Performance (in sentences per second) is the steady state throughput.

| **Num Nodes** | **Sequence Length** | **Batch size / GPU: mixed precision, FP32** | **Gradient Accumulation: mixed precision, FP32** | **Global Batch Size** | **Throughput - mixed precision** | **Throughput - FP32** | **Throughput speedup (FP32 - mixed precision)** | **Weak scaling - mixed precision** | **Weak scaling - FP32** |
|:-------------:|:-------------------:|:-------------------------------------------:|--------------------------------------------------|:---------------------:|:--------------------------------:|:---------------------:|-------------------------------------------------|------------------------------------|-------------------------|
|             1 |                 128 | 64 , 32                                     | 64 , 128                                         |                 65536 |                          3081.69 |                761.92 |                                            4.04 |                               1.00 |                    1.00 |
|             4 |                 128 | 64 , 32                                     | 16 , 32                                          |                 65536 |                         13192.00 |               3389.83 |                                            3.89 |                              4.28 |                    4.45 |
|            16 |                 128 | 64 , 32                                     | 4 , 8                                            |                 65536 |                         48223.00 |              13217.78 |                                            3.65 |                              15.65 |                   17.35 |
|            32 |                 128 | 64 , 32                                     | 2 , 4                                            |                 65536 |                         86673.64 |              25142.26 |                                            3.45 |                              28.13 |                   33.00 |
|             1 |                 512 | 8 , 8                                       | 256 , 256                                        |                 32768 |                           577.79 |                173.26 |                                            3.33 |                               1.00 |                    1.00 |
|             4 |                 512 | 8 , 8                                       | 64 , 64                                          |                 32768 |                          2284.23 |                765.04 |                                            2.99 |                               3.95 |                    4.42 |
|            16 |                 512 | 8 , 8                                       | 16 , 16                                          |                 32768 |                          8853.00 |               3001.43 |                                            2.95 |                              15.32 |                   17.32 |
|            32 |                 512 | 8 , 8                                       | 8 , 8                                            |                 32768 |                         17059.00 |               5893.14 |                                            2.89 |                              29.52 |                   34.01 |

Note: The respective values for FP32 runs that use a batch size of 64 in sequence lengths 128 are not available due to out of memory errors that arise.

###### Fine-tuning training performance for SQuAD v1.1 on DGX-2 32GB

Our results were obtained by running the `scripts/run_squad.sh` training script in the TensorFlow 20.06-py3 NGC container on NVIDIA DGX-2 with 16x V100 32GB GPUs. Performance (in sentences per second) is the steady state throughput.

| **GPUs** | **Batch size / GPU: mixed precision, FP32** | **Throughput - mixed precision** | **Throughput - FP32** | **Throughput speedup (FP32 to mixed precision)** | **Weak scaling - FP32** | **Weak scaling - mixed precision** |
|----------|---------------------------------------------|----------------------------------|-----------------------|--------------------------------------------------|-------------------------|------------------------------------|
|        1 | 24, 10                                      |                            55.28 |                 32.72 |                                             1.69 |                    1.00 |                               1.00 |
|        4 | 24, 10                                      |                           199.53 |                100.73 |                                             1.98 |                    3.61 |                               3.08 |
|        8 | 24, 10                                      |                           341.55 |                168.92 |                                             2.02 |                    6.18 |                               5.16 |
|       16 | 24, 10                                      |                           683.37 |                249.54 |                                             2.74 |                   12.36 |                               7.63 |

Note: The respective values for FP32 runs that use a batch size of 24 are not available due to out of memory errors that arise.

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

##### Training performance: NVIDIA DGX A100 (8x A100 40GB)

###### Pre-training training performance: single-node on DGX A100 40GB

Our results were obtained by running the `scripts/run_pretraining_lamb.sh` training script in the TensorFlow 20.06-py3 NGC container on NVIDIA DGX A100 with 8x A100 40GB GPUs. Performance (in sentences per second) is the steady state throughput.

| **GPUs** | **Sequence Length** | **Batch size / GPU: mixed precision, TF32** | **Gradient Accumulation: mixed precision, TF32** | **Global Batch Size** | **Throughput - mixed precision** | **Throughput - TF32** | **Throughput speedup (TF32 - mixed precision)** | **Weak scaling - mixed precision** | **Weak scaling -TF32** |
|:--------:|:-------------------:|:-------------------------------------------:|--------------------------------------------------|:---------------------:|:--------------------------------:|:---------------------:|-------------------------------------------------|------------------------------------|------------------------|
|        1 |                 128 | 64 , 64                                     | 1024 , 1024                                      |                 65536 |                          356.845 |                238.10 |                                            1.50 |                               1.00 |                   1.00 |
|        4 |                 128 | 64 , 64                                     | 256 , 256                                        |                 65536 |                          1422.25 |                952.39 |                                            1.49 |                               3.99 |                   4.00 |
|        8 |                 128 | 64 , 64                                     | 128 , 128                                        |                 65536 |                          2871.89 |               1889.71 |                                            1.52 |                               8.05 |                   7.94 |
|        1 |                 512 | 16 , 8                                      | 2048 , 4096                                      |                 32768 |                           70.856 |                 39.96 |                                            1.77 |                               1.00 |                   1.00 |
|        4 |                 512 | 16 , 8                                      | 512 , 1024                                       |                 32768 |                          284.912 |                160.16 |                                            1.78 |                               4.02 |                   4.01 |
|        8 |                 512 | 16 , 8                                      | 256 , 512                                        |                 32768 |                          572.112 |                316.51 |                                            1.81 |                               8.07 |                   7.92 |

Note: The respective values for TF32 runs that use a batch size of 16 for sequence length 512 are not available due to out of memory errors that arise.

###### Pre-training training performance: multi-node on DGX A100 40GB

Our results were obtained by running the `scripts/run_pretraining_lamb.sh` training script in the TensorFlow 20.06-py3 NGC container on NVIDIA DGX A100 with 8x A100 40GB GPUs. Performance (in sentences per second) is the steady state throughput.

| **Num Nodes** | **Sequence Length** | **Batch size / GPU: mixed precision, TF32** | **Gradient Accumulation: mixed precision, TF32** | **Global Batch Size** | **Throughput - mixed precision** | **Throughput - TF32** | **Throughput speedup (TF32 - mixed precision)** | **Weak scaling - mixed precision** | **Weak scaling -TF32** |
|:-------------:|:-------------------:|:-------------------------------------------:|--------------------------------------------------|:---------------------:|:--------------------------------:|:---------------------:|-------------------------------------------------|------------------------------------|------------------------|
|             1 |                 128 | 64 , 64                                     | 128 , 128                                        |                 65536 |                          2871.89 |               1889.71 |                                            1.52 |                               1.00 |                   1.00 |
|             4 |                 128 | 64 , 64                                     | 32 , 32                                          |                 65536 |                            11159 |               7532.00 |                                            1.48 |                               3.89 |                   3.99 |
|            16 |                 128 | 64 , 64                                     | 8 , 8                                            |                 65536 |                            41144 |              28605.62 |                                            1.44 |                              14.33 |                  15.14 |
|            32 |                 128 | 64 , 64                                     | 4 , 4                                            |                 65536 |                         77479.87 |              53585.82 |                                            1.45 |                              26.98 |                  28.36 |
|             1 |                 512 | 16 , 8                                      | 256 , 512                                        |                 32768 |                          572.112 |                316.51 |                                            1.81 |                               1.00 |                   1.00 |
|             4 |                 512 | 16 , 8                                      | 128 , 128                                        |                 65536 |                          2197.44 |               1268.43 |                                            1.73 |                               3.84 |                   4.01 |
|            16 |                 512 | 16 , 8                                      | 32 , 32                                          |                 65536 |                           8723.1 |               4903.39 |                                            1.78 |                              15.25 |                  15.49 |
|            32 |                 512 | 16 , 8                                      | 16 , 16                                          |                 65536 |                            16705 |               9463.80 |                                            1.77 |                              29.20 |                  29.90 |

Note: The respective values for TF32 runs that use a batch size of 16 for sequence length 512 are not available due to out of memory errors that arise.

###### Fine-tuning training performance for SQuAD v1.1 on DGX A100 40GB

Our results were obtained by running the `scripts/run_squad.sh` training script in the TensorFlow 20.06-py3 NGC container on NVIDIA DGX A100 with 8x A100 40GB GPUs. Performance (in sentences per second) is the steady state throughput.

| **GPUs** | **Batch size / GPU: mixed precision, TF32** | **Throughput - mixed precision** | **Throughput - TF32** | **Throughput speedup (TF32 to mixed precision)** | **Weak scaling - TF32** | **Weak scaling - mixed precision** |
|----------|---------------------------------------------|----------------------------------|-----------------------|--------------------------------------------------|-------------------------|------------------------------------|
|        1 | 32, 16                                      |                           102.26 |                61.364 |                                             1.67 |                    1.00 |                               1.00 |
|        4 | 32, 16                                      |                          366.353 |               223.187 |                                             1.64 |                    3.58 |                               3.64 |
|        8 | 32, 16                                      |                          518.898 |                440.47 |                                             1.18 |                    5.07 |                               7.18 |

Note: The respective values for TF32 runs that use a batch size of 32 are not available due to out of memory errors that arise.

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

#### Inference performance results

##### Inference performance: NVIDIA DGX-1 (1x V100 16GB)

###### Fine-tuning inference performance for SQuAD v1.1 on 16GB

Our results were obtained by running the `scripts/finetune_inference_benchmark.sh` script in the TensorFlow 20.06-py3 NGC container on NVIDIA DGX-1 with 1x V100 16GB GPUs. Performance numbers (throughput in sentences per second and latency in milliseconds) were averaged from 1024 iterations. Latency is computed as the time taken for a batch to process as they are fed in one after another in the model ie no pipelining.

| Model | Sequence Length | Batch Size | Precision | Throughput-Average(sent/sec) | Latency-Average(ms) | Latency-90%(ms) | Latency-95%(ms) | Latency-99%(ms) |
|-------|-----------------|------------|-----------|------------------------------|---------------------|-----------------|-----------------|-----------------|
| base  |             128 |          1 | fp16      |                       206.82 |                7.96 |            4.98 |            5.04 |            5.23 |
| base  |             128 |          2 | fp16      |                       376.75 |                8.68 |            5.42 |            5.49 |            5.64 |
| base  |             128 |          4 | fp16      |                          635 |               12.31 |            6.46 |            6.55 |            6.83 |
| base  |             128 |          8 | fp16      |                       962.83 |               13.64 |            8.47 |            8.56 |            8.75 |
| base  |             384 |          1 | fp16      |                       167.01 |               12.77 |            6.12 |            6.23 |            6.52 |
| base  |             384 |          2 | fp16      |                       252.12 |               21.05 |            8.03 |            8.09 |            8.61 |
| base  |             384 |          4 | fp16      |                       341.95 |               25.09 |           11.88 |           11.96 |           12.52 |
| base  |             384 |          8 | fp16      |                       421.26 |               33.16 |            19.2 |           19.37 |           19.91 |
|       |                 |            |           |                              |                     |                 |                 |                 |
| base  |             128 |          1 | fp32      |                       174.48 |                8.17 |            5.89 |            5.95 |            6.12 |
| base  |             128 |          2 | fp32      |                       263.67 |               10.33 |            7.66 |            7.69 |            7.92 |
| base  |             128 |          4 | fp32      |                       349.34 |               16.31 |           11.57 |           11.62 |           11.87 |
| base  |             128 |          8 | fp32      |                       422.88 |               23.27 |           19.23 |           19.38 |           20.38 |
| base  |             384 |          1 | fp32      |                        99.52 |               14.99 |           10.19 |           10.23 |           10.78 |
| base  |             384 |          2 | fp32      |                       118.01 |               25.98 |           17.12 |           17.18 |           17.78 |
| base  |             384 |          4 | fp32      |                        128.1 |                  41 |           31.56 |            31.7 |           32.39 |
| base  |             384 |          8 | fp32      |                        136.1 |               69.77 |           59.44 |           59.66 |           60.51 |
|       |                 |            |           |                              |                     |                 |                 |                 |
| large |             128 |          1 | fp16      |                        98.63 |               15.86 |           10.27 |           10.31 |           10.46 |
| large |             128 |          2 | fp16      |                       172.59 |               17.78 |           11.81 |           11.86 |           12.13 |
| large |             128 |          4 | fp16      |                       272.86 |               25.66 |           14.86 |           14.94 |           15.18 |
| large |             128 |          8 | fp16      |                       385.64 |               30.74 |           20.98 |            21.1 |           21.68 |
| large |             384 |          1 | fp16      |                        70.74 |               26.85 |           14.38 |           14.47 |            14.7 |
| large |             384 |          2 | fp16      |                         99.9 |               45.29 |           20.26 |           20.43 |           21.11 |
| large |             384 |          4 | fp16      |                       128.42 |               56.94 |           31.44 |           31.71 |           32.45 |
| large |             384 |          8 | fp16      |                       148.57 |               81.69 |           54.23 |           54.54 |           55.53 |
|       |                 |            |           |                              |                     |                 |                 |                 |
| large |             128 |          1 | fp32      |                        76.75 |               17.06 |           13.21 |           13.27 |            13.4 |
| large |             128 |          2 | fp32      |                       100.82 |               24.34 |           20.05 |           20.13 |           21.13 |
| large |             128 |          4 | fp32      |                       117.59 |               41.76 |           34.42 |           34.55 |           35.29 |
| large |             128 |          8 | fp32      |                       130.42 |               68.59 |              62 |           62.23 |           62.98 |
| large |             384 |          1 | fp32      |                        33.95 |               37.89 |           29.82 |           29.98 |           30.56 |
| large |             384 |          2 | fp32      |                        38.47 |               68.35 |           52.56 |           52.74 |           53.89 |
| large |             384 |          4 | fp32      |                        41.11 |              114.27 |           98.19 |           98.54 |           99.54 |
| large |             384 |          8 | fp32      |                        41.32 |              213.84 |          194.92 |          195.36 |          196.94 |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

##### Inference performance: NVIDIA DGX-1 (1x V100 32GB)

###### Fine-tuning inference performance for SQuAD v1.1 on 32GB

Our results were obtained by running the `scripts/finetune_inference_benchmark.sh` training script in the TensorFlow 20.06-py3 NGC container on NVIDIA DGX-1 with 1x V100 32GB GPUs. Performance numbers (throughput in sentences per second and latency in milliseconds) were averaged from 1024 iterations. Latency is computed as the time taken for a batch to process as they are fed in one after another in the model ie no pipelining.

| Model | Sequence Length | Batch Size | Precision | Throughput-Average(sent/sec) | Latency-Average(ms) | Latency-90%(ms) | Latency-95%(ms) | Latency-99%(ms) |
|-------|-----------------|------------|-----------|------------------------------|---------------------|-----------------|-----------------|-----------------|
| base  |             128 |          1 | fp16      |                       207.87 |                7.63 |            4.94 |            5.03 |            5.32 |
| base  |             128 |          2 | fp16      |                       376.44 |                8.47 |            5.44 |             5.5 |            5.68 |
| base  |             128 |          4 | fp16      |                       642.55 |               11.63 |             6.3 |            6.36 |            6.68 |
| base  |             128 |          8 | fp16      |                       943.85 |               13.24 |            8.56 |            8.68 |            8.92 |
| base  |             384 |          1 | fp16      |                       162.62 |               12.24 |            6.31 |             6.4 |            6.73 |
| base  |             384 |          2 | fp16      |                       244.15 |               20.05 |            8.34 |            8.41 |            8.93 |
| base  |             384 |          4 | fp16      |                       338.68 |               23.53 |           11.88 |           11.92 |           12.63 |
| base  |             384 |          8 | fp16      |                       407.46 |               32.72 |           19.84 |           20.06 |           20.89 |
|       |                 |            |           |                              |                     |                 |                 |                 |
| base  |             128 |          1 | fp32      |                       175.16 |                8.31 |            5.85 |            5.89 |            6.04 |
| base  |             128 |          2 | fp32      |                       261.31 |               10.48 |            7.75 |            7.81 |            8.08 |
| base  |             128 |          4 | fp32      |                       339.45 |               16.67 |           11.95 |           12.02 |           12.46 |
| base  |             128 |          8 | fp32      |                       406.67 |               24.12 |           19.86 |           19.97 |           20.41 |
| base  |             384 |          1 | fp32      |                        98.33 |               15.28 |           10.27 |           10.32 |           10.76 |
| base  |             384 |          2 | fp32      |                       114.92 |               26.88 |           17.55 |           17.59 |           18.29 |
| base  |             384 |          4 | fp32      |                       125.76 |               41.74 |           32.06 |           32.23 |           33.72 |
| base  |             384 |          8 | fp32      |                       136.62 |               69.78 |           58.95 |           59.19 |              60 |
|       |                 |            |           |                              |                     |                 |                 |                 |
| large |             128 |          1 | fp16      |                        96.46 |               15.56 |           10.56 |           10.66 |           11.02 |
| large |             128 |          2 | fp16      |                       168.31 |               17.42 |           12.11 |           12.25 |           12.57 |
| large |             128 |          4 | fp16      |                       267.76 |               24.76 |           15.17 |           15.36 |           16.68 |
| large |             128 |          8 | fp16      |                       378.28 |               30.34 |           21.39 |           21.54 |           21.97 |
| large |             384 |          1 | fp16      |                        68.75 |               26.02 |           14.77 |           14.94 |            15.3 |
| large |             384 |          2 | fp16      |                        95.41 |               44.01 |           21.24 |           21.47 |           22.01 |
| large |             384 |          4 | fp16      |                       124.43 |               55.14 |           32.53 |           32.83 |           33.58 |
| large |             384 |          8 | fp16      |                       143.02 |               81.37 |           56.51 |           56.88 |           58.05 |
|       |                 |            |           |                              |                     |                 |                 |                 |
| large |             128 |          1 | fp32      |                        75.34 |                17.5 |           13.46 |           13.52 |            13.7 |
| large |             128 |          2 | fp32      |                        99.73 |                24.7 |           20.27 |           20.38 |           21.45 |
| large |             128 |          4 | fp32      |                       116.92 |                42.1 |           34.49 |           34.59 |           34.98 |
| large |             128 |          8 | fp32      |                       130.11 |               68.95 |           62.03 |           62.23 |            63.3 |
| large |             384 |          1 | fp32      |                        33.84 |               38.15 |           29.75 |           29.89 |           31.23 |
| large |             384 |          2 | fp32      |                        38.02 |               69.31 |            53.1 |           53.36 |           54.42 |
| large |             384 |          4 | fp32      |                         41.2 |              114.34 |           97.96 |           98.32 |           99.55 |
| large |             384 |          8 | fp32      |                        42.37 |              209.16 |          190.18 |          190.66 |          192.77 |


To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

##### Inference performance: NVIDIA DGX-2 (1x V100 32GB)

###### Fine-tuning inference performance for SQuAD v1.1 on DGX-2  32GB

Our results were obtained by running the `scripts/finetune_inference_benchmark.sh` training script in the TensorFlow 20.06-py3 NGC container on NVIDIA DGX-2 with 1x V100 32GB GPUs. Performance numbers (throughput in sentences per second and latency in milliseconds) were averaged from 1024 iterations. Latency is computed as the time taken for a batch to process as they are fed in one after another in the model ie no pipelining.

| Model | Sequence Length | Batch Size | Precision | Throughput-Average(sent/sec) | Latency-Average(ms) | Latency-90%(ms) | Latency-95%(ms) | Latency-99%(ms) |
|-------|-----------------|------------|-----------|------------------------------|---------------------|-----------------|-----------------|-----------------|
| base  |             128 |          1 | fp16      |                       220.35 |                7.82 |             4.7 |            4.83 |            5.15 |
| base  |             128 |          2 | fp16      |                       384.55 |                 8.7 |            5.49 |            5.68 |            6.01 |
| base  |             128 |          4 | fp16      |                        650.7 |                36.3 |            6.35 |            6.51 |            6.87 |
| base  |             128 |          8 | fp16      |                       992.41 |               13.59 |            8.22 |            8.37 |            8.96 |
| base  |             384 |          1 | fp16      |                       172.89 |               12.86 |            5.94 |            6.04 |            6.44 |
| base  |             384 |          2 | fp16      |                       258.48 |               20.42 |            7.89 |            8.09 |            9.15 |
| base  |             384 |          4 | fp16      |                       346.34 |               24.93 |           11.97 |           12.12 |           12.76 |
| base  |             384 |          8 | fp16      |                        430.4 |               33.08 |           18.75 |           19.27 |           20.12 |
|       |                 |            |           |                              |                     |                 |                 |                 |
| base  |             128 |          1 | fp32      |                       183.69 |                7.52 |            5.86 |            5.97 |            6.27 |
| base  |             128 |          2 | fp32      |                       282.95 |                9.51 |            7.31 |            7.49 |            7.83 |
| base  |             128 |          4 | fp32      |                       363.83 |               15.12 |           11.35 |           11.47 |           11.74 |
| base  |             128 |          8 | fp32      |                       449.12 |               21.65 |              18 |            18.1 |            18.6 |
| base  |             384 |          1 | fp32      |                       104.92 |                13.8 |             9.9 |            9.99 |           10.48 |
| base  |             384 |          2 | fp32      |                       123.55 |               24.21 |           16.29 |            16.4 |           17.61 |
| base  |             384 |          4 | fp32      |                       139.38 |               36.69 |           28.89 |           29.04 |           30.01 |
| base  |             384 |          8 | fp32      |                       146.28 |               64.69 |           55.09 |           55.32 |            56.3 |
|       |                 |            |           |                              |                     |                 |                 |                 |
| large |             128 |          1 | fp16      |                        98.34 |               15.85 |           10.61 |           10.78 |            11.5 |
| large |             128 |          2 | fp16      |                       172.95 |                17.8 |           11.91 |           12.08 |           12.42 |
| large |             128 |          4 | fp16      |                       278.82 |               25.18 |            14.7 |           14.87 |           15.65 |
| large |             128 |          8 | fp16      |                       402.28 |               30.45 |           20.21 |           20.43 |           21.24 |
| large |             384 |          1 | fp16      |                         71.1 |               26.55 |           14.44 |           14.61 |           15.32 |
| large |             384 |          2 | fp16      |                       100.48 |               44.04 |           20.31 |           20.48 |            21.6 |
| large |             384 |          4 | fp16      |                       131.68 |               56.19 |            30.8 |           31.03 |            32.3 |
| large |             384 |          8 | fp16      |                       151.81 |               81.53 |           53.22 |           53.87 |           55.34 |
|       |                 |            |           |                              |                     |                 |                 |                 |
| large |             128 |          1 | fp32      |                        77.87 |               16.33 |           13.33 |           13.45 |           13.77 |
| large |             128 |          2 | fp32      |                       105.41 |               22.77 |           19.39 |           19.52 |           19.86 |
| large |             128 |          4 | fp32      |                       124.16 |               38.61 |           32.69 |           32.88 |            33.9 |
| large |             128 |          8 | fp32      |                       137.69 |               64.61 |           58.62 |           58.89 |           59.94 |
| large |             384 |          1 | fp32      |                        36.34 |               34.94 |           27.72 |           27.81 |           28.21 |
| large |             384 |          2 | fp32      |                        41.11 |               62.54 |           49.14 |           49.32 |           50.25 |
| large |             384 |          4 | fp32      |                        43.32 |              107.53 |           93.07 |           93.47 |           94.27 |
| large |             384 |          8 | fp32      |                        44.64 |              196.28 |          180.21 |          180.75 |          182.41 |


##### Inference performance: NVIDIA DGX A100 (1x A100 40GB)

###### Fine-tuning inference performance for SQuAD v1.1 on DGX A100  40GB

Our results were obtained by running the `scripts/finetune_inference_benchmark.sh` training script in the TensorFlow 20.06-py3 NGC container on NVIDIA DGX A100 with 1x V100 40GB GPUs. Performance numbers (throughput in sentences per second and latency in milliseconds) were averaged from 1024 iterations. Latency is computed as the time taken for a batch to process as they are fed in one after another in the model ie no pipelining.

| Model | Sequence Length | Batch Size | Precision | Throughput-Average(sent/sec) | Latency-Average(ms) | Latency-90%(ms) | Latency-95%(ms) | Latency-99%(ms) |
|-------|-----------------|------------|-----------|------------------------------|---------------------|-----------------|-----------------|-----------------|
| base  |             128 |          1 | fp16      |                       231.37 |                6.43 |            4.57 |            4.68 |            4.93 |
| base  |             128 |          2 | fp16      |                       454.54 |                6.77 |            4.66 |            4.77 |            4.96 |
| base  |             128 |          4 | fp16      |                       842.34 |                 8.8 |            4.91 |            4.98 |            5.39 |
| base  |             128 |          8 | fp16      |                      1216.43 |               10.39 |            6.77 |            6.86 |            7.28 |
| base  |             384 |          1 | fp16      |                       210.59 |                9.03 |            4.83 |            4.86 |            5.06 |
| base  |             384 |          2 | fp16      |                       290.91 |               14.88 |            7.09 |            7.19 |            7.72 |
| base  |             384 |          4 | fp16      |                       407.13 |               18.04 |            9.93 |           10.05 |           10.74 |
| base  |             384 |          8 | fp16      |                       478.67 |               26.06 |           16.92 |           17.19 |           17.76 |
|       |                 |            |           |                              |                     |                 |                 |                 |
| base  |             128 |          1 | tf32      |                       223.38 |                6.94 |            4.73 |            4.86 |            5.04 |
| base  |             128 |          2 | tf32      |                       447.57 |                 7.2 |            4.68 |            4.82 |            5.07 |
| base  |             128 |          4 | tf32      |                       838.89 |                9.16 |            4.88 |            4.93 |            5.38 |
| base  |             128 |          8 | tf32      |                      1201.05 |               10.81 |            6.88 |            6.99 |            7.21 |
| base  |             384 |          1 | tf32      |                       206.46 |                9.74 |            4.93 |            4.98 |            5.25 |
| base  |             384 |          2 | tf32      |                          287 |               15.57 |            7.18 |            7.27 |            7.87 |
| base  |             384 |          4 | tf32      |                       396.59 |               18.94 |            10.3 |           10.41 |           11.04 |
| base  |             384 |          8 | tf32      |                       479.04 |               26.81 |           16.88 |           17.25 |           17.74 |
|       |                 |            |           |                              |                     |                 |                 |                 |
| base  |             128 |          1 | fp32      |                       152.92 |                9.13 |            6.76 |            6.91 |            7.06 |
| base  |             128 |          2 | fp32      |                       297.42 |                9.51 |            6.93 |            7.07 |            7.21 |
| base  |             128 |          4 | fp32      |                       448.57 |               11.81 |            9.12 |            9.25 |            9.68 |
| base  |             128 |          8 | fp32      |                       539.94 |               17.49 |              15 |            15.1 |           15.79 |
| base  |             384 |          1 | fp32      |                       115.19 |               13.69 |            8.89 |            8.98 |            9.27 |
| base  |             384 |          2 | fp32      |                       154.66 |               18.49 |           13.06 |           13.14 |           13.89 |
| base  |             384 |          4 | fp32      |                       174.28 |               28.75 |           23.11 |           23.24 |              24 |
| base  |             384 |          8 | fp32      |                       191.97 |               48.05 |           41.85 |           42.25 |            42.8 |
|       |                 |            |           |                              |                     |                 |                 |                 |
| large |             128 |          1 | fp16      |                       127.75 |               11.18 |            8.14 |            8.25 |            8.53 |
| large |             128 |          2 | fp16      |                       219.49 |               12.76 |             9.4 |            9.54 |            9.89 |
| large |             128 |          4 | fp16      |                       315.83 |               19.01 |           12.87 |           12.98 |           13.37 |
| large |             128 |          8 | fp16      |                       495.75 |               22.21 |           16.33 |           16.45 |           16.79 |
| large |             384 |          1 | fp16      |                        96.65 |               17.46 |           10.52 |            10.6 |              11 |
| large |             384 |          2 | fp16      |                       126.07 |               29.43 |           16.09 |           16.22 |           16.78 |
| large |             384 |          4 | fp16      |                       165.21 |               38.39 |           24.41 |           24.61 |           25.38 |
| large |             384 |          8 | fp16      |                       182.13 |               61.04 |           44.32 |           44.61 |           45.23 |
|       |                 |            |           |                              |                     |                 |                 |                 |
| large |             128 |          1 | tf32      |                       133.24 |               10.86 |            7.77 |            7.87 |            8.23 |
| large |             128 |          2 | tf32      |                       218.13 |               12.86 |            9.44 |            9.56 |            9.85 |
| large |             128 |          4 | tf32      |                       316.25 |               18.98 |           12.91 |           13.01 |           13.57 |
| large |             128 |          8 | tf32      |                       495.21 |               22.25 |            16.4 |           16.51 |           17.23 |
| large |             384 |          1 | tf32      |                        95.43 |                17.5 |           10.72 |           10.83 |           11.49 |
| large |             384 |          2 | tf32      |                       125.99 |               29.47 |           16.06 |           16.15 |           16.67 |
| large |             384 |          4 | tf32      |                       164.28 |               38.77 |            24.6 |           24.83 |           25.59 |
| large |             384 |          8 | tf32      |                       182.46 |                  61 |            44.2 |           44.46 |           45.15 |
|       |                 |            |           |                              |                     |                 |                 |                 |
| large |             128 |          1 | fp32      |                        50.43 |               23.83 |           20.11 |            20.2 |           20.56 |
| large |             128 |          2 | fp32      |                        94.47 |               25.53 |           21.36 |           21.49 |           21.78 |
| large |             128 |          4 | fp32      |                       141.52 |               32.51 |           28.44 |           28.57 |           28.99 |
| large |             128 |          8 | fp32      |                       166.37 |               52.07 |            48.3 |           48.43 |           49.46 |
| large |             384 |          1 | fp32      |                        44.42 |               30.54 |           22.67 |           22.74 |           23.46 |
| large |             384 |          2 | fp32      |                        50.29 |               48.74 |           39.95 |           40.06 |           40.59 |
| large |             384 |          4 | fp32      |                        55.58 |               81.55 |           72.31 |            72.6 |            73.7 |
| large |             384 |          8 | fp32      |                        58.38 |              147.63 |          137.43 |          137.82 |           138.3 |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

##### Inference performance: NVIDIA Tesla T4 (1x T4 16GB)

###### Fine-tuning inference performance for SQuAD v1.1 on Tesla T4 16GB

Our results were obtained by running the `scripts/finetune_inference_benchmark.sh` training script in the TensorFlow 20.06-py3 NGC container on NVIDIA Tesla T4 with 1x T4 16GB GPUs. Performance numbers (throughput in sentences per second and latency in milliseconds) were averaged from 1024 iterations. Latency is computed as the time taken for a batch to process as they are fed in one after another in the model ie no pipelining.

| Model | Sequence Length | Batch Size | Precision | Throughput-Average(sent/sec) | Latency-Average(ms) | Latency-50%(ms) | Latency-90%(ms) | Latency-95%(ms) | Latency-99%(ms) | Latency-100%(ms) |
|-------|-----------------|------------|-----------|------------------------------|---------------------|-----------------|-----------------|-----------------|-----------------|------------------|
| base  |             128 |          1 | fp16      |                        91.93 |               13.94 |           10.93 |           11.41 |           11.52 |           11.94 |          5491.47 |
| base  |             128 |          2 | fp16      |                       148.08 |               16.91 |           13.65 |           13.95 |           14.06 |           14.74 |          5757.12 |
| base  |             128 |          4 | fp16      |                       215.45 |               24.56 |           18.68 |           18.92 |           19.08 |           19.84 |          5894.82 |
| base  |             128 |          8 | fp16      |                       289.52 |               33.07 |           27.77 |           28.22 |           28.38 |           29.16 |          6074.47 |
| base  |             384 |          1 | fp16      |                        60.75 |               23.18 |            16.6 |           16.93 |           17.03 |           17.45 |          7006.41 |
| base  |             384 |          2 | fp16      |                        82.85 |               37.05 |           24.26 |           24.54 |           24.63 |           25.67 |          7529.94 |
| base  |             384 |          4 | fp16      |                        97.78 |                54.4 |           41.02 |           41.53 |           41.94 |           43.91 |          7995.39 |
| base  |             384 |          8 | fp16      |                       106.78 |                89.6 |           74.98 |            75.5 |           76.13 |           78.02 |          8461.93 |
|       |                 |            |           |                              |                     |                 |                 |                 |                 |                  |
| base  |             128 |          1 | fp32      |                        54.28 |               20.88 |           18.52 |            18.8 |           18.92 |           19.29 |           4401.4 |
| base  |             128 |          2 | fp32      |                        71.75 |               30.57 |           28.08 |           28.51 |           28.62 |           29.12 |          4573.47 |
| base  |             128 |          4 | fp32      |                        88.01 |               50.37 |           45.61 |           45.94 |           46.14 |           47.04 |           4992.7 |
| base  |             128 |          8 | fp32      |                        98.92 |               85.57 |           80.98 |           81.44 |           81.74 |           82.75 |          5408.97 |
| base  |             384 |          1 | fp32      |                        25.83 |               43.63 |           38.75 |           39.33 |           39.43 |           40.02 |          5148.45 |
| base  |             384 |          2 | fp32      |                        29.08 |               77.68 |           68.89 |           69.26 |           69.55 |           72.08 |           5462.5 |
| base  |             384 |          4 | fp32      |                        30.33 |              141.45 |          131.86 |          132.53 |          133.14 |           136.7 |          5975.63 |
| base  |             384 |          8 | fp32      |                         31.8 |              262.88 |          251.62 |          252.23 |          253.08 |          255.56 |             7124 |
|       |                 |            |           |                              |                     |                 |                 |                 |                 |                  |
| large |             128 |          1 | fp16      |                        40.31 |               30.61 |           25.14 |           25.62 |           25.87 |           27.61 |         10395.87 |
| large |             128 |          2 | fp16      |                        63.79 |               37.43 |           31.66 |           32.31 |           32.66 |           34.36 |          10302.2 |
| large |             128 |          4 | fp16      |                         87.4 |                56.5 |           45.97 |            46.6 |           47.01 |           48.71 |         10391.17 |
| large |             128 |          8 | fp16      |                        107.5 |               84.29 |           74.59 |           75.25 |           75.64 |           77.73 |          10945.1 |
| large |             384 |          1 | fp16      |                        23.05 |               55.73 |           43.72 |           44.28 |           44.74 |            46.8 |         12889.23 |
| large |             384 |          2 | fp16      |                        29.59 |               91.61 |           67.94 |            68.8 |           69.45 |           71.64 |         13876.35 |
| large |             384 |          4 | fp16      |                        34.27 |              141.56 |          116.67 |          118.02 |           119.1 |           122.1 |         14570.73 |
| large |             384 |          8 | fp16      |                        38.29 |              237.85 |          208.95 |          210.08 |          211.33 |          214.61 |         16626.02 |
|       |                 |            |           |                              |                     |                 |                 |                 |                 |                  |
| large |             128 |          1 | fp32      |                        21.52 |               50.46 |           46.48 |           47.63 |           47.94 |           49.63 |          7150.38 |
| large |             128 |          2 | fp32      |                         25.4 |                83.3 |           79.06 |           79.61 |           80.06 |           81.77 |          7763.11 |
| large |             128 |          4 | fp32      |                        28.19 |              149.49 |          142.15 |           143.1 |          143.65 |          145.43 |          7701.38 |
| large |             128 |          8 | fp32      |                        30.14 |              272.84 |           265.6 |          266.57 |          267.21 |          269.37 |           8246.3 |
| large |             384 |          1 | fp32      |                         8.46 |              126.81 |          118.44 |          119.42 |          120.31 |          122.74 |          9007.96 |
| large |             384 |          2 | fp32      |                         9.29 |                 231 |          215.54 |          216.64 |          217.71 |          220.35 |          9755.69 |
| large |             384 |          4 | fp32      |                         9.55 |               436.5 |          418.71 |          420.05 |          421.27 |           424.3 |         11766.45 |
| large |             384 |          8 | fp32      |                         9.75 |               840.9 |          820.39 |          822.19 |          823.69 |          827.99 |         12856.99 |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

## Release notes

### Changelog
June 2020
- Results obtained using 20.06 and on DGX A100 40GB

Janurary 2020
- Added inference with TensorRT

November 2019
- Pre-training and Finetuning on BioMedical tasks and corpus

October 2019
- Disabling Grappler Optimizations for improved performance

September 2019
- Pre-training using LAMB
- Multi Node support
- Fine Tuning support for GLUE (CoLA, MNLI, MRPC)

July 2019
- Results obtained using 19.06
- Inference Studies using Triton Inference Server

March 2019
- Initial release

### Known issues

There are no known issues with this model.

# ELECTRA For TensorFlow2
 
This repository provides a script and recipe to train the ELECTRA model for TensorFlow2 to achieve state-of-the-art accuracy, and is tested and maintained by NVIDIA.
 
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
    + [Fine tuning parameters](#fine-tuning-parameters)
  * [Command-line options](#command-line-options)
  * [Getting the data](#getting-the-data)
  * [Training process](#training-process)
    + [Fine-tuning](#fine-tuning)
  * [Inference process](#inference-process)
    + [Fine-tuning inference](#fine-tuning-inference)
- [Performance](#performance)
  * [Benchmarking](#benchmarking)
    + [Training performance benchmark](#training-performance-benchmark)
    + [Inference performance benchmark](#inference-performance-benchmark)
  * [Results](#results)
    + [Training accuracy results](#training-accuracy-results)
      - [Fine-tuning accuracy: NVIDIA DGX A100 (8x A100 40GB)](#fine-tuning-accuracy-nvidia-dgx-a100-8x-a100-40gb)
      - [Fine-tuning accuracy: NVIDIA DGX-1 (8x V100 16GB)](#fine-tuning-accuracy-nvidia-dgx-1-8x-v100-16gb)
      - [Fine-tuning accuracy: NVIDIA DGX-2 (16x V100 32GB)](#fine-tuning-accuracy-nvidia-dgx-2-16x-v100-32gb)
      - [Training stability test](#training-stability-test)
        * [Fine-tuning stability test: NVIDIA DGX-1 (8x V100 16GB)](#fine-tuning-stability-test-nvidia-dgx-1-8x-v100-16gb)
    + [Training performance results](#training-performance-results)
      - [Training performance: NVIDIA DGX A100 (8x A100 40GB)](#training-performance-nvidia-dgx-a100-8x-a100-40gb)
        * [Fine-tuning NVIDIA DGX A100 (8x A100 40GB)](#fine-tuning-nvidia-dgx-a100-8x-a100-40gb)
      - [Training performance: NVIDIA DGX-1 (8x V100 16GB)](#training-performance-nvidia-dgx-1-8x-v100-16gb)
        * [Fine-tuning NVIDIA DGX-1 (8x V100 16GB)](#fine-tuning-nvidia-dgx-1-8x-v100-16gb)
      - [Training performance: NVIDIA DGX-2 (16x V100 32GB)](#training-performance-nvidia-dgx-2-16x-v100-32gb)
        * [Fine-tuning NVIDIA DGX-2 With 32GB](#fine-tuning-nvidia-dgx-2-with-32gb)
    + [Inference performance results](#inference-performance-results)
      - [Inference performance: NVIDIA DGX A100 (1x A100 40GB)](#inference-performance-nvidia-dgx-a100-1x-a100-40gb)
        * [Fine-tuning inference on NVIDIA DGX A100 (1x A100 40GB)](#fine-tuning-inference-on-nvidia-dgx-a100-1x-a100-40gb)
      - [Inference performance: NVIDIA DGX-1 (1x V100 16GB)](#inference-performance-nvidia-dgx-1-1x-v100-16gb)
        * [Fine-tuning inference on NVIDIA DGX-1 with 16GB](#fine-tuning-inference-on-nvidia-dgx-1-with-16gb)
      - [Inference performance: NVIDIA DGX-2 (1x V100 32GB)](#inference-performance-nvidia-dgx-2-1x-v100-32gb)
        * [Fine-tuning inference on NVIDIA DGX-2 with 32GB](#fine-tuning-inference-on-nvidia-dgx-2-with-32gb)
- [Release notes](#release-notes)
  * [Changelog](#changelog)
  * [Known issues](#known-issues)
 
## Model overview
 
Electra, Efficiently Learning an Encoder that Classifies Token Replacements Accurately, is novel pre-training language representations which outperforms existing techniques given the same compute budget on a wide array of Natural Language Processing (NLP) tasks. This model is based on the [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/forum?id=r1xMH1BtvB) paper. NVIDIA's implementation of ELECTRA is an optimized version of the [Hugging Face implementation](https://huggingface.co/transformers/model_doc/electra.html), leveraging mixed precision arithmetic and Tensor Cores on Volta, Turing, and the NVIDIA Ampere GPU architectures for faster training times while maintaining target accuracy.
 
This repository contains scripts to interactively launch data download, training, benchmarking and inference routines in a Docker container for fine-tuning for tasks such as question answering. The major differences between the original implementation of the paper and this version of ELECTRA are as follows:
 
-   Fused Adam optimizer for fine tuning tasks
-   Fused CUDA kernels for better performance LayerNorm
-   Automatic mixed precision (AMP) training support
 
Other publicly available implementations of Electra include:
1. [Hugging Face](https://huggingface.co/transformers/model_doc/electra.html)
2. [Google's implementation](https://github.com/google-research/electra)
 
This model trains with mixed precision Tensor Cores on Volta and provides a push-button solution to pretraining on a corpus of choice. As a result, researchers can get results 4x faster than training without Tensor Cores. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.
 
### Model architecture
 
ELECTRA is a combination of two transformer models: a generator and a discriminator. The generator’s role is to replace tokens in a sequence, and is therefore trained as a masked language model. The discriminator, which is the model we’re interested in, tries to identify which tokens were replaced by the generator in the sequence. Both generator and discriminator use the same architecture as the encoder of the Transformer. The encoder is simply a stack of Transformer blocks, which consist of a multi-head attention layer followed by successive stages of feed-forward networks and layer normalization. The multi-head attention layer accomplishes self-attention on multiple input representations.
 
![Figure 1-1](https://1.bp.blogspot.com/-sHybc03nJRo/XmfLongdVYI/AAAAAAAAFbI/a0t5w_zOZ-UtxYaoQlVkmTRsyFJyFddtQCLcBGAsYHQ/s1600/image1.png "ELECTRA architecture")
 
 
 
### Default configuration
 
ELECTRA uses a new pre-training task, called replaced token detection (RTD), that trains a bidirectional model (like a MLM) while learning from all input positions (like a LM). Inspired by generative adversarial networks (GANs), instead of corrupting the input by replacing tokens with “[MASK]” as in BERT, the generator is trained to corrupt the input by replacing some input tokens with incorrect, but somewhat plausible, fakes. On the other hand, the discriminator is trained to distinguish between “real” and “fake” input data. 
 
The [Google ELECTRA repository](https://github.com/google-research/electra) reports the results for three configurations of ELECTRA, each corresponding to a unique model size. This implementation provides the same configurations by default, which are described in the table below.
 
| **Model** | **Hidden layers** | **Hidden unit size** | **Parameters** |
|:---------:|:----------:|:---:|:----:|
|ELECTRA_SMALL|12 encoder| 256 | 14M|
|ELECTRA_BASE |12 encoder| 768 |110M|
|ELECTRA_LARGE|24 encoder|1024 |335M|
 
The following features were implemented in this model:
-   General:
  - Mixed precision support with TensorFlow Automatic Mixed Precision (TF-AMP)
  - Multi-GPU support using Horovod
  - XLA support
  
-   Inference:
  - Joint predictions with beam search. The default beam size is 4.
 
 
### Feature support matrix
 
The following features are supported by this model.
 
| **Feature** | **ELECTRA** |
|:---------:|:----------:|
|Automatic mixed precision (AMP)|Yes|
|Horovod Multi-GPU|Yes|
 
 
 
#### Features
 
[AMP](https://nvidia.github.io/apex/amp.html) is an abbreviation used for automatic mixed precision training.
 
 
### Mixed precision training
 
Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format, while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in Volta, and following with both the Turing and Ampere architectures, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using mixed precision training requires two steps:
 
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
 
In this repository, Mixed precision is enabled in TensorFlow by using the Automatic Mixed Precision (TF-AMP) extension which casts variables to half-precision upon retrieval, while storing variables in single-precision format. Furthermore, to preserve small gradient magnitudes in backpropagation, a [loss scaling](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#lossscaling) step must be included when applying gradients. In TensorFlow, loss scaling can be applied statically by using simple multiplication of loss by a constant value or automatically, by TF-AMP. Automatic mixed precision makes all the adjustments internally in TensorFlow, providing two benefits over manual operations. First, programmers need not modify network model code, reducing development and maintenance effort. Second, using AMP maintains forward and backward compatibility with all the APIs for defining and running TensorFlow models.
 
To enable mixed precision, you can simply add the `--amp` to the command-line used to run the model.
 
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
 
The following section lists the requirements that you need to meet in order to start training the ELECTRA model.
 
### Requirements
 
This repository contains Dockerfile which extends the TensorFlow2 NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:
 
-   [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
-   [TensorFlow2 20.06-py3 NGC container or later](https://ngc.nvidia.com/registry/nvidia-tensorflow)
-   Supported GPUs:
    - [NVIDIA Volta architecture](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
    - [NVIDIA Turing architecture](https://www.nvidia.com/en-us/geforce/turing/)
    - [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)
 
For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
-   [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
-   [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/dgx/user-guide/index.html#accessing_registry)
-   [Running TensorFlow2](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html#running)
 
For those unable to use the TensorFlow 2 NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/dgx/support-matrix/index.html).
 
## Quick Start Guide
 
To train your model using mixed precision with Tensor Cores or using FP32, perform the following steps using the default parameters of the ELECTRA model. The default parameters for pretraining have been set to run on 8x A100 40G cards. For the specifics concerning training and inference, see the [Advanced](#advanced) section.
 
1. Clone the repository.
 
`git clone https://github.com/NVIDIA/DeepLearningExamples.git`
 
`cd DeepLearningExamples/TensorFlow2/LanguageModeling/ELECTRA`
 
2. Build ELECTRA on top of the NGC container.
`bash scripts/docker/build.sh`
 
3. Start an interactive session in the NGC container to run training/inference.
`bash scripts/docker/launch.sh`
 
Resultant logs of pretraining and fine-tuning routines are stored in the `results/` folder. Checkpoints are stored in the `checkpoints/`
 
Required data are downloaded in the `data/` directory by default.
 
4. Download and preprocess the dataset.
 
This repository provides scripts to download, verify, and extract the following datasets:
 
-   [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (fine-tuning for question answering)
 
To download, verify, extract the datasets, and create the shards in `.hdf5` format, run:
`/workspace/electra/data/create_datasets_from_start.sh`
 
5. Start fine-tuning with the SQuAD dataset.
 
The above pretrained ELECTRA representations can be fine tuned with just one additional output layer for a state-of-the-art question answering system. Running the following script launches fine-tuning for question answering with the SQuAD dataset.
`bash scripts/run_squad.sh $(source scripts/configs/squad_config.sh && dgxa100_8gpu_amp) train_eval`

More configs for different V100 and A100 hardware setups can be found in `scripts/configs/squad_config.sh`
 
6. Start validation/evaluation.
 
Validation can be performed with the `bash scripts/run_squad.sh $(source scripts/configs/squad_config.sh && dgxa100_8gpu_amp) eval`. Running training first is required to generate needed checkpoints.
 
7. Start inference/predictions.
 
Inference can be performed with the `bash scripts/run_squad.sh $(source scripts/configs/squad_config.sh && dgxa100_8gpu_amp) prediction`. Inference predictions are saved to `<OUTPUT_DIRECTORY>/predictions.json`.
 
## Advanced
 
The following sections provide greater details of the dataset, running training and inference, and the training results.
 
### Scripts and sample code
 
Descriptions of the key scripts and folders are provided below.
 
-   `data/` - Contains scripts for downloading and preparing individual datasets, and will contain downloaded and processed datasets.
-   `scripts/` - Contains shell scripts to launch data download, pre-training, and fine-tuning.
-   `run_squad.sh`  - Interface for launching question answering fine-tuning with `run_squad.py`.
-   `modeling.py` - Implements the ELECTRA pre-training and fine-tuning model architectures with TensorFlow2.
-   `optimization.py` - Implements the Adam optimizer with TensorFlow2.
-   `run_squad.py` - Implements fine tuning training and evaluation for question answering on the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset.
 
 
 
### Parameters
 
#### Fine tuning parameters
 
Default arguments are listed below in the order the scripts expects:
 
-   ELECTRA MODEL - The default is `"google/electra-base-discriminator"`.
-   Number of training Epochs - The default is `2`.
-   Batch size - The default is `16`.
-   Learning rate - The default is `4e-4`.
-   Precision (either `amp` or `fp32`) - The default is `amp`.
-   Number of GPUs - The default is `8`.
-   Seed - The default is `1`.
-   SQuAD version - The default is `1.1`
-   SQuAD directory -  The default is `/workspace/electra/data/download/squad/v$SQUAD_VERSION`.
-   Output directory for result - The default is `results/`.
-   Initialize checkpoint - The default is `"None"`
-   Mode (`train`, `eval`, `train_eval`, `prediction`) - The default is `train_eval`.
 
The script saves the checkpoint at the end of each epoch to the `checkpoints/` folder.
 
 
 
The main script `run_tf_squad.py` specific parameters are:
 
```
 --electra_model ELECTRA_MODEL     - Specifies the type of ELECTRA model to use;
                                     should be one of the following:
     google/electra-small-generator
     google/electra-base-generator
     google/electra-large-generator
     google/electra-small-discriminator
     google/electra-base-discriminator
     google/electra-large-discriminator
 
 --data_dir DATA_DIR          - Path to the SQuAD json for training and evaluation.
 
 --max_seq_length MAX_SEQ_LENGTH
                              - The maximum total input sequence length
                                after WordPiece tokenization.
                                Sequences longer than this will be truncated,
                                and sequences shorter than this will be padded.
 
 --doc_stride DOC_STRIDE      - When splitting up a long document into chunks
                                this parameters sets how much stride to take
                                between chunks of tokens.
 
 --max_query_length MAX_QUERY_LENGTH
                              - The maximum number of tokens for the question.
                                Questions longer than <max_query_length>
                                will be truncated to the value specified.
 
 --n_best_size N_BEST_SIZE       - The total number of n-best predictions to
                                generate in the nbest_predictions.json
                                output file.
 
 --max_answer_length MAX_ANSWER_LENGTH
                              - The maximum length of an answer that can be
                                generated. This is needed because the start and
                                end predictions are not conditioned on one another.
    
 --joint_head <True|False>    - If true, beam search will be used to jointly predict
                                the start end end positions. Default is True.
 
 --beam_size BEAM_SIZE        - The beam size used to do joint predictions. 
 
 --verbose_logging            - If true, all the warnings related to data
                                processing will be printed. A number of warnings
                                are expected for a normal SQuAD evaluation.
 
 --do_lower_case              - Whether to lower case the input text. Set to
                                true for uncased models and false for cased models.
 
 --version_2_with_negative       - If true, the SQuAD examples contain questions
                                that do not have an answer.
 
 --null_score_diff_threshold NULL_SCORE_DIFF_THRES HOLD
                              - A null answer will be predicted if null_score
                                is greater than NULL_SCORE_DIFF_THRESHOLD.
```
 
### Command-line options
 
To see the full list of available options and their descriptions, use the `-h` or `--help` command line option, for example:
 
`python run_tf_squad.py --help`
 
Detailed descriptions of command-line options can be found in the [Parameters](#parameters) section.
 
### Getting the data
 
For fine-tuning a pre-trained ELECTRA model for specific tasks, by default this repository prepares the following dataset:
 
-   [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/): for question answering
 
 
### Training process
 
The training process consists of two steps: pre-training and fine-tuning.
 
#### Fine-tuning
 
Fine-tuning is provided for a variety of tasks. The following tasks are included with this repository through the following scripts:
 
-   Question Answering (`scripts/run_squad.sh`)
 
By default, each Python script implements fine-tuning a pre-trained ELECTRA model for a specified number of training epochs as well as evaluation of the fine-tuned model. Each shell script invokes the associated Python script with the following default parameters:
 
-   Uses 8 GPUs
-   Has FP16 precision enabled
-   HAS XLA enabled
-   Saves a checkpoint at the end of training to the `checkpoints/` folder
 
Fine-tuning Python scripts implement support for mixed precision and multi-GPU training through [Horovod](https://github.com/horovod/horovod). For a full list of parameters and associated explanations, see the [Parameters](#parameters) section.
 
All fine-tuning shell scripts have the same positional arguments, outlined below:
 
```bash scripts/run_squad.sh <pretrained electra model> <epochs> <batch size> <learning rate> <amp|fp32> <num_gpus> <seed> <SQuAD version> <path to SQuAD dataset> <results directory> <checkpoint_to_load> <mode (either `train`, `eval` or `train_eval`)>```
 
By default, the mode positional argument is set to train_eval. See the [Quick Start Guide](#quick-start-guide) for explanations of each positional argument.
 
Note: The first positional argument (the path to the checkpoint to load) is required.
 
Each fine-tuning script assumes that the corresponding dataset files exist in the `data/` directory or separate path can be a command-line input to `run_squad.sh`.
 
### Inference process
 
#### Fine-tuning inference
 
Evaluation fine-tuning is enabled by the same scripts as training:
 
-   Question Answering (`scripts/run_squad.sh`)
 
The mode positional argument of the shell script is used to run in evaluation mode. The fine-tuned ELECTRA model will be run on the evaluation dataset, and the evaluation loss and accuracy will be displayed.
 
Each inference shell script expects dataset files to exist in the same locations as the corresponding training scripts. The inference scripts can be run with default settings. By setting the `mode` variable in the script to either `eval` or `prediction` flag, you can choose between running predictions and evaluating them on a given dataset or just the former.
 
`bash scripts/run_squad.sh <pretrained electra model> <epochs> <batch size> <learning rate> <amp|fp32> <num_gpus> <seed> <SQuAD version> <path to SQuAD dataset> <results directory> <path to fine-tuned model checkpoint> <eval or prediction>`
 
To run inference interactively on question-context pairs, use the script `run_inference.py` as follows:
 
`python run_inference.py --electra_model <electra_model_type> --init_checkpoint <fine_tuned_checkpoint>  --question="What food does Harry like?" --context="My name is Harry and I grew up in Canada. I love apples."`
 
 
## Performance
 
### Benchmarking
 
The following section shows how to run benchmarks measuring the model performance in training and inference modes.
 
#### Training performance benchmark
 
Training performance benchmarks for fine-tuning can be obtained by running `scripts/benchmark.sh`. The required parameters can be passed through the command-line as described in [Training process](#training-process). The performance information is printed at the end of each epoch.
 
To benchmark the training performance on a specific batch size, run:
`bash scripts/benchmark.sh train <num_gpus> <batch size> <infer_batch_size> <amp|fp32> <SQuAD version> <path to SQuAD dataset> <results directory> <checkpoint_to_load> <cache_Dir>`
 
An example call used to generate throughput numbers:
`bash scripts/benchmark.sh train 8 16`
 
#### Inference performance benchmark
 
Inference performance benchmarks fine-tuning can be obtained by running `scripts/benchmark.sh`. The required parameters can be passed through the command-line as described in [Inference process](#inference-process). This script runs one epoch by default on the SQuAD v1.1 dataset and extracts the averaged performance for the given configuration. 
 
To benchmark the training performance on a specific batch size, run:
`bash scripts/benchmark.sh train <num_gpus> <batch size> <infer_batch_size> <amp|fp32> <SQuAD version> <path to SQuAD dataset> <results directory> <checkpoint_to_load> <cache_Dir>`
 
An example call used to generate throughput numbers:
`bash scripts/benchmark.sh eval 8 256`
  
 
### Results
 
The following sections provide details on how we achieved our performance and accuracy in training and inference. All results are on ELECTRA-base model and on SQuAD v1.1 dataset with a sequence length of 384 unless otherwise mentioned.
 
#### Training accuracy results
 
##### Fine-tuning accuracy: NVIDIA DGX A100 (8x A100 40GB)
Our results were obtained by running the `scripts/run_squad.sh` training script in the tensorflow:20.06-tf2-py3 NGC container on NVIDIA DGX A100 (8x A100 40GB) GPUs.
 
| GPUs    | Batch size / GPU    | Accuracy / F1 - FP32  | Accuracy / F1 - mixed precision  |   Time to train - TF32 (sec) |  Time to train - mixed precision (sec) | Time to train speedup (FP32 to mixed precision) | 
|---------|---------------------|------------------|-----------------------------|--------------------------|---------------------------------|-------------------------------------------------|
|   1   |       32            |           87.19 / 92.85       |              87.19 / 92.84               |          1699                |               749                  |                    2.27         |
|   8   |       32            |           86.84 / 92.57      |                86.83 / 92.56            |          263                |               201                  |                    1.30         |
 
 
##### Fine-tuning accuracy: NVIDIA DGX-1 (8x V100 16GB)
Our results were obtained by running the `scripts/run_squad.sh` training script in the tensorflow:20.06-tf2-py3 NGC container on NVIDIA DGX-1 with (8x V100 16GB) GPUs.
 
| GPUs    | Batch size / GPU (FP32 : mixed precision)   | Accuracy / F1 - FP32  | Accuracy / F1 - mixed precision  |   Time to train - FP32 (sec) |  Time to train - mixed precision (sec) | Time to train speedup (FP32 to mixed precision) |       
|---------|---------------------|------------------|-----------------------------|--------------------------|---------------------------------|-------------------------------------------------|
|   1   |          8 : 16           |         87.36 / 92.82        |             87.32 / 92.74              |              5136            |                  1378               |          3.73                 |
|   8   |          8 : 16           |         87.02 / 92.73       |             87.02 / 92.72              |              730            |                  334               |          2.18                 |
 
##### Fine-tuning accuracy: NVIDIA DGX-2 (16x V100 32GB)
Our results were obtained by running the `scripts/run_squad.sh` training script in the tensorflow:20.06-tf2-py3 NGC container on NVIDIA DGX-2 (16x V100 32G) GPUs.
 
| GPUs    | Batch size / GPU    | Accuracy / F1 - FP32  | Accuracy / F1 - mixed precision  |   Time to train - FP32 (sec) |  Time to train - mixed precision (sec) | Time to train speedup (FP32 to mixed precision) |       
|---------|---------------------|------------------|-----------------------------|--------------------------|---------------------------------|-------------------------------------------------|
|   1   |          32           |         87.14 / 92.69        |             86.95 / 92.69              |              4478            |                  1162               |          3.85                 |
|   16   |          32           |         86.95 / 90.58         |             86.93 / 92.48               |              333            |                  229               |          1.45                 |
 
 
 
 
##### Training stability test
 
###### Fine-tuning stability test: NVIDIA DGX-1 (8x V100 16GB)
 
Training stability with 8 GPUs, FP16 computations, batch size of 16 on SQuAD v1.1:
  
| Accuracy Metric | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 | Mean | Standard Deviation
|---|---|---|---|---|---|---|---
|Exact Match %| 86.99 | 86.81 | 86.95 | 87.10 | 87.26 | 87.02 | 0.17
| f1 % | 92.7 | 92.66 | 92.65 |  92.61 | 92.97 | 92.72 | 0.14
 
 Training stability with 8 GPUs, FP16 computations, batch size of 16 on SQuAD v2.0:
 
| Accuracy Metric | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 | Mean | Standard Deviation
|---|---|---|---|---|---|---|---
|Exact Match %| 83.00 | 82.84 | 83.11 | 82.70 | 82.94 | 82.91 | 0.15
| f1 % | 85.63 | 85.48 | 85.69 | 85.31 | 85.57 | 85.54 | 0.15
 
 
#### Training performance results
 
##### Training performance: NVIDIA DGX A100 (8x A100 40GB)
 
Our results were obtained by running the `scripts/benchmark.sh` training script in the tensorflow:20.06-tf2-py3 NGC container on NVIDIA DGX A100 (8x A100 40GB) GPUs. Performance numbers (in items/images per second) were averaged over an entire training epoch.
 
###### Fine-tuning NVIDIA DGX A100 (8x A100 40GB)
  
| GPUs | Batch size / GPU | Throughput - FP32 (sequences/sec) | Throughput - mixed precision (sequences/sec) | Throughput speedup (TF32 - mixed precision) | Weak scaling - TF32 | Weak scaling - mixed precision |
|------------------|----------------------|-----------------------------------------------|------------------------------------|---------------------------------|----------------------|----------------------------------------------
| 1 |  32 |  104    | 285 |  2.73  |  1.00  | 1.00
| 4 |  32 |  405    | 962 |  2.37  |  3.88  | 3.37
| 8 |  32 |   809   | 1960|  2.42  |  7.75  | 6.87
 
 
 
##### Training performance: NVIDIA DGX-1 (8x V100 16GB)
 
Our results were obtained by running the `scripts/benchmark.sh` training scripts in the tensorflow:20.06-tf2-py3 NGC container on NVIDIA DGX-1 with (8x V100 32GB) GPUs. Performance numbers (in sequences per second) were averaged over an entire training epoch.
 
 
###### Fine-tuning NVIDIA DGX-1 (8x V100 16GB)
 
| GPUs | Batch size / GPU (FP32 : mixed precision)   | Throughput - FP32 (sequences/sec) | Throughput - mixed precision (sequences/sec) | Throughput speedup (FP32 - mixed precision) | Weak scaling - FP32 | Weak scaling - mixed precision |
|------------------|----------------------|-----------------------------------------------|------------------------------------|---------------------------------|----------------------|----------------------------------------------
|1 | 8 : 16|  35| 144| 4.11| 1.00| 1.00
|4 | 8 : 16| 133| 508| 3.81| 3.80| 3.52
|8 | 8 : 16| 263| 965| 3.67| 7.51| 6.70
 
To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).
 
##### Training performance: NVIDIA DGX-2 (16x V100 32GB)
 
Our results were obtained by running the `scripts/benchmark.sh` training scripts in the tensorflow:20.06-tf2-py3 NGC container on NVIDIA DGX-2 with (16x V100 32G) GPUs. Performance numbers (in sequences per second) were averaged over an entire training epoch.
 
###### Fine-tuning NVIDIA DGX-2 With 32GB
 
| GPUs | Batch size / GPU | Throughput - FP32 (sequences/sec) | Throughput - mixed precision (sequences/sec) | Throughput speedup (FP32 - mixed precision) | Weak scaling - FP32 | Weak scaling - mixed precision |
|------|------------------|----------------------------------|---------------------------------------------|---------------------------------------------|---------------------|--------------------------------|
|    1 |               16 |                               40 |                                         173 |                                        4.33 |                1.00 |                           1.00 |
|    4 |               16 |                              157 |                                         625 |                                        3.98 |                3.93 |                           3.61 |
|    8 |               16 |                              311 |                                        1209 |                                        3.89 |                7.78 |                           6.99 |
|   16 |               16 |                              611 |                                        2288 |                                        3.74 |               15.28 |                          13.23 |
 
To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).
 
#### Inference performance results
 
##### Inference performance: NVIDIA DGX A100 (1x A100 40GB)
 
Our results were obtained by running the `scripts/benchmark.sh` inferencing benchmarking script in the tensorflow:20.06-tf2-py3 NGC container on NVIDIA DGX A100 (1x A100 40GB) GPU.
 
###### Fine-tuning inference on NVIDIA DGX A100 (1x A100 40GB)
 
FP16
 
| Batch size | Sequence length | Throughput Avg (sequences/sec) | Latency Avg (ms) | Latency 90% (ms) | Latency 95% (ms) | Latency 99% (ms) |
|------------|-----------------|--------------------------------|------------------|------------------|------------------|------------------|
|          1 |             384 |                            178 |            5.630 |            5.500 |            5.555 |            5.608 |
|        256 |             384 |                            857 |            1.112 |            1.111 |            1.111 |            1.112 |
|        512 |             384 |                            864 |            1.054 |            1.051 |            1.053 |            1.053 |
 
TF32
 
| Batch size | Sequence length | Throughput Avg (sequences/sec) | Latency Avg (ms) | Latency 90% (ms) | Latency 95% (ms) | Latency 99% (ms) |
|------------|-----------------|--------------------------------|------------------|------------------|------------------|------------------|
|          1 |             384 |                            123 |            8.186 |            7.995 |            8.078 |            8.152 |
|        256 |             384 |                            344 |            2.832 |            2.822 |            2.826 |            2.830 |
|        512 |             384 |                            351 |            2.787 |            2.781 |            2.784 |            2.784 |
 
 
 
##### Inference performance: NVIDIA DGX-1 (1x V100 16GB)
 
Our results were obtained by running the `scripts/benchmark.sh` script in the tensorflow:20.06-tf2-py3 NGC container on NVIDIA DGX-1 with (1x V100 16G) GPUs.
 
 
###### Fine-tuning inference on NVIDIA DGX-1 with 16GB
 
FP16
 
| Batch size | Sequence length | Throughput Avg (sequences/sec) | Latency Avg (ms) | Latency 90% (ms) | Latency 95% (ms) | Latency 99% (ms) |
|------------|-----------------|--------------------------------|------------------|------------------|------------------|------------------|
|          1 |             384 |                            141 |            7.100 |            7.071 |            7.081 |            7.091 |
|        128 |             384 |                            517 |            1.933 |            1.930 |            1.930 |            1.932 |
|        256 |             384 |                            524 |            1.910 |            1.907 |            1.908 |            1.909 |
 
 
FP32
 
| Batch size | Sequence length | Throughput Avg (sequences/sec) | Latency Avg (ms) | Latency 90% (ms) | Latency 95% (ms) | Latency 99% (ms) |
|------------|-----------------|--------------------------------|------------------|------------------|------------------|------------------|
|          1 |             384 |                             84 |           11.869 |           11.814 |           11.832 |           11.850 |
|        128 |             384 |                            117 |            8.548 |            8.527 |            8.529 |            8.537 |
|        256 |             384 |                            141 |            7.100 |            7.071 |            7.081 |            7.091 |
 
 
##### Inference performance: NVIDIA DGX-2 (1x V100 32GB)
 
Our results were obtained by running the `scripts/benchmark.sh` scripts in the tensorflow:20.06-tf2-py3 NGC container on NVIDIA DGX-2 with (1x V100 32G) GPUs.
 
 
###### Fine-tuning inference on NVIDIA DGX-2 with 32GB
  
FP16
 
| Batch size | Sequence length | Throughput Avg (sequences/sec) | Latency Avg (ms) | Latency 90% (ms) | Latency 95% (ms) | Latency 99% (ms) |
|------------|-----------------|--------------------------------|------------------|------------------|------------------|------------------|
|          1 |             384 |                            144 |            6.953 |            6.888 |            6.910 |            6.932 |
|        128 |             384 |                            547 |            1.828 |            1.827 |            1.827 |            1.828 |
|        256 |             384 |                            557 |            1.795 |            1.792 |            1.793 |            1.794 |
 
FP32
 
| Batch size | Sequence length | Throughput Avg (sequences/sec) | Latency Avg (ms) | Latency 90% (ms) | Latency 95% (ms) | Latency 99% (ms) |
|------------|-----------------|--------------------------------|------------------|------------------|------------------|------------------|
|          1 |             384 |                             86 |           11.580 |           11.515 |           11.535 |           11.558 |
|        128 |             384 |                            124 |            8.056 |             8.05 |            8.052 |            8.055 |
|        256 |             384 |                            125 |            8.006 |            8.002 |            8.004 |            8.005 |
 
 
To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).
  
## Release notes
 
### Changelog
 
July 2020
- Initial release.
 
### Known issues
 
There are no known issues with this model.

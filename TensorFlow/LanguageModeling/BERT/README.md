# BERT For TensorFlow

This repository provides a script and recipe to train BERT to achieve state of the art accuracy, and is tested and maintained by NVIDIA.


## Table Of Contents:
* [The model](#the-model)
  * [Default configuration](#default-configuration)
* [Setup](#setup)
  * [Requirements](#requirements)
* [Quick start guide](#quick-start-guide)
* [Details](#details)
  * [Command line options](#command-line-options)
  * [Getting the data](#getting-the-data)
  * [Training process](#training-process)
  * [Pre-training](#pre-training)
  * [Fine tuning](#fine-tuning)
  * [Enabling mixed precision](#enabling-mixed-precision)
  * [Inference process](#inference-process)
* [Benchmarking](#benchmarking)
  * [Training performance benchmark](#training-performance-benchmark)
  * [Inference performance benchmark](#inference-performance-benchmark)
* [Results](#results)
  * [Training accuracy results](#training-accuracy-results)
  * [Training stability test](#training-stability-test)
  * [Training performance results](#training-performance-results)
      * [NVIDIA DGX-1 (8x V100 16G)](#nvidia-dgx-1-8x-v100-16g)
      * [NVIDIA DGX-1 (8x V100 32G)](#nvidia-dgx-1-8x-v100-32g)
      * [NVIDIA DGX-2 (16x V100 32G)](#nvidia-dgx-2-16x-v100-32g)
  * [Inference performance results](#inference-performance-results)
      * [NVIDIA DGX-1 16G (1x V100 16G)](#nvidia-dgx-1-16g-1x-v100-16g)
      * [NVIDIA DGX-1 32G (1x V100 32G)](#nvidia-dgx-1-32g-1x-v100-32g)
      * [NVIDIA DGX-2 32G (1x V100 32G)](#nvidia-dgx-2-32g-1x-v100-32g)
* [Changelog](#changelog)
* [Known issues](#known-issues)

## The model

BERT, or Bidirectional Encoder Representations from Transformers, is a new method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks. This model is based on [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) paper. NVIDIA's BERT 19.03 is an optimized version of [Google's official implementation](https://github.com/google-research/bert), leveraging mixed precision arithmetic and tensor cores on V100 GPUS for faster training times while maintaining target accuracy.


The repository also contains scripts to interactively launch data download, training, benchmarking and inference routines in a Docker container for both pretraining and fine tuning for Question Answering. The major differences between the official implementation of the paper and our version of BERT are as follows:
- Mixed precision support with TensorFlow Automatic Mixed Precision (TF-AMP), which enables mixed precision training without any changes to the code-base by performing automatic graph rewrites and loss scaling controlled by an environmental variable.
- Scripts to download dataset for 
    - Pretraining - [Wikipedia](https://dumps.wikimedia.org/),  [BooksCorpus](http://yknzhu.wixsite.com/mbweb)
    - Fine Tuning - [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (Stanford Question Answering Dataset), Pretrained Weights from Google
- Custom fused CUDA kernels for faster computations
- Multi-GPU/Multi-Node support using Horovod


The following performance optimizations were implemented in this model:
- [XLA](https://www.tensorflow.org/xla) support (experimental).


These techniques and optimizations improve model performance and reduce training time, allowing you to perform various NLP tasks with no additional effort.


Other publicly available implementations of BERT include:
1. [Hugging Face](https://github.com/huggingface/pytorch-pretrained-BERT)
2. [codertimo](https://github.com/codertimo/BERT-pytorch)
3. [gluon-nlp](https://github.com/dmlc/gluon-nlp/tree/master/scripts/bert)


This model trains with mixed precision tensor cores on Volta, therefore researchers can get results much faster than training without tensor cores. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.

### Default configuration

BERT's model architecture is a multi-layer bidirectional Transformer encoder. Based on the model size, we have the following two default configurations of BERT.

| **Model** | **Hidden layers** | **Hidden unit size** | **Attention heads** | **Feedforward filter size** | **Max sequence length** | **Parameters** |
|:---------:|:----------:|:----:|:---:|:--------:|:---:|:----:|
|BERTBASE |12 encoder| 768| 12|4 x  768|512|110M|
|BERTLARGE|24 encoder|1024| 16|4 x 1024|512|330M|

## Setup
The following section list the requirements in order to start training the BERT model.

### Requirements
This repository contains `Dockerfile` which extends the TensorFlow NGC container and encapsulates some dependencies.  Aside from these dependencies, ensure you have the following components:
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [TensorFlow 19.03-py3](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) NGC container
- [NVIDIA Volta based GPU](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)


For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
- [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
- [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/dgx/user-guide/index.html#accessing_registry)
- [Running TensorFlow](https://docs.nvidia.com/deeplearning/dgx/tensorflow-release-notes/running.html#running)

## Quick start guide
To pretrain or fine tune your model for Question Answering using mixed precision with tensor cores or using FP32, perform the following steps using the default parameters of the BERT model.

### 1. Clone the repository.

```bash
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/TensorFlow/LanguageModeling/BERT
```

### 2. Build the BERT TensorFlow NGC container.

```bash
bash scripts/docker/build.sh
```

### 3. Download and preprocess the dataset.
This repository provides scripts to download, verify and extract the SQuaD dataset+pretrained weights for fine tuning as well as Wikipedia + BookCorpus dataset for pretraining.

To download, verify, and extract required datasets: 

```bash
bash scripts/data_download.sh  
```

The script launches a docker container with current directory mounted and downloads datasets to `data/` folder on the host. 

### 4. Start an interactive session in the NGC container to run training/inference.
After you build the container image and download the data, you can start an interactive CLI session as follows:  

```bash
bash scripts/docker/launch.sh
```

The `launch.sh` script assumes that the datasets are in the following locations by default after downloading data. 
- SQuAD v1.1 - `data/squad/v1.1`
- BERT - `data/pretrained_models_google/uncased_L-24_H-1024_A-16`
- Wikipedia - `data/wikipedia_corpus/final_tfrecords_sharded`
- BooksCorpus -  `data/bookcorpus/final_tfrecords_sharded`


### 5. Start pre-training.
BERT is designed to pre-train deep bidirectional representations for language representations. The following scripts are to replicate pretraining on Wikipedia+Books Corpus from the [paper](https://arxiv.org/pdf/1810.04805.pdf). These scripts are general and can be used for pretraining language representations on any corpus of choice.

From within the container, you can use the following script to run pre-training.
```bash
bash scripts/run_pretraining.sh <train_batch_size_per_gpu> <eval_batch_size> <learning_rate_per_gpu> <precision> <num_gpus> <warmup_steps> <train_steps> <save_checkpoint_steps> <create_logfile>
```

For FP16 training with XLA using a DGX-1 V100 32G, run:
```bash
bash scripts/run_pretraining.sh 14 8 5e-5 fp16_xla 8 5000 2285000 5000 true
```

For FP32 training without XLA using a DGX-1 V100 32G, run:
```bash
bash scripts/run_pretraining.sh 6 6 2e-5 fp32 8 2000 5333333 5000 true
```

### 6. Start fine tuning.
The above pretrained BERT representations can be fine tuned with just one additional output layer for a state-of-the-art Question Answering system. From within the container, you can use the following script to run fine-training for SQuaD.

```bash
bash scripts/run_squad.sh <batch_size_per_gpu> <learning_rate_per_gpu> <precision> <use_xla> <num_gpus> <checkpoint>
```

For FP16 training with XLA using a DGX-1 V100 32G, run:
```bash
bash scripts/run_squad.sh 10 5e-6 fp16 true 8 data/pretrained_models_google/uncased_L-24_H-1024_A-16/bert_model.ckpt
```

For FP32 training without XLA using a DGX-1 V100 32G, run:
```bash
bash scripts/run_squad.sh 5 5e-6 fp32 false 8 data/pretrained_models_google/uncased_L-24_H-1024_A-16/bert_model.ckpt
```

### 7. Start validation/evaluation.
The `run_squad_inference.sh` script runs inference on a checkpoint fine tuned for SQuaD and evaluates the goodness of predictions on the basis of exact match and F1 score.

```bash
bash scripts/run_squad_inference.sh <init_checkpoint> <batch_size> <precision> <use_xla>
```

For FP16 inference with XLA using a DGX-1 V100 32G, run:
```bash
bash scripts/run_squad_inference.sh /results/model.ckpt 8 fp16 true 
```

For FP32 inference without XLA using a DGX-1 V100 32G, run:
```bash
bash scripts/run_squad_inference.sh /results/model.ckpt 8 fp32 false
```

## Details
The following sections provide greater details of the dataset, running training and inference, and the training results.

### Command line options

The `run_squad.sh` script calls the `run_squad.py` file and the `run_pretraining.sh` script calls the `run_pretraining.py` file with a set of options. To see the full list of available options and their descriptions, use the -h or --help command line option with the python file, for example:

```bash
python run_pretraining.py --help
python run_squad.py --help 
```

Aside from options to set hyperparameters, the relevant options to control the behaviour of the `run_pretraining.py` script are: 
```bash
  --[no]amp: Whether to enable AMP ops.(default: 'false')
  --bert_config_file: The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.
  --[no]do_eval: Whether to run evaluation on the dev set.(default: 'false')
  --[no]do_train: Whether to run training.(evaluation: 'false')
  --eval_batch_size: Total batch size for eval.(default: '8')(an integer)
  --[no]horovod: Whether to use Horovod for multi-gpu runs(default: 'false')
  --init_checkpoint: Initial checkpoint (usually from a pre-trained BERT model).
  --input_file: Input TF example files (can be a glob or comma separated).
  --iterations_per_loop: How many steps to make in each estimator call.(default: '1000')
```

Aside from options to set hyperparameters, some relevant options to control the behaviour of the run_squad.py script are: 
```bash
  --bert_config_file: The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.
  --[no]do_predict: Whether to run evaluation on the dev set. (default: 'false')
  --[no]do_train: Whether to run training. (default: 'false')
  --learning_rate: The initial learning rate for Adam.(default: '5e-06')(a number)
  --max_answer_length: The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.(default: '30')(an integer)
  --max_query_length: The maximum number of tokens for the question. Questions longer than this will be truncated to this length.(default: '64')(an integer)
  --max_seq_length: The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.(default: '384')(an integer)
  --predict_batch_size: Total batch size for predictions.(default: '8')(an integer)
  --train_batch_size: Total batch size for training.(default: '8')(an integer)
  --[no]use_fp16: Whether to use fp32 or fp16 arithmetic on GPU.(default: 'false')
  --[no]use_xla: Whether to enable XLA JIT compilation.(default: 'false')
  --[no]verbose_logging: If true, all of the warnings related to data processing will be printed. A number of warnings are expected for a normal SQuAD evaluation.(default: 'false')
  --[no]version_2_with_negative: If true, the SQuAD examples contain some that do not have an answer.(default: 'false')
```

### Getting the data
For pre-training BERT, we use the concatenation of Wikipedia (2500M words) as well as Books Corpus (800M words). For Wikipedia, we extract only the text passages from [here](ftp://ftpmirror.your.org/pub/wikimedia/dumps/enwiki/20190301/enwiki-20190301-pages-articles-multistream.xml.bz2) and ignore headers list and tables. It is structured as a document level corpus rather than a shuffled sentence level corpus because it is critical to extract long contiguous sentences. The next step is to run `create_pretraining_data.py` with the document level corpus as input, which generates input data and labels for the masked language modeling and next sentence prediction tasks. Pre-training can also be performed on any corpus of your choice. The collection of data generation scripts are intended to be modular to allow modifications for additional preprocessing steps or to use additional data.

We can use a pre-trained BERT model for other fine tuning tasks like Question Answering. We use SQuaD for this task. SQuaD v1.1 has 100,000+ question-answer pairs on 500+ articles. SQuaD v2.0 combines v1.1 with an additional 50,000 new unanswerable questions and must not only answer questions but also determine when that is not possible. 

### Training process
The training process consists of two steps: pre-training and fine tuning.

#### Pre-training
Pre-training is performed using the `run_pretraining.py` script along with parameters defined in the `scripts/run_pretraining.sh`.


The `run_pretraining.sh` script runs a job on a single node  that trains the BERT-large model from scratch using the Wikipedia and Book corpus datasets as training data. By default, the training script:
- Runs on 8 GPUs with training batch size of 14 and evaluation batch size of 8 per GPU.
- Has FP16 precision enabled.
- Is XLA enabled.
- Runs for 1144000 steps with 10000 warm-up steps.
- Saves a checkpoint every 5000 iterations (keeps only the latest checkpoint) and at the end of training. All checkpoints, evaluation results and training logs are saved to the `/results` directory (in the container which can be mounted to a local directory).
- Creates the log file containing all the output.
- Evaluates the model at the end of training. To skip evaluation, modify `--do_eval` to `False`.

These parameters will train Wikipedia + BooksCorpus to reasonable accuracy on a DGX1 with 32GB V100 cards. If you want to match google’s best results from the BERT paper, you should either train for twice as many steps (2,288,000 steps) on a DGX1, or train on 16 GPUs on a DGX2. The DGX2 having 16 GPUs will be able to fit a batch size twice as large as a DGX1 (224 vs 112), hence the DGX2 can finish in half as many steps. 


For example:
```bash
run_pretraining.sh <training_batch_size> <eval_batch_size> <learning-rate> <precision> <num_gpus> <warmup_steps> <training_steps> <save_checkpoint_steps> <create_logfile>
```

Where:
- <training_batch_size> is per-gpu batch size used for training. Batch size varies with <precision>, larger batch sizes run more efficiently, but require more memory.

- <eval_batch_size> per-gpu batch size used for evaluation after training.<learning_rate> Default rate of 1e-4 is good for global batch size 256.

- <precision> Type of math in your model, can be either fp32, or amp. The options mean:

    - fp32 32 bit IEEE single precision floats.

    - amp Automatic rewrite of TensorFlow compute graph to take advantage of 16 bit arithmetic whenever that is safe.

- <num_gpus> Number of GPUs to use for training. Must be equal to or smaller than the number of GPUs attached to your node.

- <warmup_steps> Number of warm-up steps at the start of training.

- <training_steps> Total number of training steps.

- <save_checkpoint_steps> Controls how often checkpoints are saved. Default is 5000 steps.

- <create_logfile> Flag indicating if output should be written to a logfile or not (acceptable values are ‘true’ or ‘false’, true indicates output should be saved to a logfile.)


For example:
```bash
bert_tf/scripts/run_pretraining.sh 14 8 1e-4 fp16_xla 16 10000 1144000 5000 true
```

Trains BERT-large from scratch on a single DGX-2 using FP16 arithmetic. This will take around 156 hours / 6.5 days. Checkpoints are written out every 5000 steps and all printouts are saved to a logfile.

#### Fine tuning
Fine tuning is performed using the `run_squad.py` script along with parameters defined in `scripts/run_squad.sh`.

The `run_squad.sh` script trains a model and performs evaluation on the SQuaD v1.1 dataset. By default, the training script: 
- Uses 8 GPUs and batch size of 10 on each GPU.
- Has FP16 precision enabled.
- Is XLA enabled.
- Runs for 2 epochs.
- Saves a checkpoint every 1000 iterations (keeps only the latest checkpoint) and at the end of training. All checkpoints, evaluation results and training logs are saved to the `/results` directory (in the container which can be mounted to a local directory).
- Evaluation is done at the end of training. To skip evaluation, modify `--do_predict` to `False`.

This script outputs checkpoints to the `/results` directory, by default, inside the container. Mount point of `/results` can be changed in the `scripts/docker/launch.sh` file. The training log contains information about:
- Loss for the final step
- Training and evaluation performance
- F1 and exact match score on the Dev Set of SQuaD after evaluation. 

The summary after training is printed in the following format:
```bash
I0312 23:10:45.137036 140287431493376 run_squad.py:1332] 0 Total Training Time = 3007.00 Training Time W/O start up overhead = 2855.92 Sentences processed = 175176
I0312 23:10:45.137243 140287431493376 run_squad.py:1333] 0 Training Performance = 61.3378 sentences/sec
I0312 23:14:00.550846 140287431493376 run_squad.py:1396] 0 Total Inference Time = 145.46 Inference Time W/O start up overhead = 131.86 Sentences processed = 10840
I0312 23:14:00.550973 140287431493376 run_squad.py:1397] 0 Inference Performance = 82.2095 sentences/sec
{"exact_match": 83.69914853358561, "f1": 90.8477003317459}
```

Multi-gpu training is enabled with the Horovod TensorFlow module. The following example runs training on 8 GPUs: 
```bash
mpi_command="mpirun -np 8 -H localhost:8 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH -mca pml ob1 -mca btl ^openib" \
     python run_squad.py --horovod
```

### Enabling mixed precision
[Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format, while storing minimal information in single-precision to retain as much information as possible in critical parts of the network.  Since the introduction of [tensor cores](https://developer.nvidia.com/tensor-cores) in the Volta and Turing architectures, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures.  Using [mixed precision training](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html) previously required two steps:
1. Porting the model to use the FP16 data type where appropriate.
2. Manually adding loss scaling to preserve small gradient values. 
This can now be achieved using Automatic Mixed Precision (AMP) for TensorFlow to enable the full [mixed precision methodology](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#tensorflow) in your existing TensorFlow model code.  AMP enables mixed precision training on Volta and Turing GPUs automatically. The TensorFlow framework code makes all necessary model changes internally.

In TF-AMP, the computational graph is optimized to use as few casts as necessary and maximize the use of FP16, and the loss scaling is automatically applied inside of supported optimizers. AMP can be configured to work with the existing `tf.contrib` loss scaling manager by disabling the AMP scaling with a single environment variable to perform only the automatic mixed-precision optimization. It accomplishes this by automatically rewriting all computation graphs with the necessary operations to enable mixed precision training and automatic loss scaling.

For information about:
- How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html) documentation.
- How to access and enable AMP for TensorFlow, see [Using TF-AMP](https://docs.nvidia.com/deeplearning/dgx/tensorflow-user-guide/index.html#tfamp) from the TensorFlow User Guide.
- Techniques used for mixed precision training, see the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.

### Inference process
Inference on a fine tuned Question Answering system is performed using the `run_squad.py` script along with parameters defined in the `scripts/run_squad_inference.sh`. Inference is supported on single GPU at this moment.

The `run_squad_inference.sh` script trains a model and performs evaluation on the SQuaD v1.1 dataset. By default, the inferencing script: 
- Has FP16 precision enabled
- Is XLA enabled
- Evaluates the latest checkpoint present in `/results` with a batch size of 8

This script outputs predictions file to `/results/predictions.json` and computes F1 score and exact match score using SQuaD's `evaluate-v1.1.py`. Mount point of `/results` can be changed in the `scripts/docker/launch.sh` file. 

The output log contains information about:
- Evaluation performance
- F1 and exact match score on the Dev Set of SQuaD after evaluation. 

The summary after inference is printed in the following format:
```bash
I0312 23:14:00.550846 140287431493376 run_squad.py:1396] 0 Total Inference Time = 145.46 Inference Time W/O start up overhead = 131.86 Sentences processed = 10840
I0312 23:14:00.550973 140287431493376 run_squad.py:1397] 0 Inference Performance = 82.2095 sentences/sec
{"exact_match": 83.69914853358561, "f1": 90.8477003317459}
```

## Benchmarking
The following section shows how to run benchmarks measuring the model performance in training and inference modes.

Benchmarking can be performed for both training and inference. Both scripts run the BERT model for fine tuning. You can specify whether benchmarking is performed in FP16 or FP32 by specifying it as an argument to the benchmarking scripts. 

Both of these benchmarking scripts enable you to run a number of epochs and extract performance numbers.

### Training performance benchmark
Training benchmarking can be performed by running the script: 
```bash
scripts/finetune_train_benchmark.sh squad <fp16/fp32> <use_xla> <num_gpu> <batch_size/gpu> <lr> 
```

### Inference performance benchmark
Inference benchmarking can be performed by running the script: 
```bash
scripts/finetune_inference_benchmark.sh squad <fp16/fp32> <use_xla> <batch_size> <path-to-checkpoint> 
```

## Results
The following sections provide details on how we achieved our performance and accuracy in training and inference for Question Answering fine tuning.
### Training accuracy results
Our results were obtained by running the `run_squad.py`  training script in the TensorFlow 19.03-py3 NGC container on NVIDIA DGX-1 with 8x V100 32G GPUs.


| **Number of GPUs** | **Batch size per GPU** | **Training time with FP16 (Hrs)** | **Training time with FP32 (Hrs)** |
|:---:|:---:|:----:|:----:|
| 8 | 4 |0.51|0.77|

#### Training stability test
The following tables compare `F1` scores across 5 different training runs with different seeds, for both FP16 and FP32 respectively.  The runs showcase consistent convergence on all 5 seeds with very little deviation.

| **FP16, 8x GPUs** | **seed #1** | **seed #2** | **seed #3** | **seed #4** | **seed #5** | **mean** | **std** |
|:-----------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|F1         |91.16|90.69|90.99|90.94|91.17|90.99|0.196|
|Exact match|84.2 |83.68|84.14|83.95|84.34|84.06|0.255|

| **FP32, 8x GPUs** | **seed #1** | **seed #2** | **seed #3** | **seed #4** | **seed #5** | **mean** | **std** |
|:-----------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|F1         |90.67|90.8 |90.94|90.83|90.93|90.83|0.11 |
|Exact match|83.56|83.96|83.99|83.95|84.12|83.92|0.21 |


### Training performance results
Our results were obtained by running batch sizes up to 3x GPUs on a 16GB V100 and up to 10x GPUs on a 32G V100 with mixed precision.

#### NVIDIA DGX-1 (8x V100 16G)
Our results were obtained by running the `scripts/run_squad.sh` training script in the TensorFlow 19.03-py3 NGC container on NVIDIA DGX-1 with 8x V100 16G GPUs. Performance (in sentences per second) is the steady-state throughput.


| **Number of GPUs** | **Batch size per GPU** | **FP32 sentences/sec** | **FP16 sentences/sec** | **Speed-up with mixed precision** | **Multi-gpu weak scaling with FP32** | **Multi-gpu weak scaling with FP16** |
|:---:|:---:|:------:|:-----:|:----:|:----:|:----:|
| 1 | 2 | 8.06 |14.12|1.75 |1.0  |1.0 |
| 4 | 2 |25.71 |41.32|1.61 |3.19 |2.93|
| 8 | 2 |50.20 |80.76|1.61 |6.23 |5.72|


| **Number of GPUs** | **Batch size per GPU** | **FP32 sentences/sec** | **FP16 sentences/sec** | **Speed-up with mixed precision** | **Multi-gpu weak scaling with FP32** | **Multi-gpu weak scaling with FP16** |
|:---:|:---:|:-----:|:-----:|:---:|:---:|:----:|
| 1 | 3 |  -  |17.14| - | - |1.0 |
| 4 | 3 |  -  |51.59| - | - |3.0 |
| 8 | 3 |  -  |98.75| - | - |5.76|

Note: The respective values for FP32 runs that use a batch size of 3 are not available due to out of memory errors that arise. Batch size of 3 is only available on using FP16.

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

#### NVIDIA DGX-1 (8x V100 32G)
Our results were obtained by running the `scripts/run_squad.sh` training script in the TensorFlow 19.03-py3 NGC container on NVIDIA DGX-1 with 8x V100 32G GPUs. Performance (in sentences per second) is the steady-state throughput.


| **Number of GPUs** | **Batch size per GPU** | **FP32 sentences/sec** | **FP16 sentences/sec** | **Speed-up with mixed precision** | **Multi-gpu weak scaling with FP32** | **Multi-gpu weak scaling with FP16** |
|---|---|-----|------|----|----|----|
| 1 | 4 | 8.96|20.91 |2.33|1.0 |1.0 |
| 4 | 4 |33.66|64.89 |1.93|3.76|3.10|
| 8 | 4 |66.65|129.16|1.94|7.44|6.18|


| **Number of GPUs** | **Batch size per GPU** | **FP32 sentences/sec** | **FP16 sentences/sec** | **Speed-up with mixed precision** | **Multi-gpu weak scaling with FP32** | **Multi-gpu weak scaling with FP16** |
|---|---|-----|-------|---|---|----|
| 1 | 10|  -  | 31.80 | - | - |1.0 |
| 4 | 10|  -  | 107.65| - | - |3.39|
| 8 | 10|  -  | 220.88| - | - |6.95|


Note: The respective values for FP32 runs that use a batch size of 10 are not available due to out of memory errors that arise. Batch size of 10 is only available on using FP16.

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above. 

#### NVIDIA DGX-2 (16x V100 32G)
Our results were obtained by running the `scripts/run_squad.sh` training script in the TensorFlow 19.03-py3 NGC container on NVIDIA DGX-2 with 16x V100 32G GPUs. Performance (in sentences per second) is the steady-state throughput.


| **Number of GPUs** | **Batch size per GPU** | **FP32 sentences/sec** | **FP16 sentences/sec** | **Speed-up with mixed precision** | **Multi-gpu weak scaling with FP32** | **Multi-gpu weak scaling with FP16** |
|---|---|------|------|----|-----|-----|
|  1| 4 | 10.85| 21.83|2.01| 1.0 | 1.0 |
|  4| 4 | 38.85| 71.87|1.85| 3.58| 3.29|
|  8| 4 | 74.65|140.66|1.88| 6.88| 6.44|
| 16| 4 |132.71|251.26|1.89|12.23|11.51|


| **Number of GPUs** | **Batch size per GPU** | **FP32 sentences/sec** | **FP16 sentences/sec** | **Speed-up with mixed precision** | **Multi-gpu weak scaling with FP32** | **Multi-gpu weak scaling with FP16** |
|---|---|---|------|---|---|-----|
|  1| 10| - | 34.38| - | - | 1.0 |
|  4| 10| - |119.19| - | - | 3.47|
|  8| 10| - |233.86| - | - | 6.8 |
| 16| 10| - |427.19| - | - |12.43|


Note: The respective values for FP32 runs that use a batch size of 10 are not available due to out of memory errors that arise. Batch size of 10 is only available on using FP16.

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above. 
### Inference performance results

#### NVIDIA DGX-1 16G (1x V100 16G)
Our results were obtained by running the `scripts/run_squad_inference.sh` training script in the TensorFlow 19.03-py3 NGC container on NVIDIA DGX-1 with 1x V100 16G GPUs. Performance numbers (in sentences per second) were averaged over an entire training epoch.

| **Number of GPUs** | **Batch size per GPU** | **FP32 sentences/sec** | **FP16 sentences/sec** | **Speedup** |
|---|---|-----|------|----|
| 1 | 8 |41.04|112.55|2.74|

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.


#### NVIDIA DGX-1 32G (1x V100 32G)
Our results were obtained by running the `scripts/run_squad_inference.sh` training script in the TensorFlow 19.03-py3 NGC container on NVIDIA DGX-1 with 1x V100 32G GPUs. Performance numbers (in sentences per second) were averaged over an entire training epoch.

| **Number of GPUs** | **Batch size per GPU** | **FP32 sentences/sec** | **FP16 sentences/sec** | **Speedup** |
|---|---|-----|------|----|
| 1 | 8 |33.95|108.45|3.19|

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

#### NVIDIA DGX-2 32G (1x V100 32G)
Our results were obtained by running the `scripts/run_squad_inference.sh` training script in the TensorFlow 19.03-py3 NGC container on NVIDIA DGX-2 with 1x V100 32G GPUs. Performance numbers (in sentences per second) were averaged over an entire training epoch.

| **Number of GPUs** | **Batch size per GPU** | **FP32 sentences/sec** | **FP16 sentences/sec** | **Speedup** |
|---|---|-----|------|----|
| 1 | 8 |36.78|118.54|3.22|

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

## Changelog
March 2019
- Initial release

## Known issues
There are no known issues with this model.



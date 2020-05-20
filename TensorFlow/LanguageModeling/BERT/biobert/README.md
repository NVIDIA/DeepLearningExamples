# BioBert For TensorFlow

This folder provides a script and recipe to train BERT for TensorFlow to achieve state-of-the-art accuracy on *biomedical text-mining* and is tested and maintained by NVIDIA.

## Table Of Contents

* [Model overview](#model-overview)
* [Quick Start Guide](#quick-start-guide)
* [Advanced](#advanced)
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
* [Performance](#performance)
  * [Benchmarking](#benchmarking)
    * [Training performance benchmark](#training-performance-benchmark)
    * [Inference performance benchmark](#inference-performance-benchmark)
* [Results](#results)
  * [Training accuracy results](#training-accuracy-results)
    * [Pre-training accuracy](#pre-training-accuracy)
    * [Fine-tuning accuracy](#fine-tuning-accuracy)
      * [Fine-tuning accuracy for NER Chem](#fine-tuning-accuracy-for-ner-chem)
  * [Training stability test](#training-stability-test)
    * [Fine-tuning stability test](#fine-tuning-stability-test)
  * [Training performance results](#training-performance-results)
    * [Training performance: NVIDIA DGX-1 (8x V100 16G)](#training-performance-nvidia-dgx-1-8x-v100-16g)
      * [Pre-training training performance: multi-node on 16G](#pre-training-training-performance-multi-node-on-16g)
      * [Fine-tuning training performance for NER on 16G](#fine-tuning-training-performance-for-ner-on-16g)
    * [Training performance: NVIDIA DGX-1 (8x V100 32G)](#training-performance-nvidia-dgx-1-8x-v100-32g)
      * [Fine-tuning training performance for NER on 32G](#fine-tuning-training-performance-for-ner-on-32g)
    * [Training performance: NVIDIA DGX-2 (16x V100 32G)](#training-performance-nvidia-dgx-2-16x-v100-32g)
      * [Pre-training training performance: multi-node on DGX-2 32G](#pre-training-training-performance-multi-node-on-dgx-2-32g)
      * [Fine-tuning training performance for NER on DGX-2 32G](#fine-tuning-training-performance-for-ner-on-dgx-2-32g)
* [Release notes](#release-notes)
  * [Changelog](#changelog)
  * [Known issues](#known-issues)



## Model overview

In the original [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) paper, pre-training is done on [Wikipedia](https://dumps.wikimedia.org/) and [Books Corpus](http://yknzhu.wixsite.com/mbweb), with state-of-the-art results demonstrated on [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (Stanford Question Answering Dataset) benchmark.

Meanwhile, many works, including [BioBERT](https://arxiv.org/pdf/1901.08746.pdf), [SciBERT](https://arxiv.org/pdf/1903.10676.pdf), [NCBI-BERT](https://arxiv.org/pdf/1906.05474.pdf), [ClinicalBERT (MIT)](https://arxiv.org/pdf/1904.03323.pdf), [ClinicalBERT (NYU, Princeton)](https://arxiv.org/pdf/1904.05342.pdf), and others at [BioNLP’19 workshop](https://aclweb.org/aclwiki/BioNLP_Workshop), show that additional pre-training of BERT on large biomedical text corpus such as [PubMed](https://www.ncbi.nlm.nih.gov/pubmed/) results in better performance in biomedical text-mining tasks.

This repository provides scripts and recipe to adopt the [NVIDIA BERT code-base](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) to achieve state-of-the-art results in the following biomedical text-mining benchmark tasks:

- [BC5CDR-disease](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/) A Named-Entity-Recognition task to recognize diseases mentioned in a collection of 1500 PubMed titles and abstracts ([Li et al., 2016](https://academic.oup.com/database/article/doi/10.1093/database/baw068/2630414))

- [BC5CDR-chemical](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/) A Named-Entity-Recognition task to recognize chemicals mentioned in a collection of 1500 PubMed titles and abstracts ([Li et al., 2016](https://academic.oup.com/database/article/doi/10.1093/database/baw068/2630414))

- [ChemProt](https://biocreative.bioinformatics.udel.edu/news/corpora/) A Relation-Extraction task to determine chemical-protein interactions in a collection of 1820 PubMed abstracts ([Krallinger et al., 2017](https://biocreative.bioinformatics.udel.edu/media/store/files/2017/ProceedingsBCVI_v2.pdf?page=141))


## Quick Start Guide

To pretrain or fine tune your model for BioMedical tasks using mixed precision with Tensor Cores or using FP32, perform the following steps using the default parameters of the BERT model.

1. Clone the repository.

```bash
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/TensorFlow/LanguageModeling/BERT
```

2. Build the BERT TensorFlow NGC container.

```bash
bash scripts/docker/build.sh
```

3. Download and preprocess the PubMed dataset.

To download and preprocess pre-training data as well as the required vocab files, run the following script:


```bash
bash biobert/scripts/biobert_data_download.sh
```

Datasets for finetuning for NER can be obtained from this [repository](https://github.com/ncbi-nlp/BLUE_Benchmark/releases/tag/0.1)
Datasets for finetuning for RE can be obtained from this [repository](https://github.com/arwhirang/recursive_chemprot/tree/master/Demo/tree_LSTM/data)

Place them both in `/workspace/bert/data/biobert/` to be automatically picked up by our scripts.

4. Start an interactive session in the NGC container to run training/inference.

After you build the container image and download the data, you can start an interactive CLI session as follows:

```bash
bash scripts/docker/launch.sh
```

5. Download the pre-trained checkpoint, vocabulary, and configuration files.

We have uploaded checkpoints for fine tuning and pre-training on BioMedical Corpus’s on the NGC Model Registry. You can download them directly from the [NGC model catalog](https://ngc.nvidia.com/catalog/models).

Place our `BioBERT checkpoints` in the `results/` to easily access it in your scripts.

6. Start pre-training.

From within the container, you can use the following script to run the 1st phase of the pre-training using cased vocabulary:

```bash
bash biobert/scripts/run_pretraining-pubmed_base_phase_1.sh <train_batch_size> <learning_rate> <cased> <precision> <use_xla> <num_gpus> <warmup_steps> <train_steps> <num_accumulation_steps> <save_checkpoint_steps> <eval_batch_size>
```

For the 2nd phase of the pre-training, issue:

```bash
bash biobert/scripts/run_pretraining-pubmed_base_phase_2.sh <path_to_phase_1_checkpoint> <train_batch_size> <learning_rate> <cased> <precision> <use_xla> <num_gpus> <warmup_steps> <train_steps> <num_accumulation_steps> <save_checkpoint_steps> <eval_batch_size>
```


Refer to (MultiNode Section)[multi-node] for details on utilizing multiple nodes for faster pretraining.

6. Start fine tuning.

The above pretrained BERT representations can be fine tuned with just one additional output layer for a state-of-the-art biomedical text-mining system.
From within the container, you can use the following script to run fine-training for NER.

Note: The scripts assume you are running on 16 V100 32GB GPUs. If you are running on GPU having less than 32GB memory or fewer GPUs, batch size, learning rate and number of GPUs needs to be adjusted.

For NER on disease entities:

```bash
bash biobert/scripts/ner_bc5cdr-disease.sh  <init_checkpoint> <train_batch_size> <learning_rate> <cased> <precision> <use_xla> <num_gpu> <seq_length> <bert_model> <eval_batch_size> <epochs>
```

For NER on chemical entities:

```bash
bash biobert/scripts/ner_bc5cdr-chem.sh  <init_checkpoint> <train_batch_size> <learning_rate> <cased> <precision> <use_xla> <num_gpu> <seq_length> <bert_model> <eval_batch_size> <epochs>
```

For relation extraction, issue:

```
bash biobert/scripts/rel_chemprot.sh <init_checkpoint> <train_batch_size> <learning_rate> <cased> <precision> <use_xla> <num_gpu> <seq_length> <bert_model> <eval_batch_size> <epochs>
```

8. Start validation/evaluation.

The `biobert/scripts/run_biobert_finetuning_inference.sh` script runs inference on a checkpoint fine tuned for a specific task and evaluates the validity of predictions on the basis of F1, precision and recall scores.

```bash
bash biobert/scripts/run_biobert_finetuning_inference.sh <task> <init_checkpoint> <bert_model> <cased> <precision> <use_xla> <batch_size>
```

For FP16 inference for NER on BC5DR Chemical task with XLA using a DGX-2 V100 32G, run:
```bash
bash biobert/scripts/run_biobert_finetuning_inference.sh ner_bc5cdr-chem /results/model.ckpt base false fp16 true 16
```

Tasks `ner_bc5cdr-chem`, `ner_bc5cdr-disease` and `rel_chemprot` are currently supported.

## Advanced

The following sections provide greater details of the dataset, running training and inference, and the training results.

### Scripts and sample code

In addition to BERT TensorFlow files, the most important files added for NER and RE fine tuning tasks are:
* `run_ner.py` - Serves as an entry point for NER training.
* `run_re.py` - Serves as an entry point for RE training.

The `biobert/scripts/` folder encapsulates all the one-click scripts required for running various functionalities supported such as:
* `ner_bc5cdr-chem.sh` - Runs NER training and inference on the BC5CDR Chemical dataset using the `run_ner.py` file.
* `ner_bc5cdr-disease.sh` - Runs NER training and inference on the BC5CDR Disease dataset using the `run_ner.py` file.
* `rel_chemprot.sh` - Runs RE training and inference on the ChemProt dataset using the `run_re.py` file.
* `run_pretraining_pubmed_base_phase_*.sh` - Runs pre-training with LAMB optimizer using the `run_pretraining.py` file in two phases. Phase 1 does training with sequence length = 128. In phase 2, the remaining 10% of the training is done with sequence length = 512.
* `biobert_data_download.sh` - Downloads the PubMed dataset and Vocab files using files in the `data/` folder.
* `run_biobert_finetuning_inference.sh` - Runs task specific inference using a fine tuned checkpoint.


### Parameters

Aside from the options to set hyperparameters, some relevant options to control the behaviour of the `run_ner.py` and `run_re.py` scripts are:

```
  --bert_config_file: The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.
--vocab_file: The vocabulary file that the BERT model was trained on.
  --output_dir: The output directory where the model checkpoints will be written.
  --[no]do_eval: Whether to run evaluation on the dev set. (default: 'false')
  --[no]do_predict: Whether to run evaluation on the test set. (default: 'false')
  --[no]do_train: Whether to run training. (default: 'false')
  --learning_rate: The initial learning rate for Adam.(default: '5e-06')(a number)
  --max_seq_length: The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.(default: '384')(an integer)
  --predict_batch_size: Total batch size for predictions.(default: '8')(an integer)
  --train_batch_size: Total batch size for training (default: '8')(an integer)
  --[no]use_fp16: Whether to enable AMP ops.(default: 'false')
  --[no]use_xla: Whether to enable XLA JIT compilation.(default: 'false')
--init_checkpoint: Initial checkpoint (usually from a pre-trained BERT model).
--num_train_epochs: Total number of training epochs to perform.(default: '3.0')(a number)

```

Note: When initializing from a checkpoint using `--init_checkpoint` and a corpus of your choice, keep in mind that `bert_config_file` and `vocab_file` should remain unchanged.

### Command-line options

To see the full list of available options and their descriptions, use the `-h` or `--help` command-line option with the Python file, for example:

```bash
python run_ner.py --help
python run_re.py --help
```
### Getting the data

For pre-training BERT, we use the PubMed Dataset. For PubMed, we extract the xml files which are structured as a document level corpus rather than a shuffled sentence level corpus because it is critical to extract long contiguous sentences.

The next step is to run `create_pretraining_data.py` with the document level corpus as input, which generates input data and labels for the masked language modeling and next sentence prediction tasks. Pre-training can also be performed on any corpus of your choice. The collection of data generation scripts are intended to be modular to allow modifications for additional preprocessing steps or to use additional data. They can hence easily be modified for an arbitrary corpus.

The preparation of an individual pre-training dataset is described in the `create_biobert_datasets_from_start.sh ` script found in the `data/` folder. The component steps to prepare the datasets are as follows:

1.  Data download and extract - the dataset is downloaded and extracted.
2.  Clean and format - document tags, etc. are removed from the dataset. The end result of this step is a `{dataset_name_one_article_per_line}.txt` file that contains the entire corpus. Each line in the text file contains an entire document from the corpus. One file per dataset is created in the `formatted_one_article_per_line` folder.
3.  Sharding - the sentence segmented corpus file is split into a number of smaller text documents. The sharding is configured so that a document will not be split between two shards. Sentence segmentation is performed at this time using NLTK.
4.  TFRecord file creation - each text file shard is processed by the `create_pretraining_data.py` script to produce a corresponding TFRecord file. The script generates input data and labels for masked language modeling and sentence prediction tasks for the input text shard.


For fine tuning BioBERT for the task of Named Entity Recognition and Relation Extraction Tasks, we use BC5CDR and Chemprot Datasets. BC5CDR corpus consists of 1500 PubMed articles with 4409 annotated chemicals, 5818 diseases and 3116 chemical-disease interactions.
ChemProt corpus consists of text exhaustively annotated by hand with mentions of chemical compounds/drugs and genes/proteins, as well as 22 different types of compound-protein relations focussing on 5 important relation classes. It was preprocessed following [Lim and Kang](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6014134/) guidelines.

#### Dataset guidelines

The procedure to prepare a text corpus for pre-training is described in the previous section. This section provides additional insight into how exactly raw text is processed so that it is ready for pre-training.

First, raw text is tokenized using [WordPiece tokenization](https://arxiv.org/pdf/1609.08144.pdf). A [CLS] token is inserted at the start of every sequence, and the two sentences in the sequence are separated by a [SEP] token.

Note: BERT pre-training looks at pairs of sentences at a time. A sentence embedding token [A] is added to the first sentence and token [B] to the next.

BERT pre-training optimizes for two unsupervised classification tasks. The first is Masked Language Modelling (Masked LM). One training instance of Masked LM is a single modified sentence. Each token in the sentence has a 15% chance of being replaced by a [MASK] token. The chosen token is replaced with [MASK] 80% of the time, 10% with another random token and the remaining 10% with the same token. The task is then to predict the original token.

The second task is next sentence prediction. One training instance of BERT pre-training is two sentences (a sentence pair). A sentence pair may be constructed by simply taking two adjacent sentences from a single document, or by pairing up two random sentences with equal probability. The goal of this task is to predict whether or not the second sentence followed the first in the original document.

The `create_pretraining_data.py` script takes in raw text and creates training instances for both pre-training tasks.

#### Multi-dataset

We are able to combine multiple datasets into a single dataset for pre-training on a diverse text corpus. Once TFRecords have been created for each component dataset, you can create a combined dataset by adding the directory to `*FILES_DIR` in `run_pretraining_*.sh`. This will feed all matching files to the input pipeline in `run_pretraining.py`. However, in the training process, only one TFRecord file is consumed at a time, therefore, the training instances of any given training batch will all belong to the same source dataset.



### Training process

The training process consists of two steps: pre-training and fine tuning.

#### Pre-training

BERT is designed to pre-train deep bidirectional representations for language representations. The following scripts are to pre-train BERT on PubMed dataset. These scripts are general and can be used for pre-training language representations on additional corpus of biomedical text.

Pre-training is performed using the `run_pretraining.py` script along with parameters defined in the `biobert/scripts/run_pretraining_pubmed_base_phase_1.sh` and `biobert/scripts/run_pretraining_pubmed_base_phase_2.sh` scripts.

The `biobert/scripts/run_pretraining_pubmed_base_phase*.sh` scripts run a job on a single node that trains the BERT-base model from scratch using the PubMed Corpus dataset as training data. By default, the training script:
- Runs on 16 GPUs
- Has FP16 precision enabled
- Is XLA enabled
- Creates a log file containing all the output
- Saves a checkpoint every 5000 iterations (keeps only the latest checkpoint) and at the end of training. All checkpoints, evaluation results, and training logs are saved to the `/results` directory (in the container which can be mounted to a local directory).
- Evaluates the model at the end of each phase

- Phase 1
    - Runs 19531 steps with 1953 warmup steps
    - Sets Maximum sequence length as 128
    - Sets Global Batch size as 64K

- Phase 2
    - Runs 4340 steps with 434 warm-up steps
    - Sets Maximum sequence length as 512
    - Sets Global Batch size as 32K
    - Should start from Phase1's final checkpoint

These parameters train PubMed with reasonable accuracy on a DGX-2 with 32GB V100 cards.

For example:
```bash
biobert/scripts/run_pretraining-pubmed_base_phase_1.sh <train_batch_size> <learning_rate> <cased> <precision> <use_xla> <num_gpus> <warmup_steps> <train_steps> <num_accumulation_steps> <save_checkpoint_steps> <eval_batch_size>
```

Where:
- `<training_batch_size>` is per-GPU batch size used for training. Batch size varies with precision, larger batch sizes run more efficiently, but require more memory.

- `<learning_rate>` is the default rate of 3.2e-5 is good for global batch size 64k.

- `<cased>` is set to `true` or `false` depending on whether the model should be trained on cased or uncased data.

- `<precision>` is the type of math in your model, can be either `fp32` or `fp16`. Specifically:

    - `fp32` is 32-bit IEEE single precision floats.
    - `fp16` is Automatic rewrite of TensorFlow compute graph to take advantage of 16-bit arithmetic whenever it is safe.

- `<num_gpus>` is the number of GPUs to use for training. Must be equal to or smaller than the number of GPUs attached to your node.

- `<warmup_steps>` is the number of warm-up steps at the start of training.

- `<training_steps>` is the total number of training steps.

- `<save_checkpoint_steps>` controls how often checkpoints are saved. Default is 5000 steps.

- `<num_accumulation_steps>` is used to mimic higher batch sizes in the respective phase by accumulating gradients N times before weight update.

- `<bert_model>` is used to indicate whether to pretrain BERT Large or BERT Base model.

- `<eval_batch_size>` is per-GPU batch size used for evaluation after training.

The following sample code trains phase 1 of BERT-base from scratch on a single DGX-2 using FP16 arithmetic and uncased data.

```bash
biobert/scripts/run_pretraining-pubmed_base_phase_1.sh 128 3.2e-5 false fp16 true 16 1953 19531 32 5000 80
```

#### Fine tuning

Fine tuning is performed using the `run_ner.py` script along with parameters defined in `biobert/scripts/ner_bc5cdr*.sh`.

For example, `biobert/scripts/ner_bc5cdr-chem.sh` script trains a model and performs evaluation on the BC5CDR Chemical dataset. By default, the training script:

- Trains on BERT Base Uncased Model
- Uses 16 GPUs and batch size of 8 on each GPU
- Has FP16 precision enabled
- Is XLA enabled
- Runs for 10 epochs
- Evaluation is done at the end of training. To skip evaluation, modify `--do_eval` and  `--do_predict` to `False`.

This script outputs checkpoints to the `/results` directory, by default, inside the container. Mount point of `/results` can be changed in the `scripts/docker/launch.sh` file. The training log contains information about:
- Loss for the final step
- Training and evaluation performance
- F1, Precision and Recall on the Test Set of BC5CDR Chemical after evaluation.

The summary after training is printed in the following format:
```bash
 0: /results/biobert_finetune_ner_chem_191028154209/test_labels.txt
 0: /results/biobert_finetune_ner_chem_191028154209/test_labels_errs.txt
 0: processed 124669 tokens with 5433 phrases; found: 5484 phrases; correct: 5102.
 0: accuracy:  99.26%; precision:  93.03%; recall:  93.91%; FB1:  93.47
 0:                  : precision:  93.03%; recall:  93.91%; FB1:  93.47  5484
```

Multi-GPU training is enabled with the Horovod TensorFlow module. The following example runs training on 16 GPUs:

```bash
BERT_DIR=data/download/google_pretrained_weights/uncased_L-12_H-768_A-12
DATA_DIR=data/biobert/BC5CDR/chem

mpi_command="mpirun -np 16 -H localhost:16 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH -mca pml ob1 -mca btl ^openib" \
     python run_ner.py --horovod --use_fp16 --use_xla \
      --vocab_file=$BERT_DIR/vocab.txt \
     --bert_config_file=$BERT_DIR/bert_config.json \
     --output_dir=/results --data_dir=$DATA_DIR"
```

#### Multi-node

Multi-node runs can be launched on a pyxis/enroot Slurm cluster (see [Requirements](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT#requirements)) with the `biobert/scripts/run_biobert.sub` script with the following command for a 4-node DGX2 example for both phase 1 and phase 2:

```bash
BATCHSIZE=128 LEARNING_RATE='8e-6' NUM_ACCUMULATION_STEPS=8 PHASE=1 sbatch -N4 --ntasks-per-node=16 biobert/scripts/run_biobert.sub
BATCHSIZE=16 LEARNING_RATE='3.2e-5' NUM_ACCUMULATION_STEPS=32 PHASE=1 sbatch -N4 --ntasks-per-node=16 biobert/scripts/run_biobert.sub
```

Checkpoint after phase 1 will be saved in `checkpointdir` specified in `biobert/scripts/run_biobert.sub`. The checkpoint will be automatically picked up to resume training on phase 2. Note that phase 2 should be run after phase 1.

Variables to re-run the [Training performance results](#training-performance-results) are available in the `configurations.yml` file.

The batch variables `BATCHSIZE`, `LEARNING_RATE`, `NUM_ACCUMULATION_STEPS` refer to the Python arguments `train_batch_size`, `learning_rate`, `num_accumulation_steps` respectively.
The variable `PHASE` refers to phase specific arguments available in `biobert/scripts/run_biobert.sub`.

Note that the `biobert/scripts/run_biobert.sub` script is a starting point that has to be adapted depending on the environment. In particular, variables such as `datadir` handle the location of the files for each phase.

Refer to the file contents to see the full list of variables to adjust for your system.

### Inference process

Inference on a fine tuned model for Bio Medical tasks is performed using the `run_ner.py` or `run_re.py` script along with parameters defined in `biobert/scripts/run_biobert_finetuning_inference.sh`. Inference is supported on a single GPU.

The `biobert/scripts/run_biobert_finetuning_inference.sh` script performs evaluation on ChemProt or BC5CDR datasets depending on the task specified. By default, the inferencing script:

- Uses BC5CDR Chemical dataset
- Has FP16 precision enabled
- Is XLA enabled
- Evaluates the latest checkpoint present in `/results` with a batch size of 16.

This script computes F1, Precision and Recall scores. Mount point of `/results` can be changed in the `scripts/docker/launch.sh` file.

## Performance

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in training and inference modes.

Both of these benchmarking scripts enable you to run a number of epochs, extract performance numbers, and run the BERT model for fine tuning.

#### Training performance benchmark

Training benchmarking can be performed by running the script:
``` bash
biobert/scripts/biobert_finetune_training_benchmark.sh <task> <num_gpu> <bert_model> <cased>
```

This script runs 2 epochs by default on the NER BC5CDR dataset and extracts performance numbers for various batch sizes and sequence lengths in both FP16 and FP32. These numbers are saved at `/results/tf_bert_biobert_<task>_training_benchmark__<bert_model>_<cased/uncased>_num_gpu_<num_gpu>_<DATESTAMP>`

#### Inference performance benchmark

Training benchmarking can be performed by running the script:
``` bash
biobert/scripts/biobert_finetune_inference_benchmark.sh <task> <bert_model> <cased>
```

This script runs inference on the test and dev sets and extracts performance and latency numbers for various batch sizes and sequence lengths in both FP16 with XLA and FP32 without XLA. These numbers are saved at `/results/tf_bert_biobert_<task>_training_benchmark__<bert_model>_<cased/uncased>_num_gpu_<num_gpu>_<DATESTAMP>`

## Results

The following sections provide detailed results of downstream fine-tuning task on NER and RE benchmark tasks.

### Training accuracy results

#### Pre-training accuracy

Our results were obtained by running the `scripts/run_pretraining_lamb.sh` training script in the TensorFlow 19.08-py3 NGC container.

| **DGX System** | **Nodes** | **Precision** | **Batch Size/GPU: Phase1, Phase2** | **Accumulation Steps: Phase1, Phase2** | **Time to Train (Hrs)** | **Final Loss** |
|----------------|-----------|---------------|------------------------------------|----------------------------------------|----------------|-------------------------|
| DGX2H | 4  | FP16 | 128, 16 | 8, 32 | 19.14  | 0.88 |
| DGX2H | 16 | FP16 | 128, 16 | 2, 8  | 4.81   | 0.86 |
| DGX2H | 32 | FP16 | 128, 16 | 1, 4  | 2.65   | 0.87 |
| DGX1  | 1  | FP16 | 64, 8   |128,512| 174.58 | 0.87 |
| DGX1  | 4  | FP16 | 64, 8   |32, 128| 57.71  | 0.85 |
| DGX1  | 16 | FP16 | 64, 8   |8,  32 | 12.62  | 0.87 |
| DGX1  | 32 | FP16 | 64, 8   |4,  16 | 6.97   | 0.87 |

#### Fine-tuning accuracy

| **Task** | **F1** | **Precision** | **Recall** |
|:-------:|:----:|:----:|:----:|
| NER BC5CDR-chemical | 93.47 | 93.03 | 93.91 |
| NER BC5CDR-disease | 86.22 | 85.05 | 87.43 |
| RE Chemprot | 76.27 | 77.62 | 74.98 |

##### Fine-tuning accuracy for NER Chem

Our results were obtained by running the `biobert/scripts/ner_bc5cdr-chem.sh` training script in the TensorFlow 19.08-py3 NGC container.

| **DGX System** | **Batch size / GPU** | **F1 - FP32** | **F1- mixed precision** | **Time to Train - FP32 (Minutes)** | **Time to Train - mixed precision (Minutes)** |
|:---:|:----:|:----:|:---:|:----:|:----:|
| DGX-1 16G | 64 |93.33|93.40|23.95|14.13|
| DGX-1 32G | 64 |93.31|93.36|24.35|12.63|
| DGX-2 32G | 64 |93.66|93.47|12.26|8.16|


### Training stability test

#### Fine-tuning stability test:

The following tables compare F1 scores scores across 5 different training runs on the NER Chemical task with different seeds, for both FP16 and FP32.  The runs showcase consistent convergence on all 5 seeds with very little deviation.

| **16 x V100 GPUs** | **seed 1** | **seed 2** | **seed 3** | **seed 4** | **seed 5** | **mean** | **std** |
|:-----------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| F1 Score (FP16)  | 93.13     | 92.92   | 93.34   | 93.66   | 93.47   | 93.3  | 0.29 |
| F1 Score (FP32)  | 93.1      | 93.28   | 93.33   | 93.45   | 93.17   | 93.27 | 0.14 |


### Training performance results

#### Training performance: NVIDIA DGX-1 (8x V100 16G)

##### Pre-training training performance: multi-node on DGX-1 16G

Our results were obtained by running the `biobert/scripts/run_biobert.sub` training script in the TensorFlow 19.08-py3 NGC container using multiple NVIDIA DGX-1 with 8x V100 16G GPUs. Performance (in sentences per second) is the steady state throughput.

| **Nodes** | **Sequence Length**| **Batch size / GPU: mixed precision, FP32** | **Throughput - mixed precision** | **Throughput - FP32** | **Throughput speedup (FP32 to mixed precision)** | **Weak scaling - mixed precision** | **Weak scaling - FP32** |
|:-------:|:-----:|:-------:|:-------:|:-------:|:-------------:|:------:|:------:|
| 1  | 128 | 64,32 | 2762.06  | 744.48   | 3.71 | 1.00  | 1.00  |
| 4  | 128 | 64,32 | 10283.08 | 2762.88  | 3.72 | 3.72  | 3.71  |
| 16 | 128 | 64,32 | 39051.69 | 10715.14 | 3.64 | 14.14 | 14.39 |
| 32 | 128 | 64,32 | 76077.39 | 21104.87 | 3.60 | 27.54 | 28.35 |
| 1  | 512 | 8,8   | 432.33   | 160.38   | 2.70 | 1.00  | 1.00  |
| 4  | 512 | 8,8   | 1593.00  | 604.36   | 2.64 | 3.68  | 3.77  |
| 16 | 512 | 8,8   | 5941.82  | 2356.44  | 2.52 | 13.74 | 14.69 |
| 32 | 512 | 8,8   | 11483.73 | 4631.29  | 2.48 | 26.56 | 28.88 |

Note: The respective values for FP32 runs that use a batch size of 16, 2 in sequence lengths 128 and 512 respectively are not available due to out of memory errors that arise.

##### Fine-tuning training performance for NER on DGX-1 16G

Our results were obtained by running the `biobert/scripts/ner_bc5cdr-chem.sh` training script in the TensorFlow 19.08-py3 NGC container on NVIDIA DGX-1 with 8x V100 16G GPUs. Performance (in sentences per second) is the mean throughput from 2 epochs.

| **GPUs** | **Batch size / GPU** | **Throughput - FP32** | **Throughput - mixed precision** | **Throughput speedup (FP32 to mixed precision)** | **Weak scaling - FP32** | **Weak scaling - mixed precision** |
|:---:|:---:|:------:|:-----:|:----:|:----:|:----:|
| 1 | 64 | 147.71 | 348.84  | 2.36 | 1.00 | 1.00 |
| 4 | 64 | 583.78 | 1145.46 | 1.96 | 3.95 | 3.28 |
| 8 | 64 | 981.22 | 1964.85 | 2.00 | 6.64 | 5.63 |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

#### Training performance: NVIDIA DGX-1 (8x V100 32G)


##### Fine-tuning training performance for NER on DGX-1 32G

Our results were obtained by running the `biobert/scripts/ner_bc5cdr-chem.sh` training script in the TensorFlow 19.08-py3 NGC container on NVIDIA DGX-1 with 8x V100 32G GPUs. Performance (in sentences per second) is the mean throughput from 2 epochs.


| **GPUs** | **Batch size / GPU** | **Throughput - FP32** | **Throughput - mixed precision** | **Throughput speedup (FP32 to mixed precision)** | **Weak scaling - FP32** | **Weak scaling - mixed precision** |
|:---:|:---:|:------:|:-----:|:----:|:----:|:----:|
| 1 | 64 | 144.1 | 417.39  | 2.89 | 1.00 | 1.00 |
| 4 | 64 | 525.15 | 1354.14 | 2.57 | 3.64 | 3.24 |
| 8 | 64 | 969.4 | 2341.39 | 2.41 | 6.73 | 5.61 |


To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

#### Training performance: NVIDIA DGX-2 (16x V100 32G)


##### Pre-training training performance: multi-node on DGX-2H 32G

Our results were obtained by running the `biobert/scripts/run_biobert.sub` training script in the TensorFlow 19.08-py3 NGC container using multiple NVIDIA DGX-2H with 16x V100 32G GPUs. Performance (in sentences per second) is the steady state throughput.


| **Nodes** | **Sequence Length**| **Batch size / GPU: mixed precision, FP32** | **Throughput - mixed precision** | **Throughput - FP32** | **Throughput speedup (FP32 to mixed precision)** | **Weak scaling - mixed precision** | **Weak scaling - FP32** |
|:-------:|:-----:|:-------:|:-------:|:-------:|:-------------:|:------:|:------:|
| 1  | 128 | 128,128 | 7772.18   | 2165.04   | 3.59 | 1.00  | 1.00  |
| 4  | 128 | 128,128 | 29785.31  | 8516.90   | 3.50 | 3.83  | 3.93  |
| 16 | 128 | 128,128 | 115581.29 | 33699.15  | 3.43 | 14.87 | 15.57 |
| 32 | 128 | 128,128 | 226156.53 | 66996.73  | 3.38 | 29.10 | 30.94 |
| 64 | 128 | 128,128 | 444955.74 | 133424.95 | 3.33 | 57.25 | 61.63 |
| 1  | 512 | 16,16   | 1260.06   | 416.92    | 3.02 | 1.00  | 1.00  |
| 4  | 512 | 16,16   | 4781.19   | 1626.76   | 2.94 | 3.79  | 3.90  |
| 16 | 512 | 16,16   | 18405.65  | 6418.09   | 2.87 | 14.61 | 15.39 |
| 32 | 512 | 16,16   | 36071.06  | 12713.67  | 2.84 | 28.63 | 30.49 |
| 64 | 512 | 16,16   | 69950.86  | 25245.96  | 2.77 | 55.51 | 60.55 |


##### Fine-tuning training performance for NER on DGX-2 32G

Our results were obtained by running the `biobert/scripts/ner_bc5cdr-chem.sh` training script in the TensorFlow 19.08-py3 NGC container on NVIDIA DGX-2 with 16x V100 32G GPUs. Performance (in sentences per second) is the mean throughput from 2 epochs.

| **GPUs** | **Batch size / GPU** | **Throughput - FP32** | **Throughput - mixed precision** | **Throughput speedup (FP32 to mixed precision)** | **Weak scaling - FP32** | **Weak scaling - mixed precision** |
|:---:|:---:|:------:|:-----:|:----:|:----:|:----:|
| 1 | 64 | 139.59 | 475.54  | 3.4 | 1.00 | 1.00 |
| 4 | 64 | 517.08 | 1544.01 | 2.98 | 3.70 | 3.25 |
| 8 | 64 | 1009.84 | 2695.34 | 2.66 | 7.23 | 5.67 |
| 16 | 64 | 1997.73 | 4268.81 | 2.13 | 14.31 | 8.98 |

To achieve these same results, follow the [Quick Start Guide](#quick-start-guide) outlined above.

## Release notes

### Changelog

November 2019
- Initial release

### Known issues


- There are no known issues with the model.




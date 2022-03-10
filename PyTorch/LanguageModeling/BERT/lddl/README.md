# Language Datasets and Data Loaders

Language Datasets and Data Loaders (LDDL) is an utility library that minimizes
the friction during dataset retrieval, preprocessing and loading for the 
language models in 
_[NVIDIA Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples)_.

The current capabilities of LDDL include:
- Data preprocessing at scale via [Dask](https://dask.org/) and 
[MPI for Python (mpi4py)](https://mpi4py.readthedocs.io/en/stable/): it would 
even take less than 2 mins to finish preprocessing 
[BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)'s 
dataset for pretraining on 32 DGXA100 nodes.
- Data loading for [PyTorch](https://pytorch.org/) multi-node training workloads 
with minimum overhead.
- Sequence binning that can reduce end-to-end training latency.

## Installation

The steps to install LDDL are listed below:

**Step 1 [optional but recommended]:** The [jemalloc](http://jemalloc.net/) 
memory allocator is an alternative (to the glibc memory allocator) that might 
offer better performance during data preprocessing. You can install jemalloc 
via:
```bash
conda install jemalloc
```
**Step 2 [required]:** LDDL can be installed from the source by running 
`pip install <target>` where `<target>` is the project root directory of LDDL or
an URL thereof. For examples:
```bash
pip install git+https://github.com/NVIDIA/DeepLearningExamples.git#subdirectory=DeepLearningExamples/Tools/lddl
```
or
```bash
git clone https://github.com/NVIDIA/DeepLearningExamples.git
pip install DeepLearningExamples/Tools/lddl
```
`pip` would also automatically install all LDDL's other Python package 
dependencies.
> LDDL is only tested to work with Python 3!

**Step 3 [required]:** After installing [NLTK](https://www.nltk.org/index.html) 
(either manually or automatically when `pip install` LDDL), the model data of 
the NLTK's Punkt Sentence Tokenizer needs to be downloaded before the Punkt 
Sentence Tokenizer can be used:
```bash
python -m nltk.downloader punkt
```

### Example Dockerfile

For your own tasks, it is highly likely that you would need to use a Docker
container whose image you customize and build. As an example to show how to
install LDDL inside a Docker container, we provide
[a Dockerfile](docker/ngc_pyt.Dockerfile) which follows the above installation
steps to install LDDL in a
[NGC Container](https://developer.nvidia.com/ai-hpc-containers). You can build
NGC Container images with LDDL installed using this example Dockerfile by
```bash
bash docker/build.sh <Dockerfile name without extension> <tag of the base image> <output image name/URL>
```

> NGC Containers are **not** one of LDDL's dependencies. You can install LDDL in
> your customized Docker image, local virtualenv or conda environments too.

For example, to build the image with LDDL installed based on the
[NGC PyTorch Container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
[Version 21.11](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_21-11.html#rel_21-11)
and name the output image as `lddl:latest`:
```bash
bash docker/build.sh ngc_pyt 21.11-py3 lddl:latest
```
where:
- `ngc_pyt` refers to using
[docker/ngc_pyt.Dockerfile](docker/ngc_pyt.Dockerfile);
- `21.11-py3` means that `nvcr.io/nvidia/pytorch:21.11-py3` is used as the base
image;
- `lddl:latest` is the name of the built image. After building this image, you 
can find its name via `docker image list`.

You can launch a container in interactive mode by:
```bash
bash docker/interactive.sh <mount specifications> <shell command> <output image name/URL>
```
For example, to launch a container using the `lddl:latest` image and mount 
`datasets/` under your home directory to `/datasets` inside the container:
```bash
bash docker/interactive.sh "-v ${HOME}/datasets:/datasets" /bin/bash lddl:latest
```

## Introduction

To clarify, we define the following terms:
- Offline vs. Online: By "Offline", we mean something to be run as a separate
entity only once with respect to the training jobs; in contrast, "Online" refers 
to being run as part of the training jobs.
> As an analogy, buying kitchen appliances is "Offline" with respect to cooking,
> because you only need to buy them once, and the action of buying often happens
> when you are not cooking (i.e., when you are going to the stores or shopping
> online). In contrast, washing the vegetables is "Online" with respect to
> cooking, because it happens every time before you cook the vegetables.
- Ahead-of-training vs. During-training: By "Ahead-of-training", we mean 
something to be run before the training processes start; in contrast, 
"During-training" refers to being run during the training processes.

![Summary](./docs/images/summary.gif "Summary")

In summary, LDDL consists of four components:
1. *Stage 1 [Offline]* **Downloaders** that download the raw text of datasets
from public and online sources.
2. *[Offline or Online and Ahead-of-training]* Preprocessing:
   1. *Stage 2* **Preprocessors** that preprocesses the raw text into unbalanced
   [Parquet](https://arrow.apache.org/docs/python/parquet.html#) shards.
   2. *Stage 3* **Load Balancer** that balance the Parquet shards and makes sure
   every shard has the same amount of samples.
3. *Stage 4 [Online and During-training]* **Data Loaders** that load the 
balanced shards into memory and perform additional preprocessing steps during
training.

Depending on the specific usage, a certain step can be performed in different 
stages. For example, if you want to experiment with static masking, you can 
request the preprocessor to mask each samples; however, if you want to 
experiment with dynamic masking, you can request the data loader to mask each 
samples.

![Sequence Binning](./docs/images/binning.gif "Sequence Binning")

LDDL supports the technique of *sequence binning* in order to reduce redundant 
computation on the padded tokens:
1. The maximum sequence length is divided into several *bins*. For example, if
we want `4` bins out of a maximum sequence length of `512`, then:
   1. The first bin contains samples whose sequence lengths are between 
   `[0, 128]`;
   2. The second bin contains samples whose sequence lengths are between
     `[129, 256]`;  
   3. The third bin contains samples whose sequence lengths are between
     `[257, 384]`;
   4. The third bin contains samples whose sequence lengths are between
     `[385, 512]`;
2. At each training iteration, a bin is randomly selected based on the sequence 
distribution of the entire dataset, and all ranks are fed with mini-batches 
whose samples all belong to this selected bin.
3. Each mini-batch is only padded to the longest sequence within this 
mini-batch.

![Preprocessing Performance](./docs/images/preprocess_perf.gif "Preprocessing Performance")

The preprocessor and load balancer can speedup preprocessing large corpora 
significantly by scaling to multi-node via Dask and MPI.

![Sequence Binning Performance](./docs/images/binning_perf.gif "Sequence Binning Performance")

Meanwhile, the technique of sequence binning can significantly reduce the 
end-to-end training latency by reducing redundant computation on the padded 
tokens.

### Coverage

Downloaders, preprocessors and the load balancer can be launched via shell 
commands:
- Downloaders:
  - [Wikipedia dumps](https://dumps.wikimedia.org/): `download_wikipedia`
  - [Bookcorpus](https://github.com/soskek/bookcorpus/issues/27#issuecomment-716104208) `download_books`
  - [Common Crawl](https://github.com/fhamborg/news-please#news-archive-from-commoncrawlorg) `download_common_crawl`
- Preprocessors:
  - BERT
    - Pretraining: `preprocess_bert_pretrain`
- Load Balancer: `balance_dask_output`

Please use the `--help` flag to check the exact usage of each command (e.g., 
`download_wikipedia --help`).

> An implementation of MPI is required to be already installed on your system 
> for the preprocessors and the load balancer. NGC containers come with a
> pre-installed MPI implementation.

LDDL currently supports the following data loaders:
- BERT
  - [PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)
    - Pretraining: [`lddl.torch.get_bert_pretrain_data_loader`](lddl/torch/bert.py:175)

Please refer to their in-code documentation for more details.

## Quick Start Guide

### BERT

We provide two working example scripts to demonstrate how to use LDDL end-to-end 
(i.e., from downloading the datasets to loading input batches during training) 
for a (mock) BERT Phase 2 pretraining task:
- Running on a local machine: [local_example.sh](examples/local_example.sh).
  You can run this script by `bash examples/local_example.sh`.
- Running on a Slurm cluster and scale to multi-nodes: 
  [slurm_example.sub](examples/slurm_example.sub). Before running this script, 
  you need to download and move the datasets to the right location in the NFS of 
  your Slurm cluster. You might also need to customize this script to match the 
  specific settings of your Slurm cluster. You can run this script and submit 
  jobs to Slurm by `sbatch -N<number of nodes> examples/slurm_example.sub` 
  (e.g., `sbatch -N2 examples/slurm_example.sub` if you want to 
  run on 2 nodes).

We assume that these two example scripts could be run without interruption so 
that they would work out-of-the-box. You can also comment out the commands in 
these two scripts to run each step individually. Important steps in the above 
working example scripts are summarized and highlighted below:

#### Offline Downloader

The Wikipedia corpus can be downloaded via:
```bash
download_wikipedia --outdir <Wikipedia output path>
```
where `<Wikipedia output path>` is where you want to store the raw text of the 
Wikipedia corpus. For example,
```bash
download_wikipedia --outdir data/wikipedia
```
would download the raw text of the Wikipedia corpus to `data/wikipedia`.

#### Offline, or Online and Ahead-of-training Preprocessing

##### Preprocessor

The dataset for BERT pretraining can be preprocessed via:
```bash
mpirun \
  -np $(nproc) \
  --oversubscribe \
  --allow-run-as-root \
  -x LD_PRELOAD=<path to libjemalloc.so> \
    preprocess_bert_pretrain \
      --schedule mpi \
      --target-seq-length <128 for Phase 1; 512 for Phase 2> \
      --wikipedia <Wikipedia output path>/source \
      --sink <BERT pretraining input path> \
      --vocab-file <path to the vocab file> \
      --num-blocks <number of input shards> \
      --bin-size <bin size>
```
where:
- `<BERT pretraining input path>` is where you want to store the preprocessed 
but unbalanced Parquet shards generated by this BERT pretraining preprocessor.
- `<path to the vocab file>` is the path to the 
[vocab file](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/vocab/vocab) 
that is used by the WordPiece tokenizer.
- `<number of input shards>` is the total number of input shards; this number 
needs to be a positive integer multiple of 
`<world size> * <max(DataLoader's num_workers, 1)>`.
- if you want to enable sequence binning, you can set the `--bin-size` flag. The
`i`-th bin contains sequences that have from `(i - 1) * <bin size> + 1` to 
`i * <bin size>` tokens.
- `<path to libjemalloc.so>` depends on the conda environment that `jemalloc` 
was installed with. If installed by `root`, it can often be found at 
`/opt/conda/lib/libjemalloc.so`; if installed by an non-root user, it can often
be found at `$CONDA_PREFIX/lib/libjemalloc.so`.

If you want to use the memory allocator from glibc instead of jemalloc, you 
can omit the `-x LD_PRELOAD=<path to libjemalloc.so>` flag to `mpirun`. Either 
jemalloc or glibc could be more suitable to your specific system, so we 
recommend trying both and select the one that yields the best performance.

By default, masking is deferred to the online and during-training data loading
and performed dynamically. During dynamic masking, each sample can be masked 
differently among epochs. If you want to enable static masking that takes place
during the preprocessor stage, you need to add the `--masking` flag.

For example,
```bash
mpirun \
  --oversubscribe \
  --allow-run-as-root \
  -np 64 \
  -x LD_PRELOAD=/opt/conda/lib/libjemalloc.so \
    preprocess_bert_pretrain \
      --schedule mpi \
      --vocab-file data/vocab/bert-en-uncased.txt \
      --wikipedia data/wikipedia/source/ \
      --sink data/bert/pretrain/phase2/bin_size_64/ \
      --target-seq-length 512 \
      --num-blocks 4096 \
      --bin-size 64 \
      --masking
```
would use a total of `64` processes to run the preprocessor for BERT Phase 2 
pretraining (whose maximum sequence length is `512`). In this case, the shared
object of jemalloc is located at `/opt/conda/lib/libjemalloc.so`; the vocab file
is located at `data/vocab/bert-en-uncased.txt`; the Wikipedia corpus is 
downloaded at `data/wikipedia/`; the preprocessor would store the unbalanced
(roughly `4096`) Parquet shards at `data/bert/pretrain/phase2/bin_size_64/`;
the sequence binning is enabled with the bin size of `64`; and the static 
masking is enabled as well.

##### Load Balancer

We can balance the number of samples among the preprocessed but unbalanced 
Parquet shards via:
```bash
mpirun -np $(nproc) --oversubscribe --allow-run-as-root \
  balance_dask_output \
    --indir <BERT pretraining input path> \
    --num-shards <number of input shards>
```
After the load balancer finishes, all shards will have exactly the same number 
of samples, or some shards are different by only 1 sample if the total number of 
shards does not divide the total number of samples. If you don't specify a path 
to the `--outdir` flag, the Parquet shards in `<BERT pretraining input path>` 
will be modified in-place. For example,
```bash
mpirun \
  --oversubscribe \
  --allow-run-as-root \
  -np 64 \
    balance_dask_output \
      --indir data/bert/pretrain/phase2/bin_size_64/ \
      --num-shards 4096
```
would use `64` processes to run the load balancer which balances the Parquet 
shards (located at `data/bert/pretrain/phase2/bin_size_64/`) into exactly `4096`
shards. Among these shards, the number of samples could differ by at most 1.

> The above `mpirun` commands show how the preprocessor and load balancer could 
> be run on a single node. The flags passed into `mpirun` often need to be 
> adjusted based on the configuration of your compute cluster.

> We also provide an [example](examples/slurm_example.sub) to demonstrate how to
> run the preprocessor and load balancer on SLURM clusters that support
> [MPI](https://slurm.schedmd.com/mpi_guide.html),
> [Pyxis](https://github.com/NVIDIA/pyxis) and
> [Enroot](https://github.com/NVIDIA/enroot).

#### Online and During-training Data Loading

##### PyTorch

We can get the LDDL dataloader for BERT pretraining via 
`lddl.torch.get_bert_pretrain_data_loader` (please refer to the 
[in-code documentation](lddl/torch/bert.py:175) of this function). Afterwards,
we can use it like a normal PyTorch DataLoader instance. For example,
```python
import argparse
import logging
import os

import lddl.torch

parser = argparse.ArgumentParser()
parser.add_argument(
    '--local_rank',
    type=int,
    default=os.getenv('LOCAL_RANK', 0),
    help='local_rank is set by torch.distributed.launch or SLURM',
)
args = parser.parse_args()

# Contains the balanced Parquet shards generated by the load balancer.
input_dir = 'data/bert/pretrain/phase2/bin_size_64/'
# Path to the vocab file.
vocab_file = 'data/vocab/bert-en-uncased.txt'
# Number of samples in a single mini-batch per rank.
batch_size = 64 
# Number of DataLoader worker processes per rank.
num_workers = 4
# Epoch number to start with.
start_epoch = 0
# Total number of epochs to train. One epoch refers to going through the entire
# dataset once.
epochs = 2

train_dataloader = lddl.torch.get_bert_pretrain_data_loader(
  input_dir,
  local_rank=args.local_rank,
  vocab_file=vocab_file,
  data_loader_kwargs={
    'batch_size': batch_size,
    'num_workers': num_workers,
    'pin_memory': True,
  },
  log_level=logging.WARNING,
  start_epoch=start_epoch,
)
...
for epoch in range(start_epoch, start_epoch + epochs):
  for i, batch in enumerate(train_dataloader):
    prediction_scores, seq_relationship_score = model(
        input_ids=batch['input_ids'].to(device), 
        token_type_ids=batch['token_type_ids'].to(device), 
        attention_mask=batch['attention_mask'].to(device),
    )
    loss = criterion(  
        prediction_scores, 
        seq_relationship_score, 
        batch['labels'].to(device), 
        batch['next_sentence_labels'].to(device),
    )
    ...
```
We provide a [(mock) training script](benchmarks/torch_train.py) that shows how
the LDDL dataloader should be used. For example, if the balanced Parquet shards
are located at `data/bert/pretrain/phase2/bin_size_64/` and the vocab file is 
located at `data/vocab/bert-en-uncased.txt`, you can run this (mock) training 
script with a world size of `2` on a single machine via:
```bash
python -m torch.distributed.launch --nproc_per_node=2 \
  benchmarks/torch_train.py \
    --path data/bert/pretrain/phase2/bin_size_64/ \
    --vocab-file data/vocab/bert-en-uncased.txt
```
Once the (mock) training processes are up and running, and the first rank starts
to print output, these processes simply emulate the training loop which could
take some time to go through one epoch. You can kill these processes at any 
time.

## Contribution

We welcome any form of contribution! The simplest contribution would be to try
LDDL on your own NLP tasks where data preprocessing and loading is a headache 
for you. If you find rough edges for your specific use case, please file a 
GitHub issue and tag `@shangw-nvidia`.
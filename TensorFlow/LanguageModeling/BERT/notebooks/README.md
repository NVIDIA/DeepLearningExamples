```
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
```
<img src="http://developer.download.nvidia.com/compute/machine-learning/frameworks/nvidia_logo.png" style="width: 90px; float: right;">


# Table Of Contents
- [BERT Question Answering Fine-Tuning and Inference with Mixed Precision](#bert-question-answering-inference/fine-tuning-with-mixed-precision)
- [BioBERT Named-Entity Recognition Inference with Mixed Precision](#biobert-named-entity-recognition-inference-with-mixed-precision)


# BERT Question Answering Inference/Fine-Tuning with Mixed Precision

## 1. Overview

Bidirectional Embedding Representations from Transformers (BERT), is a method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks.

The original paper can be found here: https://arxiv.org/abs/1810.04805.

NVIDIA's BERT 19.10 is an optimized version of Google's official implementation, leveraging mixed precision arithmetic and tensor cores on V100 GPUS for faster training times while maintaining target accuracy.

### 1.a Learning objectives

This repository contains multiple notebooks which demonstrate:
- Inference on QA task with BERT Large model
- The use/download of pretrained NVIDIA BERT models
- Fine-Tuning on SQuaD 2.0 Dataset
- Use of Mixed Precision for Inference and Fine-Tuning

Here is a short description of each relevant file:
 - _bert_squad_tf_inference.ipynb_ : BERT Q&A Inference with TF Checkpoint model
 - _bert_squad_tf_finetuning.ipynb_ : BERT Fine-Tuning on SQuaD dataset

## 2. Quick Start Guide

### 2.a Build the BERT TensorFlow NGC container:
To run the notebook you first need to build the Bert TensorFlow container using the following command from the main directory of this repository:

``` bash
docker build . --rm -t bert
```
### 2.b Dataset

We need to download the vocabulary and the bert_config files:

``` python3
python3 /workspace/bert/data/bertPrep.py --action download --dataset google_pretrained_weights  # Includes vocab
```

This is only needed during fine-tuning in order to download the Squad dataset:

``` python3
python3 /workspace/bert/data/bertPrep.py --action download --dataset squad
```

### 2.c Start of the NGC container to run inference:
Once the image is built, you need to run the container with the `--publish
0.0.0.0:8888:8888` option to publish Jupyter's port `8888` to the host machine
at port `8888` over all network interfaces (`0.0.0.0`):

```bash
nvidia-docker run \
  -v $PWD:/workspace/bert \
  -v $PWD/results:/results \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --publish 0.0.0.0:8888:8888 \
  -it bert:latest bash
```

Then you can use the following command within the BERT Tensorflow container under
`/workspace/bert`:

```bash
jupyter notebook --ip=0.0.0.0 --allow-root
```

And navigate a web browser to the IP address or hostname of the host machine
at port `8888`:

```
http://[host machine]:8888
```

Use the token listed in the output from running the `jupyter` command to log
in, for example:

```
http://[host machine]:8888/?token=aae96ae9387cd28151868fee318c3b3581a2d794f3b25c6b
```


# BioBERT Named-Entity Recognition Inference with Mixed Precision

## 1. Overview

Bidirectional Embedding Representations from Transformers (BERT), is a method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks. 

BioBERT is a domain specific version of BERT that has been trained on PubMed abstracts.

The original BioBERT paper can be found here: https://arxiv.org/abs/1901.08746

NVIDIA's BioBERT is an optimized version of the implementation presented in the paper, leveraging mixed precision arithmetic and tensor cores on V100 GPUS for faster training times while maintaining target accuracy.

### 1.a Learning objectives

This repository contains an example notebook that demonstrates:
- Inference on NER task with BioBERT model
- The use/download of fine-tuned NVIDIA BioBERT models
- Use of Mixed Precision for Inference

Here is a short description of the relevant file:
 - _biobert_ner_tf_inference.ipynb_ : BioBERT Inference with TF Checkpoint model
 
## 2. Quick Start Guide

### 2.a Build the BERT TensorFlow NGC container:
To run the notebook you first need to build the Bert TensorFlow container using the following command from the main directory of this repository:

``` bash
docker build . --rm -t bert
```
### 2.b Start of the NGC container to run inference:
Once the image is built, you need to run the container with the `--publish
0.0.0.0:8888:8888` option to publish Jupyter's port `8888` to the host machine
at port `8888` over all network interfaces (`0.0.0.0`):

```bash
nvidia-docker run \
  -v $PWD:/workspace/bert \
  -v $PWD/results:/results \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --publish 0.0.0.0:8888:8888 \
  -it bert:latest bash
```

Then you can use the following commands within the BERT Tensorflow container under
`/workspace/bert`:


Install spaCy. You'll use this to pre-process text and to visualize the results using displaCy.
```
pip install spacy
python -m spacy download en_core_web_sm
```

Launch Jupyter.
```bash
jupyter notebook --ip=0.0.0.0 --allow-root
```

And navigate a web browser to the IP address or hostname of the host machine
at port `8888`:

```
http://[host machine]:8888
```

Use the token listed in the output from running the `jupyter` command to log
in, for example:

```
http://[host machine]:8888/?token=aae96ae9387cd28151868fee318c3b3581a2d794f3b25c6b
```


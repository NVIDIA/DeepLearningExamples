# FasterTransformer

This repository provides a script and recipe to run the highly optimized transformer-based encoder and decoder component, and it is tested and maintained by NVIDIA.

## Table Of Contents

  - [Model overview](#model-overview)
    - [Configuration support matrix](#configuration-support-matrix)
    - [Model architecture](#model-architecture)
      - [Encoder](#encoder)
      - [Decoder](#decoder)
      - [Decoding](#decoding)
      - [Decoder and Decoding](#decoder-and-decoding)
  - [Setup](#setup)
    - [Requirements](#requirements)
  - [Quick Start Guide](#quick-start-guide)
    - [Build the FasterTransformer](#build-the-fastertransformer)
    - [Execute the encoder demos](#execute-the-encoder-demos)
    - [Execute the decoding demos](#execute-the-decoding-demos)
  - [Advanced](#advanced)
    - [Scripts and sample codes](#scripts-and-sample-codes)
    - [Command-line options](#command-line-options)
    - [Inference process](#inference-process)
      - [Encoder process](#encoder-process)
      - [Decoder and decoding process](#decoder-and-decoding-process)
      - [Translation process](#translation-process)
  - [Performance](#performance)
    - [Encoder performance](#encoder-performance)
    - [Decoder performance on T4](#decoder-performance-on-t4)
    - [Decoding performance on T4](#decoding-performance-on-t4)
    - [Decoding performance on V100](#decoding-performance-on-v100)
  - [Release notes](#release-notes)
    - [Changelog](#changelog)
    - [Known issues](#known-issues)

## Model overview


In NLP, encoder and decoder are two important components, with the transformer layer becoming a popular architecture for both components. FasterTransformer implements a highly optimized transformer layer for both the encoder and decoder for inference. On Volta and Turing GPUs, the computing power of Tensor Cores are used automatically when the precision of the data and weights are FP16. 

In FasterTransformer 1.0, we implemented a highly optimized BERT transformer layer, which is used in the encoder. 

In FasterTransformer 2.0, we have added a highly optimized decoder and decoding models based on OpenNMT-TF, an open-source library. Here, the decoder is the model that contains some transformer layers. On the other hand, decoding refers to the whole translating process, including the lookup embedding table, position encoding, a decoder and beam search. 

The following graph demonstrates the model architecture. 

![](images/encoder-decoding.png)

FasterTransformer is built on top of CUDA and cuBLAS, providing the C++ API and TensorFlow OP. Users can integrate them into TensorFlow or other inference service codes that are built in native C++. We also provide some simple sample code to demonstrate how to use the encoder, decoder and to carry out decoding in C++ and TensorFlow. 

### Configuration support matrix

The following configurations are supported in the FasterTransformer encoder. 
- Batch size (B<sub>1</sub>): smaller or equal to 512
- Sequence length (S): larger than 3 and smaller or equal to 1024 
- Head number (H) and size per head (N): 
  - 12 heads * 64 per heads
  - 4 heads * 32 per heads
  - 8 heads * 96 per heads
- Data type: FP32 and FP16

The following configurations are supported in the FasterTransformer decoder and decoding.
- Batch size (B<sub>1</sub>) * beam width (B<sub>2</sub>): smaller than 1024
- Sequence length (S): smaller than 1024
- Head number (H): 8 and 12
- Size per head (N): 64
- Vocabulary size (V): from 64 to 30000
- Data type: FP32 and FP16

Note: For Encoder-Decoding structure, the sequence length of Encoder and Decoding must be the same. 

### Model architecture

#### Encoder
The encoder requires the following inputs:
  1. An input tensor. The shape is \[ B<sub>1</sub>, S, H x N\].
  2. An attention mask.
  3. The weights of all parameters.

The encoder will return the following outputs:
  1. The encoder output feature. The shape is \[ B<sub>1</sub>, S, H x N \].

#### Decoder
The decoder requires the following inputs:
  1. The features vector obtained by looking up the embedding table, or the previous result of the decoder. The shape is \[ B<sub>1</sub> x B<sub>2</sub>, 1, H x N \].
  2. The output of the encoder.
  3. The sequence length of the source sentence. Note that the lengths should be expanded by beam width times. 
  4. A memory cache space to store the K, V of masked multi-head attention. The size will grow for each step.
  5. A memory cache space to store the K, V of cross attention. Since K, V is computed by the encoder result, we only compute them in the first step, storing them into the cache, and then reuse in the other steps. 
  6. The weights of all parameters.
  7. In order to prevent the parallel computing of TensorFlow decoder and FasterTransformer Decoder, we put the TensorFlow result as a pseudo input in the TensorFlow OP. Otherwise, the results of FasterTransformer Decoder will incorrect. This input is useless for computing. Users can remove it when applying Decoder into a real application.  

The decoder will return the following outputs:
  1. Memory cache of masked multi-head attention. 
  2. Memory cache of cross attention. 
  3. The decoder output feature. The shape is \[ B<sub>1</sub> x B<sub>2</sub>, 1, H x N \].

#### Decoding
Decoding refers to the whole translating process, including position encoding, embedding lookup, and a simple beam search kernel.

Decoding requires the following inputs:
  1. The output of the encoder. The shape is \[ B<sub>1</sub>, memory sequence length, H x N \].
  2. The sequence length of the source sentence. Note that the lengths should be expanded by beam width times.
  3. The table for embedding lookup. The shape is \[ V, H x N \].
  4. The start id and end id for the vocabulary. 
  5. The weights of all parameters.

Decoding returns the following outputs:
  1. The output ids. The shape is \[ B<sub>1</sub> x B<sub>2</sub> \].
  2. The parent ids, which are the chosen beam ids.
  3. The sequence lengths of each sentence. 

Note that these results are required to be finalized by TensorFlow's `tf.contrib.seq2seq.gather_tree` or other progress. 

#### Decoder and Decoding
Although the decoding process of most methods is similar, we find that there are lots of different kinds to compute the probability and implement the beam search. Therefore, if your chosen beam search algorithm is different from our implementation and it is hard for you to modify the beam search kernel, TensorFlow decoding with FasterTransformer Decoder is the recommended choice. However, the performance of the TensorFlow decoding with the FasterTransformer Decoder is worse than the performance of the FasterTransformer Decoding, especially for small batch sizes.

## Setup

The following section lists the requirements in order to use FasterTransformer.

### Requirements

- CMake >= 3.8 
- CUDA 10.1
- Python 2.7
- Tensorflow 1.14
These components are readily available within the NGC TensorFlow Docker image below.

Ensure you have the following components:
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [TensorFlow 19.07-py2+](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) NGC container
- [NVIDIA Pascal](https://www.nvidia.com/en-us/data-center/pascal-gpu-architecture/) or [Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) based GPU

For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
- [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
- [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)
- [Running TensorFlow](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html#running)

For those unable to use the TensorFlow NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## Quick Start Guide 

The following section shows how to use FasterTransformer on the NGC container. 

### Build the FasterTransformer

1. Run the container.

```bash
nvidia-docker run -ti nvcr.io/nvidia/tensorflow:19.07-py2 bash
```

2. Clone the repository.

```bash
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/FasterTransformer/v2
git submodule init
git submodule update
```

3. Build the project.

```bash
ln -s /usr/local/lib/python2.7/dist-packages/tensorflow/libtensorflow_framework.so.1 /usr/local/lib/python2.7/dist-packages/tensorflow/libtensorflow_framework.so
mkdir -p build
cd build
cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release .. # C++ only
cmake -DSM=xx -DCMAKE_BUILD_TYPE=Debug .. # C++ debug only
cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_TF=ON -DTF_PATH=/usr/local/lib/python2.7/dist-packages/tensorflow .. # Tensorflow mode
make
```

Note: `xx` is the compute capability of your GPU. For example, 60 (P40) or 61 (P4) or 70 (V100) or 75(T4).

### Execute the encoder demos

1. Generate the `gemm_config.in` file. 

```bash
 ./bin/encoder_gemm <batch_size> <sequence_length> <head_number> <size_per_head> <is_use_fp16>
./bin/encoder_gemm 1 32 12 64 0
``` 

2. Run the encoder.

a.	Run the encoder in C++ by running the following scripts: 

```bash
./bin/encoder_sample <batch_size> <num_layers> <sequence_length> <head_number> <size_per_head> <is_use_fp16>
./bin/encoder_sample 1 12 32 12 64 0 
```

b.	Run the encoder in TensorFlow by running the following scripts: 

```bash
python encoder_sample.py \
        --batch_size 1 \
        --seq_len 32 \
        --head_number 12 \
        --size_per_head 64 \
        --num_layer 12 \
        --data_type fp32 \
        --test_time 1
```

c.	Run the encoder in FP16:

Note that the configuration of FP32 and FP16 are different, so it is necessary to generate the configuration again. 

```bash
./bin/encoder_gemm 1 32 12 64 1
./bin/encoder_sample 1 12 32 12 64 1
python encoder_sample.py \
        --batch_size 1 \
        --seq_len 32 \
        --head_number 12 \
        --size_per_head 64 \
        --num_layer 12 \
        --data_type fp16 \
        --test_time 1
```
3. Run the FasterTransformer in BERT.

The following script demonstrates how to integrate the FasterTransformer into a BERT model. This requires the repo of [BERT](https://github.com/google-research/bert).

a.	Prepare the BERT codes, Download the BERT pretrained model.

```bash
cd tensorflow_bert
git clone https://github.com/google-research/bert.git
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
```

b. Download the GLUE MRPC dataset. Note that the file `download_glue_data.py` can only executed under python3. 

```bash
wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py
python download_glue_data.py --tasks MRPC
```

c. Finetune the pretrained model on MRPC datasets. This takes some minutes. The accuracy would be better or worse because the MRPC dataset is very small. 

```bash
export BERT_BASE_DIR=${PWD}/uncased_L-12_H-768_A-12
export GLUE_DIR=${PWD}/glue_data/

python bert/run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=mrpc_output/
```

The results would be like: 
```bash
I0403 08:52:49.721482 140547349206848 estimator.py:2039] Saving dict for global step 343: eval_accuracy = 0.87009805, eval_loss = 0.44462326, global_step = 343, loss = 0.44462326
I0403 08:52:50.128525 140547349206848 estimator.py:2099] Saving 'checkpoint_path' summary for global step 343: mrpc_output/model.ckpt-343
I0403 08:52:50.129132 140547349206848 error_handling.py:96] evaluation_loop marked as finished
I0403 08:52:50.129281 140547349206848 run_classifier.py:923] ***** Eval results *****
I0403 08:52:50.129338 140547349206848 run_classifier.py:925]   eval_accuracy = 0.87009805
I0403 08:52:50.129695 140547349206848 run_classifier.py:925]   eval_loss = 0.44462326
I0403 08:52:50.129786 140547349206848 run_classifier.py:925]   global_step = 343
I0403 08:52:50.129833 140547349206848 run_classifier.py:925]   loss = 0.44462326
```

d. Conver the finetuned checkpoint to FP16, check the accuracy of Fastertransformer under FP16. 

```bash
python ckpt_type_convert.py --init_checkpoint=mrpc_output/model.ckpt-343 --fp16_checkpoint=mrpc_output/fp16_model.ckpt
python run_classifier_wrap.py   --floatx=float16   --task_name=MRPC   --do_eval=true   --data_dir=$GLUE_DIR/MRPC   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=mrpc_output/fp16_model.ckpt   --max_seq_length=128   --eval_batch_size=8   --output_dir=mrpc_output
```

Because we do not generate the `gemm_config.ini` file, you can see many warning messages like:

```bash
gemm_config.in is not found
loading GEMM algorithms error, using default GEMM algorithms
gemm_config.in is not found
loading GEMM algorithms error, using default GEMM algorithms!
I0403 08:55:07.053885 140260684429120 evaluation.py:275] Finished evaluation at 2020-04-03-08:55:07
I0403 08:55:07.054126 140260684429120 estimator.py:2039] Saving dict for global step 343: eval_accuracy = 0.86764705, eval_loss = 0.45615184, global_step = 343, loss = 0.4561844
I0403 08:55:07.422543 140260684429120 estimator.py:2099] Saving 'checkpoint_path' summary for global step 343: mrpc_output/fp16_model.ckpt
I0403 08:55:07.423089 140260684429120 error_handling.py:96] evaluation_loop marked as finished
I0403 08:55:07.423257 140260684429120 run_classifier.py:923] ***** Eval results *****
I0403 08:55:07.423315 140260684429120 run_classifier.py:925]   eval_accuracy = 0.86764705
I0403 08:55:07.423553 140260684429120 run_classifier.py:925]   eval_loss = 0.45615184
I0403 08:55:07.423635 140260684429120 run_classifier.py:925]   global_step = 343
I0403 08:55:07.423686 140260684429120 run_classifier.py:925]   loss = 0.4561844
```

This shows that we use the FasterTransformer to run the inference successfully. In this case, using FP16 to do inference will reduce the accuracy with about 0.3%.

e. Compare the speed of BERT of TensorFlow and FasterTransformer under both FP32 and FP16.

```bash
../bin/encoder_gemm 1 32 12 64 0
python profile_transformer_inference.py --init_checkpoint=mrpc_output/model.ckpt-343 --tf_profile=false --output_dir=mrpc_output --profiling_output_file=time_elapsed --xla=false --floatx=float32
../bin/encoder_gemm 1 32 12 64 1
python profile_transformer_inference.py --init_checkpoint=mrpc_output/fp16_model.ckpt --tf_profile=false --output_dir=mrpc_output --profiling_output_file=time_elapsed --xla=false --floatx=float16
```

The results of FP16 under V100 would be like:

```bash
average time (seconds) elasped original tensorflow: 0.011663460731506347
average time (seconds) elasped fast transformer: 0.007064676284790039
```

### Execute the decoding demos

1. Generate the `decoding_gemm_config.in` file. 

```bash
./bin/decoding_gemm <batch_size> <beam_width> <head_number> <size_per_head> <sequence_length> <encoder_hidden_dim> <is_use_fp16>
./bin/decoding_gemm 32 4 8 64 30000 32 768 0
```

2. Run the decoder and decoding. 
 
a.	Run the decoding in C++ by running the following script: 

```bash
./bin/decoding_sample <batch_size> <beam_width> <head_number> <size_per_head> <sequence_length> <num_layers> <encoder_hidden_dim> <is_use_fp16>
./bin/decoding_sample 32 4 8 64 30000 32 6 768 0
```

b.	Run the decoder in TensorFlow by running the following script: 

```bash
python decoder_sample.py \
        --batch_size 32 \
        --beam_width 4 \
        --max_seq_len 32 \
        --head_number 8 \
        --size_per_head 64 \
        --memory_hidden_dim 768 \
        --num_layer 6 \
        --data_type fp32 \
        --decoder_type 2
```

c.	Run the decoding in TensorFlow by running the following script: 
    
```bash
python decoding_sample.py \
        --batch_size 32 \
        --beam_width 4 \
        --max_seq_len 32 \
        --head_number 8 \
        --size_per_head 64 \
        --memory_hidden_dim 768 \
        --num_layer 6 \
        --data_type fp32
```

3. Run the encoder and decoding at the same time.

```bash
python encoder_decoding_sample.py \
        --batch_size 32 \
        --beam_width 4 \
        --max_seq_len 32 \
        --encoder_head_number 12 \
        --encoder_size_per_head 64 \
        --decoder_head_number 8 \
        --decoder_size_per_head 64 \
        --encoder_num_layer 6 \
        --decoder_num_layer 6 \
        --data_type fp32
```

## Advanced

The following sections provide greater details.

### Scripts and sample codes

The following code lists the directory structure of FasterTransformer: 

```bash
/fastertransformer: source code of transformer
   |--/cuda: some CUDA kernels and multi-head attention implementation, both are compiled with cuda/cuBLAS. 
   |--/tf_op: custom Tensorflow OP implementation
   |--/trt_plugin: TensorRT plugin implementation
/sample: c++ and tensorflow transformer interface samples
   |--/cpp: c++ interface samples
   |--/tensorflow_bert: samples that show of how to integrate our Tensorflow OP into the open source BERT model for sentence (and sentence-pair) classification tasks (GLUE), the samples support both FP16 and FP32, see readme file within this folder more details
   |--/tensorflow:TensorFlow OP samples
   |--/tensorRT: both FP16 and FP32 tensorRT plugin samples
/tools/gemm_test: loop over all GEMM algorithms to pick the best one
```

In the root directory of FasterTransformer, the most important directories are:
* `fastertransformer/`
* `sample/`
* `tools/`

The `fastertransformer/` folder encapsulates all the source codes of FasterTransformer:
* `tf_op/` - Contains the TensorFlow Op source files of encoder, decoder and decoding 
* `cuda/` - Contains all CUDA kernels of FasterTransformer
* `bert_encoder_transformer.h` - Contains the encoder transformer layer 
* `open_decoder.h` - Contains the decoder transformer layer
* `beam_search_opennmt.h` - Contains the beam search progress for decoding
* `decoding_opennmt.h` - Contains the decoding progress

The `tools/` folder contains the tools to generate the GEMM configuration of FasterTransformer for different settings: 
* `tools/gemm_test/encoder_gemm.cc` - Encoder GEMM config
* `tools/gemm_test/decoding_gemm.cc` - Decoder and decoding GEMM config 

The `sample/` folder contains useful sample codes for FasterTransformer:
* `sample/cpp/encoder_sample.cc` - C encoder sample codes 
* `sample/cpp/decoding_sample.cc` - C decoding sample codes 
* `sample/tensorflow/encoder_sample.py` - TensorFlow encoder sample codes 
* `sample/tensorflow/decoder_sample.py` - TensorFlow decoder sample codes 
* `sample/tensorflow/decoding_sample.py` - TensorFlow decoding sample codes 
* `sample/tensorflow/encoder_decoder_sample.py` - TensorFlow `encoder_decoder` sample codes 
* `sample/tensorflow/encoder_decoding_sample.py` - TensorFlow `encoder_decoding` sample codes 
* `sample/tensorflow/translate_sample.py` - TensorFlow translation sample codes

### Command-line options

To see the full list of available options and their descriptions, use the `-h` or `--help` command-line option with the Python file, for example:

```bash
python encoder_sample.py --help
python decoder_sample.py --help
python decoding_sample.py --help
python encoder_decoder_sample.py --help
python encoder_decoding_sample.py --help
python translate_sample.py --help
```

### Inference process

This subsection provides the details about how to use the encoder, the decoder and the decoding. 

#### Encoder process

1. Generate the `gemm_config.in` file. 

`./bin/encoder_gemm` can generate the best GEMM configuration. The arguments of `encoder_gemm` is:

```bash
./bin/encoder_gemm <batch_size> <sequence_length> <head_number> <size_per_head> <is_use_fp16>
```

Assume the settings of the encoder are as follows:
- `batch_size`=1
- `sequence_length`=32
- `head_number`=12
- `size_per_head`=64 
- `data_type`=FP32

Then the following scripts can generate the best GEMM configuration under such settings, and record the configuration into the `gemm_config.in.in` file.

```bash
./bin/encoder_gemm 1 32 12 64 0
```

2. Run the encoder.

Assume the settings are the same as above, and the encoder contains 12 transformer layers. 

a.	Run the encoder in C++ by running the following scripts: 
		
`./bin/encoder_sample` runs the encoder in the `cpp`. The arguments of `encoder_sample` is:

```bash
./bin/encoder_sample <batch_size> <num_layers> <sequence_length> <head_number> <size_per_head> <is_use_fp16>
```

Then the following scripts can run the encoder under the above settings. 

```bash
./bin/encoder_sample 1 12 32 12 64 0 
```

The outputs should be similar to the following:  
    
```bash 
Device Tesla T4
before allocate free 14.65 GB total 14.76 GB
After allocate free 14.61 GB used 0.14 GB total 14.76 GB
[batch_size 1 seq_len 32 12 transformer layers] costs 3.08 ms
```

b.	Run the encoder in TensorFlow by running the following scripts: 

The following script demonstrates the cross check between the encoder of TensorFlow and the encoder of FasterTransformer, and the execution time of them.

```bash
python encoder_sample.py \
        --batch_size 1 \
        --seq_len 32 \
        --head_number 12 \
        --size_per_head 64 \
        --num_layer 12 \
        --data_type fp32 \
        --test_time 1
```

The outputs should be similar to the following:

```bash
[INFO] Encoder Cross check True
[INFO] Max diff 3.57627868652e-06
[INFO] min diff 0.0
[INFO] TF decoder time costs: 6.63149 ms
[INFO] OP decoder time costs: 4.64135 ms
```

c.	Run the encoder in FP16:

Note that the configuration of FP32 and FP16 are different, so it is necessary to generate the configuration again. 

For C, users only need to set the `<is_use_fp16>` flag as 1. 

For TensorFlow, users can use the arguments `--data_type fp16` to change the computing mode. 

```bash
./bin/encoder_gemm 1 32 12 64 1
./bin/encoder_sample 1 12 32 12 64 1
python encoder_sample.py \
        --batch_size 1 \
        --seq_len 32 \
        --head_number 12 \
        --size_per_head 64 \
        --num_layer 12 \
        --data_type fp16 \
        --test_time 1
```

#### Decoder and decoding process

1. Generate the `decoding_gemm_config.in` file. 

`./bin/decoding_gemm` can generate the best GEMM configuration. The arguments of `decoding_gemm` are:

```bash
./bin/decoding_gemm <batch_size> <beam_width> <head_number> <size_per_head> <sequence_length> <encoder_hidden_dim> <is_use_fp16>
```

Assume the settings of decoding are as follows.

- `batch_size`=32
- `beam_width`=4
- `head_number`=8
- `size_per_head`=64 
- `vocabulary_size`=30000
- `sequence_length`=32
- `encoder's hidden dimension`=768
- `data_type`=FP32

Then the following scripts can generate the best GEMM configuration under such settings, and record the configuration into the `decoding_gemm_config.in` file.

```bash
./bin/decoding_gemm 32 4 8 64 30000 32 768 0
```

2. Run the decoder and decoding. 

Assume the settings are the same as above, and the decoder contains 6 transformer layers. 

a.	Run the decoding in C++ by running the following script: 

`./bin/decoding_sample` runs the decoding in the `cpp`. The arguments of `encoder_sample` is:

```bash
./bin/decoding_sample <batch_size> <beam_width> <head_number> <size_per_head> <sequence_length> <num_layers> <encoder_hidden_dim> <is_use_fp16>
```

Then the following scripts can run the decoding under the above settings. 

```bash
./bin/decoding_sample 32 4 8 64 30000 32 6 768 0
```

The outputs should be similar to the following:
    
```bash 
Device Tesla T4
[batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000] costs 191.21 ms
done
```

b.	Run the decoder in TensorFlow by running the following script: 

```bash
python decoder_sample.py \
        --batch_size 32 \
        --beam_width 4 \
        --max_seq_len 32 \
        --head_number 8 \
        --size_per_head 64 \
        --memory_hidden_dim 768 \
        --num_layer 6 \
        --data_type fp32 \
        --decoder_type 2
```

The outputs should be similar to the following:

```bash 
[[INFO][PYTHON] step:][0][max diff: ][5.00679e-06][ op val: ][2.3735888][ tf val: ][2.37359381][True]
[[INFO][PYTHON] step:][1][max diff: ][4.64916229e-06][ op val: ][-0.588810563][ tf val: ][-0.588815212][True]
[[INFO][PYTHON] step:][2][max diff: ][5.36441803e-06][ op val: ][-1.46514082][ tf val: ][-1.46514618][True]
...
[[INFO][PYTHON] step:][29][max diff: ][4.529953e-06][ op val: ][2.88768935][ tf val: ][2.88769388][True]
[[INFO][PYTHON] step:][30][max diff: ][4.17232513e-06][ op val: ][-1.28717053][ tf val: ][-1.2871747][True]
[[INFO][PYTHON] step:][31][max diff: ][4.05311584e-06][ op val: ][-1.01830876][ tf val: ][-1.01831281][True]
```

The results show that the differences between the decoder of TensorFlow and decoder are smaller than threshold. Note that the differences are absolute differences, so the differences may be large when the op val is large. In this case, the differences are larger than the threshold and the checking will return "False", but it may be not affect the final results.

The option `decoder_type` decides to use the decoder of TensorFlow or decoder of FasterTransformer. `decoder_type 2` uses both decoders and compares their results. 

The following script demonstrates the execution time of the FasterTransformer decoder.

```bash
python decoder_sample.py \
        --batch_size 32 \
        --beam_width 4 \
        --max_seq_len 32 \
        --head_number 8 \
        --size_per_head 64 \
        --memory_hidden_dim 768 \
        --num_layer 6 \
        --data_type fp32 \
        --decoder_type 1 \
        --test_time 1
```

The outputs should be similar to the following:

```bash
[INFO] time costs of OP decoder: 248.046 ms.
```

The following script demonstrates the execution time of the TensorFlow decoder.

```bash 
python decoder_sample.py \
        --batch_size 32 \
        --beam_width 4 \
        --max_seq_len 32 \
        --head_number 8 \
        --size_per_head 64 \
        --memory_hidden_dim 768 \
        --num_layer 6 \
        --data_type fp32 \
        --decoder_type 0 \
        --test_time 1
```

c.	Run the decoding in TensorFlow by running the following script: 
    
```bash
python decoding_sample.py \
        --batch_size 32 \
        --beam_width 4 \
        --max_seq_len 32 \
        --head_number 8 \
        --size_per_head 64 \
        --memory_hidden_dim 768 \
        --num_layer 6 \
        --data_type fp32
```

The outputs should be similar to the following:

```bash
       Output ids cross-check: True
 
       Parent ids cross-check: True
 
       Sequence lengths cross-check: True
 
       Finalized output ids cross-check: True
```

Note that the results of OP and the results of TensorFlow are often different in the random inputs and weights. 

3. Run the encoder and decoding at the same time.

```bash
python encoder_decoding_sample.py \
        --batch_size 32 \
        --beam_width 4 \
        --max_seq_len 32 \
        --encoder_head_number 12 \
        --encoder_size_per_head 64 \
        --decoder_head_number 8 \
        --decoder_size_per_head 64 \
        --encoder_num_layer 6 \
        --decoder_num_layer 6 \
        --data_type fp32
```

#### Translation process

This subsection demonstrates how to use FasterTansformer decoding to translate a sentence. We use the pretrained model and testing data in [OpenNMT-tf](https://opennmt.net/Models-tf/), which translate from English to German. 

Because the FasterTransformer Encoder is based on BERT, we cannot restore the model of encoder of OpenNMT to FasterTransformer Encoder. Therefore, we use OpenNMT-tf to build the encoder and preprocess the source sentence.

Another problem is that the implementation of FasterTransformer Decoder and decoder of OpenNMT-tf is a little different. For example, the decoder of OpenNMT-tf uses one convolution to compute query, key and value in masked-multihead-attention; but FasterTransformer Decoder splits them into three gemms. The tool `utils/dump_model.py` will convert the pretrained model to fit the model structure of FasterTransformer Decoder.

`download_model_data.sh` will install the OpenNMT-tf v1, downloads the pretrained model into the `translation` folder, and convert the model. 

```bash
bash utils/translation/download_model_data.sh
```

Then run the translation sample by the following script:

```bash
./bin/decoding_gemm 1 4 8 64 32001 100 512 0
python translate_sample.py
```

The outputs should be similar to the following:

```bash
[INFO] opennmt: ▁28 - jährige r ▁Chef koch ▁to t ▁in ▁San ▁Francisco </s>
[INFO] tf     : ▁28 - jährige r ▁Chef koch ▁to t ▁in ▁San ▁Francisco </s>
[INFO] op     : ▁28 - jährige r ▁Chef koch ▁to t ▁in ▁San ▁Francisco </s>
```

## Performance

Hardware settings: 
* CPU: Intel(R) Xeon(R) Gold 6132 CPU @ 2.60GHz
* T4 (with mclk 5000MHz, pclk 1590MHz)  
* P4 (with mclk 3003MHz, pclk 1531MHz)  
* V100 (with mclk 877MHz, pclk 1380MHz)  

In the following experiments, we updated the following parameters:  
* head_num = 8 
* size_per_head = 64 
* transformer layers = 6 
* vocabulary_size = 30000 

For Encoder, the reported time is the average inference time for 100 iterations with 100 warm-up iterations. 

For Decoder and Decoding, the reported time the is average inference time for 50 iterations with 50 warm-up iterations.

### Encoder performance

We demonstrate the inference time of FasterTransformer in C++ and compare it to the inference time of TensorFlow in Python. 

| <batch_size, layers, seq_len, head_num, size_per_head> | P4 FP32 (in ms) | T4 FP32 (in ms) | T4 FP16 (in ms) |
|:--------------------:|:----:|:---------:|:-----------:|
| (1, 12, 32, 12, 64)  | 3.43 | 2.74 | 1.56 |
| (1, 12, 64, 12, 64)  | 4.04 | 3.64 | 1.77 | 
| (1, 12, 128, 12, 64) | 6.22 | 5.93 | 2.23 |

For large batch size cases, we report both TensorFlow XLA and faster transformer's performance.

| <batch_size, layers, seq_len, head_num, size_per_head> | TensorFlow XLA on V100 FP16 (in ms) | FasterTransformer V100 FP16 (in ms) | Speedup |
|:-------------:|:-------------:|:---------:|:-----------:|
| (100, 12, 32, 12, 64)  | 13.96  | 9.57 | 1.459 |
| (200, 12, 32, 12, 64)  | 26.47  | 18.37 | 1.44 |
| (300, 12, 32, 12, 64)  | 38.4  | 27.41 | 1.401 |
| (400, 12, 32, 12, 64)  | 49.65  | 35.63 | 1.393 |
| (500, 12, 32, 12, 64)  | 62.2  | 44.57 | 1.396 |

| <batch_size, layers, seq_len, head_num, size_per_head> | TensorFlow XLA on V100 FP16 (in ms) | FasterTransformer V100 FP16 (in ms) | Speedup |
|:-------------:|:-------------:|:---------:|:-----------:|
| (100, 12, 32, 4, 32)  | 3.49  | 1.73 | 2.017 |
| (200, 12, 32, 4, 32)  | 4.9  | 2.55 | 1.922 |
| (300, 12, 32, 4, 32)  | 6.35  | 3.356 | 1.892 |
| (400, 12, 32, 4, 32)  | 8  | 4.31 | 1.856 |
| (500, 12, 32, 4, 32)  | 9.93  | 5.13 | 1.936 |

### Decoder performance on T4

We do not demonstrate the performance of TensorFlow with XLA since we did not find that using XLA has obvious speedup. 

The following results of FasterTransformer are generated by 

```bash
bash scripts/profile_decoder_op_performance.sh
```

* We set beam_width = 1
* We replace the decoder of tensorflow with our decoder op. 

| <batch_size, seq_len> | TensorFlow FP32 (in ms) | Decoder FP32 (in ms) | FP32 Speedup | TensorFlow FP16 (in ms) | Decoder FP16 (in ms) | FP16 Speedup |
|:---------:|:-------:|:------:|:----:|:-------:|:------:|:----:|
| (1, 32)   | 441.68  | 111.14 | 3.97 | 508.81  | 165.88 | 3.06 |
| (1, 64)   | 872.39  | 207.37 | 4.20 | 1038.71 | 326.69 | 3.18 |
| (1, 128)  | 1714.01 | 457.62 | 3.74 | 2082.92 | 661.00 | 3.41 |
| (32, 32)  | 470.93  | 119.87 | 3.92 | 568.83  | 167.42 | 3.39 |
| (64, 32)  | 503.57  | 153.62 | 3.27 | 579.21  | 183.74 | 3.15 |
| (128, 32) | 614.59  | 245.94 | 2.50 | 641.98  | 238.27 | 2.69 |
| (256, 32) | 802.18  | 439.33 | 2.01 | 735.67  | 348.74 | 2.11 |

### Decoding performance on T4 

We do not demonstrate the performance of TensorFlow with XLA since we did not find that using XLA has obvious speedup. 

The following results are generated by 

```bash
bash scripts/profile_decoding_op_performance.sh
```

* We set beam_width = 4

| <batch_size, seq_len> | TensorFlow FP32 (in ms) | Decoder FP32 (in ms) | FP32 Speedup | TensorFlow FP16 (in ms) | Decoder FP16 (in ms) | FP16 Speedup |
|:------------:|:-------:|:-------:|:----:|:-------:|:------:|:-----:|
| (1, 32)   | 430.39  | 64.16   | 6.70 | 537.95  | 49.07  | 10.96 |
| (1, 64)   | 876.24  | 135.42  | 6.47 | 1056.78 | 97.45  | 10.84 |
| (1, 128)  | 1799.16 | 318.65  | 5.64 | 2145.74 | 240.85 | 8.91  |
| (32, 32)  | 597.42  | 217.61  | 2.74 | 646.07  | 128.39 | 5.03  |
| (64, 32)  | 789.22  | 395.85  | 1.99 | 769.17  | 246.89 | 3.11  |
| (128, 32) | 1223.72 | 726.43  | 1.68 | 996.03  | 424.53 | 2.34  |
| (256, 32) | 2188.00 | 1385.60 | 1.58 | 1599.58 | 781.38 | 2.04  |

### Decoding performance on V100

We do not demonstrate the performance of TensorFlow with XLA since we did not find that using XLA has obvious speedup. 

The following results are generated by 

```bash
bash scripts/profile_decoding_op_performance.sh
```

* We set beam_width = 4

| <batch_size, seq_len> | TensorFlow FP32 (in ms) | Decoder FP32 (in ms) | FP32 Speedup | TensorFlow FP16 (in ms) | Decoder FP16 (in ms) | FP16 Speedup |
|:------------:|:-------:|:------:|:----:|:-------:|:------:|:-----:|
| (1, 32)   | 440.46  | 58.70  | 7.50 | 531.70  | 46.18  | 11.51 |
| (1, 64)   | 888.19  | 122.50 | 7.25 | 1065.76 | 93.84  | 11.35 |
| (1, 128)  | 1821.76 | 293.21 | 6.21 | 2076.63 | 293.21 | 7.08  |
| (32, 32)  | 543.27  | 101.35 | 5.36 | 630.55  | 73.37  | 8.59  |
| (64, 32)  | 648.27  | 157.54 | 4.11 | 793.83  | 106.77 | 7.43  |
| (128, 32) | 838.43  | 277.77 | 3.02 | 867.71  | 169.04 | 5.13  |
| (256, 32) | 1221.30 | 493.85 | 2.47 | 1101.36 | 290.44 | 3.79  |

## Release notes

### Changelog

March 2020
- Add feature in FasterTransformer 2.0
  - Add `translate_sample.py` to demonstrate how to translate a sentence by restoring the pretrained model of OpenNMT-tf.
- Fix bugs of Fastertransformer 2.0
  - Fix the bug of maximum sequence length of decoder cannot be larger than 128.
  - Fix the bug that decoding does not check finish or not after each step. 
  - Fix the bug of decoder about max_seq_len.
  - Modify the decoding model structure to fit the OpenNMT-tf decoding model. 
    - Add a layer normalization layer after decoder.
    - Add a normalization for inputs of decoder

Febuary 2020
- Release the FasterTransformer 2.0
  - Provide a highly optimized OpenNMT-tf based decoder and decoding, including C++ API and TensorFlow op. 
  - Refine the sample codes of encoder.
  - Add dynamic batch size feature into encoder op.

July 2019
- Release the FasterTransformer 1.0
  - Provide a highly optimized bert equivalent transformer layer, including C++ API, TensorFlow op and TensorRT plugin. 

### Known issues

- batch_size should be smaller or equal to 1024 in Decoder.
- batch_size x beam_width should be smaller or equal to 1024 in Decoding.
- Results of TensorFlow and OP would be different in decoding. This problem is caused by the accumulated log probability, and we do not avoid this problem. 
- Cmake 15 or Cmake 16 fail to build this project. Cmake 14 is no problem. 
- Max sequence length of encoder and decoder should be the same. 

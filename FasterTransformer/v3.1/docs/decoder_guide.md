# FasterTransformer Decoder

The FasterTransformer Decoder contains the transformer decoder block, whole decoding progress, and GPT-2 model. 

## Table Of Contents

- [FasterTransformer Decoder](#fastertransformer-decoder)
  - [Table Of Contents](#table-of-contents)
  - [Model architecture](#model-architecture)
    - [Decoder](#decoder)
    - [Decoding progress](#decoding-progress)
    - [Decoder and Decoding](#decoder-and-decoding)
    - [GPT-2](#gpt-2)
  - [Setup](#setup)
    - [Requirements](#requirements)
  - [How to use](#how-to-use)
    - [Decoder and decoding process](#decoder-and-decoding-process)
    - [Translation process](#translation-process)
    - [GPT-2 process](#gpt-2-process)
  - [Performance](#performance)
    - [Decoder performance](#decoder-performance)
      - [Decoder performance on T4 and TensorFlow](#decoder-performance-on-t4-and-tensorflow)
      - [Decoder performance on V100 and TensorFlow](#decoder-performance-on-v100-and-tensorflow)
    - [Decoding performance](#decoding-performance)
      - [Decoding performance on T4 and TensorFlow](#decoding-performance-on-t4-and-tensorflow)
      - [Decoding performance on V100 and TensorFlow](#decoding-performance-on-v100-and-tensorflow)
      - [Decoder and decoding performance on T4 and PyTorch](#decoder-and-decoding-performance-on-t4-and-pytorch)
      - [Decoder and decoding performance on V100 and PyTorch](#decoder-and-decoding-performance-on-v100-and-pytorch)
      - [TensorFlow performance on translation](#tensorflow-performance-on-translation)
      - [PyTorch performance on translation](#pytorch-performance-on-translation)
      - [GPT-2 performance on V100 and TensorFlow](#gpt-2-performance-on-v100-and-tensorflow)

## Model architecture

<div align=center><img  width='616' height='1256' src ="images/decoding_flowchart.png "/></div>
<div align=center>Fig. 1 Flowchart of Decoding and GPT-2.</div>

### Decoder

Here, Decoder means the decoder transformer block, which contains two attention block and a feed-forward network. The modules in the red block of Fig. 1 demonstrate Decoder block.

The arguments, inputs, and outputs of decoder: 

* Arguments:
  1. Head number (H)
  2. size per head (N)
* Inputs:
  1. The features vector obtained by looking up the embedding table, or the previous result of the decoder. The shape is \[ B<sub>1</sub> x B<sub>2</sub>, 1, H x N \].
  2. The output of the encoder.
  3. The sequence length of the source sentence. Note that the lengths should be expanded by beam width times. 
  4. A memory cache space to store the K, V of masked multi-head attention. The size will grow for each step.
  5. A memory cache space to store the K, V of cross attention. Since K, V is computed by the encoder result, we only compute them in the first step, storing them into the cache, and then reuse in the other steps. 
  6. The weights of all parameters.
  7. To prevent the parallel computing of TensorFlow decoder and FasterTransformer Decoder, we put the TensorFlow result as a pseudo input in the TensorFlow OP. Otherwise, the results of FasterTransformer Decoder will incorrect. This input is useless for computing. Users can remove it when applying Decoder into a real application.  
* Outputs:
  1. Memory cache of masked multi-head attention. 
  2. Memory cache of cross attention. 
  3. The decoder output feature. The shape is \[ B<sub>1</sub> x B<sub>2</sub>, 1, H x N \].

### Decoding progress

Decoding refers to the whole translating process, including position encoding, embedding lookup, and beam search or sampling method to choose the token. Fig. 1 shows the different between decoding with beam search and sampling.

The arguments, inputs, and outputs of decoding with beam search: 

* Arguments:
  1. Beam width (B<sub>2</sub>)
  2. Maximum sequence length (S)
  3. Head number (H)
  4. Size per head (N)
  5. Number of decoder layers
  6. Start id of the vocabulary
  7. End id of the vocabulary
  8. Beam search diversity rate of [simple diverse decoding](https://arxiv.org/pdf/1611.08562.pdf)
* Inputs:
  1. The output of the encoder. The shape is \[ B<sub>1</sub>, memory sequence length, H x N \]. Where B<sub>1</sub> means the batch size.
  2. The sequence length of the source sentence. Note that the lengths should be expanded by beam width times.
  3. The table for embedding lookup. The shape is \[ V, H x N \].
  4. The weights of all parameters.
  5. Position encoding table. The shape is \[ S, H x N \].
* Outputs:
  1. The output ids. The shape is \[ S, B<sub>1</sub> x B<sub>2</sub> \].
  2. The parent ids, which are the chosen beam ids. The shape is \[ S, B<sub>1</sub> x B<sub>2</sub> \].
  3. The sequence lengths of each sentence. The shape is \[B<sub>1</sub> x B<sub>2</sub> \].

Note that the results of decoding with beam search are required to be finalized by TensorFlow's `tf.contrib.seq2seq.gather_tree` or other progress. 

The arguments, inputs, and outputs of decoding with sampling: 

* Arguments:
  1. Maximum sequence length (S)
  2. Top k value (K)
  3. Top p value (P)
  4. Head number (H)
  5. Size per head (N)
  6. Number of decoder layers
  7. Start id of the vocabulary
  8. End id of the vocabulary
* Inputs:
  1. The output of the encoder. The shape is \[ B, memory sequence length, H x N \]. Where B means the batch size.
  2. The sequence length of the source sentence. Note that the lengths should be expanded by beam width times.
  3. The table for embedding lookup. The shape is \[ V, H x N \].
  4. The weights of all parameters.
  5. Position encoding table. The shape is \[ S, H x N \].
* Outputs:
  1. The output ids. The shape is \[ S, B \].
  2. The sequence lengths of each sentence. The shape is \[ B \].

Note that K and P cannot be zero or non-zero value at the same time. FasterTransformer chooses the non-zero one to determine to use top k sampling or top p sampling. 

### Decoder and Decoding

Although the decoding process of most methods is similar, we find that there are lots of different kinds to compute the probability and implement the beam search. Therefore, if your chosen beam search algorithm is different from our implementation and it is hard for you to modify the beam search kernel, TensorFlow decoding with FasterTransformer Decoder is the recommended choice. However, the performance of the TensorFlow decoding with the FasterTransformer Decoder is worse than the performance of the FasterTransformer Decoding, especially for small batch sizes.

### GPT-2

The GPT-2 model is based on This project is based on [OpenAI gpt-2 project](https://github.com/openai/gpt-2). GPT-2 is a special case of Decoding, it does not require the cross-attention block and the results from encoder. Users can put some started words into GPT-2, and GPT-2 will use these words to generate the next word. By this method, GPT-2 can translate the sentence, reply the questions, and do many different applications. Fig. 1 shows the difference between GPT-2 standard decoding model. 

* Arguments:
  1. batch size (B)
  2. Top k value (K)
  3. Top p value (P)
  4. maximum sequence length (S)
  5. Head number (H)
  6. Size per head (N)
  7. Number of decoder layers
  8. Start id of the vocabulary
  9. Start ids, which are converted from the input sentences. If “start ids” is empty list, then FasterTransformer will use the “start id” as the first token.
  10. End id of the vocabulary
  11. temperature: 
* Inputs:
  1. The table for embedding lookup. The shape is \[ V, H x N \].
  2. The weights of all parameters.
  3. Position encoding table. The shape is \[ S, H x N \].
* Outputs:
  1. The output ids. The shape is \[ B \].

## Setup

The following section lists the requirements to use FasterTransformer.

### Requirements

- CMake >= 3.8 for Tensorflow, CMake >= 3.13 for PyTorch
- CUDA 10.1 or newer version
- Python 3 is recommended because some features are not supported in python 2
- Tensorflow 1.13 or 1.14 or 1.15
- PyTorch >= 1.4.0

These components are readily available within the NGC TensorFlow Docker image below.

Ensure you have the following components:
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) and NGC container are recommended
- [NVIDIA Pascal](https://www.nvidia.com/en-us/data-center/pascal-gpu-architecture/) or [Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) or [Ampere](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/) based GPU 

For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:

- [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
- [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)
- [Running TensorFlow](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html#running)
- [Running PyTorch](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html)

For those unable to use the NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## How to use

### Decoder and decoding process

1. Run FasterTransformer decoding on C++

    1.1 Generate the `decoding_gemm_config.in` file. 

    `./bin/decoding_gemm` can generate the best GEMM configuration. The arguments of `decoding_gemm` are:

    ```bash
    ./bin/decoding_gemm <batch_size> <beam_width> <head_number> <size_per_head> <vocab_size> <sequence_length> <encoder_hidden_dim> <is_use_fp16>
    ```

    Assume the settings of decoding are as follows.

    - `batch_size`=32
    - `beam_width`=4
    - `head_number`=8
    - `size_per_head`=64 
    - `vocabulary_size`=30000
    - `sequence_length`=32
    - `encoder's hidden dimension`=512
    - `data_type`=FP32

    Then the following scripts can generate the best GEMM configuration under such settings, and record the configuration into the `decoding_gemm_config.in` file.

    ```bash
    ./bin/decoding_gemm 32 4 8 64 30000 32 512 0
    ```

    1.2 Run decoding under FP32 on C++

    Assume the settings are the same as above, and the decoder contains 6 transformer layers. 

    In the decoding, we provide two kinds of methods to choose the tokens from the candidates. The first kind of method is the beam search algorithm. The second kind of method is sampling algorithm. 

    For beam search, we provide a simple diverse decoding of [link](https://arxiv.org/pdf/1611.08562.pdf). When the diversity rate is set to 0, it is equivalent to the naive beam search. 

    For sampling, we provide the top k sampling and top p sampling. Here, k is an integer number and p is a float point number. Note that we cannot use both of them at the same time. So, only one of both can be non-zero value. 

    `./bin/decoding_beamsearch_sample` runs the decoding with beam search in the `C++`. The arguments of `decoding_beamsearch_sample` is:

    ```bash
    ./bin/decoding_beamsearch_sample <batch_size> <beam_width> <head_number> <size_per_head> <vocab_size> <sequence_length> <num_layers> <encoder_hidden_dim> <is_use_fp16>
    ```

    Then the following scripts can run the decoding with beam search under the above settings. 

    ```bash
    ./bin/decoding_beamsearch_sample 32 4 8 64 30000 32 6 512 0
    ```

    The outputs should be like to the following:

    ```bash 
    Device Tesla V100-PCIE-32GB
    [INFO] batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 FT-CPP-decoding-beamsearch-time 73.36 ms
    ```

    `./bin/decoding_sampling_sample` runs the decoding with sampling in the `C++`. The arguments of `decoding_sampling_sample` is:

    ```bash
    ./bin/decoding_sampling_sample <batch_size> <candidate_num> <probability_threshold> <head_number> <size_per_head> <vocab_size> <sequence_length> <num_layers> <encoder_hidden_dim> <is_use_fp16>
    ```

    where `candidate_num` is the k value of top k, while `probability_threshold` is the p value of top p.

    Note that the beam width of sampling algorithm is always 1, so we need to generate the new configuration.

    The following scripts can run the decoding with top k sampling with under the above settings. 

    ```bash
    ./bin/decoding_gemm 32 1 8 64 30000 32 512 0
    ./bin/decoding_sampling_sample 32 4 0.0 8 64 30000 32 6 512 0
    ```

    The outputs should be like to the following:

    ```bash 
    Device Tesla V100-PCIE-32GB
    [INFO] batch_size 32 topk 4 topp 0.000000 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 FT-CPP-decoding-sampling-time 41.65 ms
    ```

    The following scripts can run the decoding with top p sampling with under the above settings. 

    ```bash
    ./bin/decoding_gemm 32 1 8 64 30000 32 512 0
    ./bin/decoding_sampling_sample 32 0 0.01 8 64 30000 32 6 512 0
    ```

    The outputs should be like to the following:

    ```bash 
    Device Tesla V100-PCIE-32GB
    [INFO] batch_size 32 topk 0 topp 0.010000 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 FT-CPP-decoding-sampling-time 61.63 ms
    ```

    1.3 Run decoding under FP16 on C++

    So far, we use the FP32 to run the FasterTransformer. If we use the volta or newer NVIDIA GPU, we can use tensor core to accelerate when we use the FP16. 

    To use the FP16, we only need to set the `<is_use_fp16>` flag to 1 like following:

    ```bash
    ./bin/decoding_gemm 32 4 8 64 30000 32 512 1
    ./bin/decoding_beamsearch_sample 32 4 8 64 30000 32 6 512 1
    ```

    Note that the configuration of FP32 and FP16 are different, so we need to generate the configuration again. 

    The outputs should be like to the following:  

    ```bash
    Device Tesla V100-PCIE-32GB
    [INFO] batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 FT-CPP-decoding-beamsearch-time 47.89 ms
    ```

2. Run FasterTransformer decoder/decoding on TensorFlow

    2.1 Run FasterTransformer decoder under FP32 on TensorFlow

    ```bash
    ./bin/decoding_gemm 32 4 8 64 30000 32 512 0
    python tensorflow/decoder_sample.py \
            --batch_size 32 \
            --beam_width 4 \
            --head_number 8 \
            --size_per_head 64 \
            --vocab_size 30000 \
            --max_seq_len 32 \
            --num_layer 6 \
            --memory_hidden_dim 512 \
            --data_type fp32 \
            --decoder_type 2
    ```

    The outputs should be like to the following:

    ```bash 
    [[INFO][PYTHON] step:][29][True][max abs diff: ][4.17232513e-06][ op val: ][1.23598516][ tf val: ][1.23598933]
    [[INFO][PYTHON] step:][30][True][max abs diff: ][4.05311584e-06][ op val: ][-2.40530682][ tf val: ][-2.40531087]
    [[INFO][PYTHON] step:][31][False][max abs diff: ][3.7997961e-06][ op val: ][-0.120998174][ tf val: ][-0.121001974]
    ```

    The results show that the differences between the decoder of TensorFlow and decoder are smaller than threshold. Sometimes, the differences are larger than the threshold and the checking will return "False", but it does not affect the results.

    The argument `decoder_type` decides to use the decoder of TensorFlow or decoder of FasterTransformer. `decoder_type 2` uses both decoders and compares their results. 

    The following script demonstrates the execution time of the FasterTransformer decoder.

    ```bash
    ./bin/decoding_gemm 32 4 8 64 30000 32 512 0
    python tensorflow/decoder_sample.py \
            --batch_size 32 \
            --beam_width 4 \
            --head_number 8 \
            --size_per_head 64 \
            --vocab_size 30000 \
            --max_seq_len 32 \
            --num_layer 6 \
            --memory_hidden_dim 512 \
            --data_type fp32 \
            --decoder_type 1 \
            --test_time 1
    ```

    The outputs should be like to the following:

    ```bash
    [INFO] batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 FT-OP-decoder-time 138.90 ms.
    ```

    The following script demonstrates the execution time of the TensorFlow decoder.

    ```bash 
    python tensorflow/decoder_sample.py \
            --batch_size 32 \
            --beam_width 4 \
            --head_number 8 \
            --size_per_head 64 \
            --vocab_size 30000 \
            --max_seq_len 32 \
            --num_layer 6 \
            --memory_hidden_dim 512 \
            --data_type fp32 \
            --decoder_type 0 \
            --test_time 1
    ```

    The outputs should be like to the following:

    ```bash
    [INFO] batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 TF-decoding-beamsearch-time 564.37 ms.
    ```

    2.2 Run FasterTransformer decoder under FP16 on TensorFlow

    To use the FP16 in TensorFlow, we only need to set the `--data_type fp16` like following:

    ```bash
    ./bin/decoding_gemm 32 4 8 64 30000 32 512 1
    python tensorflow/decoder_sample.py \
            --batch_size 32 \
            --beam_width 4 \
            --head_number 8 \
            --size_per_head 64 \
            --vocab_size 30000 \
            --max_seq_len 32 \
            --num_layer 6 \
            --memory_hidden_dim 512 \
            --data_type fp16 \
            --decoder_type 2
    ```

    The outputs should be like to the following:

    ```bash 
    [[INFO][PYTHON] step:][29][True][max abs diff: ][0.01171875][ op val: ][2.03125][ tf val: ][2.04296875]
    [[INFO][PYTHON] step:][30][True][max abs diff: ][0.01171875][ op val: ][2.3671875][ tf val: ][2.35546875]
    [[INFO][PYTHON] step:][31][True][max abs diff: ][0.01171875][ op val: ][2.33398438][ tf val: ][2.32226562]
    ```

    The following script demonstrates the execution time of the FasterTransformer decoder.

    ```bash
    ./bin/decoding_gemm 32 4 8 64 30000 32 512 1
    python tensorflow/decoder_sample.py \
            --batch_size 32 \
            --beam_width 4 \
            --head_number 8 \
            --size_per_head 64 \
            --vocab_size 30000 \
            --max_seq_len 32 \
            --num_layer 6 \
            --memory_hidden_dim 512 \
            --data_type fp16 \
            --decoder_type 1 \
            --test_time 1
    ```

    The outputs should be like to the following:

    ```bash
    [INFO] batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 FT-OP-decoder-time 132.48 ms.
    ```

    The following script demonstrates the execution time of the TensorFlow decoder.

    ```bash 
    python tensorflow/decoder_sample.py \
            --batch_size 32 \
            --beam_width 4 \
            --head_number 8 \
            --size_per_head 64 \
            --vocab_size 30000 \
            --max_seq_len 32 \
            --num_layer 6 \
            --memory_hidden_dim 512 \
            --data_type fp16 \
            --decoder_type 0 \
            --test_time 1
    ```

    The outputs should be like to the following:

    ```bash
    [INFO] batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 TF-decoding-beamsearch-time 503.52 ms.
    ```

    Note that when the batch size is small, using FP16 may cause the inference speed to become slower. This is because that decoding is not computing bound and using FP16 in TensorFlow leads to some additional operation and casting. 

    2.3 Run FasterTransformer decoding under FP32 on TensorFlow

    In the decoding, we provide two kinds of methods to choose the tokens from the candidates. The first kind of method is the beam search algorithm. The second kind of method is sampling algorithm. 

    For beam search, we provide a simple diverse decoding of [link](https://arxiv.org/pdf/1611.08562.pdf). When the `--beam_search_diversity_rate` is set to 0, it is equivalent to the naive beam search. 

    For sampling, we provide the top k sampling and top p sampling, which are set by the arguments `--sampling_topk` and `--sampling_topp`. Here, k is an integer number and p is a float point number. Note that we cannot use both at the same time. So, only one of both can be non-zero value. 

    The following script uses diverse decoding with diversity rate 0 and top k sampling with k = 4:

    ```bash
    ./bin/decoding_gemm 32 4 8 64 30000 32 512 0
    python tensorflow/decoding_sample.py \
            --batch_size 32 \
            --beam_width 4 \
            --head_number 8 \
            --size_per_head 64 \
            --vocab_size 30000 \
            --max_seq_len 32 \
            --num_layer 6 \
            --memory_hidden_dim 512 \
            --data_type fp32 \
            --beam_search_diversity_rate 0 \
            --sampling_topk 4 \
            --sampling_topp 0.0 \
            --test_time 0123
    ```
    
    The outputs should be like to the following:

    ```bash
    [INFO] batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 TF-decoding-beamsearch-time 555.87 ms.
    [INFO] batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 FT-OP-decoding-beamsearch-time  75.80 ms.
    [INFO] batch_size 32 topk 4 topp 0.0 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 TF-decoding-sampling-time 432.40 ms.
    [INFO] batch_size 32 topk 4 topp 0.0 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 FT-OP-decoding-sampling-time  46.68 ms.
    ```

    Note that the results of FasterTransformer may be different, especially when the batch size is larger.

    Here, we use same configuration to run the decoding with beam search and sampling at the same time. This is not correct because the beam width of decoding with sampling is always 1, so the configurations of them are same only when the beam width is 1. However, this only little reduce the speed of decoding with sampling, so we ignore this problem here. 

    Here, the meaning of argument `--test_time` is different. 0 means testing the TensorFlow with beam search; 1 means testing the FasterTransformer with beam search; 2 means testing the TensorFlow with sampling; 3 means testing the FasterTransformer with sampling. 

    The following script uses diverse decoding with diversity rate -1.3 and top p sampling with p = 0.01:

    ```bash
    ./bin/decoding_gemm 32 4 8 64 30000 32 512 0
    python tensorflow/decoding_sample.py \
            --batch_size 32 \
            --beam_width 4 \
            --head_number 8 \
            --size_per_head 64 \
            --vocab_size 30000 \
            --max_seq_len 32 \
            --num_layer 6 \
            --memory_hidden_dim 512 \
            --data_type fp32 \
            --beam_search_diversity_rate -1.3 \
            --sampling_topk 0 \
            --sampling_topp 0.01 \
            --test_time 0123
    ```

    The outputs should be like to the following:

    ```bash
    [INFO] batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 TF-decoding-beamsearch-time 525.55 ms.
    [INFO] batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 FT-OP-decoding-beamsearch-time  76.79 ms.
    [INFO] batch_size 32 topk 4 topp 0.0 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 TF-decoding-sampling-time 420.98 ms.
    [INFO] batch_size 32 topk 4 topp 0.0 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 FT-OP-decoding-sampling-time  46.37 ms.
    ```

    For the sampling algorithm, the results of TensorFlow and FasterTransformer are often different. 

    2.4 Run FasterTransformer decoding under FP16 on TensorFlow

    ```bash
    ./bin/decoding_gemm 32 4 8 64 30000 32 512 1
    python tensorflow/decoding_sample.py \
            --batch_size 32 \
            --beam_width 4 \
            --head_number 8 \
            --size_per_head 64 \
            --vocab_size 30000 \
            --max_seq_len 32 \
            --num_layer 6 \
            --memory_hidden_dim 512 \
            --data_type fp16 \
            --beam_search_diversity_rate 0.0 \
            --sampling_topk 4 \
            --sampling_topp 0.00 \
            --test_time 0123
    ```

    The outputs should be like to the following:

    ```bash
    [INFO] batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 TF-decoding-beamsearch-time 494.23 ms.
    [INFO] batch_size 32 beam_width 4 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 FT-OP-decoding-beamsearch-time  50.43 ms.
    [INFO] batch_size 32 topk 4 topp 0.0 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 TF-decoding-sampling-time 382.34 ms.
    [INFO] batch_size 32 topk 4 topp 0.0 head_num 8 size_per_head 64 seq_len 32 decoder_layers 6 vocab_size 30000 FT-OP-decoding-sampling-time  33.19 ms.
    ```

    Note that the results of FasterTransformer may be different, especially when the batch size is larger.

    2.5 Run FasterTransformer encoder and decoder/decoding on TensorFlow at the same time

    In this subsection, we demonstrate how to use the FasterTransformer encoder and decoder/decoding at the same time. 

    ```bash
    ./bin/encoder_gemm 32 32 8 64 0 0
    ./bin/decoding_gemm 32 4 8 64 30000 32 512 0
    python tensorflow/encoder_decoder_sample.py \
            --batch_size 32 \
            --beam_width 4 \
            --encoder_head_number 8 \
            --encoder_size_per_head 64 \
            --decoder_head_number 8 \
            --decoder_size_per_head 64 \
            --vocab_size 30000 \
            --max_seq_len 32 \
            --encoder_num_layer 6 \
            --decoder_num_layer 6 \
            --data_type fp32 
    ```

    The `encoder_decoder_sample.py` files show the results of "TensorFlow encoder + FasterTransformer decoder" and the results of "FasterTransformer encoder + FasterTransformer decoder. The usage is similar to `decoder_sample.py`. 

    ```bash
    ./bin/encoder_gemm 32 32 8 64 0 0
    ./bin/decoding_gemm 32 4 8 64 30000 32 512 0
    python tensorflow/encoder_decoding_sample.py \
            --batch_size 32 \
            --beam_width 4 \
            --encoder_head_number 8 \
            --encoder_size_per_head 64 \
            --decoder_head_number 8 \
            --decoder_size_per_head 64 \
            --vocab_size 30000 \
            --max_seq_len 32 \
            --encoder_num_layer 6 \
            --decoder_num_layer 6 \
            --data_type fp32 
    ```

    For convenience, we only show how to use the FasterTransformer encoder and decoding with beam search in the `encoder_decoding_sample.py`. The usage is like `decoding_sample.py`.

3. Run FasterTransformer decoder/decoding on PyTorch

    Please install OpenNMT-py first before running the demos by
    ```bash
    pip install opennmt-py==1.1.1
    ```

    3.1 Generate the `decoding_gemm_config.in` file:

    ```bash
    ./bin/decoding_gemm <batch_size> <beam_size> <head_number> <size_per_head> <vocab_size> <seq_len> <memory_hidden_dim> <is_fp16>
    ./bin/decoding_gemm 8 4 8 64 31538 32 512 1
    ```
    If you want to use the library in other directory, please generate this file according to your setting and copy it to your working directory.

    3.2 Run the PyTorch decoder sample: 

    ```bash
    python pytorch/decoder_sample.py <batch_size> <layer_num> <sequence_length> <head_number> <size_per_head> <--fp16> <--time>
    python pytorch/decoder_sample.py 8 6 32 8 64 --fp16 --time
    ```
    Remove `--fp16` for fp32 mode. `--ths` will use the TorchScript custom class.

    The outputs should be like to the following:

    ```bash
    step: 30     Mean relative diff: 0.01395416259765625     Max relative diff: 1.38671875     Min relative diff: 0.0
    step: 31     Mean relative diff: 0.0148468017578125     Max relative diff: 2.880859375     Min relative diff: 0.0
    [INFO] ONMTDecoder time costs: 218.37 ms
    [INFO] FTDecoder time costs: 25.15 ms
    ```

    Note that the relative diff is very large. It is caused by the random initial weights and inputs, and it does not affect the result of translation.

    3.3 Run the PyTorch decoding sample: 

    ```bash
    python pytorch/decoding_sample.py <batch_size> <layer_num> <sequence_length> <head_number> <size_per_head> <beam_size> <vocab_size> <--fp16> <--time>
    python pytorch/decoding_sample.py 8 6 32 8 64 4 31538 --fp16 --time
    ```
    Remove `--fp16` for fp32 mode.  `--ths` will use the TorchScript custom class.

    The outputs should be like to the following:

    ```bash
    [INFO] TorchDecoding time costs: 289.08 ms
    [INFO] TorchDecoding (with FTDecoder) time costs: 104.15 ms
    [INFO] FTDecoding time costs: 30.57 ms
    ```

    Random initialized parameters may lead to different results. You can download the pretrained model following the instruction in the next part, and add `--use_pretrained`, then you can get the same results.

### Translation process

1. Translation with FasterTransformer on TensorFlow

    This subsection demonstrates how to use FasterTransformer decoding to translate a sentence. We use the pretrained model and testing data in [OpenNMT-tf](https://opennmt.net/Models-tf/), which translates from English to German. 

    Because the FasterTransformer Encoder is based on BERT, we cannot restore the model of encoder of OpenNMT to FasterTransformer Encoder. Therefore, we use OpenNMT-tf to build the encoder and preprocess the source sentence.

    Another problem is that the implementation of FasterTransformer Decoder and decoder of OpenNMT-tf is a little different. For example, the decoder of OpenNMT-tf uses one convolution to compute query, key, and value in masked-multi-head-attention; but FasterTransformer Decoder splits them into three gemms. One method is using the tool `utils/dump_model.py` to convert the pretrained model to fit the model structure of FasterTransformer Decoder. Another method is Splitting the weights during inference.

    `download_model_data.sh` will install the OpenNMT-tf v1, downloading the pretrained model into the `translation` folder, and convert the model. 

    ```bash
    bash tensorflow/utils/translation/download_model_data.sh
    ```

    Then run the translation sample by the following script:

    ```bash
    ./bin/decoding_gemm 128 4 8 64 32001 100 512 0
    python tensorflow/translate_sample.py \
            --batch_size 128 \
            --beam_width 4 \
            --encoder_head_number 8 \
            --encoder_size_per_head 64 \
            --decoder_head_number 8 \
            --decoder_size_per_head 64 \
            --max_seq_len 32 \
            --encoder_num_layer 6 \
            --decoder_num_layer 6 \
            --data_type fp32 \
            --beam_search_diversity_rate 0.0 \
            --sampling_topk 1 \
            --sampling_topp 0.00 \
            --test_time 012345
    ```

    The outputs of should be similar to the following:

    ```bash
    [INFO] tf-decoding-beamsearch translates 24 batches taking 31.39 ms to translate 67092 tokens, BLEU score: 26.29, 2137 tokens/sec.
    [INFO] op-decoder-beamsearch translates 24 batches taking 10.37 ms to translate 67092 tokens, BLEU score: 26.29, 6473 tokens/sec.
    [INFO] op-decoding-beamsearch translates 24 batches taking 7.88 ms to translate 67124 tokens, BLEU score: 26.31, 8513 tokens/sec.
    [INFO] tf-decoding-sampling translates 24 batches taking 16.23 ms to translate 67813 tokens, BLEU score: 25.79, 4178 tokens/sec.
    [INFO] op-decoder-sampling translates 24 batches taking 6.29 ms to translate 67813 tokens, BLEU score: 25.79, 10781 tokens/sec.
    [INFO] op-decoding-sampling translates 24 batches taking 4.10 ms to translate 67813 tokens, BLEU score: 25.79, 16524 tokens/sec.
    ```

    The scripts of running under FP16 is following:

    ```bash
    python tensorflow/tensorflow_bert/ckpt_type_convert.py --init_checkpoint=translation/ckpt/model.ckpt-500000 --fp16_checkpoint=translation/ckpt/fp16_model.ckpt-500000
    ./bin/decoding_gemm 128 4 8 64 32001 100 512 1
    python tensorflow/translate_sample.py \
          --batch_size 128 \
          --beam_width 4 \
          --encoder_head_number 8 \
          --encoder_size_per_head 64 \
          --decoder_head_number 8 \
          --decoder_size_per_head 64 \
          --max_seq_len 32 \
          --encoder_num_layer 6 \
          --decoder_num_layer 6 \
          --data_type fp16 \
          --beam_search_diversity_rate 0.0 \
          --sampling_topk 1 \
          --sampling_topp 0.00 \
          --test_time 012345
    ```

    The outputs of should be similar to the following:

    ```bash
    [INFO] tf-decoding-beamsearch translates 24 batches taking 22.75 ms to translate 67094 tokens, BLEU score: 26.31, 2949 tokens/sec.
    [INFO] op-decoder-beamsearch translates 24 batches taking 7.73 ms to translate 67089 tokens, BLEU score: 26.30, 8682 tokens/sec.
    [INFO] op-decoding-beamsearch translates 24 batches taking 5.27 ms to translate 67130 tokens, BLEU score: 26.33, 12746 tokens/sec.
    [INFO] tf-decoding-sampling translates 24 batches taking 13.65 ms to translate 67828 tokens, BLEU score: 25.83, 4968 tokens/sec.
    [INFO] op-decoder-sampling translates 24 batches taking 4.92 ms to translate 67831 tokens, BLEU score: 25.80, 13773 tokens/sec.
    [INFO] op-decoding-sampling translates 24 batches taking 2.54 ms to translate 67844 tokens, BLEU score: 25.82, 26718 tokens/sec.
    ```

2.  Translation with FasterTransformer on PyTorch

    We have a translation demo for En-De translation.

    You need to download the pretrained_model first by:

    ```bash
    bash pytorch/scripts/download_translation_model.sh
    ```

    Then you can run the demo by:

    ```bash
    python pytorch/run_translation.py --batch_size <batch_size> --beam_size <beam_size> --model_type <model_type> --data_type <data_type> --output_file <output_file>
    ```
    you can also use `--module_path` to set the FasterTransformer module `.so` file path, and use `--input_file` to set the input file to be translated.

    the `<model_type>` can be:
    - `ori`: original OpenNMT model
    - `decoder_ext`: replace the decoder in OpenNMT model with our FasterTransformer decoder
    - `decoding_ext`: using our FasterTransformer decoding module
    - `torch_decoding`: PyTorch version decoding with the method FasterTransformer decoding uses
    - `torch_decoding_with_decoder_ext`: PyTorch version decoding with the method FasterTransformer decoding uses but replace the decoder with the FasterTransformer decoder

    the `<data_type>` can be `fp32` or `fp16`

    if you do not specify the output file, it only print to the stdout.

    If you want to evaluate the BLEU score, please recover the BPE first by:
    ```bash
    python pytorch/utils/recover_bpe.py <ref_file> <debpe_ref_file>
    python pytorch/utils/recover_bpe.py <output_file> <debpe_output_file>
    ```
    the `<ref_file>` for our demo is `pytorch/translation/data/test.de`, the `<output_file>` is the output from `run_translation.py`.

    Then you can evalute the BLEU score, for example, through `sacrebleu`:
    ```bash
    pip install sacrebleu
    cat <debpe_output_file> | sacrebleu <debpe_ref_file>
    ```

    The following scripts run translation under FP32 and get the bleu score:

    ```bash
    ./bin/decoding_gemm 128 4 8 64 31538 100 512 0
    python pytorch/run_translation.py --batch_size 128 --beam_size 4 --model_type decoding_ext --data_type fp32 --output_file output.txt
    python pytorch/utils/recover_bpe.py pytorch/translation/data/test.de debpe_ref.txt
    python pytorch/utils/recover_bpe.py output.txt debpe_output.txt
    pip install sacrebleu
    cat debpe_output.txt | sacrebleu debpe_ref.txt
    ```

### GPT-2 process

1. Install required tools

```bash
pip install -r tensorflow/utils/gpt2_requirement.txt
```

2. Download the gpt-2 model by following scripts.

```bash
python tensorflow/utils/download_gpt2_model.py <model_name>
e.g. python tensorflow/utils/download_gpt2_model.py 124M
```

In the repo of OpenAI, they provide many models, including `124M`, `355M`, `774M` and `1558M`

Next, using the following script to convert the model to csv files, which are used in C++ sample codes.

```bash
mkdir tmp
python tensorflow/utils/convert_tf_ckpt_to_csv.py  -o tmp -i models/124M/model.ckpt
```

3. Run GPT-2

    3.1 Generate the `decoding_gemm_config.in` file.


    `./bin/decoding_gemm` can generate the best GEMM configuration. The arguments of `decoding_gemm` are:

    ```bash
    ./bin/decoding_gemm <batch_size> <beam_width> <head_number> <size_per_head> <vocab_size> <sequence_length> <encoder_hidden_dim> <is_use_fp16>
    ```

    Assume the settings of decoding are as follows (the hyper-parameters of GPT-2 124M model).

    - `batch_size`=4
    - `top k`=4
   	- `top p`=0.6
    - `head_number`=12
    - `size_per_head`=64 
    - `vocabulary_size`=50257
    - `sequence_length`=32
    - `data_type`=FP32

    Then the following scripts can generate the best GEMM configuration under such settings, and record the configuration into the `decoding_gemm_config.in` file.

    ```bash
    ./bin/decoding_gemm 4 1 12 64 50257 32 768 0
    ```

    Note that the beam width of sampling algorithm is always 1, so we need to generate the new configuration.

      3.2 Run GPT-2 under FP32 on C++

    Assume the settings are the same as above, and the decoder contains 12 transformer layers.

    The arguments of `gpt2_sample` is 

    ```bash
    ./bin/decoding_sample <batch_size> <candidate_num> <probability_threshold> <head_num> <size_per_head> <vocab_size> <seq_len> <num_layer> <is_fp16>
    ```

    where `candidate_num` is the k value of top k, while `probability_threshold` is the p value of top p.

    ```bash
    ./bin/gpt2_sample 4 4 0.6 12 64 50257 32 12 0
	  ```

    The outputs should be like to the following:

    ```bash 
    Device Tesla V100-PCIE-32GB
    [INFO] batch_size 4 head_num 12 size_per_head 64 seq_len 32 decoder_layers 12 vocab_size 50257 FT-CPP-gpt2-time 52.67 ms
    ```

	  3.3 Run GPT-2 under FP16 on C++

	  So far, we use the FP32 to run the FasterTransformer. If we use the volta or newer NVIDIA GPU, we can use tensor core to accelerate when we use the FP16. 

    To use the FP16, we only need to set the `<is_use_fp16>` flag to 1 like following:

    ```bash
    ./bin/decoding_gemm 4 1 12 64 50257 32 768 1
    ./bin/gpt2_sample 4 4 0.6 12 64 50257 32 12 1
    ```

    Note that the configuration of FP32 and FP16 are different, so we need to generate the configuration again. 

    The outputs should be like to the following:  

    ```bash
    Device Tesla V100-PCIE-32GB
    [INFO] batch_size 4 head_num 12 size_per_head 64 seq_len 32 decoder_layers 12 vocab_size 50257 FT-CPP-gpt2-time 40.65 ms
    ```

    3.4 Run FasterTransformer GPT-2 under FP32 on TensorFlow

    ```bash
    ./bin/decoding_gemm 4 1 12 64 50257 32 768 0
    python tensorflow/gpt2_sample.py --batch_size=4 \
                                  --length=32 \
                                  --top_k=4 \
                                  --top_p=0.6 \
                                  --data_type=fp32
    ```

    The outputs should be like to the following:

    ```bash 
    [INFO] FT op time: 57.9542
    ======================================== SAMPLE 1 ========================================
    The first of three films from the acclaimed director, who is also a producer on the upcoming Star Wars: The Force Awakens, is set to be released on December 8
    ======================================== SAMPLE 2 ========================================

    The first time I saw the new "The Last of Us" trailer, I was blown away. I had never seen anything like it before. It was so
    ======================================== SAMPLE 3 ========================================

    A new study from the University of California, Berkeley, and the University of California, Santa Barbara found that the number of people who have been diagnosed with schizophrenia in
    ======================================== SAMPLE 4 ========================================

    The U.S. government has been accused of using a secret surveillance program to spy on American citizens.

    The Justice Department has said that the program was
    ```

    It generates four different sentences due to different random seeds.

    3.5 Run FasterTransformer GPT-2 under FP16 on TensorFlow

    ```bash
    ./bin/decoding_gemm 4 1 12 64 50257 32 768 1
    python tensorflow/gpt2_sample.py --batch_size=4 \
                                  --length=32 \
                                  --top_k=4 \
                                  --top_p=0.6 \
                                  --data_type=fp16
    ```

    The outputs should be like to the following:

    ```bash
    [INFO] FT op time: 47.6268
    ======================================== SAMPLE 1 ========================================

    The U.S. Department of Justice has filed a lawsuit against the company that owns the video game company, Electronic Arts, alleging that the company violated antitrust laws
    ======================================== SAMPLE 2 ========================================
    The U.S. government has been trying to find ways to prevent the spread of Zika virus in the Americas, but it has been unable to do so.

    ======================================== SAMPLE 3 ========================================

    The U.S. Department of Justice has filed a lawsuit against the company that owns the video game company, Electronic Arts, alleging that the company violated antitrust laws
    ======================================== SAMPLE 4 ========================================

    The first of two episodes of the new season of "The Walking Dead" is here, and it's a great one.

    The first episode of the
	  ```

    User can check the arguments in `sample/tensorflow/gpt2_sample.py` file.

## Performance

Hardware settings: 
* CPU: Intel(R) Xeon(R) Gold 6132 CPU @ 2.60GHz
* T4 (with mclk 5000MHz, pclk 1590MHz) with Intel(R) Xeon(R) CPU E5-2603 v4 @ 1.70GHz
* V100 (with mclk 877MHz, pclk 1380MHz) with Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz (dgx-1 server)

To run the following benchmark, we need to install the unix computing tool "bc" by

```bash
apt-get install bc
```

### Decoder performance 

We demonstrate the inference time of FasterTransformer in C++, TensorFlow, and compare to the performance of pure TensorFlow on T4 and V100. The performance of PyTorch are put in the "Decoding performance" subsection.

In this benchmark, we compare the performance of TensorFlow decoding with beam search method (TF), and the performance of replacing the decoder of TensorFlow by FasterTransformer (FT-OP), and show the speedup of FT-OP compare to TF.  

We do not demonstrate the performance of TensorFlow with XLA since we did not find that using XLA has obvious speedup. 

Our results of C++ and TensorFlow were obtained by running the `sample/tensorflow/scripts/profile_decoder_performance.sh`

In the experiments of decoding, we updated the following parameters:

* head_num = 8
* size_per_head = 64 
* num_layers = 6
* vocabulary_size = 30000 for TensorFlow sample codes, 31538 for PyTorch sample codes
* memory_hidden_dim = 512

#### Decoder performance on T4 and TensorFlow

* Performance on FP32

| <Batch_size, beam_width, Seq_len> | TF (ms) | FT-OP (ms) | FT-OP Speedup | 
|:---------------------------------:|:-------:|:----------:|:-------------:| 
| <1, 1, 32> | 509.16 | 107.98 | 4.71 | 
| <1, 1, 64> | 951.49 | 223.69 | 4.25 | 
| <1, 1, 128> | 1943.97 | 425.28 | 4.57 | 
| <1, 4, 32> | 497.88 | 126.70 | 3.92 | 
| <1, 4, 64> | 1050.92 | 243.64 | 4.31 | 
| <1, 4, 128> | 2064.92 | 508.16 | 4.06 | 
| <8, 1, 32> | 510.90 | 125.96 | 4.05 | 
| <8, 1, 64> | 995.81 | 244.18 | 4.07 | 
| <8, 1, 128> | 2041.21 | 479.02 | 4.26 | 
| <8, 4, 32> | 539.70 | 129.21 | 4.17 | 
| <8, 4, 64> | 1100.77 | 267.75 | 4.11 | 
| <8, 4, 128> | 2100.58 | 558.91 | 3.75 | 
| <32, 1, 32> | 575.80 | 123.16 | 4.67 | 
| <32, 1, 64> | 1070.51 | 251.52 | 4.25 | 
| <32, 1, 128> | 2172.67 | 554.32 | 3.91 | 
| <32, 4, 32> | 673.70 | 204.51 | 3.29 | 
| <32, 4, 64> | 1335.84 | 492.47 | 2.71 | 
| <32, 4, 128> | 3136.18 | 1331.35 | 2.35 | 
| <64, 1, 32> | 582.22 | 142.49 | 4.08 | 
| <64, 1, 64> | 1243.74 | 312.54 | 3.97 | 
| <64, 1, 128> | 2420.20 | 791.30 | 3.05 | 
| <64, 4, 32> | 850.54 | 350.63 | 2.42 | 
| <64, 4, 64> | 1833.49 | 874.46 | 2.09 | 
| <64, 4, 128> | 4586.01 | 2450.19 | 1.87 | 
| <128, 1, 32> | 656.85 | 208.91 | 3.14 | 
| <128, 1, 64> | 1461.70 | 499.76 | 2.92 | 
| <128, 1, 128> | 3209.60 | 1361.95 | 2.35 | 
| <128, 4, 32> | 1260.55 | 656.29 | 1.92 | 
| <128, 4, 64> | 2875.73 | 1663.91 | 1.72 | 
| <128, 4, 128> | 8018.63 | 4718.32 | 1.69 | 

* Performance on FP16

| <Batch_size, beam_width, Seq_len> | TF (ms) | FT-OP (ms) | FT-OP Speedup | 
|:---------------------------------:|:-------:|:----------:|:-------------:| 
| <1, 1, 32> | 400.02 | 121.19 | 3.30 | 
| <1, 1, 64> | 823.41 | 233.93 | 3.51 | 
| <1, 1, 128> | 1616.38 | 422.73 | 3.82 | 
| <1, 4, 32> | 476.33 | 128.45 | 3.70 | 
| <1, 4, 64> | 868.67 | 261.18 | 3.32 | 
| <1, 4, 128> | 1857.95 | 464.51 | 3.99 | 
| <8, 1, 32> | 452.70 | 119.73 | 3.78 | 
| <8, 1, 64> | 906.15 | 222.74 | 4.06 | 
| <8, 1, 128> | 1789.19 | 428.80 | 4.17 | 
| <8, 4, 32> | 484.09 | 127.14 | 3.80 | 
| <8, 4, 64> | 973.28 | 252.81 | 3.84 | 
| <8, 4, 128> | 1907.93 | 527.98 | 3.61 | 
| <32, 1, 32> | 476.66 | 124.72 | 3.82 | 
| <32, 1, 64> | 933.16 | 240.70 | 3.87 | 
| <32, 1, 128> | 1953.02 | 518.10 | 3.76 | 
| <32, 4, 32> | 607.62 | 159.24 | 3.81 | 
| <32, 4, 64> | 1280.93 | 352.51 | 3.63 | 
| <32, 4, 128> | 2511.20 | 882.21 | 2.84 | 
| <64, 1, 32> | 501.07 | 135.40 | 3.70 | 
| <64, 1, 64> | 1020.40 | 281.34 | 3.62 | 
| <64, 1, 128> | 2243.14 | 627.33 | 3.57 | 
| <64, 4, 32> | 692.42 | 213.80 | 3.23 | 
| <64, 4, 64> | 1517.27 | 542.75 | 2.79 | 
| <64, 4, 128> | 3351.21 | 1554.97 | 2.15 | 
| <128, 1, 32> | 593.39 | 163.73 | 3.62 | 
| <128, 1, 64> | 1258.93 | 358.26 | 3.51 | 
| <128, 1, 128> | 2672.11 | 910.34 | 2.93 | 
| <128, 4, 32> | 989.35 | 364.63 | 2.71 | 
| <128, 4, 64> | 2216.00 | 962.84 | 2.30 | 
| <128, 4, 128> | 5515.29 | 2913.02 | 1.89 | 

#### Decoder performance on V100 and TensorFlow

* Performance on FP32

| <Batch_size, beam_width, Seq_len> | TF (ms) | FT-OP (ms) | FT-OP Speedup | 
|:---------------------------------:|:-------:|:----------:|:-------------:| 
| <1, 1, 32> | 239.38 | 68.88 | 3.47 | 
| <1, 1, 64> | 500.20 | 133.88 | 3.73 | 
| <1, 1, 128> | 1021.87 | 261.55 | 3.90 | 
| <1, 4, 32> | 242.70 | 74.93 | 3.23 | 
| <1, 4, 64> | 509.43 | 145.60 | 3.49 | 
| <1, 4, 128> | 893.73 | 296.82 | 3.01 | 
| <8, 1, 32> | 241.06 | 68.85 | 3.50 | 
| <8, 1, 64> | 494.16 | 145.88 | 3.38 | 
| <8, 1, 128> | 1028.89 | 285.51 | 3.60 | 
| <8, 4, 32> | 274.33 | 73.38 | 3.73 | 
| <8, 4, 64> | 534.15 | 152.04 | 3.51 | 
| <8, 4, 128> | 1090.66 | 321.77 | 3.38 | 
| <32, 1, 32> | 249.78 | 71.74 | 3.48 | 
| <32, 1, 64> | 527.18 | 150.84 | 3.49 | 
| <32, 1, 128> | 1053.79 | 313.93 | 3.35 | 
| <32, 4, 32> | 313.01 | 114.31 | 2.73 | 
| <32, 4, 64> | 666.00 | 252.23 | 2.64 | 
| <32, 4, 128> | 1376.10 | 593.28 | 2.31 | 
| <64, 1, 32> | 288.73 | 86.66 | 3.33 | 
| <64, 1, 64> | 553.34 | 177.65 | 3.11 | 
| <64, 1, 128> | 1125.72 | 404.00 | 2.78 | 
| <64, 4, 32> | 377.06 | 156.55 | 2.40 | 
| <64, 4, 64> | 806.34 | 373.36 | 2.15 | 
| <64, 4, 128> | 1913.47 | 974.17 | 1.96 | 
| <128, 1, 32> | 319.11 | 110.49 | 2.88 | 
| <128, 1, 64> | 666.36 | 243.54 | 2.73 | 
| <128, 1, 128> | 1426.32 | 591.99 | 2.40 | 
| <128, 4, 32> | 528.52 | 256.18 | 2.06 | 
| <128, 4, 64> | 1215.82 | 620.55 | 1.95 | 
| <128, 4, 128> | 3167.89 | 1733.38 | 1.82 | 

* Performance on FP16

| <Batch_size, beam_width, Seq_len> | TF (ms) | FT-OP (ms) | FT-OP Speedup | 
|:---------------------------------:|:-------:|:----------:|:-------------:| 
| <1, 1, 32> | 209.70 | 70.37 | 2.97 | 
| <1, 1, 64> | 423.41 | 141.34 | 2.99 | 
| <1, 1, 128> | 775.10 | 287.64 | 2.69 | 
| <1, 4, 32> | 215.05 | 81.37 | 2.64 | 
| <1, 4, 64> | 449.72 | 146.28 | 3.07 | 
| <1, 4, 128> | 910.03 | 291.50 | 3.12 | 
| <8, 1, 32> | 226.01 | 68.60 | 3.29 | 
| <8, 1, 64> | 437.30 | 153.32 | 2.85 | 
| <8, 1, 128> | 915.96 | 286.39 | 3.19 | 
| <8, 4, 32> | 248.44 | 75.81 | 3.27 | 
| <8, 4, 64> | 463.51 | 154.71 | 2.99 | 
| <8, 4, 128> | 960.88 | 293.46 | 3.27 | 
| <32, 1, 32> | 233.93 | 69.80 | 3.35 | 
| <32, 1, 64> | 482.73 | 147.54 | 3.27 | 
| <32, 1, 128> | 922.02 | 294.40 | 3.13 | 
| <32, 4, 32> | 279.34 | 88.29 | 3.16 | 
| <32, 4, 64> | 582.95 | 193.42 | 3.01 | 
| <32, 4, 128> | 1198.26 | 454.66 | 2.63 | 
| <64, 1, 32> | 245.73 | 76.29 | 3.22 | 
| <64, 1, 64> | 463.44 | 158.65 | 2.92 | 
| <64, 1, 128> | 1007.24 | 332.69 | 3.02 | 
| <64, 4, 32> | 331.58 | 114.84 | 2.88 | 
| <64, 4, 64> | 699.38 | 262.69 | 2.66 | 
| <64, 4, 128> | 1618.15 | 695.07 | 2.32 | 
| <128, 1, 32> | 270.86 | 82.38 | 3.28 | 
| <128, 1, 64> | 537.55 | 181.03 | 2.96 | 
| <128, 1, 128> | 1183.11 | 442.73 | 2.67 | 
| <128, 4, 32> | 433.38 | 165.23 | 2.62 | 
| <128, 4, 64> | 928.87 | 410.96 | 2.26 | 
| <128, 4, 128> | 2297.10 | 1175.40 | 1.95 | 

<!-- #### Decoder performance on A100 and TensorFlow

* Performance on FP32

* Performance on FP16 -->

### Decoding performance

We demonstrate the inference time of FasterTransformer in C++, TensorFlow and PyTorch, and compare to the performance of pure TensorFlow and PyTorch on T4 and V100.

For the benchmark of TensorFlow, we compare the performance of TensorFlow (TF), the performance of TensorFlow with FasterTransformer OP (FT-OP) and the performance of FasterTransformer on C++ (TF-CPP), and show the speedup of FT-OP and FT-CPP compare to the TensorFlow. 

We do not demonstrate the performance of TensorFlow with XLA since we did not find that using XLA has obvious speedup. 

For the benchmark of PyTorch, we compare the performance of PyTorch decoding with beam search (PyTorch), the performance of replacing the decoder of PyTorch by FasterTransformer (Decoder) and performance of FasterTransformer Decoding with beam search (Decoding), and show the speedup Decoder and Decoding compare to the PyTorch. Due to the dynamic property, it is hard to trace/script the PyTorch decoder/decoding model, so we only test on plain PyTorch.

The results of C++ and TensorFlow were obtained by running the `sample/tensorflow/scripts/profile_decoding_performance.sh`.

The results of PyTorch were obtained by running the `../sample/pytorch/scripts/profile_decoder_decoding.sh`. 

In the experiments of decoding, we updated the following parameters:

* head_num = 8
* size_per_head = 64 
* num_layers = 6
* vocabulary_size = 30000 for TensorFlow sample codes, 31538 for PyTorch sample codes
* memory_hidden_dim = 512

#### Decoding performance on T4 and TensorFlow

* Performance on FP32

| <Batch_size, beam_width, Seq_len> | TF (ms) | FT-OP (ms) | FT-OP Speedup | FT-CPP (ms) | FT-CPP Speedup | 
|:---------------------------------:|:-------:|:----------:|:-------------:|:-----------:|:--------------:| 
| <1, 1, 32> | 453.10 | 31.84 | 14.23 | 28.00 | 16.18 | 
| <1, 1, 64> | 882.08 | 61.51 | 14.34 | 57.33 | 15.38 | 
| <1, 1, 128> | 1843.03 | 126.54 | 14.56 | 122.76 | 15.01 | 
| <1, 4, 32> | 471.63 | 40.71 | 11.58 | 36.44 | 12.94 | 
| <1, 4, 64> | 937.28 | 79.41 | 11.80 | 75.54 | 12.40 | 
| <1, 4, 128> | 1926.79 | 166.26 | 11.58 | 160.75 | 11.98 | 
| <8, 1, 32> | 482.82 | 43.48 | 11.10 | 39.85 | 12.11 | 
| <8, 1, 64> | 921.57 | 87.21 | 10.56 | 83.39 | 11.05 | 
| <8, 1, 128> | 1894.78 | 184.38 | 10.27 | 183.43 | 10.32 | 
| <8, 4, 32> | 515.76 | 56.47 | 9.13 | 53.63 | 9.61 | 
| <8, 4, 64> | 1014.02 | 119.61 | 8.47 | 120.85 | 8.39 | 
| <8, 4, 128> | 2020.41 | 277.44 | 7.28 | 300.16 | 6.73 | 
| <32, 1, 32> | 534.25 | 56.06 | 9.52 | 53.65 | 9.95 | 
| <32, 1, 64> | 1034.65 | 121.27 | 8.53 | 121.52 | 8.51 | 
| <32, 1, 128> | 1966.53 | 285.25 | 6.89 | 300.35 | 6.54 | 
| <32, 4, 32> | 640.24 | 154.65 | 4.13 | 154.34 | 4.14 | 
| <32, 4, 64> | 1354.65 | 350.07 | 3.86 | 367.81 | 3.68 | 
| <32, 4, 128> | 3027.38 | 859.86 | 3.52 | 947.46 | 3.19 | 
| <64, 1, 32> | 553.85 | 86.66 | 6.39 | 85.61 | 6.46 | 
| <64, 1, 64> | 1114.51 | 192.89 | 5.77 | 198.66 | 5.61 | 
| <64, 1, 128> | 2318.32 | 472.83 | 4.90 | 512.98 | 4.51 | 
| <64, 4, 32> | 825.52 | 285.46 | 2.89 | 289.26 | 2.85 | 
| <64, 4, 64> | 1752.80 | 653.98 | 2.68 | 685.59 | 2.55 | 
| <64, 4, 128> | 4390.23 | 1631.13 | 2.69 | 1798.83 | 2.44 | 
| <128, 1, 32> | 620.29 | 151.94 | 4.08 | 153.28 | 4.04 | 
| <128, 1, 64> | 1366.14 | 342.94 | 3.98 | 358.99 | 3.80 | 
| <128, 1, 128> | 2987.18 | 868.05 | 3.44 | 945.11 | 3.16 | 
| <128, 4, 32> | 1170.25 | 542.47 | 2.15 | 552.39 | 2.11 | 
| <128, 4, 64> | 2760.15 | 1257.03 | 2.19 | 1334.39 | 2.06 | 
| <128, 4, 128> | 7774.93 | 3155.91 | 2.46 | 3445.01 | 2.25 | 

* Performance on FP16

| <Batch_size, beam_width, Seq_len> | TF (ms) | FT-OP (ms) | FT-OP Speedup | FT-CPP (ms) | FT-CPP Speedup | 
|:---------------------------------:|:-------:|:----------:|:-------------:|:-----------:|:--------------:| 
| <1, 1, 32> | 396.28 | 34.38 | 11.52 | 26.66 | 14.86 | 
| <1, 1, 64> | 768.43 | 63.88 | 12.02 | 56.44 | 13.61 | 
| <1, 1, 128> | 1543.99 | 129.90 | 11.88 | 123.63 | 12.48 | 
| <1, 4, 32> | 419.53 | 35.09 | 11.95 | 26.25 | 15.98 | 
| <1, 4, 64> | 806.38 | 59.80 | 13.48 | 54.02 | 14.92 | 
| <1, 4, 128> | 1570.90 | 123.67 | 12.70 | 115.83 | 13.56 | 
| <8, 1, 32> | 410.31 | 36.86 | 11.13 | 26.83 | 15.29 | 
| <8, 1, 64> | 795.15 | 63.40 | 12.54 | 58.65 | 13.55 | 
| <8, 1, 128> | 1639.86 | 132.13 | 12.41 | 127.12 | 12.90 | 
| <8, 4, 32> | 439.64 | 38.89 | 11.30 | 35.99 | 12.21 | 
| <8, 4, 64> | 891.54 | 82.09 | 10.86 | 79.82 | 11.16 | 
| <8, 4, 128> | 1766.03 | 182.58 | 9.67 | 193.54 | 9.12 | 
| <32, 1, 32> | 466.24 | 40.58 | 11.48 | 35.76 | 13.03 | 
| <32, 1, 64> | 886.57 | 82.15 | 10.79 | 80.28 | 11.04 | 
| <32, 1, 128> | 1837.41 | 187.04 | 9.82 | 195.01 | 9.42 | 
| <32, 4, 32> | 536.00 | 84.37 | 6.35 | 82.82 | 6.47 | 
| <32, 4, 64> | 1116.74 | 189.16 | 5.90 | 198.95 | 5.61 | 
| <32, 4, 128> | 2473.57 | 470.40 | 5.25 | 518.77 | 4.76 | 
| <64, 1, 32> | 480.88 | 53.39 | 9.00 | 50.89 | 9.44 | 
| <64, 1, 64> | 939.87 | 114.97 | 8.17 | 118.25 | 7.94 | 
| <64, 1, 128> | 2051.09 | 280.67 | 7.30 | 305.32 | 6.71 | 
| <64, 4, 32> | 668.45 | 143.41 | 4.66 | 144.53 | 4.62 | 
| <64, 4, 64> | 1476.17 | 332.89 | 4.43 | 351.14 | 4.20 | 
| <64, 4, 128> | 3282.27 | 860.21 | 3.81 | 966.68 | 3.39 | 
| <128, 1, 32> | 587.50 | 80.61 | 7.28 | 80.79 | 7.27 | 
| <128, 1, 64> | 1107.02 | 182.72 | 6.05 | 193.22 | 5.72 | 
| <128, 1, 128> | 2635.13 | 467.93 | 5.63 | 518.73 | 5.07 | 
| <128, 4, 32> | 996.88 | 265.51 | 3.75 | 271.80 | 3.66 | 
| <128, 4, 64> | 2157.85 | 627.24 | 3.44 | 671.76 | 3.21 | 
| <128, 4, 128> | 5389.81 | 1646.64 | 3.27 | 1848.24 | 2.91 | 

#### Decoding performance on V100 and TensorFlow

* Performance of FP32

| <Batch_size, beam_width, Seq_len> | TF (ms) | FT-OP (ms) | FT-OP Speedup | FT-CPP (ms) | FT-CPP Speedup | 
|:---------------------------------:|:-------:|:----------:|:-------------:|:-----------:|:--------------:| 
| <1, 1, 32> | 247.70 | 20.99 | 11.80 | 19.17 | 12.92 | 
| <1, 1, 64> | 495.89 | 43.63 | 11.36 | 39.93 | 12.41 | 
| <1, 1, 128> | 936.57 | 90.46 | 10.35 | 87.20 | 10.74 | 
| <1, 4, 32> | 234.78 | 30.85 | 7.61 | 28.12 | 8.34 | 
| <1, 4, 64> | 464.19 | 54.83 | 8.46 | 52.79 | 8.79 | 
| <1, 4, 128> | 909.90 | 117.46 | 7.74 | 113.13 | 8.04 | 
| <8, 1, 32> | 231.98 | 28.18 | 8.23 | 25.61 | 9.05 | 
| <8, 1, 64> | 457.38 | 56.72 | 8.06 | 53.44 | 8.55 | 
| <8, 1, 128> | 923.71 | 121.91 | 7.57 | 117.66 | 7.85 | 
| <8, 4, 32> | 249.10 | 31.72 | 7.85 | 29.34 | 8.49 | 
| <8, 4, 64> | 503.95 | 65.72 | 7.66 | 64.22 | 7.84 | 
| <8, 4, 128> | 1020.94 | 147.66 | 6.91 | 149.51 | 6.82 | 
| <32, 1, 32> | 245.18 | 31.71 | 7.73 | 29.16 | 8.40 | 
| <32, 1, 64> | 521.13 | 65.71 | 7.93 | 64.31 | 8.10 | 
| <32, 1, 128> | 968.92 | 149.11 | 6.49 | 149.72 | 6.47 | 
| <32, 4, 32> | 290.96 | 67.00 | 4.34 | 66.66 | 4.36 | 
| <32, 4, 64> | 662.04 | 147.43 | 4.49 | 155.35 | 4.26 | 
| <32, 4, 128> | 1445.38 | 352.77 | 4.09 | 382.38 | 3.77 | 
| <64, 1, 32> | 267.80 | 42.61 | 6.28 | 42.18 | 6.34 | 
| <64, 1, 64> | 573.75 | 93.68 | 6.12 | 94.01 | 6.10 | 
| <64, 1, 128> | 1204.28 | 217.32 | 5.54 | 228.94 | 5.26 | 
| <64, 4, 32> | 369.10 | 113.17 | 3.26 | 114.41 | 3.22 | 
| <64, 4, 64> | 811.20 | 251.04 | 3.23 | 265.57 | 3.05 | 
| <64, 4, 128> | 1896.34 | 615.58 | 3.08 | 687.73 | 2.75 | 
| <128, 1, 32> | 300.77 | 67.01 | 4.48 | 66.01 | 4.55 | 
| <128, 1, 64> | 619.74 | 150.08 | 4.12 | 151.31 | 4.09 | 
| <128, 1, 128> | 1406.48 | 356.22 | 3.94 | 387.80 | 3.62 | 
| <128, 4, 32> | 497.61 | 202.93 | 2.45 | 207.86 | 2.39 | 
| <128, 4, 64> | 1194.74 | 463.58 | 2.57 | 496.50 | 2.40 | 
| <128, 4, 128> | 3068.19 | 1135.37 | 2.70 | 1259.20 | 2.43 | 

* Performance of FP16

| <Batch_size, beam_width, Seq_len> | TF (ms) | FT-OP (ms) | FT-OP Speedup | FT-CPP (ms) | FT-CPP Speedup | 
|:---------------------------------:|:-------:|:----------:|:-------------:|:-----------:|:--------------:| 
| <1, 1, 32> | 179.29 | 22.79 | 7.86 | 19.90 | 9.00 | 
| <1, 1, 64> | 424.71 | 46.31 | 9.17 | 42.07 | 10.09 | 
| <1, 1, 128> | 800.49 | 106.68 | 7.50 | 102.70 | 7.79 | 
| <1, 4, 32> | 215.21 | 22.99 | 9.36 | 20.42 | 10.53 | 
| <1, 4, 64> | 426.36 | 47.33 | 9.00 | 42.67 | 9.99 | 
| <1, 4, 128> | 842.32 | 105.93 | 7.95 | 105.07 | 8.01 | 
| <8, 1, 32> | 218.83 | 22.45 | 9.74 | 20.29 | 10.78 | 
| <8, 1, 64> | 429.64 | 46.16 | 9.30 | 42.66 | 10.07 | 
| <8, 1, 128> | 827.80 | 96.64 | 8.56 | 94.76 | 8.73 | 
| <8, 4, 32> | 228.45 | 25.30 | 9.02 | 23.36 | 9.77 | 
| <8, 4, 64> | 434.26 | 51.36 | 8.45 | 49.95 | 8.69 | 
| <8, 4, 128> | 879.69 | 113.05 | 7.78 | 115.80 | 7.59 | 
| <32, 1, 32> | 224.73 | 25.34 | 8.86 | 23.12 | 9.72 | 
| <32, 1, 64> | 447.28 | 51.98 | 8.60 | 50.01 | 8.94 | 
| <32, 1, 128> | 887.31 | 114.14 | 7.77 | 114.74 | 7.73 | 
| <32, 4, 32> | 249.40 | 43.55 | 5.72 | 43.17 | 5.77 | 
| <32, 4, 64> | 549.04 | 96.69 | 5.67 | 101.74 | 5.39 | 
| <32, 4, 128> | 1182.18 | 225.50 | 5.24 | 248.09 | 4.76 | 
| <64, 1, 32> | 227.12 | 30.99 | 7.32 | 29.93 | 7.58 | 
| <64, 1, 64> | 494.82 | 67.05 | 7.37 | 67.49 | 7.33 | 
| <64, 1, 128> | 1000.46 | 154.54 | 6.47 | 160.94 | 6.21 | 
| <64, 4, 32> | 304.52 | 68.84 | 4.42 | 69.72 | 4.36 | 
| <64, 4, 64> | 666.90 | 154.89 | 4.30 | 164.80 | 4.04 | 
| <64, 4, 128> | 1494.30 | 373.57 | 4.00 | 425.44 | 3.51 | 
| <128, 1, 32> | 252.69 | 43.08 | 5.86 | 42.74 | 5.91 | 
| <128, 1, 64> | 535.56 | 93.53 | 5.72 | 97.05 | 5.51 | 
| <128, 1, 128> | 1134.44 | 225.94 | 5.02 | 245.81 | 4.61 | 
| <128, 4, 32> | 410.80 | 114.56 | 3.58 | 118.16 | 3.47 | 
| <128, 4, 64> | 934.86 | 263.50 | 3.54 | 283.36 | 3.29 | 
| <128, 4, 128> | 2236.95 | 653.69 | 3.42 | 746.66 | 2.99 | 

<!-- #### Decoding performance on A100 and TensorFlow

* Performance of FP32

* Performance of FP16 -->

#### Decoder and decoding performance on T4 and PyTorch

* Performance on FP32

| <batch_size, seq_len, beam_size> | PyTorch (ms) | Decoder (ms) | Decoding (ms) | Decoder Speedup | Decoding Speedup | 
|:-----------------------:|:------:|:------:|:------:|:---------:|:---------:| 
| <1, 32, 1> | 484.75 | 144.20 | 29.08 | 3.36 | 16.66 | 
| <1, 64, 1> | 964.91 | 295.16 | 57.97 | 3.26 | 16.64 | 
| <1, 128, 1> | 2482.00 | 716.21 | 118.97 | 3.46 | 20.86 | 
| <8, 32, 1> | 640.09 | 198.37 | 41.27 | 3.22 | 15.50 | 
| <8, 64, 1> | 1026.29 | 326.66 | 86.32 | 3.14 | 11.88 | 
| <8, 128, 1> | 2077.31 | 683.36 | 180.75 | 3.03 | 11.49 | 
| <32, 32, 1> | 539.02 | 182.05 | 55.35 | 2.96 | 9.73 | 
| <32, 64, 1> | 1060.14 | 368.43 | 121.32 | 2.87 | 8.73 | 
| <32, 128, 1> | 2198.63 | 822.78 | 294.63 | 2.67 | 7.46 | 
| <64, 32, 1> | 544.38 | 216.06 | 87.28 | 2.51 | 6.23 | 
| <64, 64, 1> | 1359.49 | 483.68 | 196.35 | 2.81 | 6.92 | 
| <64, 128, 1> | 2409.26 | 1239.34 | 487.91 | 1.94 | 4.93 | 
| <128, 32, 1> | 705.29 | 321.99 | 157.30 | 2.19 | 4.48 | 
| <128, 64, 1> | 1490.15 | 765.70 | 359.43 | 1.94 | 4.14 | 
| <128, 128, 1> | 3328.75 | 2032.92 | 900.86 | 1.63 | 3.69 | 
| <1, 32, 4> | 519.91 | 170.90 | 37.49 | 3.04 | 13.86 | 
| <1, 64, 4> | 1022.17 | 329.85 | 75.47 | 3.09 | 13.54 | 
| <1, 128, 4> | 2087.35 | 654.85 | 156.97 | 3.18 | 13.29 | 
| <8, 32, 4> | 653.81 | 212.86 | 55.83 | 3.07 | 11.71 | 
| <8, 64, 4> | 1056.50 | 363.22 | 121.80 | 2.90 | 8.67 | 
| <8, 128, 4> | 2187.94 | 842.20 | 298.90 | 2.59 | 7.31 | 
| <32, 32, 4> | 588.74 | 320.21 | 160.45 | 1.83 | 3.66 | 
| <32, 64, 4> | 1280.28 | 773.54 | 363.31 | 1.65 | 3.52 | 
| <32, 128, 4> | 2869.27 | 2116.43 | 916.30 | 1.35 | 3.13 | 
| <64, 32, 4> | 694.86 | 530.53 | 297.42 | 1.30 | 2.33 | 
| <64, 64, 4> | 1777.26 | 1331.30 | 687.77 | 1.33 | 2.58 | 
| <64, 128, 4> | 4769.54 | 3960.06 | 1740.75 | 1.20 | 2.73 | 
| <128, 32, 4> | 990.83 | 975.95 | 576.75 | 1.01 | 1.71 | 
| <128, 64, 4> | 2794.30 | 2610.29 | 1310.25 | 1.07 | 2.13 | 

* Performance on FP16

| <batch_size, seq_len, beam_size> | PyTorch (ms) | Decoder (ms) | Decoding (ms) | Decoder Speedup | Decoding Speedup | 
|:-----------------------:|:------:|:------:|:------:|:---------:|:---------:| 
| <1, 32, 1> | 636.17 | 187.04 | 28.32 | 3.40 | 22.46 | 
| <1, 64, 1> | 1030.81 | 313.46 | 53.82 | 3.28 | 19.15 | 
| <1, 128, 1> | 2029.57 | 612.47 | 121.08 | 3.31 | 16.76 | 
| <8, 32, 1> | 546.08 | 163.20 | 34.43 | 3.34 | 15.86 | 
| <8, 64, 1> | 1112.37 | 315.34 | 73.64 | 3.52 | 15.10 | 
| <8, 128, 1> | 2237.78 | 638.65 | 160.04 | 3.50 | 13.98 | 
| <32, 32, 1> | 546.68 | 171.72 | 40.91 | 3.18 | 13.36 | 
| <32, 64, 1> | 1374.25 | 342.27 | 89.34 | 4.01 | 15.38 | 
| <32, 128, 1> | 2219.99 | 712.94 | 206.78 | 3.11 | 10.73 | 
| <64, 32, 1> | 557.29 | 196.28 | 60.96 | 2.83 | 9.14 | 
| <64, 64, 1> | 1127.56 | 423.53 | 133.64 | 2.66 | 8.43 | 
| <64, 128, 1> | 2431.01 | 1024.73 | 324.01 | 2.37 | 7.50 | 
| <128, 32, 1> | 604.19 | 260.15 | 100.36 | 2.32 | 6.02 | 
| <128, 64, 1> | 1252.95 | 594.85 | 228.57 | 2.10 | 5.48 | 
| <128, 128, 1> | 2727.85 | 1526.56 | 567.00 | 1.78 | 4.81 | 
| <1, 32, 4> | 568.26 | 165.05 | 33.89 | 3.44 | 16.76 | 
| <1, 64, 4> | 1099.60 | 321.63 | 68.78 | 3.41 | 15.98 | 
| <1, 128, 4> | 2177.06 | 630.75 | 146.24 | 3.45 | 14.88 | 
| <8, 32, 4> | 558.22 | 173.52 | 41.02 | 3.21 | 13.60 | 
| <8, 64, 4> | 1105.78 | 343.64 | 88.14 | 3.21 | 12.54 | 
| <8, 128, 4> | 2240.45 | 728.21 | 205.81 | 3.07 | 10.88 | 
| <32, 32, 4> | 606.68 | 267.60 | 104.44 | 2.26 | 5.80 | 
| <32, 64, 4> | 1254.07 | 606.08 | 237.79 | 2.06 | 5.27 | 
| <32, 128, 4> | 2741.17 | 1553.44 | 580.81 | 1.76 | 4.71 | 
| <64, 32, 4> | 669.47 | 399.96 | 192.19 | 1.67 | 3.48 | 
| <64, 64, 4> | 1424.02 | 966.43 | 436.73 | 1.47 | 3.26 | 
| <64, 128, 4> | 3638.59 | 2843.25 | 1091.42 | 1.27 | 3.33 | 
| <128, 32, 4> | 968.40 | 690.89 | 369.87 | 1.40 | 2.61 | 
| <128, 64, 4> | 2087.75 | 1808.63 | 838.92 | 1.15 | 2.48 | 
| <128, 128, 4> | 6735.41 | 5440.68 | 2082.84 | 1.23 | 3.23 | 

#### Decoder and decoding performance on V100 and PyTorch

* Performance on FP32

| <batch_size, seq_len, beam_size> | PyTorch (ms) | Decoder (ms) | Decoding (ms) | Decoder Speedup | Decoding Speedup | 
|:-----------------------:|:------:|:------:|:------:|:---------:|:---------:| 
| <1, 32, 1> | 353.90 | 103.39 | 19.72 | 3.42 | 17.94 | 
| <1, 64, 1> | 698.88 | 212.27 | 40.61 | 3.29 | 17.20 | 
| <1, 128, 1> | 1449.20 | 441.20 | 79.19 | 3.28 | 18.30 | 
| <8, 32, 1> | 439.07 | 139.12 | 27.43 | 3.15 | 16.00 | 
| <8, 64, 1> | 761.94 | 237.07 | 55.40 | 3.21 | 13.75 | 
| <8, 128, 1> | 1731.31 | 535.99 | 117.83 | 3.23 | 14.69 | 
| <32, 32, 1> | 373.02 | 124.94 | 30.53 | 2.98 | 12.21 | 
| <32, 64, 1> | 771.97 | 250.84 | 66.12 | 3.07 | 11.67 | 
| <32, 128, 1> | 1563.37 | 527.23 | 147.27 | 2.96 | 10.61 | 
| <64, 32, 1> | 391.65 | 166.63 | 43.54 | 2.35 | 8.99 | 
| <64, 64, 1> | 763.75 | 347.91 | 95.53 | 2.19 | 7.99 | 
| <64, 128, 1> | 1626.91 | 734.35 | 225.06 | 2.21 | 7.22 | 
| <128, 32, 1> | 399.32 | 205.76 | 65.84 | 1.94 | 6.06 | 
| <128, 64, 1> | 845.62 | 428.30 | 147.87 | 1.97 | 5.71 | 
| <128, 128, 1> | 1780.45 | 1061.66 | 362.33 | 1.67 | 4.91 | 
| <1, 32, 4> | 361.21 | 113.60 | 29.08 | 3.17 | 12.42 | 
| <1, 64, 4> | 733.17 | 220.84 | 52.21 | 3.31 | 14.04 | 
| <1, 128, 4> | 1489.75 | 467.02 | 125.59 | 3.18 | 11.86 | 
| <8, 32, 4> | 382.98 | 124.76 | 30.43 | 3.06 | 12.58 | 
| <8, 64, 4> | 768.14 | 248.43 | 64.50 | 3.09 | 11.90 | 
| <8, 128, 4> | 1535.88 | 532.08 | 149.88 | 2.88 | 10.24 | 
| <32, 32, 4> | 401.86 | 196.38 | 69.34 | 2.04 | 5.79 | 
| <32, 64, 4> | 842.37 | 435.26 | 151.97 | 1.93 | 5.54 | 
| <32, 128, 4> | 1758.36 | 1076.28 | 367.99 | 1.63 | 4.77 | 
| <64, 32, 4> | 433.80 | 283.74 | 114.21 | 1.52 | 3.79 | 
| <64, 64, 4> | 955.72 | 698.55 | 256.37 | 1.36 | 3.72 | 
| <64, 128, 4> | 2137.94 | 1777.37 | 642.46 | 1.20 | 3.32 | 
| <128, 32, 4> | 510.07 | 456.99 | 213.86 | 1.11 | 2.38 | 
| <128, 64, 4> | 1140.04 | 1192.74 | 485.95 | .95 | 2.34 | 

* Performance on FP16

| <batch_size, seq_len, beam_size> | PyTorch (ms) | Decoder (ms) | Decoding (ms) | Decoder Speedup | Decoding Speedup | 
|:-----------------------:|:------:|:------:|:------:|:---------:|:---------:| 
| <1, 32, 1> | 364.93 | 104.67 | 23.59 | 3.48 | 15.46 | 
| <1, 64, 1> | 730.63 | 219.29 | 48.02 | 3.33 | 15.21 | 
| <1, 128, 1> | 1448.80 | 435.08 | 90.06 | 3.32 | 16.08 | 
| <8, 32, 1> | 396.70 | 113.47 | 28.43 | 3.49 | 13.95 | 
| <8, 64, 1> | 766.96 | 213.44 | 58.41 | 3.59 | 13.13 | 
| <8, 128, 1> | 1508.97 | 430.11 | 123.92 | 3.50 | 12.17 | 
| <32, 32, 1> | 380.00 | 113.32 | 30.81 | 3.35 | 12.33 | 
| <32, 64, 1> | 755.43 | 230.70 | 56.28 | 3.27 | 13.42 | 
| <32, 128, 1> | 1592.17 | 481.88 | 140.00 | 3.30 | 11.37 | 
| <64, 32, 1> | 385.02 | 150.23 | 36.38 | 2.56 | 10.58 | 
| <64, 64, 1> | 1006.94 | 352.55 | 77.56 | 2.85 | 12.98 | 
| <64, 128, 1> | 1647.93 | 669.11 | 174.38 | 2.46 | 9.45 | 
| <128, 32, 1> | 393.47 | 172.10 | 49.39 | 2.28 | 7.96 | 
| <128, 64, 1> | 846.32 | 371.34 | 109.92 | 2.27 | 7.69 | 
| <128, 128, 1> | 1812.89 | 892.29 | 260.72 | 2.03 | 6.95 | 
| <1, 32, 4> | 403.72 | 111.89 | 28.33 | 3.60 | 14.25 | 
| <1, 64, 4> | 758.80 | 215.31 | 58.97 | 3.52 | 12.86 | 
| <1, 128, 4> | 1565.94 | 431.89 | 113.51 | 3.62 | 13.79 | 
| <8, 32, 4> | 388.91 | 117.17 | 31.56 | 3.31 | 12.32 | 
| <8, 64, 4> | 768.24 | 232.11 | 61.85 | 3.30 | 12.42 | 
| <8, 128, 4> | 1618.71 | 497.68 | 136.25 | 3.25 | 11.88 | 
| <32, 32, 4> | 415.84 | 183.10 | 51.08 | 2.27 | 8.14 | 
| <32, 64, 4> | 874.10 | 390.93 | 112.19 | 2.23 | 7.79 | 
| <32, 128, 4> | 1806.96 | 876.53 | 255.26 | 2.06 | 7.07 | 
| <64, 32, 4> | 453.94 | 234.66 | 84.20 | 1.93 | 5.39 | 
| <64, 64, 4> | 948.13 | 517.52 | 185.68 | 1.83 | 5.10 | 
| <64, 128, 4> | 2071.99 | 1333.14 | 446.57 | 1.55 | 4.63 | 
| <128, 32, 4> | 486.71 | 349.62 | 146.36 | 1.39 | 3.32 | 
| <128, 64, 4> | 1084.80 | 808.79 | 330.19 | 1.34 | 3.28 | 
| <128, 128, 4> | 2638.70 | 2248.28 | 800.58 | 1.17 | 3.29 | 

#### TensorFlow performance on translation

We test with batch_size 128, beam width 4 on V100.

| Type | tokens per seconds | BLEU |
|:----:|:------------------:|:----:|
| TensorFlow, beam search, FP32 | 2137  | BLEU 26.29 |
| Decoder, beam search, FP32    | 6473  | BLEU 26.29 |
| Decoding, beam search, FP32   | 8513  | BLEU 26.31 |
| TensorFlow, sampling, FP32    | 4178  | BLEU 25.79 |
| Decoder, sampling, FP32       | 10781 | BLEU 25.79 |
| Decoding, sampling, FP32      | 16524 | BLEU 25.79 |
| TensorFlow, beam search, FP16 | 2949  | BLEU 26.31 |
| Decoder, beam search, FP16    | 8682  | BLEU 26.30 |
| Decoding, beam search, FP16   | 12746 | BLEU 26.33 |
| TensorFlow, sampling, FP16    | 6968  | BLEU 25.83 |
| Decoder, sampling, FP16       | 13773 | BLEU 25.80 |
| Decoding, sampling, FP16      | 26718 | BLEU 25.82 |

#### PyTorch performance on translation

batch size 128, beam width 4, max_seq_len 32, beam search algorithm on V100:

| Type | tokens per seconds | BLEU |
|:----:|:------------------:|:----:|
| PyTorch, FP32  | 2462   | BLEU 24.1 |
| Decoder, FP32  | 3358   | BLEU 24.1 |
| Decoding, FP32 | 8959   | BLEU 24.1 |
| PyTorch, FP16  | 4019   | BLEU 24.1 |
| Decoder, FP16  | 4377   | BLEU 24.1 |
| Decoding, FP16 | 15048  | BLEU 24.1 |

#### GPT-2 performance on V100 and TensorFlow

* Performance on 124M GPT-2 model, top 1 sampling

We use the source codes of [here](https://github.com/openai/gpt-2) as the baseline, and compare to FasterTransformer TensorFlow OP and CPP under FP16.

| <Batch_size, Seq_len> | TF (ms) | FT-OP (ms) | FT-OP Speedup | FT-CPP (ms) | FT-CPP Speedup | 
|:---------------------------------:|:-------:|:----------:|:-------------:|:-----------:|:--------------:| 
| <1, 128> | 1887.46 | 171.04 | 11.03 | 151.12 | 12.49 | 
| <1, 1024> | 15734.21 | 1676.85 | 9.38 | 1651.21 | 9.53 | 
| <8, 128> | 1756.14 | 180.87 | 9.71 | 173.97 | 10.09 | 
| <8, 1024> | 17227.00 | 1917.78 | 8.98 | 1906.97 | 9.03 | 
| <32, 128> | 2275.04 | 214.84 | 10.59 | 204.82 | 11.11 | 
| <32, 1024> | 31674.81 | 2462.69 | 12.86 | 2440.94 | 12.98 | 


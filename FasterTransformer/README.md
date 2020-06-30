# FasterTransformer

This repository provides a script and recipe to run the highly optimized transformer for inference, and it is tested and maintained by NVIDIA.

## Table Of Contents
- [FasterTransformer](#fastertransformer)
  - [Table Of Contents](#table-of-contents)
  - [Model overview](#model-overview)
    - [FasterTransformer V1](#fastertransformer-v1)
    - [FasterTransformer V2](#fastertransformer-v2)
    - [FasterTransformer V2.1](#fastertransformer-v21)
    - [Architecture matrix](#architecture-matrix)
  - [Release notes](#release-notes)
    - [Changelog](#changelog)
  - [Known issues](#known-issues)

## Model overview

### FasterTransformer V1

FasterTransformer V1 provides a highly optimized BERT equivalent Transformer layer for inference, including C++ API, TensorFlow op and TensorRT plugin. The experiments show that FasterTransformer V1 can provide 1.3 ~ 2 times speedup on NVIDIA Tesla T4 and NVIDIA Tesla V100 for inference. 

### FasterTransformer V2

FastTransformer V2 adds a highly optimized OpenNMT-tf based decoder and decoding for inference in FasterTransformer V1, including C++ API and TensorFlow op. The experiments show that FasterTransformer V2 can provide 1.5 ~ 11 times speedup on NVIDIA Telsa T4 and NVIDIA Tesla V 100 for inference.

### FasterTransformer V2.1

FasterTransformer V2.1 optimizes some kernels of encoder and decoder, adding the support of PyTorch, the support of remove the padding of encoder and the support of sampling algorithm in decoding. 

### Architecture matrix

The following matrix shows the Architecture Differences between the model.

| Architecure               | Encoder             |Decoder             | Decoding with beam search | Decoding with sampling |
|---------------------------|---------------------|--------------------|---------------------------|------------------------|
|FasterTransformer V1    |  Yes | No  | No  | No  |
|FasterTransformer V2    |  Yes | Yes | Yes | No  |
|FasterTransformer V2.1  |  Yes | Yes | Yes | Yes |


## Release notes

FasterTransformer V1 will be deprecated on July 2020. 

FasterTransformer V2 will be deprecated on Dec 2020. 

### Changelog

June 2020
- **Release the FasterTransformer 2.1**
- Add [effective transformer](https://github.com/bytedance/effective_transformer) supporting into encoder.
- Optimize the beam search kernels.
- Add PyTorch op supporting

May 2020
- Fix the bug that seq_len of encoder must be larger than 3.
- Add the position_encoding of decoding as the input of FasterTransformer decoding. This is convenient to use different types of position encoding. FasterTransformer does not compute the position encoding value, but only lookup the table. 
- Modifying the method of loading model in `translate_sample.py`.

April 2020
- Rename `decoding_opennmt.h` to `decoding_beamsearch.h`
- Add DiverseSiblingsSearch for decoding.
- Add sampling into Decoding
  - The implementation is in the `decoding_sampling.h`
  - Add top_k sampling, top_p sampling for decoding.
- Refactor the tensorflow custom op codes.
  - Merge `bert_transformer_op.h`, `bert_transformer_op.cu.cc` into `bert_transformer_op.cc`
  - Merge `decoder.h`, `decoder.cu.cc` into `decoder.cc`
  - Merge `decoding_beamsearch.h`, `decoding_beamsearch.cu.cc` into `decoding_beamsearch.cc`
- Fix the bugs of finalize function decoding.py. 
- Fix the bug of tf DiverseSiblingSearch.
- Add BLEU scorer `bleu_score.py` into `utils`. Note that the BLEU score requires python3. 
- Fuse QKV Gemm of encoder and masked_multi_head_attention of decoder.
- Add dynamic batch size and dynamic sequence length features into all ops.

March 2020
- Add feature in FasterTransformer 2.0
  - Fix the bug of maximum sequence length of decoder cannot be larger than 128.
  - Add `translate_sample.py` to demonstrate how to translate a sentence by restoring the pretrained model of OpenNMT-tf.
  - Fix the bug that decoding does not check finish or not after each step. 
  - Fix the bug of decoder about max_seq_len.
  - Modify the decoding model structure to fit the OpenNMT-tf decoding model. 
    - Add a layer normalization layer after decoder.
    - Add a normalization for inputs of decoder
    
February 2020
 * Release the FasterTransformer 2.0
 * Provide a highly optimized OpenNMT-tf based decoder and decoding, including C++ API and TensorFlow OP.
 * Refine the sample codes of encoder.
 * Add dynamic batch size feature into encoder op.

July 2019
 * Release the FasterTransformer 1.0
 * Provide a highly optimized bert equivalent transformer layer, including C++ API, TensorFlow OP and TensorRT plugin.
 

## Known issues

There are no known issues with this model.

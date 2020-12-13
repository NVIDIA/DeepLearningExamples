# FasterTransformer

This repository provides a script and recipe to run the highly optimized transformer for inference, and it is tested and maintained by NVIDIA.

## Table Of Contents
- [FasterTransformer](#fastertransformer)
  - [Table Of Contents](#table-of-contents)
  - [Model overview](#model-overview)
    - [FasterTransformer v1](#fastertransformer-v1)
    - [FasterTransformer v2](#fastertransformer-v2)
    - [FasterTransformer v2.1](#fastertransformer-v21)
    - [FasterTransformer v3.0](#fastertransformer-v30)
    - [FasterTransformer v3.1](#fastertransformer-v31)
    - [Architecture matrix](#architecture-matrix)
  - [Release notes](#release-notes)
    - [Changelog](#changelog)
  - [Known issues](#known-issues)

## Model overview

### FasterTransformer v1

FasterTransformer v1 provides a highly optimized BERT equivalent Transformer layer for inference, including C++ API, TensorFlow op and TensorRT plugin. The experiments show that FasterTransformer v1 can provide 1.3 ~ 2 times speedup on NVIDIA Tesla T4 and NVIDIA Tesla V100 for inference. 

### FasterTransformer v2

FastTransformer v2 adds a highly optimized OpenNMT-tf based decoder and decoding for inference in FasterTransformer v1, including C++ API and TensorFlow op. The experiments show that FasterTransformer v2 can provide 1.5 ~ 11 times speedup on NVIDIA Telsa T4 and NVIDIA Tesla V 100 for inference.

### FasterTransformer v2.1

FasterTransformer v2.1 optimizes some kernels of encoder and decoder, adding the support of PyTorch, the support of remove the padding of encoder and the support of sampling algorithm in decoding. 

### FasterTransformer v3.0 

FasterTransformer v3.0 adds the supporting of INT8 quantization for cpp and TensorFlow encoder model on Turing and Ampere GPUs. 

### FasterTransformer v3.1

First, FasterTransformer v3.1 adds the supporting of INT8 quantization of PyTorch encoder model on Turing and Ampere GPUs. Second, v3.1 improve the performance of encoder on FP16 and INT8. Compared to v3.0, v3.1 provides at most 1.2x speedup on T4 FP16, and 1.7x speedup on T4 INT8. Third, v3.1 supports the inference of GPT-2 model.

### Architecture matrix

The following matrix shows the Architecture Differences between the model.

| Architecure               | Encoder           | Encoder INT8 quantization  | Decoder             | Decoding with beam search | Decoding with sampling | GPT-2 |
|---------------------------|-------------------|----------------------------|---------------------|---------------------------|------------------------|-------|
| v1   | Yes | No  | No  | No  | No  | No  |
| v2   | Yes | No  | Yes | Yes | No  | No  |
| v2.1 | Yes | No  | Yes | Yes | Yes | No  |
| v3.0 | Yes | Yes | Yes | Yes | Yes | No  |
| v3.1 | Yes | Yes | Yes | Yes | Yes | Yes |

## Release notes

FasterTransformer v1 was deprecated on July 2020. 

FasterTransformer v2 will be deprecated on Dec 2020. 

FasterTransformer v2.1 will be deprecated on July 2021. 

FasterTransformer v3.0 will be deprecated on Sep 2021. 

### Changelog

Dec 2020
- **Release the FasterTransformer 3.1**

Nov 2020
- Optimize the INT8 inference.
- Support PyTorch INT8 inference.
- Provide PyTorch INT8 quantiztion tools.
- Integrate the fused multi-head attention kernel of TensorRT into FasterTransformer.
- Add unit test of SQuAD. 
- Update the missed NGC checkpoints.

Sep 2020
- Support GPT2
- **Release the FasterTransformer 3.0**
  - Support INT8 quantization of encoder of cpp and TensorFlow op.
  - Add bert-tf-quantization tool.
  - Fix the issue that Cmake 15 or Cmake 16 fail to build this project.

Aug 2020
- Fix the bug of trt plugin.

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
- **Release the FasterTransformer 2.0**
  - Provide a highly optimized OpenNMT-tf based decoder and decoding, including C++ API and TensorFlow OP.
  - Refine the sample codes of encoder.
  - Add dynamic batch size feature into encoder op.

July 2019
- **Release the FasterTransformer 1.0**
  - Provide a highly optimized bert equivalent transformer layer, including C++ API, TensorFlow OP and TensorRT plugin.
 

## Known issues

There are no known issues with this model.

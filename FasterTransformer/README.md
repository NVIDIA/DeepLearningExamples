# FasterTransformer

This repository provides a script and recipe to run the highly optimized transformer for inference, and it is tested and maintained by NVIDIA.

## Table Of Contents
- [Models overview](#model-overview)
    * [FasterTransformer V1](#model-architecture)
    * [FasterTransformer V2](#default-configuration)
    * [Architecture matrix](#feature-support-matrix)
- [Release notes](#release-notes)
    * [Changelog](#changelog)
    * [Known issues](#known-issues)


## Model overview

### FasterTransformer V1

FasterTransformer V1 provides a highly optimized BERT equivalent Transformer layer for inference, including C++ API, TensorFlow op and TensorRT plugin. The experiments show that FasterTransformer V1 can provide 1.3 ~ 2 times speedup on NVIDIA Tesla T4 and NVIDIA Tesla V100 for inference. 

### FasterTransformer V2

FastTransformer V2 adds a highly optimized OpenNMT-tf based decoder and decoding for inference in FasterTransformer V1, including C++ API and TensorFlow op. The experiments show that FasterTransformer V2 can provide 1.5 ~ 11 times speedup on NVIDIA Telsa T4 and NVIDIA Tesla V 100 for inference.

### Architecture matrix

The following matrix shows the Architecture Differences between the model.

| Architecure               | Encoder             |Decoder             |
|-----------------------|--------------------------|---------------|
|FasterTransformer V1  |  Yes |No |
|FasterTransformer V2  |  Yes |Yes |


## Release notes
FasterTransformer V1 will be deprecated on July 2020. 

### Changelog

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

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
This is a sample code to demonstrate how to replace the decoder transformer 
layer of TensorFlow by the decoder of FasterTransformer.

This sample code builds a decoding model by TensorFlow, and user can replace
the decoder transformer layer of model by the decoder of FasterTransformer. 
The other parts, including the embedding lookup, position encoder and beam 
search, are still computed by TensorFlow. 

Namely, the baseline model is like:
    embedding-lookup -> position encoding -> TensorFlow decoder -> beam search
    User can build this model by using "-decoder 0" to set the decoder type
and the new model is like:
    embedding-lookup -> position encoding -> FasterTransformer decoder -> beam search
    User can build this model by using "-decoder 1" to set the decoder type
    
If user wants to verify the correctness of decoder, they can use "-decoder 2",
which will run the both TensorFlow decoder and FasterTransformer in one model, 
and compare their difference. 

Users are also able to use this sample code to test the average forward time of 
TensorFlow and FasterTransformer. 
'''

from __future__ import print_function
import numpy as np
import tensorflow as tf
import argparse
from utils.common import time_test
from utils.common import TransformerArgument
from utils.common import DecodingBeamsearchArgument
from utils.decoding import tf_beamsearch_decoding
from utils.decoding import generate_encoder_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-batch', '--batch_size', type=int, default=1, metavar='NUMBER',
                        help='batch size (default: 1)')
    parser.add_argument('-beam', '--beam_width', type=int, default=4, metavar='NUMBER',
                        help='beam width (default: 4)')
    parser.add_argument('-s', '--max_seq_len', type=int, default=5, metavar='NUMBER',
                        help='max sequence length (default: 5)')
    parser.add_argument('-n', '--head_number', type=int, default=8, metavar='NUMBER',
                        help='head number (default: 8)')
    parser.add_argument('-size', '--size_per_head', type=int, default=64, metavar='NUMBER',
                        help='size per head (default: 64)')
    parser.add_argument('-l', '--num_layer', type=int, default=6, metavar='NUMBER',
                        help='number of layers (default: 6)')
    parser.add_argument('-mem_hidden', '--memory_hidden_dim', type=int, default=768, metavar='NUMBER',
                        help='memory hidden dimension (default: 768)')
    parser.add_argument('-v', '--vocab_size', type=int, default=30000, metavar='BOOL',
                        help='vocabulary size. (default: 30000).')
    parser.add_argument('-d', '--data_type', type=str, default="fp32", metavar='STRING',
                        help='data type (default: fp32)', choices=['fp32', 'fp16'])
    parser.add_argument('-time', '--test_time', type=int, default=0, metavar='BOOL',
                        help='test the time or not. (default: False (0)), True is 1.', 
                        choices=[0, 1])
    parser.add_argument('-decoder', '--decoder_type', type=int, default=2, metavar='NUMBER',
                        help='''
                        Decoder type:
                            type 0: only run tf decoder;
                            type 1: only run op decoder;
                            type 2: run both tf and op decoder, and compare the difference.
                            default: type 2 ''', choices=[0, 1, 2])

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print("{}: {}".format(key, vars(args)[key]))
    print("========================================")

    start_of_sentence_id = 1
    end_of_sentence_id = 2

    np.random.seed(1)
    tf.set_random_seed(1)
    kernel_initializer_range = 0.02
    bias_initializer_range = 0.02

    batch_size = args.batch_size
    beam_width = args.beam_width
    max_seq_len = args.max_seq_len
    head_num = args.head_number
    size_per_head = args.size_per_head
    num_layer = args.num_layer
    hidden_dim = head_num * size_per_head
    memory_hidden_dim = args.memory_hidden_dim
    vocab_size = args.vocab_size
    tf_datatype = tf.float32
    np_datatype = np.float32
    if args.data_type == "fp16":
        tf_datatype = tf.float16
        np_datatype = np.float16

    decoder_args = TransformerArgument(beam_width=beam_width,
                                        head_num=head_num,
                                        size_per_head=size_per_head,
                                        num_layer=num_layer,
                                        dtype=tf_datatype,
                                        kernel_init_range=kernel_initializer_range,
                                        bias_init_range=bias_initializer_range,
                                        fuse_qkv=False)
    
    decoding_args = DecodingBeamsearchArgument(vocab_size,
                                                start_of_sentence_id,
                                                end_of_sentence_id,
                                                max_seq_len,
                                                decoder_args,
                                                0.0)
    
    embedding_table = np.random.randn(vocab_size, hidden_dim).astype(np_datatype) * 0.01  # a [vocab_size, hidden_dim] table
    embedding_table = tf.convert_to_tensor(embedding_table)
    memory, memory_sequence_length = generate_encoder_result(
        batch_size, max_seq_len, memory_hidden_dim, tf_datatype)
    
    finalized_tf_output_ids, finalized_tf_sequence_lengths, _, _, _ = tf_beamsearch_decoding(memory,
                                                                                            memory_sequence_length,
                                                                                            embedding_table,
                                                                                            decoding_args,
                                                                                            decoder_type=args.decoder_type)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(finalized_tf_output_ids)

        if args.test_time == 1:
            
            time_cost = time_test(sess, finalized_tf_output_ids, iterations=10)
            types = ["TF-decoding-beamsearch", "FT-OP-decoder", "TF+FT-OP"]

            print("[INFO] batch_size {} beam_width {} head_num {} size_per_head {} seq_len {} " \
                    "decoder_layers {} vocab_size {} {}-time {:6.2f} ms.".format(batch_size, beam_width, head_num, size_per_head, 
                                                                        max_seq_len, num_layer, vocab_size, types[args.decoder_type], time_cost))

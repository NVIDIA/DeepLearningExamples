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

import numpy as np
import argparse
import tensorflow as tf
import os
from utils.common import time_test, DecodingArgument, int_result_cross_check
from utils.decoding import tf_decoding, generate_encoder_result, op_decoding

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', '--batch_size', type=int, default=1, metavar='NUMBER',
                        help='batch size (default: 1)')
    parser.add_argument('-beam', '--beam_width', type=int, default=4, metavar='NUMBER',
                        help='beam width (default: 4)')
    parser.add_argument('-s', '--max_seq_len', type=int, default=30, metavar='NUMBER',
                        help='max sequence length (default: 30)')
    parser.add_argument('-n', '--head_number', type=int, default=8, metavar='NUMBER',
                        help='head number (default: 8)')
    parser.add_argument('-size', '--size_per_head', type=int, default=64, metavar='NUMBER',
                        help='size per head (default: 64)')
    parser.add_argument('-l', '--num_layer', type=int, default=6, metavar='NUMBER',
                        help='number of layers (default: 6)')
    parser.add_argument('-mem_hidden', '--memory_hidden_dim', type=int, default=768, metavar='NUMBER',
                        help='memory hidden dim (default: 768)')
    parser.add_argument('-v', '--vocab_size', type=int, default=30000, metavar='BOOL',
                        help='vocabulary size. (default: 30000).')
    parser.add_argument('-d', '--data_type', type=str, default="fp32", metavar='STRING',
                        help='data type (default: fp32)')
    parser.add_argument('-x', '--use_XLA', type=int, default=0, metavar='BOOL',
                        help='use XLA (default: False 0)')
    parser.add_argument('-time', '--test_time', type=int, default=0, metavar='BOOL',
                        help='test the time or not. (default: False (0)), True is 1.')
    parser.add_argument('-check', '--cross_check', type=int, default=1, metavar='BOOL',
                        help='cross check the answer of TF and OP. (default: True (1)), False is 0.')
    parser.add_argument('-op_time', '--test_op_time', type=int, default=0, metavar='BOOL',
                        help='test the op time or not. (default: False (0)), True is 1.')
    parser.add_argument('-tf_time', '--test_tf_time', type=int, default=0, metavar='BOOL',
                        help='test the tf time or not. (default: False (0)), True is 1.')

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    print(args)
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
    vocab_size = args.vocab_size
    tf_datatype = tf.float32
    np_datatype = np.float32
    if args.data_type == "fp16":
        tf_datatype = tf.float16
        np_datatype = np.float16
    use_XLA = args.use_XLA

    hidden_dim = head_num * size_per_head
    memory_hidden_dim = args.memory_hidden_dim

    decoding_args = DecodingArgument(batch_size=batch_size,
                                     beam_width=beam_width,
                                     head_num=head_num,
                                     size_per_head=size_per_head,
                                     num_layer=num_layer,
                                     max_seq_len=max_seq_len,
                                     vocab_size=vocab_size,
                                     start_id=start_of_sentence_id,
                                     end_id=end_of_sentence_id,
                                     encoder_hidden_dim=memory_hidden_dim,
                                     dtype=tf_datatype)

    embedding_table = np.random.rand(vocab_size, hidden_dim).astype(
        np_datatype)  # a [vocab_size, hidden_dim] table
    embedding_table = tf.convert_to_tensor(embedding_table)
    memory, memory_sequence_length = generate_encoder_result(
        batch_size, max_seq_len, memory_hidden_dim, tf_datatype)

    finalized_tf_output_ids, finalized_tf_sequence_lengths, tf_output_ids, \
        tf_parent_ids, tf_sequence_lengths = tf_decoding(memory,
                                                         memory_sequence_length,
                                                         embedding_table,
                                                         decoding_args,
                                                         0,
                                                         kernel_initializer_range,
                                                         bias_initializer_range)

    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    finalized_op_output_ids, finalized_op_sequence_lengths, op_output_ids, \
        op_parent_ids, op_sequence_lengths = op_decoding(memory,
                                                         memory_sequence_length,
                                                         embedding_table,
                                                         all_vars,
                                                         decoding_args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if use_XLA == 1:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        
        if args.cross_check == 1:
            finalized_tf_output_ids_result, tf_output_ids_result, tf_parent_ids_result, \
                tf_sequence_lengths_result = sess.run(
                    [finalized_tf_output_ids, tf_output_ids, tf_parent_ids, tf_sequence_lengths])
            finalized_op_output_ids_result, op_output_ids_result, op_parent_ids_result, \
                op_sequence_lengths_result = sess.run(
                    [finalized_op_output_ids, op_output_ids, op_parent_ids, op_sequence_lengths])

            int_result_cross_check("Output ids", tf_output_ids_result, op_output_ids_result, 
                                   shape=[max_seq_len, batch_size * beam_width])
            int_result_cross_check("Parent ids", tf_parent_ids_result, op_parent_ids_result, 
                                   shape=[max_seq_len, batch_size * beam_width])
            int_result_cross_check("Sequence lengths", tf_sequence_lengths_result, 
                                   op_sequence_lengths_result, shape=[1, batch_size * beam_width])
            int_result_cross_check("Finalized output ids", finalized_tf_output_ids_result.T, 
                                   finalized_op_output_ids_result.T,
                                   shape=[max_seq_len, batch_size * beam_width])

        if args.test_time == 1 or args.test_tf_time == 1 or args.test_op_time == 1:
            if args.test_time == 1 or args.test_tf_time == 1:
                tf_time_result = time_test(
                    sess, finalized_tf_output_ids, iterations=50, warmup=True)
            if args.test_time == 1 or args.test_op_time == 1:
                op_time_result = time_test(
                    sess, finalized_op_output_ids, iterations=50, warmup=True)

            if args.test_time == 1 or args.test_tf_time == 1:
                print("[INFO] TF execution time: {} ms".format(tf_time_result))
            if args.test_time == 1 or args.test_op_time == 1:
                print("[INFO] OP execution time: {} ms".format(op_time_result))

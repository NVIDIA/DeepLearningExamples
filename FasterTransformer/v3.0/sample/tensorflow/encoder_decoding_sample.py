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

import tensorflow as tf
import numpy as np
import argparse
from utils.common import int_result_cross_check
from utils.common import time_test
from utils.common import TransformerArgument
from utils.common import DecodingBeamsearchArgument
from utils.encoder import tf_encoder
from utils.encoder import op_encoder
from utils.encoder import build_sequence_mask
from utils.decoding import tf_beamsearch_decoding
from utils.decoding import op_beamsearch_decoding


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', '--batch_size', type=int, default=1, metavar='NUMBER',
                        help='batch size (default: 1)')
    parser.add_argument('-beam', '--beam_width', type=int, default=4, metavar='NUMBER',
                        help='beam width (default: 4)')
    parser.add_argument('-s', '--max_seq_len', type=int, default=5, metavar='NUMBER',
                        help='max sequence length (default: 5)')
    parser.add_argument('-encoder_head', '--encoder_head_number', type=int, default=12, metavar='NUMBER',
                        help='encoder head number (default: 12)')
    parser.add_argument('-encoder_size', '--encoder_size_per_head', type=int, default=64, metavar='NUMBER',
                        help='encoder size per head (default: 64)')
    parser.add_argument('-decoder_head', '--decoder_head_number', type=int, default=8, metavar='NUMBER',
                        help='decoder head number (default: 8)')
    parser.add_argument('-decoder_size', '--decoder_size_per_head', type=int, default=64, metavar='NUMBER',
                        help='decoder size per head (default: 64)')
    parser.add_argument('-encoder_layer', '--encoder_num_layer', type=int, default=12, metavar='NUMBER',
                        help='number of layers (default: 12)')
    parser.add_argument('-decoder_layer', '--decoder_num_layer', type=int, default=6, metavar='NUMBER',
                        help='number of layers (default: 6)')
    parser.add_argument('-v', '--vocab_size', type=int, default=30000, metavar='BOOL',
                        help='vocabulary size. (default: 30000).')
    parser.add_argument('-d', '--data_type', type=str, default="fp32", metavar='STRING',
                        help='data type (default: fp32)')
    parser.add_argument('-time', '--test_time', type=int, default=0, metavar='BOOL',
                        help='test the time or not. (default: False (0)), True is 1.')
    parser.add_argument("-remove_padding", "--remove_padding", type=str, default="False", metavar="BOOL",
                        choices=["True", "False"],
                        help="remove the padding of sentence or not. This brings speedups when the average of \
                            sequence length is smaller than the maximum sequence length.")

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
    initializer_range = 0.02
    bias_initializer_range = 0.02

    batch_size = args.batch_size
    beam_width = args.beam_width
    max_seq_len = args.max_seq_len
    seq_len = max_seq_len
    encoder_head_num = args.encoder_head_number
    encoder_size_per_head = args.encoder_size_per_head
    decoder_head_num = args.decoder_head_number
    decoder_size_per_head = args.decoder_size_per_head
    encoder_num_layer = args.encoder_num_layer
    decoder_num_layer = args.decoder_num_layer
    encoder_hidden_dim = encoder_head_num * encoder_size_per_head
    decoder_hidden_dim = decoder_head_num * decoder_size_per_head
    vocab_size = args.vocab_size
    remove_padding = True if args.remove_padding.lower() == "true" else False
    tf_datatype = tf.float32
    np_datatype = np.float32
    atol_threshold = 2e-5
    if args.data_type == "fp16":
        tf_datatype = tf.float16
        np_datatype = np.float16
        atol_threshold = 2e-2

    from_data = np.random.randn(batch_size, seq_len, encoder_hidden_dim)
    from_tensor = tf.convert_to_tensor(from_data, dtype=tf_datatype)
    memory_sequence_length = np.random.randint(
        1, max_seq_len + 1, size=batch_size).astype(np.int32)
    memory_sequence_length[np.random.randint(0, batch_size)] = max_seq_len
    embedding_table = np.random.randn(vocab_size, decoder_hidden_dim).astype(np_datatype) * initializer_range  # a [vocab_size, decoder_hidden_dim] table

    attention_mask = build_sequence_mask(memory_sequence_length, num_heads=encoder_head_num, maximum_length=max_seq_len, dtype=tf_datatype)

    encoder_args = TransformerArgument(beam_width=1,
                                    head_num=encoder_head_num,
                                    size_per_head=encoder_size_per_head,
                                    num_layer=encoder_num_layer,
                                    dtype=tf_datatype,
                                    remove_padding=remove_padding)

    decoder_args = TransformerArgument(beam_width=beam_width,
                                    head_num=decoder_head_num,
                                    size_per_head=decoder_size_per_head,
                                    num_layer=decoder_num_layer,
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

    tf_encoder_result = tf_encoder(input_tensor=from_tensor,
                                   encoder_args=encoder_args,
                                   attention_mask=attention_mask)
    tf_encoder_result = tf.reshape(
        tf_encoder_result, [batch_size, max_seq_len, encoder_hidden_dim])

    tf_encoder_result = tf_encoder_result * tf.expand_dims(tf.sequence_mask(memory_sequence_length, maxlen=max_seq_len, dtype=tf_datatype), axis=-1)
    
    finalized_tf_output_ids, finalized_tf_sequence_lengths, tf_output_ids, \
        tf_parent_ids, tf_sequence_lengths = tf_beamsearch_decoding(tf_encoder_result,
                                                                    memory_sequence_length,
                                                                    embedding_table,
                                                                    decoding_args,
                                                                    decoder_type=0)
        

    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    decoder_var_start_id = 0
    while all_vars[decoder_var_start_id].name.find("transformer/decoder") == -1:
        decoder_var_start_id += 1
    
    encoder_variables = all_vars[:decoder_var_start_id]
    decoder_variables = all_vars[decoder_var_start_id:]

    encoder_variables_dict = {}
    for v in encoder_variables:
        encoder_variables_dict[v.name] = v
        
    op_encoder_result = op_encoder(inputs=from_tensor,
                                   encoder_args=encoder_args,
                                   attention_mask=attention_mask,
                                   encoder_vars_dict=encoder_variables_dict,
                                   sequence_length=memory_sequence_length)
    op_encoder_result = tf.reshape(
        op_encoder_result, [batch_size, max_seq_len, encoder_hidden_dim])
    op_encoder_result = op_encoder_result * tf.expand_dims(tf.sequence_mask(memory_sequence_length, maxlen=max_seq_len, dtype=tf_datatype), axis=-1)
    
    finalized_op_output_ids, finalized_op_sequence_lengths, op_output_ids, \
        op_parent_ids, op_sequence_lengths = op_beamsearch_decoding(op_encoder_result,
                                                                    memory_sequence_length,
                                                                    embedding_table,
                                                                    decoder_variables,
                                                                    decoding_args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        finalized_tf_output_ids_result, tf_output_ids_result, tf_parent_ids_result, \
            tf_sequence_lengths_result = sess.run(
                [finalized_tf_output_ids, tf_output_ids, tf_parent_ids, tf_sequence_lengths])
        finalized_op_output_ids_result, op_output_ids_result, op_parent_ids_result, \
            op_sequence_lengths_result = sess.run(
                [finalized_op_output_ids, op_output_ids, op_parent_ids, op_sequence_lengths])

        int_result_cross_check("Output ids", tf_output_ids_result, op_output_ids_result, 
                                   shape=[batch_size, beam_width, max_seq_len])
        int_result_cross_check("Parent ids", tf_parent_ids_result, op_parent_ids_result, 
                                   shape=[batch_size, beam_width, max_seq_len])
        int_result_cross_check("Sequence lengths", tf_sequence_lengths_result, 
                                op_sequence_lengths_result, shape=[batch_size, beam_width, 1])
        int_result_cross_check("Finalized output ids", finalized_tf_output_ids_result.T, 
                                finalized_op_output_ids_result.T,
                                shape=[batch_size, beam_width, max_seq_len])

        if args.test_time == 1:
            tf_time_result = time_test(
                sess, finalized_tf_output_ids, iterations=50, warmup=True)
            op_time_result = time_test(
                sess, finalized_op_output_ids, iterations=50, warmup=True)

            print("[INFO] TF execution time: {} ms".format(tf_time_result))
            print("[INFO] OP execution time: {} ms".format(op_time_result))

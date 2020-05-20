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
import numpy as np
from utils.common import DecodingArgument, TransformerArgument
from utils.decoding import tf_decoding
from utils.encoder import tf_encoder, op_encoder

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
    parser.add_argument('-decoder', '--decoder_type', type=int, default=2, metavar='NUMBER',
                        help='Decoder type:'
                        + ' type 0: only run tf decoder;'
                        + ' type 1: only run op decoder;'
                        + ' type 2: run both tf and op decoder, and compare the difference.'
                        + ' default: type 2')

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    print(args)

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
    encoder_head_num = args.encoder_head_number
    encoder_size_per_head = args.encoder_size_per_head
    decoder_head_num = args.decoder_head_number
    decoder_size_per_head = args.decoder_size_per_head
    encoder_num_layer = args.encoder_num_layer
    decoder_num_layer = args.decoder_num_layer
    encoder_hidden_dim = encoder_head_num * encoder_size_per_head
    decoder_hidden_dim = decoder_head_num * decoder_size_per_head
    vocab_size = args.vocab_size
    tf_datatype = tf.float32
    np_datatype = np.float32
    atol_threshold = 2e-5
    if args.data_type == "fp16":
        tf_datatype = tf.float16
        np_datatype = np.float16
        atol_threshold = 2e-2

    initializer_range = 0.02
    from_data = np.random.randn(batch_size, max_seq_len, encoder_hidden_dim)
    from_tensor = tf.convert_to_tensor(from_data, dtype=tf_datatype)
    memory_sequence_length = np.random.randint(
        1, max_seq_len + 1, size=batch_size).astype(np.int32)
    embedding_table = np.random.randn(vocab_size, decoder_hidden_dim).astype(
        np_datatype)  # a [vocab_size, decoder_hidden_dim] table
    embedding_table = tf.convert_to_tensor(embedding_table)

    mask = np.random.randint(2, size=(batch_size, max_seq_len, max_seq_len))
    attention_mask = tf.convert_to_tensor(mask, dtype=tf_datatype)

    encoder_args = TransformerArgument(batch_size=batch_size,
                                       beam_width=1,
                                       head_num=encoder_head_num,
                                       size_per_head=encoder_size_per_head,
                                       num_layer=encoder_num_layer,
                                       max_seq_len=max_seq_len,
                                       dtype=tf_datatype)

    decoding_args = DecodingArgument(batch_size=batch_size,
                                     beam_width=beam_width,
                                     head_num=decoder_head_num,
                                     size_per_head=decoder_size_per_head,
                                     num_layer=decoder_num_layer,
                                     max_seq_len=max_seq_len,
                                     vocab_size=vocab_size,
                                     start_id=start_of_sentence_id,
                                     end_id=end_of_sentence_id,
                                     encoder_hidden_dim=encoder_head_num * encoder_size_per_head,
                                     dtype=tf_datatype)

    tf_encoder_result = tf_encoder(input_tensor=from_tensor,
                                   encoder_args=encoder_args,
                                   attention_mask=attention_mask)
    tf_encoder_result = tf.reshape(
        tf_encoder_result, [batch_size, max_seq_len, encoder_hidden_dim])

    tf_decoding_result, _, _, _, _ = tf_decoding(tf_encoder_result,
                                                 memory_sequence_length,
                                                 embedding_table,
                                                 decoding_args,
                                                 args.decoder_type,
                                                 kernel_initializer_range,
                                                 bias_initializer_range,
                                                 atol_threshold)

    encoder_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    op_encoder_result = op_encoder(inputs=from_tensor,
                                   encoder_args=encoder_args,
                                   encoder_vars=encoder_variables,
                                   attention_mask=attention_mask)
    op_encoder_result = tf.reshape(
        op_encoder_result, [batch_size, max_seq_len, encoder_hidden_dim])

    op_decoding_result, _, _, _, _ = tf_decoding(op_encoder_result,
                                                 memory_sequence_length,
                                                 embedding_table,
                                                 decoding_args,
                                                 args.decoder_type,
                                                 kernel_initializer_range,
                                                 bias_initializer_range,
                                                 atol_threshold)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        print("[INFO] TF encoder + TF-OP decoder: ")
        sess.run(tf_decoding_result)
        print("[INFO] OP encoder + TF-OP decoder: ")
        sess.run(op_decoding_result)

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
from utils.common import TransformerArgument, time_test, cross_check
from utils.encoder import tf_encoder, op_encoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', '--batch_size', type=int, default=1, metavar='NUMBER',
                        help='batch size (default: 1)')
    parser.add_argument('-l', '--num_layer', type=int, default=12, metavar='NUMBER',
                        help='number of layers (default: 12)')
    parser.add_argument('-s', '--seq_len', type=int, default=32, metavar='NUMBER',
                        help='sequence length (default: 32)')
    parser.add_argument('-n', '--head_number', type=int, default=12, metavar='NUMBER',
                        help='head number (default: 12)')
    parser.add_argument('-size', '--size_per_head', type=int, default=64, metavar='NUMBER',
                        help='size per head (default: 64)')
    parser.add_argument('-d', '--data_type', type=str, default="fp32", metavar='STRING',
                        help='data type (default: fp32)')
    parser.add_argument('-time', '--test_time', type=int, default=0, metavar='BOOL',
                        help='test the time or not. (default: False (0)), True is 1.')

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    print(args)

    np.random.seed(1)
    tf.set_random_seed(1)
    kernel_initializer_range = 0.02
    bias_initializer_range = 0.02

    batch_size = args.batch_size
    num_layer = args.num_layer
    seq_len = args.seq_len
    head_num = args.head_number
    size_per_head = args.size_per_head
    tf_datatype = tf.float32
    np_datatype = np.float32
    atol_threshold = 2e-5
    if args.data_type == "fp16":
        tf_datatype = tf.float16
        np_datatype = np.float16
        atol_threshold = 2e-2

    hidden_dim = head_num * size_per_head
    initializer_range = 0.02
    from_data = np.random.randn(batch_size, seq_len, hidden_dim)
    from_tensor = tf.convert_to_tensor(from_data, dtype=tf_datatype)

    mask = np.random.randint(2, size=(batch_size, seq_len, seq_len))
    attention_mask = tf.convert_to_tensor(mask, dtype=tf_datatype)

    encoder_args = TransformerArgument(batch_size=batch_size,
                                       beam_width=1,
                                       head_num=head_num,
                                       size_per_head=size_per_head,
                                       num_layer=num_layer,
                                       max_seq_len=seq_len,
                                       dtype=tf_datatype)

    tf_encoder_result = tf_encoder(input_tensor=from_tensor,
                                   encoder_args=encoder_args,
                                   attention_mask=attention_mask)

    encoder_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    op_encoder_result = op_encoder(inputs=from_tensor,
                                   encoder_args=encoder_args,
                                   encoder_vars=encoder_variables,
                                   attention_mask=attention_mask)

    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        for idx, var in enumerate(encoder_variables):
            print((str(idx) + " " + str(var.name) + " " +
                   str(var.shape)) + " " + str(var.dtype))

        print("#################################")
        tf_encoder_result_val = sess.run(tf_encoder_result)
        op_encoder_result_val = sess.run(op_encoder_result)
        cross_check("Encoder", tf_encoder_result_val,
                    op_encoder_result_val, atol_threshold)

        if args.test_time == 1:
            ite = 100
            tf_time = time_test(sess, tf_encoder_result, ite)
            op_time = time_test(sess, op_encoder_result, ite)

            print("[INFO] TF decoder time costs: {} ms".format(tf_time))
            print("[INFO] OP decoder time costs: {} ms".format(op_time))

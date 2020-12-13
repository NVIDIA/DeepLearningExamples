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
This is a sample code to demonstrate how to use the TensorFlow custom op with 
FasterTransformer library in encoder.

This sample code builds a BERT transformer model by TensorFlow and TensorFlow 
custom op. Then compare the maximum difference of them to verify the correctness
of FasterTransformer. 

Users are also able to use this sample code to test the average forward time of 
TensorFlow and FasterTransformer. 
'''

import tensorflow as tf
import numpy as np
import argparse
import time
from utils.common import TransformerArgument
from utils.common import time_test
from utils.common import cross_check
from utils.encoder import tf_encoder
from utils.encoder import op_encoder
from utils.encoder import build_sequence_mask

def encoder_sample(args_dict):
    print("\n=============== Argument ===============")
    for key in args_dict:
        print("{}: {}".format(key, args_dict[key]))
    print("========================================")

    np.random.seed(1)
    tf.set_random_seed(1)

    batch_size = args_dict['batch_size']
    num_layer = args_dict['num_layer']
    max_seq_len = args_dict['max_seq_len']
    avg_seq_len = args_dict['avg_seq_len']
    head_num = args_dict['head_number']
    size_per_head = args_dict['size_per_head']
    remove_padding = True if args_dict['remove_padding'].lower() == "true" else False
    tf_datatype = tf.float32
    np_datatype = np.float32
    atol_threshold = 3e-5
    int8_mode = args_dict['int8_mode']
    allow_gemm_test = True if args_dict['allow_gemm_test'].lower() == "true" else False
    if args_dict['data_type'] == "fp16":
        tf_datatype = tf.float16
        np_datatype = np.float16
        atol_threshold = 3e-2

    hidden_dim = head_num * size_per_head

    sequence_length = np.random.randint(1, max_seq_len + 1, size=batch_size)
    if avg_seq_len != -1 and remove_padding == True:
        # This means we use "remove_padding" and set a smaller average sequence length
        sequence_length = np.ones(batch_size) * avg_seq_len
    else:
        sequence_length = np.ones(batch_size) * (max_seq_len / 2)
    sequence_length = sequence_length.astype(np.int32)

    from_data = np.random.randn(batch_size, max_seq_len, hidden_dim)
    from_tensor = tf.convert_to_tensor(from_data, dtype=tf_datatype)
    
    attention_mask = build_sequence_mask(sequence_length, num_heads=head_num, maximum_length=max_seq_len, dtype=tf_datatype)
    
    encoder_args = TransformerArgument(beam_width=1,
                                       head_num=head_num,
                                       size_per_head=size_per_head,
                                       num_layer=num_layer,
                                       dtype=tf_datatype,
                                       remove_padding=remove_padding,
                                       int8_mode=int8_mode,
                                       allow_gemm_test=allow_gemm_test)

    tf_encoder_result = tf_encoder(input_tensor=from_tensor,
                                   encoder_args=encoder_args,
                                   attention_mask=attention_mask)

    encoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    encoder_variables_dict = {}
    for v in encoder_vars:
        encoder_variables_dict[v.name] = v
    
    op_encoder_result = op_encoder(inputs=from_tensor,
                                   encoder_args=encoder_args,
                                   attention_mask=attention_mask,
                                   encoder_vars_dict=encoder_variables_dict,
                                   sequence_length=sequence_length)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        for idx, name in enumerate(encoder_variables_dict):
            print((str(idx) + " " + str(name) + " " +
                   str(encoder_variables_dict[name].shape)) + " " + str(encoder_variables_dict[name].dtype))
            
        print("#################################")
        '''
        the scales of int8 mode are random, so we don't cross_check for int8.
        
        tf_encoder_result_val = sess.run(tf_encoder_result)
        op_encoder_result_val = sess.run(op_encoder_result)

        cross_check("Encoder TF v.s. FT with tensor input", tf_encoder_result_val,
                    op_encoder_result_val, atol_threshold)
        '''
        
        ''' 
            Use the numpy array as inputs of FasterTransformer OP. 
            
            This method require more time for the op initialization (especially for FP16), 
            but the inference time would be little faster than using tensor as input. 
        '''
        encoder_variables_dict_2 = {}
        for var, val in zip(encoder_vars, sess.run(encoder_vars)):
            encoder_variables_dict_2[var.name] = val

        if args_dict['test_time'] == 1:
            
            ite = 50
            tf_time = time_test(sess, tf_encoder_result, ite)
            time.sleep(60)
            op_time = time_test(sess, op_encoder_result, ite)
            
            print("[INFO] batch_size {} max_seq_len {} {} layer TF-time {:6.2f} ms".format(batch_size, max_seq_len, num_layer, tf_time))
            print("[INFO] batch_size {} max_seq_len {} {} layer FT-INT8-v{}-OP-tensor-time {:6.2f} ms".format(batch_size, max_seq_len, num_layer, int8_mode, op_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', '--batch_size', type=int, default=4, metavar='NUMBER',
                        help='batch size (default: 4)')
    parser.add_argument('-l', '--num_layer', type=int, default=12, metavar='NUMBER',
                        help='number of layers (default: 12)')
    parser.add_argument('-s', '--max_seq_len', type=int, default=32, metavar='NUMBER',
                        help='max sequence length (default: 32)')
    parser.add_argument('-n', '--head_number', type=int, default=12, metavar='NUMBER',
                        help='head number (default: 12)')
    parser.add_argument('-size', '--size_per_head', type=int, default=64, metavar='NUMBER',
                        help='size per head (default: 64)')
    parser.add_argument('-d', '--data_type', type=str, default="fp32", metavar='STRING',
                        help='data type (default: fp32)', choices=['fp32', 'fp16'])
    parser.add_argument('-int8_mode', '--int8_mode', type=int, default=0, metavar='NUMBER',
                        help='int8 mode (default: 0)', choices=[0, 1, 2])
    parser.add_argument('-allow_gemm_test', '--allow_gemm_test', type=str, default="False", metavar='BOOL',
                        help='whether allow gemm test inside FT (default: False)', choices=["True", "False"])
    parser.add_argument('-time', '--test_time', type=int, default=0, metavar='BOOL',
                        help='test the time or not. (default: False (0)), True is 1.',
                        choices=[0, 1])
    parser.add_argument("-remove_padding", "--remove_padding", type=str, default="False", metavar="BOOL",
                        choices=["True", "False"],
                        help="remove the padding of sentence or not. This brings speedups when the average of \
                            sequence length is smaller than the maximum sequence length.")
    parser.add_argument('-avg_seq', '--avg_seq_len', type=int, default=-1, metavar='NUMBER',
                        help='average sequence length (default: -1)')

    args = parser.parse_args()
    encoder_sample(vars(args))

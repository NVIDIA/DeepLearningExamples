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
FasterTransformer library in decoding. 

This sample code builds a decoding model by TensorFlow and TensorFlow custom 
op. Compare 1. the results of TensorFlow decoding with beam search and 
the results FasterTransformer decoding with beam search; and 2. the results 
of TensorFlow decoding with sampling and the results FasterTransformer decoding 
with sampling.

Users are also able to use this sample code to test the average forward time of 
TensorFlow and FasterTransformer. 
'''

import copy
import numpy as np
import argparse
import tensorflow as tf
from utils.common import time_test
from utils.common import DecodingBeamsearchArgument
from utils.common import DecodingSamplingArgument
from utils.common import TransformerArgument
from utils.common import int_result_cross_check
from utils.decoding import tf_beamsearch_decoding
from utils.decoding import op_beamsearch_decoding
from utils.decoding import tf_sampling_decoding
from utils.decoding import op_sampling_decoding
from utils.decoding import generate_encoder_result

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
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
                        help='data type (default: fp32)', choices=['fp32', 'fp16'])
    parser.add_argument('-x', '--use_XLA', type=int, default=0, metavar='BOOL',
                        help='use XLA (default: False 0)', choices=[0, 1])
    parser.add_argument('-time', '--test_time', type=str, default='', metavar='STRING',
                        help='''
                        Test the time of which one (default: '' (not test anyone) ); 
                        '': not test anyone 
                        '0': test tf_decoding_beamsearch  
                        '1': test op_decoding_beamsearch 
                        '2': test tf_decoding_sampling 
                        '3': test op_decoding_sampling 
                        'e.g., if you want to test tf_decoding_beamsearch and op_decoding_sampling, 
                               then you need to use -time '02' ''')
    parser.add_argument('-check', '--cross_check', type=int, default=1, metavar='BOOL',
                        help='cross check the answer of TF and OP. (default: True (1)), False is 0.',
                        choices=[0, 1])
    parser.add_argument('-diversity_rate', '--beam_search_diversity_rate', type=float, default=0.0, metavar='NUMBER',
                        help='deviersity rate of beam search. default is 0. When diversity rate = 0, it is equivalent to the naive beams earch.')
    parser.add_argument('-topk', '--sampling_topk', type=int, default=1, metavar='NUMBER',
                        help='Candidate (k) value of top k sampling in decoding. Default is 1.')
    parser.add_argument('-topp', '--sampling_topp', type=float, default=0.0, metavar='NUMBER',
                        help='Probability (p) value of top p sampling in decoding. Default is 0.0. ')
    
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
    vocab_size = args.vocab_size
    tf_datatype = tf.float32
    np_datatype = np.float32
    if args.data_type == "fp16":
        tf_datatype = tf.float16
        np_datatype = np.float16
    use_XLA = args.use_XLA
    beam_search_diversity_rate = args.beam_search_diversity_rate
    sampling_topk = args.sampling_topk
    sampling_topp = args.sampling_topp

    hidden_dim = head_num * size_per_head
    memory_hidden_dim = args.memory_hidden_dim
    
    decoder_args = TransformerArgument(beam_width=beam_width,
                                        head_num=head_num,
                                        size_per_head=size_per_head,
                                        num_layer=num_layer,
                                        dtype=tf_datatype,
                                        kernel_init_range=kernel_initializer_range,
                                        bias_init_range=bias_initializer_range)
    
    decoding_args = DecodingBeamsearchArgument(vocab_size,
                                                start_of_sentence_id,
                                                end_of_sentence_id,
                                                max_seq_len,
                                                decoder_args,
                                                beam_search_diversity_rate)
    
    decoder_args_2 = copy.deepcopy(decoder_args) # for beam search
    decoder_args_2.__dict__ = copy.deepcopy(decoder_args.__dict__)
    decoder_args_2.beam_width = 1 # for sampling
    
    decoding_sampling_args = DecodingSamplingArgument(vocab_size,
                                                    start_of_sentence_id,
                                                    end_of_sentence_id,
                                                    max_seq_len,
                                                    decoder_args_2,
                                                    sampling_topk,
                                                    sampling_topp)

    embedding_table = np.random.rand(vocab_size, hidden_dim).astype(
        np_datatype)  # a [vocab_size, hidden_dim] table
    embedding_table = tf.convert_to_tensor(embedding_table)
    memory, memory_sequence_length = generate_encoder_result(
        batch_size, max_seq_len, memory_hidden_dim, tf_datatype)
    
    finalized_tf_output_ids, finalized_tf_sequence_lengths, tf_output_ids, \
        tf_parent_ids, tf_sequence_lengths = tf_beamsearch_decoding(memory,
                                                                    memory_sequence_length,
                                                                    embedding_table,
                                                                    decoding_args,
                                                                    decoder_type=0)

    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    finalized_op_output_ids, finalized_op_sequence_lengths, op_output_ids, \
        op_parent_ids, op_sequence_lengths = op_beamsearch_decoding(memory,
                                                         memory_sequence_length,
                                                         embedding_table,
                                                         all_vars,
                                                         decoding_args)
    
    tf_sampling_target_ids, tf_sampling_target_length = tf_sampling_decoding(memory,
                                                                            memory_sequence_length,
                                                                            embedding_table,
                                                                            decoding_sampling_args,
                                                                            decoder_type=0)
    
    op_sampling_target_ids, op_sampling_target_length = op_sampling_decoding(memory,
                                                                            memory_sequence_length,
                                                                            embedding_table,
                                                                            all_vars,
                                                                            decoding_sampling_args)

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
                
            print("[INFO] BeamSearch cross check:")
            int_result_cross_check("Output ids", tf_output_ids_result, op_output_ids_result, 
                                   shape=[batch_size, beam_width, max_seq_len])
            int_result_cross_check("Parent ids", tf_parent_ids_result, op_parent_ids_result, 
                                   shape=[batch_size, beam_width, max_seq_len])
            int_result_cross_check("Sequence lengths", tf_sequence_lengths_result, 
                                   op_sequence_lengths_result, shape=[batch_size, beam_width, 1])
            int_result_cross_check("Finalized output ids", finalized_tf_output_ids_result.T, 
                                   finalized_op_output_ids_result.T,
                                   shape=[batch_size, beam_width, max_seq_len])
            
            tf_sampling_ids, tf_sampling_length = sess.run([tf_sampling_target_ids,
                                                           tf_sampling_target_length])
            op_sampling_ids, op_sampling_length = sess.run([op_sampling_target_ids,
                                                           op_sampling_target_length])
            print("[INFO] Sampling cross check:")
            int_result_cross_check("Output ids", tf_sampling_ids, op_sampling_ids,
                                   shape=[batch_size, max_seq_len])
            int_result_cross_check("Sequence length", tf_sampling_length, op_sampling_length,
                                   shape=[batch_size])
            

        time_args = args.test_time
        test_lists = []
        test_names = []
        if time_args.find("0") != -1:
            test_lists.append(finalized_tf_output_ids)
            test_names.append("TF-decoding-beamsearch")
        if time_args.find("1") != -1:
            test_lists.append(finalized_op_output_ids)
            test_names.append("FT-OP-decoding-beamsearch")
        if time_args.find("2") != -1:
            test_lists.append(tf_sampling_target_ids)
            test_names.append("TF-decoding-sampling")
        if time_args.find("3") != -1:
            test_lists.append(op_sampling_target_ids)
            test_names.append("FT-OP-decoding-sampling")
            
        test_time_result = []
        for op in test_lists:
            test_time_result.append(time_test(sess, op, iterations=10, warmup=True))
        
        for name, t_result in zip(test_names, test_time_result):
            if name.find("beamsearch") != -1:
                print("[INFO] batch_size {} beam_width {} head_num {} size_per_head {} seq_len {} " \
                    "decoder_layers {} vocab_size {} {}-time {:6.2f} ms.".format(batch_size, beam_width, head_num, size_per_head, 
                                                                        max_seq_len, num_layer, vocab_size, name, t_result))
            elif name.find("sampling") != -1:
                print("[INFO] batch_size {} topk {} topp {} head_num {} size_per_head {} seq_len {} " \
                    "decoder_layers {} vocab_size {} {}-time {:6.2f} ms.".format(batch_size, sampling_topk, sampling_topp, head_num, size_per_head, 
                                                                        max_seq_len, num_layer, vocab_size, name, t_result))

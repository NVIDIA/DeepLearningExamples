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
This is a sample code to demonstrate how to use the Fastertransformer op 
to translate sentence from English to German. 

This sample code builds then encoder model by TensorFlow, which has the same 
model structure to OpenNMT-tf encoder. Next, building the decoder model by
TensorFlow and FasterTransformer op, which has the same model structure to
OpenNMT-tf decoder. So, we can restore the checkpoint of OpenNMT-tf 
transformer model directly. 

We compare the bleu scores and the times of translating all sentences in test 
dataset of TensorFlow and FasterTransformer op. 
'''

from __future__ import print_function
import copy
from datetime import datetime
import tensorflow as tf
import numpy as np
import argparse
import os
from utils.common import TransformerArgument
from utils.common import DecodingSamplingArgument
from utils.common import DecodingBeamsearchArgument
from utils.encoder import tf_encoder_opennmt
from utils.decoding import tf_beamsearch_decoding
from utils.decoding import tf_sampling_decoding
from utils.decoding import op_beamsearch_decoding
from utils.decoding import op_sampling_decoding
from utils.bleu_score import bleu_score
from opennmt.utils import misc
from opennmt.inputters import WordEmbedder
from opennmt.inputters import ExampleInputter

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', '--batch_size', type=int, default=1, metavar='NUMBER',
                        help='batch size (default: 1)')
    parser.add_argument('-beam', '--beam_width', type=int, default=4, metavar='NUMBER',
                        help='beam width (default: 4)')
    parser.add_argument('-s', '--max_seq_len', type=int, default=200, metavar='NUMBER',
                        help='max sequence length (default: 200)')
    parser.add_argument('-encoder_head', '--encoder_head_number', type=int, default=8, metavar='NUMBER',
                        help='encoder head number (default: 8)')
    parser.add_argument('-encoder_size', '--encoder_size_per_head', type=int, default=64, metavar='NUMBER',
                        help='encoder size per head (default: 64)')
    parser.add_argument('-decoder_head', '--decoder_head_number', type=int, default=8, metavar='NUMBER',
                        help='decoder head number (default: 8)')
    parser.add_argument('-decoder_size', '--decoder_size_per_head', type=int, default=64, metavar='NUMBER',
                        help='decoder size per head (default: 64)')
    parser.add_argument('-encoder_layer', '--encoder_num_layer', type=int, default=6, metavar='NUMBER',
                        help='number of layers (default: 6)')
    parser.add_argument('-decoder_layer', '--decoder_num_layer', type=int, default=6, metavar='NUMBER',
                        help='number of layers (default: 6)')
    parser.add_argument('-d', '--data_type', type=str, default="fp32", metavar='STRING',
                        help='data type (default: fp32)', choices=['fp32', 'fp16'])
    parser.add_argument('-time', '--test_time', type=str, default='', metavar='STRING',
                        help='''
                        Test the time of which one (default: '' (not test anyone) ); 
                        '': not test anyone 
                        '0': test tf_decoding_beamsearch  
                        '1': test op_decoder_beamsearch 
                        '2': test op_decoding_beamsearch 
                        '3': test tf_decoding_sampling 
                        '4': test op_decoder_sampling 
                        '5': test op_decoding_sampling 
                        'e.g., if you want to test op_decoder_beamsearch and op_decoding_sampling, 
                               then you need to use -time '15' ''')
    parser.add_argument('-diversity_rate', '--beam_search_diversity_rate', type=float, default=0.0, metavar='NUMBER',
                        help='deviersity rate of beam search. default is 0. When diversity rate = 0, it is equivalent to the naive beams earch.')
    parser.add_argument('-topk', '--sampling_topk', type=int, default=1, metavar='NUMBER',
                        help='Candidate (k) value of top k sampling in decoding. Default is 1.')
    parser.add_argument('-topp', '--sampling_topp', type=float, default=0.0, metavar='NUMBER',
                        help='Probability (p) value of top p sampling in decoding. Default is 0.0. ')
    
    parser.add_argument('--source_vocabulary', type=str, default="./tensorflow/utils/translation/wmtende.vocab", metavar='STRING',
                        help='Source vocabulary file path. Default is ./tensorflow/utils/translation/wmtende.vocab ')
    parser.add_argument('--target_vocabulary', type=str, default="./tensorflow/utils/translation/wmtende.vocab", metavar='STRING',
                        help='Target vocabulary file path. Default is ./tensorflow/utils/translation/wmtende.vocab ')
    parser.add_argument('--source', type=str, default="./tensorflow/utils/translation/test.en", metavar='STRING',
                        help='Source file path. Default is ./tensorflow/utils/translation/test.en ')
    parser.add_argument('--target', type=str, default="./tensorflow/utils/translation/test.de", metavar='STRING',
                        help='Target file path. Default is ./tensorflow/utils/translation/test.de ')

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print("{}: {}".format(key, vars(args)[key]))
    print("========================================")

    start_of_sentence_id = 1
    end_of_sentence_id = 2

    kernel_initializer_range = 0.02
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
    tf_datatype = tf.float32
    np_datatype = np.float32
    if args.data_type == "fp16":
        tf_datatype = tf.float16
        np_datatype = np.float16
    beam_search_diversity_rate = args.beam_search_diversity_rate
    sampling_topk = args.sampling_topk
    sampling_topp = args.sampling_topp

    source_inputter = WordEmbedder("source_vocabulary", embedding_size=encoder_hidden_dim, dtype=tf_datatype)
    target_inputter = WordEmbedder("target_vocabulary", embedding_size=decoder_hidden_dim, dtype=tf_datatype)
    inputter = ExampleInputter(source_inputter, target_inputter)
    inputter.initialize({
        "source_vocabulary": args.source_vocabulary,
        "target_vocabulary": args.target_vocabulary
        })
    vocab_size = target_inputter.vocabulary_size
    source_file = args.source
    
    encoder_args = TransformerArgument(beam_width=1,
                                        head_num=encoder_head_num,
                                        size_per_head=encoder_size_per_head,
                                        num_layer=encoder_num_layer,
                                        dtype=tf_datatype,
                                        kernel_init_range=kernel_initializer_range,
                                        bias_init_range=bias_initializer_range)
    
    decoder_args = TransformerArgument(beam_width=beam_width,
                                        head_num=decoder_head_num,
                                        size_per_head=decoder_size_per_head,
                                        num_layer=decoder_num_layer,
                                        dtype=tf_datatype,
                                        kernel_init_range=kernel_initializer_range,
                                        bias_init_range=bias_initializer_range)
    
    decoder_args_2 = copy.deepcopy(decoder_args) # for beam search
    decoder_args_2.__dict__ = copy.deepcopy(decoder_args.__dict__)
    decoder_args_2.beam_width = 1 # for sampling
        
    decoding_beamsearch_args = DecodingBeamsearchArgument(vocab_size,
                                                        start_of_sentence_id,
                                                        end_of_sentence_id,
                                                        max_seq_len,
                                                        decoder_args,
                                                        beam_search_diversity_rate)
    
    decoding_sampling_args = DecodingSamplingArgument(vocab_size,
                                                    start_of_sentence_id,
                                                    end_of_sentence_id,
                                                    max_seq_len,
                                                    decoder_args_2,
                                                    sampling_topk,
                                                    sampling_topp)

    mode = tf.estimator.ModeKeys.PREDICT
    with tf.variable_scope("transformer/encoder", reuse=tf.AUTO_REUSE):
        dataset = inputter.make_inference_dataset(source_file, batch_size)
        iterator = dataset.make_initializable_iterator()
        source = iterator.get_next()
        source_embedding = source_inputter.make_inputs(source)
        source_embedding = tf.cast(source_embedding, tf_datatype)
        memory_sequence_length = source["length"]

        tf_encoder_result = tf_encoder_opennmt(source_embedding, encoder_args, sequence_length=memory_sequence_length)
        tf_encoder_result = tf.cast(tf_encoder_result, tf_datatype) 
        
    tf_encoder_result = tf.reshape(tf_encoder_result, tf.shape(source_embedding))

    with tf.variable_scope("transformer/decoder", reuse=tf.AUTO_REUSE):
        target_inputter.build()
    target_vocab_rev = target_inputter.vocabulary_lookup_reverse()
    
    ### TF BeamSearch Decoding ###    
    tf_beamsearch_target_ids, tf_beamsearch_target_length, _, _, _ = tf_beamsearch_decoding(tf_encoder_result,
                                                                                            memory_sequence_length,
                                                                                            target_inputter.embedding,
                                                                                            decoding_beamsearch_args,
                                                                                            decoder_type=0)
        
    # tf_beamsearch_target_tokens: [batch_size, beam_width, seq_len]
    tf_beamsearch_target_tokens = target_vocab_rev.lookup(tf.cast(tf_beamsearch_target_ids, tf.int64))
    tf_beamsearch_target_length = tf.minimum(tf_beamsearch_target_length + 1, tf.shape(tf_beamsearch_target_ids)[-1])
    ### end of TF BeamSearch Decoding ###
    
    ### TF Sampling Decoding ###    
    tf_sampling_target_ids, tf_sampling_target_length = tf_sampling_decoding(tf_encoder_result,
                                                                            memory_sequence_length,
                                                                            target_inputter.embedding,
                                                                            decoding_sampling_args,
                                                                            decoder_type=0)
        
    # tf_sampling_target_tokens: [batch_size, seq_len]
    tf_sampling_target_tokens = target_vocab_rev.lookup(tf.cast(tf_sampling_target_ids, tf.int64))
    tf_sampling_target_length = tf.minimum(tf_sampling_target_length + 1, tf.shape(tf_sampling_target_ids)[-1])
    ### end of TF BeamSearch Decoding ###
    
    ### OP BeamSearch Decoder ###    
    op_decoder_beamsearch_target_ids, op_decoder_beamsearch_target_length, _, _, _ = tf_beamsearch_decoding(tf_encoder_result,
                                                                                            memory_sequence_length,
                                                                                            target_inputter.embedding,
                                                                                            decoding_beamsearch_args,
                                                                                            decoder_type=1)
        
    # op_decoder_beamsearch_target_tokens: [batch_size, beam_width, seq_len]
    op_decoder_beamsearch_target_tokens = target_vocab_rev.lookup(tf.cast(op_decoder_beamsearch_target_ids, tf.int64))
    op_decoder_beamsearch_target_length = tf.minimum(op_decoder_beamsearch_target_length + 1, tf.shape(op_decoder_beamsearch_target_ids)[-1])
    ### end of OP BeamSearch Decoder ###
    
    ### OP Sampling Decoder ###    
    op_decoder_sampling_target_ids, op_decoder_sampling_target_length = tf_sampling_decoding(tf_encoder_result,
                                                                            memory_sequence_length,
                                                                            target_inputter.embedding,
                                                                            decoding_sampling_args,
                                                                            decoder_type=1)
        
    op_decoder_sampling_target_tokens = target_vocab_rev.lookup(tf.cast(op_decoder_sampling_target_ids, tf.int64))
    op_decoder_sampling_target_length = tf.minimum(op_decoder_sampling_target_length + 1, tf.shape(op_decoder_sampling_target_ids)[-1])
    ### end of OP BeamSearch Decoder ###
    
    ### Prepare Decoding variables for FasterTransformer  ###
    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    decoder_var_start_id = 0
    
    while all_vars[decoder_var_start_id].name.find("transformer/decoder") == -1:
        decoder_var_start_id += 1
    encoder_variables = all_vars[:decoder_var_start_id]
    decoder_variables = all_vars[decoder_var_start_id + 1:] # decoder_var_start_id + 1 means skip the embedding table
    
    ### OP BeamSearch Decoding ###
    op_beamsearch_target_ids, op_beamsearch_target_length, _, _, _ = op_beamsearch_decoding(tf_encoder_result,
                                                                                            memory_sequence_length,
                                                                                            target_inputter.embedding,
                                                                                            decoder_variables,
                                                                                            decoding_beamsearch_args)

    op_beamsearch_target_tokens = target_vocab_rev.lookup(tf.cast(op_beamsearch_target_ids, tf.int64))
    op_beamsearch_target_length = tf.minimum(op_beamsearch_target_length + 1, tf.shape(op_beamsearch_target_ids)[-1])
    ### end of OP BeamSearch Decoding ###

    ### OP Sampling Decoding ###
    op_sampling_target_ids, op_sampling_target_length = op_sampling_decoding(tf_encoder_result,
                                                                            memory_sequence_length,
                                                                            target_inputter.embedding,
                                                                            decoder_variables,
                                                                            decoding_sampling_args)

    op_sampling_target_tokens = target_vocab_rev.lookup(tf.cast(op_sampling_target_ids, tf.int64))
    op_sampling_target_length = tf.minimum(op_sampling_target_length + 1, tf.shape(op_sampling_target_ids)[-1])
    ### end of OP Sampling Decoding ###

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    time_args = args.test_time
    
    class TranslationResult(object):
        def __init__(self, token_op, length_op, name):
            self.token_op = token_op
            self.length_op = length_op
            self.name = name
            self.file_name = name + ".txt"
            
            self.token_list = []
            self.length_list = []
            self.batch_num = 0
            self.execution_time = 0.0 # seconds
            self.sentence_num = 0
            self.bleu_score = None
    
    translation_result_list = []
    
    if time_args.find("0") != -1:
        translation_result_list.append(TranslationResult(
            tf_beamsearch_target_tokens, tf_beamsearch_target_length, "tf-decoding-beamsearch"))
    if time_args.find("1") != -1:
        translation_result_list.append(TranslationResult(
            op_decoder_beamsearch_target_tokens, op_decoder_beamsearch_target_length, "op-decoder-beamsearch"))
    if time_args.find("2") != -1:
        translation_result_list.append(TranslationResult(
            op_beamsearch_target_tokens, op_beamsearch_target_length, "op-decoding-beamsearch"))
    if time_args.find("3") != -1:
        translation_result_list.append(TranslationResult(
            tf_sampling_target_tokens, tf_sampling_target_length, "tf-decoding-sampling"))
    if time_args.find("4") != -1:
        translation_result_list.append(TranslationResult(
            op_decoder_sampling_target_tokens, op_decoder_sampling_target_length, "op-decoder-sampling"))
    if time_args.find("5") != -1:
        translation_result_list.append(TranslationResult(
            op_sampling_target_tokens, op_sampling_target_length, "op-decoding-sampling"))
    
    float_var_list = []
    half_var_list = []
    for var in tf.global_variables()[:-1]:
        if var.dtype.base_dtype == tf.float32:
            float_var_list.append(var)
        elif var.dtype.base_dtype == tf.float16:
            half_var_list.append(var)
    
    for i in range(len(translation_result_list)):
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())        
            sess.run(tf.tables_initializer())
            sess.run(iterator.initializer)

            if(len(float_var_list) > 0):
                float_saver = tf.train.Saver(float_var_list)
                float_saver.restore(sess, "translation/ckpt/model.ckpt-500000")
            if(len(half_var_list) > 0):
                half_saver = tf.train.Saver(half_var_list)
                half_saver.restore(sess, "translation/ckpt/fp16_model.ckpt-500000")
                
            t1 = datetime.now()
            while True:
                try:
                    batch_tokens, batch_length = sess.run([translation_result_list[i].token_op, 
                                                           translation_result_list[i].length_op])
                    for tokens, length in zip(batch_tokens, batch_length):
                        if translation_result_list[i].name.find("beamsearch") != -1:
                            translation_result_list[i].token_list.append(b" ".join(tokens[0][:length[0] - 2]).decode("UTF-8"))
                        else:
                            translation_result_list[i].token_list.append(b" ".join(tokens[:length - 2]).decode("UTF-8"))
                    translation_result_list[i].batch_num += 1
                except tf.errors.OutOfRangeError:
                    break
            t2 = datetime.now()
            time_sum = (t2 - t1).total_seconds()
            translation_result_list[i].execution_time = time_sum
            
            with open(translation_result_list[i].file_name, "w") as file_b:
                for s in translation_result_list[i].token_list:
                    file_b.write(s)
                    file_b.write("\n")
                    
            ref_file_path = "./.ref_file.txt"
            os.system("head -n %d %s > %s" % (len(translation_result_list[i].token_list), args.target, ref_file_path))
            translation_result_list[i].bleu_score = bleu_score(translation_result_list[i].file_name, ref_file_path)
            os.system("rm {}".format(ref_file_path))

    for t in translation_result_list:
        print("[INFO] {} translates {} batches taking {:.2f} ms to translate {} tokens, BLEU score: {:.2f}, {:.0f} tokens/sec.".format(
            t.name, t.batch_num, t.execution_time, t.bleu_score.sys_len, t.bleu_score.score, t.bleu_score.sys_len / t.execution_time))

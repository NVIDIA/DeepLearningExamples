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
import tensorflow as tf
import os
import pickle 
import sys
from utils.decoder import tf_decoder
from utils.decoder import op_decoder
from utils.decoder import init_op_cache
from utils.decoder import init_tf_cache
from utils.common import create_initializer
from utils.common import _get_shape_invariants
from utils.position import SinusoidalPositionEncoder
from utils.beam_search import search_word
from utils.sampling import Sampling

def initialize_decoding_variables(decoding_args, batchxbeam):

    start_ids = tf.fill([batchxbeam], decoding_args.start_id)  # [batch_size * beam_width]

    step = tf.constant(0, dtype=tf.int32)
    # save the output ids for each step
    outputs = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

    cache = init_tf_cache(batchxbeam,
                          decoding_args.decoder_args.head_num, decoding_args.decoder_args.size_per_head,
                          decoding_args.decoder_args.num_layer, dtype=decoding_args.decoder_args.dtype, num_sources=1)

    finished = tf.zeros([batchxbeam], dtype=tf.bool)  # [batch_size * beam_width], record that a sentence is finished or not
    initial_log_probs = tf.cast(tf.tile([0.] + [-float("inf")] * (decoding_args.decoder_args.beam_width - 1),
                                        [batchxbeam / decoding_args.decoder_args.beam_width]), dtype=tf.float32)  # [batch_size * beam_width]
    # [batch_size * beam_width], record the lengths of all sentences
    sequence_lengths = tf.zeros([batchxbeam], dtype=tf.int32)
    # record the beam search indices, used for rebuild the whole sentence in the final
    parent_ids = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    extra_vars = tuple([parent_ids, sequence_lengths])

    return start_ids, step, outputs, cache, finished, initial_log_probs, sequence_lengths, extra_vars

def generate_encoder_result(batch_size,
                            max_seq_len,
                            memory_hidden_dim,
                            dtype):

    memory_sequence_length = np.random.randint(
        1, max_seq_len + 1, size=batch_size).astype(np.int32)
    memory_sequence_length[np.random.randint(0, batch_size)] = max_seq_len
    outter_embbeding = np.random.randn(memory_hidden_dim) * 0.01

    memory = []
    mem_max_seq_len = np.max(memory_sequence_length)
    for i in range(batch_size):
        data = np.random.randn(mem_max_seq_len, memory_hidden_dim) * 0.01
        for j in range(memory_sequence_length[i], mem_max_seq_len):
            data[j] = outter_embbeding
        memory.append(data)
    memory = np.asarray(memory)
    memory = tf.convert_to_tensor(memory, dtype=dtype)

    return memory, memory_sequence_length

def finalize(beam_width, parent_ids, sequence_lengths, outputs, end_id, max_seq_len=None):
    maximum_lengths = tf.reduce_max(tf.reshape(
        sequence_lengths, [-1, beam_width]), axis=-1)
    
    if max_seq_len != None:
        array_shape = [max_seq_len, -1, beam_width]
    else:
        array_shape = [tf.reduce_max(maximum_lengths), -1, beam_width]
        
    step_ids = tf.reshape(outputs, array_shape)
    parent_ids = tf.reshape(parent_ids, array_shape)

    ids = tf.contrib.seq2seq.gather_tree(
        step_ids, parent_ids, maximum_lengths, end_id)

    ids = tf.transpose(ids, perm=[1, 2, 0])
    lengths = tf.not_equal(ids, end_id)
    lengths = tf.cast(lengths, tf.int32)
    lengths = tf.reduce_sum(lengths, axis=-1)
    return ids, lengths

def decoding_body(word_ids,
                  step,
                  memory,
                  memory_sequence_length,
                  my_cache,
                  op_self_cache,
                  op_mem_cache,
                  embedding_table,
                  decoding_args,
                  decoder_type):
    
    decoder_args = decoding_args.decoder_args
    hidden_dim = decoder_args.hidden_dim
    k_init_range = decoder_args.kernel_init_range
    data_type = decoder_args.dtype
    
    batchxbeam = tf.shape(word_ids)[0]
    # [batch_size * beam_width, hidden_dim]
    inputs = tf.nn.embedding_lookup(embedding_table, word_ids)
    # [batch_size * beam_width, 1, hidden_dim]
    inputs = tf.expand_dims(inputs, 1)

    inputs *= hidden_dim**0.5
    position_encoder = SinusoidalPositionEncoder()
    if position_encoder is not None:
        position_encoding_table = position_encoder._create_position_encoding_table(decoding_args.max_seq_len, hidden_dim, data_type)
        position_encoding_val = position_encoding_table[step]
        position_encoding_val = tf.reshape(position_encoding_val, [1, 1, -1])
        position_encoding_val = tf.tile(position_encoding_val, [batchxbeam, 1, 1])
        inputs = inputs + position_encoding_val
        
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        tf_result = tf_decoder(decoder_args=decoder_args,
                                inputs=inputs,
                                memory=memory,
                                memory_sequence_length=memory_sequence_length,
                                step=step,
                                cache=my_cache)

        if decoder_type != 0:
            decoder_vars = tf.global_variables()
            decoder_vars_start_id = 0
            while decoder_vars_start_id < len(decoder_vars):
                if decoder_vars[decoder_vars_start_id].name.find("transformer/decoder/layer") != -1:
                    break
                decoder_vars_start_id += 1
            decoder_vars = decoder_vars[decoder_vars_start_id:]
            decoder_var_dict = {}
            for v in decoder_vars:
                decoder_var_dict[v.name] = v

            psuedo_input = []
            if decoder_type == 2:
                psuedo_input = tf_result
                
            op_result, op_self_cache, op_mem_cache = op_decoder(inputs,
                                                                memory,
                                                                memory_sequence_length,
                                                                op_self_cache,
                                                                op_mem_cache,
                                                                psuedo_input,
                                                                decoder_var_dict,
                                                                decoder_args)

        result = None
        if decoder_type == 0:
            result = tf_result
        elif decoder_type == 1:
            result = op_result
        elif decoder_type == 2:
            result = tf_result
            result_2 = op_result
            
            flatten_result = tf.reshape(result, [-1])
            flatten_result_2 = tf.reshape(result_2, [-1])
            abs_diff = tf.math.abs(flatten_result - flatten_result_2)
            abs_argmax = tf.math.argmax(abs_diff)
            result = tf.Print(result, ["[INFO][PYTHON] step:", step, 
                                        tf.cond(abs_diff[abs_argmax] / (tf.math.abs(flatten_result[abs_argmax]) + 1e-6) < decoder_args.check_threshold, 
                                                lambda: "True", lambda: "False"),
                                        "max abs diff: ", abs_diff[abs_argmax],
                                        " op val: ", flatten_result_2[abs_argmax],
                                        " tf val: ", flatten_result[abs_argmax] ])
        else:
            print("[TF][ERROR] decoder type is only 0 or 1 or 2.")
            exit(-1)

        result = tf.contrib.layers.layer_norm(result, begin_norm_axis=-1)

        # [batch_size * beam_width, hidden_dim]
        result = tf.squeeze(result, axis=1)
        logits = tf.layers.dense(result,
                                decoding_args.vocab_size,
                                use_bias=True,
                                bias_initializer=create_initializer(0.0, data_type),
                                kernel_initializer=create_initializer(k_init_range, data_type),
                                activation=None)
        
        return logits, my_cache, op_self_cache, op_mem_cache

def tf_beamsearch_decoding(memory_tensor,
                            memory_sequence_length,
                            embedding_table,
                            decoding_args,
                            decoder_type):
    '''
    Run the decoding with beam search by TensorFlow.
    
    Args:
        memory_tensor: A tf.tensor with shape [batch_size * beam_width, max(memory_sequence_length), encoder_hidden_dimension]. 
                       The results of encoder transformer layer. The rank must be 3. 
                       Note that it must be extended by beam_width times.
        memory_sequence_length: A tf.Tensor with shape [batch_size * beam_width], type tf.int. 
                                The lenght of each sentence of results of encoder. 
                                Note that it must be extended by beam_width times.
        embedding_table: A tf.Tensor with shape [vocab_size, hidden_dimension]. 
                         The embedding table of embedding lookup for each step.
        decoder_args: The arguments for decoding. The details are in the class "DecodingBeamsearchArgument" of common.py
        decoder_type: A int value. Choose to using TensorFlow decoder, FasterTransformer decoder, or both.
                      If it is 0, then using the TensorFlow decoder only.
                      If it is 1, then using the FasterTransformer decoder only.
                      If it is 2, then using both decoder and compare their result. 
    Outputs:
        finalized_tf_output_ids: A tf.Tensor with shape [batch_size, beam_width, max(tf_sequence_lengths)], with tf.int type. 
                                 Finalized tf_output_ids by beam search algorithm and tf_parent_ids.
        finalized_tf_sequence_lengths: A tf.Tensor with shape [batch_size * beam_width], with int type.
                                       Finalized tf_sequence_lengths by beam search algorithm and tf_parent_ids.
        tf_output_ids: A tf.Tensor with shape [batch_size, beam_width, max(tf_sequence_lengths)], with tf.int type. 
                       The results of decoding. It contains the id of token of vocabulary.
        tf_parent_ids: A tf.Tensor with shape [batch_size, beam_width, max(tf_sequence_lengths)], with tf.int type.
                       The beam index of output ids for each step. 
        tf_sequence_lengths: A tf.Tensor with shape [batch_size * beam_width], with int type.
    '''

    decoder_args = decoding_args.decoder_args
    beam_width = decoder_args.beam_width
    search_method = decoding_args.search_method
    with tf.variable_scope("transformer", reuse=tf.AUTO_REUSE):
        # copy memory and memory_sequence_length by beam_width times
        # if memory is [a, b, c], beam_width = 3, then the result is: [a a a b b b c c c ]
        extended_memory = tf.contrib.seq2seq.tile_batch(memory_tensor, multiplier=beam_width)
        extended_memory_sequence_length = tf.contrib.seq2seq.tile_batch(
            memory_sequence_length, multiplier=beam_width)

        def _cond(word_ids, cum_log_probs, finished, step, outputs, my_cache, extra_vars, op_self_cache, op_mem_cache):
            return tf.reduce_any(tf.logical_not(finished))

        def _body(word_ids, cum_log_probs, finished, step, outputs, my_cache, extra_vars, op_self_cache, op_mem_cache):
            logits, my_cache, op_self_cache, op_mem_cache = decoding_body(word_ids,
                                                                        step,
                                                                        extended_memory,
                                                                        extended_memory_sequence_length,
                                                                        my_cache,
                                                                        op_self_cache,
                                                                        op_mem_cache,
                                                                        embedding_table,
                                                                        decoding_args,
                                                                        decoder_type)

            end_ids = tf.fill([tf.shape(logits)[0]], decoding_args.end_id) # [batch_size * beam_width]
            eos_max_prob = tf.one_hot(end_ids, decoding_args.vocab_size,
                                      on_value=decoder_args.dtype.max,
                                      off_value=decoder_args.dtype.min)  # [batch_size * beam_width, vocab_size]
            
            # [batch_size * beam_width, vocab_size]
            logits = tf.where(finished, x=eos_max_prob, y=logits)
            logits = tf.cast(logits, tf.float32)
            
            output_id, next_cum_log_probs, finished, my_cache, \
                extra_vars, op_self_cache = search_word(beam_width,
                                                        decoding_args.vocab_size,
                                                        step,
                                                        logits,
                                                        cum_log_probs,
                                                        finished,
                                                        my_cache,
                                                        extra_vars,
                                                        op_self_cache,
                                                        search_method=search_method)
            cum_log_probs = tf.where(finished, x=cum_log_probs, y=next_cum_log_probs)
            
            outputs = outputs.write(step, output_id)
            finished = tf.logical_or(finished, tf.equal(output_id, decoding_args.end_id))

            return output_id, cum_log_probs, finished, step + 1, outputs, my_cache, extra_vars, op_self_cache, op_mem_cache

        # initialization
        batchxbeam = tf.shape(extended_memory)[0]
        start_ids, step, outputs, tf_decoder_cache, finished, initial_log_probs, \
            tf_sequence_lengths, extra_vars = initialize_decoding_variables(decoding_args, batchxbeam)

        word_ids = tf.identity(start_ids, name="word_ids")
        cum_log_probs = tf.identity(initial_log_probs, name="cum_log_probs")
        # if use_op == False, these two caches are useless
        op_self_cache, op_mem_cache = init_op_cache(decoder_args, batchxbeam, tf.reduce_max(memory_sequence_length))

        _, _, _, _, outputs, _, extra_vars, _, _ = tf.while_loop(
            _cond,
            _body,
            loop_vars=(
                word_ids,
                cum_log_probs,
                finished,
                step,
                outputs,
                tf_decoder_cache,
                extra_vars,
                op_self_cache,
                op_mem_cache
            ),
            back_prop=False,
            maximum_iterations=decoding_args.max_seq_len,
            shape_invariants=(
                start_ids.shape,
                initial_log_probs.shape,
                finished.shape,
                step.shape,
                tf.TensorShape(None),
                tf.contrib.framework.nest.map_structure(_get_shape_invariants, tf_decoder_cache),
                tf.contrib.framework.nest.map_structure(_get_shape_invariants, extra_vars),
                tf.contrib.framework.nest.map_structure(_get_shape_invariants, op_self_cache),
                tf.contrib.framework.nest.map_structure(_get_shape_invariants, op_mem_cache))
        )

        tf_parent_ids = extra_vars[0].stack()
        tf_sequence_lengths = extra_vars[1]
        tf_output_ids = outputs.stack()
        
        finalized_tf_output_ids, finalized_tf_sequence_lengths = finalize(beam_width,
                                                                          tf_parent_ids,
                                                                          tf_sequence_lengths,
                                                                          tf_output_ids,
                                                                          decoding_args.end_id)

        finalized_tf_output_ids = tf.cast(finalized_tf_output_ids, start_ids.dtype)
        finalized_tf_sequence_lengths = tf.minimum(
            finalized_tf_sequence_lengths + 1, tf.shape(finalized_tf_output_ids)[2])

        return finalized_tf_output_ids, finalized_tf_sequence_lengths, tf_output_ids, tf_parent_ids, tf_sequence_lengths

def tf_sampling_decoding(memory_tensor,
                        memory_sequence_length,
                        embedding_table,
                        decoding_args,
                        decoder_type):
    '''
    Run the decoding with sampling by TensorFlow.
    
    Args:
        memory_tensor: A tf.tensor with shape [batch_size, max(memory_sequence_length), encoder_hidden_dimension]. 
                       The results of encoder transformer layer. The rank must be 3. 
        memory_sequence_length: A tf.Tensor with shape [batch_size], type tf.int. 
                                The lenght of each sentence of results of encoder. 
        embedding_table: A tf.Tensor with shape [vocab_size, hidden_dimension]. 
                         The embedding table of embedding lookup for each step.
        decoder_args: The arguments for decoding. The details are in the class "DecodingSamplingArgument" of common.py
        decoder_type: A int value. Choose to using TensorFlow decoder, FasterTransformer decoder, or both.
                      If it is 0, then using the TensorFlow decoder only.
                      If it is 1, then using the FasterTransformer decoder only.
                      If it is 2, then using both decoder and compare their result. 
    Outputs:
        tf_output_ids: A tf.Tensor with shape [batch_size, max(sequence_lengths)], with int type. 
                    The results of decoding. It contains the id of token of vocabulary.
        sequence_lengths: A tf.Tensor with shape [batch_size], with int type.
    '''
    
    decoder_args = decoding_args.decoder_args
    
    with tf.variable_scope("transformer", reuse=tf.AUTO_REUSE):
        batch_size = tf.shape(memory_tensor)[0]

        def _cond(word_ids, finished, step, outputs, my_cache, sequence_lengths, op_self_cache, op_mem_cache):
            return tf.reduce_any(tf.logical_not(finished))

        def _body(word_ids, finished, step, outputs, my_cache, sequence_lengths, op_self_cache, op_mem_cache):
            logits, my_cache, op_self_cache, op_mem_cache = decoding_body(word_ids,
                                                                        step,
                                                                        memory_tensor,
                                                                        memory_sequence_length,
                                                                        my_cache,
                                                                        op_self_cache,
                                                                        op_mem_cache,
                                                                        embedding_table,
                                                                        decoding_args,
                                                                        decoder_type)

            end_ids = tf.fill([batch_size],decoding_args.end_id)  # [batch_size * beam_width]
            eos_max_prob = tf.one_hot(end_ids, decoding_args.vocab_size,
                                      on_value=decoder_args.dtype.max,
                                      off_value=decoder_args.dtype.min)  # [batch_size * beam_width, vocab_size]
            # [batch_size, vocab_size]
            logits = tf.where(finished, x=eos_max_prob, y=logits)
            logits = tf.cast(logits, tf.float32)

            # sampling
            if decoding_args.top_k != 0:
                sampling_method = Sampling("top_k")
                output_id = sampling_method.sample(logits, threshold=decoding_args.top_k)
            elif decoding_args.top_p != 0.0:
                sampling_method = Sampling("top_p")
                output_id = sampling_method.sample(logits, threshold=decoding_args.top_p)
            sequence_lengths = tf.where(finished, x=sequence_lengths, y=sequence_lengths + 1)
            
            outputs = outputs.write(step, output_id)
            finished = tf.logical_or(finished, tf.equal(output_id, decoding_args.end_id))

            # return output_id, cum_log_probs, finished, step + 1, outputs, my_cache, extra_vars, op_self_cache, op_mem_cache
            return output_id, finished, step + 1, outputs, my_cache, sequence_lengths, op_self_cache, op_mem_cache

        # initialization
        start_ids, step, outputs, tf_decoder_cache, finished, _, \
            _, extra_vars = initialize_decoding_variables(decoding_args, batch_size)

        sequence_lengths = extra_vars[1]
        word_ids = tf.identity(start_ids, name="word_ids")
        # if use_op == False, these two caches are useless
        op_self_cache, op_mem_cache = init_op_cache(decoder_args, batch_size, tf.reduce_max(memory_sequence_length))

        _, _, _, outputs, _, sequence_lengths, _, _ = tf.while_loop(
            _cond,
            _body,
            loop_vars=(
                word_ids,
                finished,
                step,
                outputs,
                tf_decoder_cache,
                sequence_lengths,
                op_self_cache,
                op_mem_cache
            ),
            back_prop=False,
            maximum_iterations=decoding_args.max_seq_len,
            shape_invariants=(
                start_ids.shape,
                finished.shape,
                step.shape,
                tf.TensorShape(None),
                tf.contrib.framework.nest.map_structure(
                    _get_shape_invariants, tf_decoder_cache),
                tf.contrib.framework.nest.map_structure(
                    _get_shape_invariants, sequence_lengths),
                tf.contrib.framework.nest.map_structure(
                    _get_shape_invariants, op_self_cache),
                tf.contrib.framework.nest.map_structure(_get_shape_invariants, op_mem_cache))
        )

        tf_output_ids = outputs.stack()
        tf_sequence_lengths = sequence_lengths
        tf_output_ids = tf.reshape(tf_output_ids, [-1, batch_size])
        tf_output_ids = tf.transpose(tf_output_ids, [1, 0])
        tf_output_ids = tf.cast(tf_output_ids, start_ids.dtype)

        return tf_output_ids, sequence_lengths

def preprocess_decoder_var(decoding_vars,
                            num_layer,
                            using_model_var,
                            checkpoint_filename,
                            data_type,
                            fuse_qkv=True):
    '''
    Args:
        decoding_vars: A list of tf.Tensor. The variables of decoding.  
        num_layer: A int value. The number of transformer layer of decoder in decoding
        using_model_var: A bool value. Using the model variables of TensorFlow or not.
                         If True, then putting the model variables of TensorFlow decoding model into decoding op directly. 
                            The data type is tensor of TensorFlow in this case. 
                        
                         If False, then restoring the values of variables from the checkpoint_filename, and putting
                         the values into decoding op.
                            The data type is numpy is this case. 
        checkpoint_file: A string. The checkpoint file name of storing the values of model. The checkpoint should be stored in 
                         pickle, and the name of checkpoint should be xxx.pkl.
                         The model is saved by dict. 
                         The key of the dict is the name of variables
                         The value of the dict is the values of variables
                         For example, decoding_vars[0]=<tf.Variable 'transformer/decoder/layer_0/masked_multi_head/LayerNorm/beta:0' shape=(512,) dtype=float32_ref>,
                         then the key is 'transformer/decoder/layer_0/masked_multi_head/LayerNorm/beta:0'; the value is sess.run(decoding_vars[0])
        data_type: tf.float32 or tf.float16. 
                   Only used when using_model_var is False. Convert the numpy data to the data type of model.
                   
    Outputs:
        vars_in_diff_layers_dict: A dict to store the variables by their name.
                                
                                For decoder variables, the key is like 'transformer/decoder/layer/masked_multi_head/LayerNorm/beta:0', 
                                which is similar to the name of variables, except we use 'layer' but not 'layer_x'. The value is a list, 
                                which contains 'transformer/decoder/layer_%d/masked_multi_head/LayerNorm/beta:0' % i for i in range(num_layer)
                                
                                For other variables, the key is the name of variable, and the value is the correspoding weight.
                                
                                Note that we return the concated weights. The concat operation would bring other overhead, and this should be optimized in 
                                the real application. The recommended method is pre-processing the weights as numpy format. Because TensorFlow do the operations
                                for each inference if using the TensorFlow to pre-process the weights.
    '''
    
    var_dict = {}
    if using_model_var == False:
        # restore the model from the checkpoint file
        if(checkpoint_filename == None):
            print("[ERROR] checkpoint_filename cannot be None when using_model_var is False.")
            exit(-1)
        
        with open(checkpoint_filename, 'rb') as f:
            ckpt = pickle.load(f)
            
            for var in decoding_vars:
                var_dict[var.name] = ckpt[var.name]
    else:
        for var in decoding_vars:
            var_dict[var.name] = var
    
    vars_in_diff_layers_dict = {}
    vars_in_diff_layers_dict["transformer/decoder/LayerNorm/beta:0"] = var_dict["transformer/decoder/LayerNorm/beta:0"]
    vars_in_diff_layers_dict["transformer/decoder/LayerNorm/gamma:0"] = var_dict["transformer/decoder/LayerNorm/gamma:0"]
    vars_in_diff_layers_dict["transformer/decoder/dense/kernel:0"] = var_dict["transformer/decoder/dense/kernel:0"]
    vars_in_diff_layers_dict["transformer/decoder/dense/bias:0"] = tf.cast(var_dict["transformer/decoder/dense/bias:0"], dtype=tf.float32)

    for i in range(num_layer):
        ''' 
            Handling the names of q, k, v kernel and bias because their names 
            are different for fusing the qkv or not.
        '''
        
        layer_prefix_name = "transformer/decoder/layer_%d/" % i
        if fuse_qkv == True:
            var_dict[layer_prefix_name + 'masked_multi_head/query/kernel:0'], \
            var_dict[layer_prefix_name + 'masked_multi_head/key/kernel:0'], \
            var_dict[layer_prefix_name + 'masked_multi_head/value/kernel:0'] = \
                tf.split(var_dict[layer_prefix_name + 'masked_multi_head/conv1d/kernel:0'], 3, axis=-1)
                
            var_dict[layer_prefix_name + 'masked_multi_head/query/bias:0'], \
            var_dict[layer_prefix_name + 'masked_multi_head/key/bias:0'], \
            var_dict[layer_prefix_name + 'masked_multi_head/value/bias:0'] = \
                tf.split(var_dict[layer_prefix_name + 'masked_multi_head/conv1d/bias:0'], 3, axis=-1)
            
            var_dict[layer_prefix_name + 'multi_head/query/kernel:0'] = \
                var_dict[layer_prefix_name + 'multi_head/conv1d/kernel:0']
            var_dict[layer_prefix_name + 'multi_head/query/bias:0'] = \
                var_dict[layer_prefix_name + 'multi_head/conv1d/bias:0']
            var_dict[layer_prefix_name + 'multi_head/key/kernel:0'], \
            var_dict[layer_prefix_name + 'multi_head/value/kernel:0'] = \
                tf.split(var_dict[layer_prefix_name + 'multi_head/conv1d_1/kernel:0'], 2, axis=-1)
            var_dict[layer_prefix_name + 'multi_head/key/bias:0'], \
            var_dict[layer_prefix_name + 'multi_head/value/bias:0'] = \
                tf.split(var_dict[layer_prefix_name + 'multi_head/conv1d_1/bias:0'], 2, axis=-1)
        else:
            var_dict[layer_prefix_name + 'masked_multi_head/query/kernel:0'] = \
                var_dict[layer_prefix_name + 'masked_multi_head/conv1d/kernel:0']
            var_dict[layer_prefix_name + 'masked_multi_head/key/kernel:0'] = \
                var_dict[layer_prefix_name + 'masked_multi_head/key/kernel:0']
            var_dict[layer_prefix_name + 'masked_multi_head/value/kernel:0'] = \
                var_dict[layer_prefix_name + 'masked_multi_head/value/kernel:0']
                
            var_dict[layer_prefix_name + 'masked_multi_head/query/bias:0'] = \
                var_dict[layer_prefix_name + 'masked_multi_head/conv1d/bias:0']
            var_dict[layer_prefix_name + 'masked_multi_head/key/bias:0'] = \
                var_dict[layer_prefix_name + 'masked_multi_head/key/bias:0']
            var_dict[layer_prefix_name + 'masked_multi_head/value/bias:0'] = \
                var_dict[layer_prefix_name + 'masked_multi_head/value/bias:0']
            
            var_dict[layer_prefix_name + 'multi_head/query/kernel:0'] = \
                var_dict[layer_prefix_name + 'multi_head/conv1d/kernel:0']
            var_dict[layer_prefix_name + 'multi_head/query/bias:0'] = \
                var_dict[layer_prefix_name + 'multi_head/conv1d/bias:0']
            var_dict[layer_prefix_name + 'multi_head/key/kernel:0'] = \
                var_dict[layer_prefix_name + 'multi_head/conv1d_1/kernel:0']
            var_dict[layer_prefix_name + 'multi_head/key/bias:0'] = \
                var_dict[layer_prefix_name + 'multi_head/conv1d_1/bias:0']
            var_dict[layer_prefix_name + 'multi_head/value/kernel:0'] = \
                var_dict[layer_prefix_name + 'multi_head/value/kernel:0']
            var_dict[layer_prefix_name + 'multi_head/value/bias:0'] = \
                var_dict[layer_prefix_name + 'multi_head/value/bias:0']

    layer_prefix_name = 'transformer/decoder/layer'
    vars_in_diff_layers_dict[layer_prefix_name + '/masked_multi_head/LayerNorm/beta:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/masked_multi_head/LayerNorm/beta:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/masked_multi_head/LayerNorm/gamma:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/masked_multi_head/LayerNorm/gamma:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/masked_multi_head/query/kernel:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/masked_multi_head/query/kernel:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/masked_multi_head/query/bias:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/masked_multi_head/query/bias:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/masked_multi_head/key/kernel:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/masked_multi_head/key/kernel:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/masked_multi_head/key/bias:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/masked_multi_head/key/bias:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/masked_multi_head/value/kernel:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/masked_multi_head/value/kernel:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/masked_multi_head/value/bias:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/masked_multi_head/value/bias:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/masked_multi_head/conv1d_1/kernel:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/masked_multi_head/conv1d_1/kernel:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/masked_multi_head/conv1d_1/bias:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/masked_multi_head/conv1d_1/bias:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
        
    vars_in_diff_layers_dict[layer_prefix_name + '/multi_head/LayerNorm/beta:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/multi_head/LayerNorm/beta:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/multi_head/LayerNorm/gamma:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/multi_head/LayerNorm/gamma:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/multi_head/query/kernel:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/multi_head/query/kernel:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/multi_head/query/bias:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/multi_head/query/bias:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/multi_head/key/kernel:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/multi_head/key/kernel:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/multi_head/key/bias:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/multi_head/key/bias:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/multi_head/value/kernel:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/multi_head/value/kernel:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/multi_head/value/bias:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/multi_head/value/bias:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/multi_head/conv1d_2/kernel:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/multi_head/conv1d_2/kernel:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/multi_head/conv1d_2/bias:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/multi_head/conv1d_2/bias:0' % i] for i in range(num_layer) ], axis=0), dtype=data_type)
    
    vars_in_diff_layers_dict[layer_prefix_name + '/ffn/LayerNorm/beta:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/ffn/LayerNorm/beta:0' % i] for i in range(num_layer)], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/ffn/LayerNorm/gamma:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/ffn/LayerNorm/gamma:0' % i] for i in range(num_layer)], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/ffn/conv1d/kernel:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/ffn/conv1d/kernel:0' % i] for i in range(num_layer)], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/ffn/conv1d/bias:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/ffn/conv1d/bias:0' % i] for i in range(num_layer)], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/ffn/conv1d_1/kernel:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/ffn/conv1d_1/kernel:0' % i] for i in range(num_layer)], axis=0), dtype=data_type)
    vars_in_diff_layers_dict[layer_prefix_name + '/ffn/conv1d_1/bias:0'] = \
        tf.cast(tf.concat([ var_dict[layer_prefix_name + '_%d/ffn/conv1d_1/bias:0' % i] for i in range(num_layer)], axis=0), dtype=data_type)
    
    return vars_in_diff_layers_dict

def op_beamsearch_decoding(memory_tensor,
                        memory_sequence_length,
                        embedding_table,
                        decoding_vars,
                        decoding_args,
                        using_model_var=True,
                        checkpoint_filename=None):
    '''
    Run the decoding with beam search by TensorFlow.
    
    Args:
        memory_tensor: A tf.tensor with shape [batch_size * beam_width, max(memory_sequence_length), encoder_hidden_dimension]. 
                       The results of encoder transformer layer. The rank must be 3. 
                       Note that it must be extended by beam_width times.
        memory_sequence_length: A tf.Tensor with shape [batch_size * beam_width], type tf.int. 
                                The lenght of each sentence of results of encoder. 
                                Note that it must be extended by beam_width times.
        embedding_table: A tf.Tensor with shape [vocab_size, hidden_dimension]. 
                         The embedding table of embedding lookup for each step.
        decoder_vars: A list of tf.Tensor. The variables for decoding. A list of model variables of TensorFlow model. 
        decoder_args: The arguments for decoding. The details are in the class "DecodingBeamsearchArgument" of common.py
        using_model_var: A bool value. Using the model variables of TensorFlow or not. 
                         The details are described in 'preprocess_decoder_var' function in the following.
        checkpoint_filename: A string. The checkpoint file name of storing the values of model.
                             The details are described in 'preprocess_decoder_var' function in the following.
    Outputs:
        finalized_output_ids: A tf.Tensor with shape [batch_size, beam_width, max(sequence_lengths)], with tf.int type. 
                                 Finalized output_ids by beam search algorithm and parent_ids.
        finalized_sequence_lengths: A tf.Tensor with shape [batch_size * beam_width], with int type.
                                       Finalized sequence_lengths by beam search algorithm and parent_ids.
        output_ids: A tf.Tensor with shape [batch_size, beam_width, max(sequence_lengths)], with tf.int type. 
                       The results of decoding. It contains the id of token of vocabulary.
        parent_ids: A tf.Tensor with shape [batch_size, beam_width, max(sequence_lengths)], with tf.int type.
                       The beam index of output ids for each step. 
        sequence_lengths: A tf.Tensor with shape [batch_size * beam_width], with int type.
    '''

    decoder_args = decoding_args.decoder_args
    decoding_op_module = tf.load_op_library(os.path.join('./lib/libtf_decoding_beamsearch.so'))
    
    vars_dict_in_differ_layers = preprocess_decoder_var(decoding_vars,
                                                        decoder_args.num_layer,
                                                        using_model_var,
                                                        checkpoint_filename,
                                                        decoder_args.dtype,
                                                        decoder_args.fuse_qkv)
    
    extended_memory = tf.contrib.seq2seq.tile_batch(
        memory_tensor, multiplier=decoder_args.beam_width)
    extended_memory_sequence_length = tf.contrib.seq2seq.tile_batch(
        memory_sequence_length, multiplier=decoder_args.beam_width)
    
    position_encoder = SinusoidalPositionEncoder()
    position_encoding_table = position_encoder._create_position_encoding_table(
        decoding_args.max_seq_len, decoder_args.head_num * decoder_args.size_per_head, decoder_args.dtype)
    # shape of position_encoding_table: [max_seq_len, hidden_dim]

    output_ids, parent_ids, sequence_lengths = decoding_op_module.decoding(
        extended_memory, # 0
        extended_memory_sequence_length, # 1
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/LayerNorm/beta:0'], # 2
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/LayerNorm/gamma:0'], # 3
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/query/kernel:0'], # 4
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/query/bias:0'], # 5
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/key/kernel:0'], # 6
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/key/bias:0'], # 7
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/value/kernel:0'], # 8
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/value/bias:0'], # 9
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/conv1d_1/kernel:0'], # 10
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/conv1d_1/bias:0'],  # 11
        vars_dict_in_differ_layers['transformer/decoder/layer/multi_head/LayerNorm/beta:0'], # 12
        vars_dict_in_differ_layers['transformer/decoder/layer/multi_head/LayerNorm/gamma:0'], # 13
        vars_dict_in_differ_layers['transformer/decoder/layer/multi_head/query/kernel:0'], # 14
        vars_dict_in_differ_layers['transformer/decoder/layer/multi_head/query/bias:0'], # 15
        vars_dict_in_differ_layers['transformer/decoder/layer/multi_head/key/kernel:0'], # 16
        vars_dict_in_differ_layers['transformer/decoder/layer/multi_head/key/bias:0'], # 17
        vars_dict_in_differ_layers['transformer/decoder/layer/multi_head/value/kernel:0'], # 18
        vars_dict_in_differ_layers['transformer/decoder/layer/multi_head/value/bias:0'], # 19
        vars_dict_in_differ_layers['transformer/decoder/layer/multi_head/conv1d_2/kernel:0'], # 20
        vars_dict_in_differ_layers['transformer/decoder/layer/multi_head/conv1d_2/bias:0'], # 21
        vars_dict_in_differ_layers['transformer/decoder/layer/ffn/LayerNorm/beta:0'], # 22
        vars_dict_in_differ_layers['transformer/decoder/layer/ffn/LayerNorm/gamma:0'], # 23
        vars_dict_in_differ_layers['transformer/decoder/layer/ffn/conv1d/kernel:0'], # 24
        vars_dict_in_differ_layers['transformer/decoder/layer/ffn/conv1d/bias:0'], # 25
        vars_dict_in_differ_layers['transformer/decoder/layer/ffn/conv1d_1/kernel:0'], # 26
        vars_dict_in_differ_layers['transformer/decoder/layer/ffn/conv1d_1/bias:0'], # 27
        vars_dict_in_differ_layers['transformer/decoder/LayerNorm/beta:0'], # 28
        vars_dict_in_differ_layers['transformer/decoder/LayerNorm/gamma:0'], # 29
        embedding_table, # 30
        vars_dict_in_differ_layers['transformer/decoder/dense/kernel:0'], # 31
        vars_dict_in_differ_layers['transformer/decoder/dense/bias:0'], # 32
        position_encoding_table, # 33 
        beam_width=decoder_args.beam_width,
        max_seq_len=decoding_args.max_seq_len,
        head_num=decoder_args.head_num, 
        size_per_head=decoder_args.size_per_head,
        num_layer=decoder_args.num_layer,
        start_id=decoding_args.start_id, 
        end_id=decoding_args.end_id,
        beam_search_diversity_rate=decoding_args.beam_search_diversity_rate
    )
    parent_ids = parent_ids % decoder_args.beam_width
    
    finalized_output_ids, finalized_sequence_lengths = finalize(decoder_args.beam_width,
                                                                parent_ids,
                                                                sequence_lengths,
                                                                output_ids,
                                                                decoding_args.end_id,
                                                                decoding_args.max_seq_len)

    finalized_sequence_lengths = tf.minimum(
        finalized_sequence_lengths + 1, tf.shape(finalized_output_ids)[2])
    
    return finalized_output_ids, finalized_sequence_lengths, output_ids, parent_ids, sequence_lengths

def op_sampling_decoding(memory_tensor,
                        memory_sequence_length,
                        embedding_table,
                        decoding_vars,
                        decoding_args,
                        using_model_var=True,
                        checkpoint_filename=None):
    '''
    Run the decoding with sampling by FasterTransformer.
    
    Args:
        memory_tensor: A tf.tensor with shape [batch_size, max(memory_sequence_length), encoder_hidden_dimension]. 
                       The results of encoder transformer layer. The rank must be 3. 
        memory_sequence_length: A tf.Tensor with shape [batch_size], type tf.int. 
                                The lenght of each sentence of results of encoder. 
        embedding_table: A tf.Tensor with shape [vocab_size, hidden_dimension]. 
                         The embedding table of embedding lookup for each step.
        decoder_vars: A list of tf.Tensor. The variables for decoding. A list of model variables of TensorFlow model. 
        decoder_args: The arguments for decoding. The details are in the class "DecodingSamplingArgument" of common.py
        using_model_var: A bool value. Using the model variables of TensorFlow or not. 
                         The details are described in 'preprocess_decoder_var' function in the following.
        checkpoint_filename: A string. The checkpoint file name of storing the values of model.
                             The details are described in 'preprocess_decoder_var' function in the following.
    Outputs:
        output_ids: A tf.Tensor with shape [batch_size, max(sequence_lengths)], with int type. 
                    The results of decoding. It contains the id of token of vocabulary.
        sequence_lengths: A tf.Tensor with shape [batch_size], with int type.
    '''
    
    decoder_args = decoding_args.decoder_args
    decoding_op_module = tf.load_op_library(os.path.join('./lib/libtf_decoding_sampling.so'))
    
    vars_dict_in_differ_layers = preprocess_decoder_var(decoding_vars,
                                                        decoding_args.decoder_args.num_layer,
                                                        using_model_var,
                                                        checkpoint_filename,
                                                        decoder_args.dtype,
                                                        decoder_args.fuse_qkv)
    
    position_encoder = SinusoidalPositionEncoder()
    position_encoding_table = position_encoder._create_position_encoding_table(
        decoding_args.max_seq_len, decoder_args.head_num * decoder_args.size_per_head, decoder_args.dtype)
    # shape of position_encoding_table: [max_seq_len, hidden_dim]
    
    output_ids, sequence_lengths = decoding_op_module.decoding_sampling(
        memory_tensor, # 0
        memory_sequence_length, # 1
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/LayerNorm/beta:0'], # 2
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/LayerNorm/gamma:0'], # 3
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/query/kernel:0'], # 4
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/query/bias:0'], # 5
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/key/kernel:0'], # 6
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/key/bias:0'], # 7
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/value/kernel:0'], # 8
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/value/bias:0'], # 9
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/conv1d_1/kernel:0'], # 10
        vars_dict_in_differ_layers['transformer/decoder/layer/masked_multi_head/conv1d_1/bias:0'],  # 11
        vars_dict_in_differ_layers['transformer/decoder/layer/multi_head/LayerNorm/beta:0'], # 12
        vars_dict_in_differ_layers['transformer/decoder/layer/multi_head/LayerNorm/gamma:0'], # 13
        vars_dict_in_differ_layers['transformer/decoder/layer/multi_head/query/kernel:0'], # 14
        vars_dict_in_differ_layers['transformer/decoder/layer/multi_head/query/bias:0'], # 15
        vars_dict_in_differ_layers['transformer/decoder/layer/multi_head/key/kernel:0'], # 16
        vars_dict_in_differ_layers['transformer/decoder/layer/multi_head/key/bias:0'], # 17
        vars_dict_in_differ_layers['transformer/decoder/layer/multi_head/value/kernel:0'], # 18
        vars_dict_in_differ_layers['transformer/decoder/layer/multi_head/value/bias:0'], # 19
        vars_dict_in_differ_layers['transformer/decoder/layer/multi_head/conv1d_2/kernel:0'], # 20
        vars_dict_in_differ_layers['transformer/decoder/layer/multi_head/conv1d_2/bias:0'], # 21
        vars_dict_in_differ_layers['transformer/decoder/layer/ffn/LayerNorm/beta:0'], # 22
        vars_dict_in_differ_layers['transformer/decoder/layer/ffn/LayerNorm/gamma:0'], # 23
        vars_dict_in_differ_layers['transformer/decoder/layer/ffn/conv1d/kernel:0'], # 24
        vars_dict_in_differ_layers['transformer/decoder/layer/ffn/conv1d/bias:0'], # 25
        vars_dict_in_differ_layers['transformer/decoder/layer/ffn/conv1d_1/kernel:0'], # 26
        vars_dict_in_differ_layers['transformer/decoder/layer/ffn/conv1d_1/bias:0'], # 27
        vars_dict_in_differ_layers['transformer/decoder/LayerNorm/beta:0'], # 28
        vars_dict_in_differ_layers['transformer/decoder/LayerNorm/gamma:0'], # 29
        embedding_table, # 30
        vars_dict_in_differ_layers['transformer/decoder/dense/kernel:0'], # 31
        vars_dict_in_differ_layers['transformer/decoder/dense/bias:0'], # 32
        position_encoding_table, # 33 
        max_seq_len=decoding_args.max_seq_len,
        candidate_num=decoding_args.top_k,
        probability_threshold=decoding_args.top_p,
        head_num=decoder_args.head_num, 
        size_per_head=decoder_args.size_per_head,
        num_layer=decoder_args.num_layer,
        start_id=decoding_args.start_id, 
        end_id=decoding_args.end_id
    )
    batch_size = tf.shape(memory_tensor)[0]
    output_ids = tf.reshape(output_ids, [-1, batch_size])
    output_ids = tf.transpose(output_ids, [1, 0])
    
    return output_ids, sequence_lengths
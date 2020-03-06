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
from decoder import tf_decoder, op_decoder, init_op_cache, init_tf_cache
from common import create_initializer, _get_shape_invariants
from utils.position import SinusoidalPositionEncoder

def initialize_decoding_variables(decoding_args):

    start_ids = tf.fill([decoding_args.decoder_args.batch_size * decoding_args.decoder_args.beam_width],
                        decoding_args.start_id)  # [batch_size * beam_width]

    step = tf.constant(0, dtype=tf.int32)
    # save the output ids for each step
    outputs = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

    cache = init_tf_cache(decoding_args.decoder_args.batch_size * decoding_args.decoder_args.beam_width,
                          decoding_args.decoder_args.head_num, decoding_args.decoder_args.size_per_head,
                          decoding_args.decoder_args.num_layer, dtype=decoding_args.decoder_args.dtype, num_sources=1)

    finished = tf.zeros([decoding_args.decoder_args.batch_size * decoding_args.decoder_args.beam_width],
                        dtype=tf.bool)  # [batch_size * beam_width], record that a sentence is finished or not
    initial_log_probs = tf.cast(tf.tile([0.] + [-float("inf")] * (decoding_args.decoder_args.beam_width - 1),
                                        [decoding_args.decoder_args.batch_size]), dtype=tf.float32)  # [batch_size * beam_width]
    # [batch_size * beam_width], record the lengths of all sentences
    sequence_lengths = tf.zeros(
        [decoding_args.decoder_args.batch_size * decoding_args.decoder_args.beam_width], dtype=tf.int32)
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
    outter_embbeding = np.random.randn(memory_hidden_dim) * 0.01

    memory = []
    for i in range(batch_size):
        data = np.random.randn(max_seq_len, memory_hidden_dim) * 0.01
        for j in range(memory_sequence_length[i], max_seq_len):
            data[j] = outter_embbeding
        memory.append(data)
    memory = np.asarray(memory)
    memory = tf.convert_to_tensor(memory, dtype=dtype)

    return memory, memory_sequence_length


def beam_search(beam_width,
                vocab_size,
                step,
                log_probs,
                cum_log_probs,
                finished,
                cache,
                extra_vars,
                op_self_cache=None):

    parent_ids = extra_vars[0]
    sequence_lengths = extra_vars[1]

    # [batch_size * beam_width, vocab_size] + [batch_size * beam_width], has to broadcast
    total_probs = log_probs + tf.expand_dims(cum_log_probs, 1)
    # [batch_size, beam_width * vocab_size], can skip in cuda
    total_probs = tf.reshape(total_probs, [-1, beam_width * vocab_size])

    # both shapes are: [batch_size, beam_width]
    _, sample_ids = tf.nn.top_k(total_probs, beam_width)
    # [batch_size * beam_width], can skip in cuda
    sample_ids = tf.reshape(sample_ids, [-1])
    word_ids = sample_ids % vocab_size  # [batch_size * beam_width]
    beam_ids = sample_ids // vocab_size  # [batch_size * beam_width]
    # [batch_size * beam_width]
    beam_indices = (
        tf.range(sample_ids.shape[0]) // beam_width) * beam_width + beam_ids

    sequence_lengths = tf.where(
        finished, x=sequence_lengths, y=sequence_lengths + 1)

    # [batch_size * beam_width]
    batch_pos = tf.range(sample_ids.shape[0]) // beam_width
    cum_log_probs = tf.gather_nd(total_probs, tf.stack(
        [batch_pos, sample_ids], axis=-1))  # [batch_size * beam_width]
    finished = tf.gather(finished, beam_indices)
    sequence_lengths = tf.gather(sequence_lengths, beam_indices)

    cache = tf.contrib.framework.nest.map_structure(
        lambda s: tf.gather(s, beam_indices), cache)
    if op_self_cache != None:
        op_self_cache = tf.contrib.framework.nest.map_structure(
            lambda s: tf.gather(s, beam_indices, axis=3), op_self_cache)

    parent_ids = parent_ids.write(step, beam_ids)
    extra_vars = [parent_ids, sequence_lengths]

    return word_ids, cum_log_probs, finished, cache, tuple(extra_vars), op_self_cache


def finalize(beam_width, parent_ids, sequence_lengths, outputs, end_id, max_seq_len=None):
    maximum_lengths = tf.reduce_max(tf.reshape(
        sequence_lengths, [-1, beam_width]), axis=-1)
    if max_seq_len != None:
        array_shape = [max_seq_len, -1, beam_width]
    else:
        array_shape = [maximum_lengths[0], -1, beam_width]

    step_ids = tf.reshape(outputs, array_shape)
    parent_ids = tf.reshape(parent_ids, array_shape)

    ids = tf.contrib.seq2seq.gather_tree(
        step_ids, parent_ids, maximum_lengths, end_id)

    ids = tf.transpose(ids, perm=[1, 2, 0])
    lengths = tf.not_equal(ids, end_id)
    lengths = tf.cast(lengths, tf.int32)
    lengths = tf.reduce_sum(lengths, axis=-1)
    return ids, lengths


def op_decoding(memory_tensor,
                memory_sequence_length,
                embedding_table,
                decoding_vars,
                decoding_args):

    decoding_op_module = tf.load_op_library(
        os.path.join('./lib/libtf_decoding.so'))

    val_off = 26
    decoding_vars_in_differ_layers = []
    for i in range(val_off):
        par = []
        for j in range(decoding_args.decoder_args.num_layer):
            par.append(decoding_vars[i + j * val_off])
        decoding_vars_in_differ_layers.append(par)

    extended_memory = tf.contrib.seq2seq.tile_batch(
        memory_tensor, multiplier=decoding_args.decoder_args.beam_width)
    extended_memory_sequence_length = tf.contrib.seq2seq.tile_batch(
        memory_sequence_length, multiplier=decoding_args.decoder_args.beam_width)

    output_ids, parent_ids, sequence_lengths = decoding_op_module.decoding(
        extended_memory, extended_memory_sequence_length,
        decoding_vars_in_differ_layers[0], decoding_vars_in_differ_layers[1], 
        decoding_vars_in_differ_layers[2], decoding_vars_in_differ_layers[3], 
        decoding_vars_in_differ_layers[4], decoding_vars_in_differ_layers[5], 
        decoding_vars_in_differ_layers[6], decoding_vars_in_differ_layers[7],
        decoding_vars_in_differ_layers[8], decoding_vars_in_differ_layers[9], 
        decoding_vars_in_differ_layers[10], decoding_vars_in_differ_layers[11], 
        decoding_vars_in_differ_layers[12], decoding_vars_in_differ_layers[13], 
        decoding_vars_in_differ_layers[14], decoding_vars_in_differ_layers[15], 
        decoding_vars_in_differ_layers[16], decoding_vars_in_differ_layers[17], 
        decoding_vars_in_differ_layers[18], decoding_vars_in_differ_layers[19], 
        decoding_vars_in_differ_layers[20], decoding_vars_in_differ_layers[21], 
        decoding_vars_in_differ_layers[22], decoding_vars_in_differ_layers[23], 
        decoding_vars_in_differ_layers[24], decoding_vars_in_differ_layers[25],
        decoding_vars[-4], decoding_vars[-3], embedding_table, 
        decoding_vars[-2], tf.cast(decoding_vars[-1], dtype=tf.float32),
        batch_size=decoding_args.decoder_args.batch_size, 
        beam_width=decoding_args.decoder_args.beam_width,
        max_seq_len=decoding_args.decoder_args.max_seq_len,
        head_num=decoding_args.decoder_args.head_num, 
        size_per_head=decoding_args.decoder_args.size_per_head,
        num_layer=decoding_args.decoder_args.num_layer,
        memory_hidden_dim=decoding_args.encoder_hidden_dim, 
        vocab_size=decoding_args.vocab_size,
        start_id=decoding_args.start_id, end_id=decoding_args.end_id
    )
    parent_ids = parent_ids % decoding_args.decoder_args.beam_width

    finalized_output_ids, finalized_sequence_lengths = finalize(decoding_args.decoder_args.beam_width,
                                                                parent_ids,
                                                                sequence_lengths,
                                                                output_ids,
                                                                decoding_args.end_id,
                                                                decoding_args.decoder_args.max_seq_len)

    finalized_sequence_lengths = tf.minimum(
        finalized_sequence_lengths + 1, tf.shape(finalized_output_ids)[2])
    
    return finalized_output_ids, finalized_sequence_lengths, output_ids, parent_ids, sequence_lengths


def tf_decoding(memory_tensor,
                memory_sequence_length,
                embedding_table,
                decoding_args,
                decoder_type,
                kernel_initializer_range,
                bias_initializer_range,
                atol_threshold=1e-6):

    with tf.variable_scope("transformer/decoding", reuse=tf.AUTO_REUSE):
        # copy memory and memory_sequence_length by beam_width times
        # if memory is [a, b, c], beam_width = 3, then the result is: [a a a b b b c c c ]
        extended_memory = tf.contrib.seq2seq.tile_batch(
            memory_tensor, multiplier=decoding_args.decoder_args.beam_width)
        extended_memory_sequence_length = tf.contrib.seq2seq.tile_batch(
            memory_sequence_length, multiplier=decoding_args.decoder_args.beam_width)

        def _cond(word_ids, cum_log_probs, finished, step, outputs, my_cache, extra_vars, op_self_cache, op_mem_cache):
            return tf.reduce_any(tf.logical_not(finished))

        def _body(word_ids, cum_log_probs, finished, step, outputs, my_cache, extra_vars, op_self_cache, op_mem_cache):
            # [batch_size * beam_width, hidden_dim]
            inputs = tf.nn.embedding_lookup(embedding_table, word_ids)
            # [batch_size * beam_width, 1, hidden_dim]
            inputs = tf.expand_dims(inputs, 1)
            
            inputs *= decoding_args.decoder_args.hidden_dim**0.5
            position_encoder = SinusoidalPositionEncoder()
            if position_encoder is not None:
                inputs = position_encoder(
                    inputs, position=step + 1 if step is not None else None)
                
            with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
                tf_result = tf_decoder(decoder_args=decoding_args.decoder_args,
                                    inputs=inputs,
                                    memory=extended_memory,
                                    memory_sequence_length=extended_memory_sequence_length,
                                    step=step,
                                    cache=my_cache,
                                    kernel_initializer_range=kernel_initializer_range,
                                    bias_initializer_range=bias_initializer_range)


            if decoder_type != 0:
                decoder_vars = tf.global_variables()
                decoder_vars_start_id = 0
                while decoder_vars_start_id < len(decoder_vars):
                    if decoder_vars[decoder_vars_start_id].name.find("transformer/decoding/decoder") != -1:
                        break
                    decoder_vars_start_id += 1
                decoder_vars = decoder_vars[decoder_vars_start_id:]

                psuedo_input = []
                if decoder_type == 2:
                    psuedo_input = tf_result
                    
                op_result, op_self_cache, op_mem_cache = op_decoder(inputs,
                                                                    step,
                                                                    extended_memory,
                                                                    extended_memory_sequence_length,
                                                                    op_self_cache,
                                                                    op_mem_cache,
                                                                    psuedo_input,
                                                                    decoder_vars,
                                                                    decoding_args.decoder_args,
                                                                    decoding_args.encoder_hidden_dim)

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
                argmax = tf.math.argmax(abs_diff)
                result = tf.Print(result, ["[INFO][PYTHON] step:", step, "max diff: ", abs_diff[argmax],
                                           " op val: ", flatten_result_2[argmax],
                                           " tf val: ", flatten_result[argmax], 
                                           tf.cond(abs_diff[argmax] < atol_threshold, lambda: "True", lambda: "False")])
            else:
                print("[TF][ERROR] decoder type is only 0 or 1 or 2.")
                exit(-1)

            result = tf.contrib.layers.layer_norm(result, begin_norm_axis=-1)
            # [batch_size * beam_width, hidden_dim]
            result = tf.squeeze(result, axis=1)
            logits = tf.layers.dense(result,
                                     decoding_args.vocab_size,
                                     use_bias=True,
                                     bias_initializer=create_initializer(
                                         bias_initializer_range, decoding_args.decoder_args.dtype),
                                     kernel_initializer=create_initializer(
                                         kernel_initializer_range, decoding_args.decoder_args.dtype),
                                     activation=None)

            end_ids = tf.fill([decoding_args.decoder_args.batch_size * decoding_args.decoder_args.beam_width],
                              decoding_args.end_id)  # [batch_size * beam_width]
            eos_max_prob = tf.one_hot(end_ids, decoding_args.vocab_size,
                                      on_value=decoding_args.decoder_args.dtype.max,
                                      off_value=decoding_args.decoder_args.dtype.min)  # [batch_size * beam_width, vocab_size]
            # [batch_size * beam_width, vocab_size]
            logits = tf.where(finished, x=eos_max_prob, y=logits)
            logits = tf.cast(logits, tf.float32)
            # [batch_size * beam_width, vocab_size]
            log_probs = tf.nn.log_softmax(logits)

            output_id, next_cum_log_probs, finished, my_cache, \
                extra_vars, op_self_cache = beam_search(decoding_args.decoder_args.beam_width,
                                                        decoding_args.vocab_size,
                                                        step,
                                                        log_probs,
                                                        cum_log_probs,
                                                        finished,
                                                        my_cache,
                                                        extra_vars,
                                                        op_self_cache)

            outputs = outputs.write(step, output_id)
            cum_log_probs = tf.where(
                finished, x=cum_log_probs, y=next_cum_log_probs)
            finished = tf.logical_or(finished, tf.equal(
                output_id, decoding_args.end_id))

            return output_id, cum_log_probs, finished, step + 1, outputs, my_cache, extra_vars, op_self_cache, op_mem_cache

        # initialization
        start_ids, step, outputs, tf_decoder_cache, finished, initial_log_probs, \
            tf_sequence_lengths, extra_vars = initialize_decoding_variables(
                decoding_args)

        word_ids = tf.identity(start_ids, name="word_ids")
        cum_log_probs = tf.identity(initial_log_probs, name="cum_log_probs")
        # if use_op == False, these two caches are useless
        op_self_cache, op_mem_cache = init_op_cache(decoding_args.decoder_args)

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
            maximum_iterations=decoding_args.decoder_args.max_seq_len,
            shape_invariants=(
                start_ids.shape,
                initial_log_probs.shape,
                finished.shape,
                step.shape,
                tf.TensorShape(None),
                tf.contrib.framework.nest.map_structure(
                    _get_shape_invariants, tf_decoder_cache),
                tf.contrib.framework.nest.map_structure(
                    _get_shape_invariants, extra_vars),
                tf.contrib.framework.nest.map_structure(
                    _get_shape_invariants, op_self_cache),
                tf.contrib.framework.nest.map_structure(_get_shape_invariants, op_mem_cache))
        )

        tf_parent_ids = extra_vars[0].stack()
        tf_sequence_lengths = extra_vars[1]
        tf_output_ids = outputs.stack()

        finalized_tf_output_ids, finalized_tf_sequence_lengths = finalize(decoding_args.decoder_args.beam_width,
                                                                          tf_parent_ids,
                                                                          tf_sequence_lengths,
                                                                          tf_output_ids,
                                                                          decoding_args.end_id)

        finalized_tf_output_ids = tf.cast(
            finalized_tf_output_ids, start_ids.dtype)
        finalized_tf_sequence_lengths = tf.minimum(
            finalized_tf_sequence_lengths + 1, tf.shape(finalized_tf_output_ids)[2])

        return finalized_tf_output_ids, finalized_tf_sequence_lengths, tf_output_ids, tf_parent_ids, tf_sequence_lengths

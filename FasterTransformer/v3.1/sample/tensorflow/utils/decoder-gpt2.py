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

import os
import tensorflow as tf
from utils.common import create_initializer
import numpy as np

def norm(inputs):
    """Layer normalizes :obj:`inputs`."""
    return tf.contrib.layers.layer_norm(inputs, begin_norm_axis=-1)


def split_heads(inputs, num_heads):
    """Splits a tensor in depth.

    Args:
      inputs: A ``tf.Tensor`` of shape :math:`[B, T, D]`.
      num_heads: The number of heads :math:`H`.

    Returns:
      A ``tf.Tensor`` of shape :math:`[B, H, T, D / H]`.
    """
    static_shape = inputs.get_shape().as_list()
    depth = static_shape[-1]
    outputs = tf.reshape(
        inputs, [tf.shape(inputs)[0], tf.shape(inputs)[1], num_heads, depth // num_heads])
    outputs = tf.transpose(outputs, perm=[0, 2, 1, 3])
    return outputs


def build_sequence_mask(sequence_length,
                        num_heads=None,
                        maximum_length=None,
                        data_type=tf.float32):
    """Builds the dot product mask.

    Args:
      sequence_length: The sequence length.
      num_heads: The number of heads.
      maximum_length: Optional size of the returned time dimension. Otherwise
        it is the maximum of :obj:`sequence_length`.
      dtype: The type of the mask tensor.

    Returns:
      A broadcastable ``tf.Tensor`` of type :obj:`dtype` and shape
      ``[batch_size, 1, 1, max_length]``.
    """
    mask = tf.sequence_mask(
        sequence_length, maxlen=maximum_length, dtype=data_type)
    mask = tf.expand_dims(mask, axis=1)
    if num_heads is not None:
        mask = tf.expand_dims(mask, axis=1)
    return mask

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def tf_decoder(decoder_args,
               inputs,
               memory,
               memory_sequence_length,
               step,
               cache=None):
    '''
    Run the decoder transformer layer by TensorFlow.
                      
    Args:
        decoder_args: The arguments for decoder. The details are in the class "TransformerArgument" of common.py
        inputs: A tf.Tensor with shape [batch_size * beam_width, 1, hidden_dimension].
                The inputs tensor of encoder. The rank must be 3.
        memory: A tf.tensor with shape [batch_size * beam_width, max(memory_sequence_length), encoder_hidden_dimension]. 
                The results of encoder transformer layer. The rank must be 3. 
                Note that it must be extended by beam_width times
        memory_sequence_length: A tf.Tensor with shape [batch_size * beam_width], type tf.int. 
                                The lenght of each sentence of results of encoder. 
                                Note that it must be extended by beam_width times
        step: A tf.Tensor with tf.int type. The current step in the translation process.
        cache: A dict. The cache space to store the keys and values of attention layers.

    Outputs:
        outputs: A tf.Tensor with shape [batch_size * beam_width, 1, hidden_dimension].
                 The results of decoder.
    '''
    
    k_init_range = decoder_args.kernel_init_range
    b_init_range = decoder_args.bias_init_range
    data_type = decoder_args.dtype
    
    memory_mask = None  # has something

    if memory is not None and not tf.contrib.framework.nest.is_sequence(memory):
        memory = (memory,)
        if memory_sequence_length is not None:
            if not tf.contrib.framework.nest.is_sequence(memory_sequence_length):
                memory_sequence_length = (memory_sequence_length,)
            memory_mask = [
                build_sequence_mask(
                    length, num_heads=decoder_args.head_num, maximum_length=tf.shape(m)[1], data_type=data_type)
                for m, length in zip(memory, memory_sequence_length)]

    for l in range(decoder_args.num_layer):
        layer_name = "layer_{}".format(l)
        layer_cache = cache[layer_name] if cache is not None else None
        
        with tf.variable_scope(layer_name):
            with tf.variable_scope("masked_multi_head"):
                norm_inputs = norm(inputs)
                queries, keys, values = tf.split( tf.layers.conv1d(norm_inputs, decoder_args.hidden_dim * 3, 1, 
                                                                bias_initializer=create_initializer(b_init_range, data_type),
                                                                kernel_initializer=create_initializer(k_init_range, data_type)), 3, axis=2)

                keys = tf.reshape(keys, [tf.shape(keys)[0], 1, decoder_args.head_num, decoder_args.size_per_head])
                keys = tf.transpose(keys, [0, 2, 1, 3])
                values = tf.reshape(values, [tf.shape(values)[0], 1, decoder_args.head_num, decoder_args.size_per_head])
                values = tf.transpose(values, [0, 2, 1, 3])
                keys = tf.concat([layer_cache["self_keys"], keys], axis=2)
                values = tf.concat([layer_cache["self_values"], values], axis=2)
                layer_cache["self_keys"] = keys
                layer_cache["self_values"] = values

                queries = tf.reshape(queries, [tf.shape(queries)[0], 1, decoder_args.head_num, decoder_args.size_per_head])
                queries = tf.transpose(queries, [0, 2, 1, 3])
                queries *= (decoder_args.size_per_head)**-0.5

                dot = tf.matmul(queries, keys, transpose_b=True)

                attn = tf.cast(tf.nn.softmax(tf.cast(dot, data_type)), dot.dtype)
                context = tf.matmul(attn, values)
                context = tf.transpose(context, [0, 2, 1, 3])
                context = tf.reshape(context, [tf.shape(context)[0], 1, decoder_args.head_num * decoder_args.size_per_head])

                outputs = tf.layers.conv1d(context,
                                            decoder_args.hidden_dim,
                                            1,
                                            bias_initializer=create_initializer(b_init_range, data_type),
                                            kernel_initializer=create_initializer(k_init_range, data_type))

                # drop_and_add
                input_dim = inputs.get_shape().as_list()[-1]
                output_dim = outputs.get_shape().as_list()[-1]
                if input_dim == output_dim:
                    outputs += inputs
                last_context = outputs

            # For GPT-2, we do not need cross attention
            # if memory is not None:
            #     for i, (mem, mask) in enumerate(zip(memory, memory_mask)):
            #         memory_cache = layer_cache["memory"][i] if layer_cache is not None else None

            #         with tf.variable_scope("multi_head" if i == 0 else "multi_head_%d" % i):
            #             queries = tf.layers.conv1d(
            #                 norm(last_context),
            #                 decoder_args.hidden_dim,
            #                 1,
            #                 bias_initializer=create_initializer(b_init_range, data_type),
            #                 kernel_initializer=create_initializer(k_init_range, data_type))

            #             def _project_and_split():
            #                 keys, values = tf.split( tf.layers.conv1d(mem, decoder_args.hidden_dim * 2, 1, 
            #                                                     bias_initializer=create_initializer(b_init_range, data_type),
            #                                                     kernel_initializer=create_initializer(k_init_range, data_type)), 2, axis=2)

            #                 keys = tf.reshape(keys, [tf.shape(keys)[0], tf.shape(keys)[1],
            #                                             decoder_args.head_num, decoder_args.size_per_head])
            #                 keys = tf.transpose(keys, [0, 2, 1, 3])
            #                 values = tf.reshape(values, [tf.shape(values)[0], tf.shape(values)[1],
            #                                             decoder_args.head_num, decoder_args.size_per_head])
            #                 values = tf.transpose(values, [0, 2, 1, 3])

            #                 return keys, values

            #             keys, values = tf.cond(
            #                 tf.equal(
            #                     tf.shape(memory_cache["memory_keys"])[2], 0),
            #                 true_fn=_project_and_split,
            #                 false_fn=lambda: (memory_cache["memory_keys"], memory_cache["memory_values"]))

            #             memory_cache["memory_keys"] = keys
            #             memory_cache["memory_values"] = values

            #             queries = tf.reshape(queries, [tf.shape(queries)[0], 1,decoder_args.head_num, decoder_args.size_per_head])
            #             queries = tf.transpose(queries, [0, 2, 1, 3])
            #             queries *= (decoder_args.size_per_head)**-0.5
            #             
            #             dot = tf.matmul(queries, keys, transpose_b=True)
            #             dot = tf.cast(tf.cast(dot, data_type) * mask +
            #                           ((1.0 - mask) * data_type.min), dot.dtype)

            #             attn = tf.cast(tf.nn.softmax(
            #                 tf.cast(dot, data_type)), dot.dtype)
            #             context = tf.matmul(attn, values)
            #             context = tf.transpose(context, [0, 2, 1, 3])
            #             context = tf.reshape(context, [tf.shape(context)[0], 1,
            #                                            decoder_args.head_num * decoder_args.size_per_head])
            #             context = tf.layers.conv1d(context,
            #                                         decoder_args.hidden_dim,
            #                                         1,
            #                                         bias_initializer=create_initializer(b_init_range, data_type),
            #                                         kernel_initializer=create_initializer(k_init_range, data_type))

            #             # drop_and_add
            #             input_dim = last_context.get_shape().as_list()[-1]
            #             output_dim = context.get_shape().as_list()[-1]
            #             if input_dim == output_dim:
            #                 context += last_context

            with tf.variable_scope("ffn"):
                # forward
                # For GPT-2, take the inputs from the self attention
                # normed_last_context = norm(context)
                normed_last_context = norm(last_context)
                input_dim = normed_last_context.get_shape().as_list()[-1]
                # GPT-2 uses GELU
                # inner = tf.layers.conv1d(normed_last_context,
                #                         decoder_args.hidden_dim * 4,
                #                         1,
                #                         activation=tf.nn.relu,
                #                         use_bias=True,
                #                         bias_initializer=create_initializer(b_init_range, data_type),
                #                         kernel_initializer=create_initializer(k_init_range, data_type))
                inner = gelu(tf.layers.conv1d(normed_last_context,
                                        decoder_args.hidden_dim * 4,
                                        1,
                                        use_bias=True,
                                        bias_initializer=create_initializer(b_init_range, data_type),
                                        kernel_initializer=create_initializer(k_init_range, data_type)))
                transformed = tf.layers.conv1d(inner,
                                                input_dim,
                                                1,
                                                use_bias=True,
                                                bias_initializer=create_initializer(b_init_range, data_type),
                                                kernel_initializer=create_initializer(k_init_range, data_type))

                # drop_and_add
                input_dim = context.get_shape().as_list()[-1]
                output_dim = transformed.get_shape().as_list()[-1]
                if input_dim == output_dim:
                    # For GPT-2, residual connection comes from self attention
                    # transformed += context
                    transformed += last_context
        inputs = transformed
    outputs = inputs
    return outputs


def init_tf_cache(batch_size,
                  head_num,
                  size_per_head,
                  num_layer,
                  dtype,
                  num_sources=1):
    cache = {}
    for l in range(num_layer):
        proj_cache_shape = [batch_size, head_num, 0, size_per_head]
        layer_cache = {}
        layer_cache["memory"] = [
            {
                "memory_keys": tf.zeros(proj_cache_shape, dtype=dtype, name="memory_keys"),
                "memory_values": tf.zeros(proj_cache_shape, dtype=dtype, name="memory_values")
            } for _ in range(num_sources)]
        layer_cache["self_keys"] = tf.zeros(
            proj_cache_shape, dtype=dtype, name="self_keys")
        layer_cache["self_values"] = tf.zeros(
            proj_cache_shape, dtype=dtype, name="self_values")
        cache["layer_{}".format(l)] = layer_cache
    return cache


def init_op_cache(decoder_args, batchxbeam, memory_max_seq_len):
    self_cache = tf.zeros([decoder_args.num_layer, 2, 0, batchxbeam,
                           decoder_args.hidden_dim], dtype=decoder_args.dtype, name="op_self_caches")
    mem_cache = tf.zeros([decoder_args.num_layer, 2, batchxbeam,
                          memory_max_seq_len, decoder_args.hidden_dim], dtype=decoder_args.dtype, name="op_memory_caches")

    return self_cache, mem_cache


def op_decoder(inputs,
               memory_tensor,
               memory_sequence_length,
               op_self_cache,
               op_mem_cache,
               psuedo_input,
               decoder_vars,
               decoder_args):
    '''
    Run the decoder transformer layer by FasterTransformer.
    
    Args:
        inputs: A tf.Tensor with shape [batch_size * beam_width, 1, hidden_dimension].
                The inputs tensor of encoder. The rank must be 3.
        memory_tensor: A tf.tensor with shape [batch_size * beam_width, max(memory_sequence_length), encoder_hidden_dimension]. 
                       The results of encoder transformer layer. The rank must be 3. 
                       Note that it must be extended by beam_width times
        memory_sequence_length: A tf.Tensor with shape [batch_size * beam_width], type tf.int. 
                                The lenght of each sentence of results of encoder. 
                                Note that it must be extended by beam_width times
        op_self_cache: A tf.Tensor with shape [num_layer, 2, None, batch_size * beam_width, hidden_dimension]. 
                       The cache space to store the keys and values of first attention layer in each step.
        op_mem_cache: A tf.Tensor with shape [num_layer, 2, batch_size * beam_width, max(memory_sequence_length) hidden_dimension]. 
                      The cache space to store the keys and values of second attention layer.
                      Since they are same in each step, it is only need to compute them in first time. 
        psuedo_input: A tf.Tensor or null list. 
                      Put the decoder results of TensorFlow when running the TensorFlow decoder and FasterTransformer
                      decoder in one model. This prevents the race condition. 
                      It is useless when only run the FasterTransformer decoder.
        decoder_args: The arguments for decoder. The details are in the class "TransformerArgument" of common.py
        decoder_vars: A list of tf.Tensor. The variables for decoder. They can be either some tensor or some numpy array. 
                      The number of variables should be 20 * num_layer.

    Outputs:
        outputs: A tf.Tensor with shape [batch_size * beam_width, 1, hidden_dimension].
                 The results of decoder.
    '''

    decoder_op_module = tf.load_op_library(
        os.path.join('./lib/libtf_decoder.so'))
    
    op_self_cache = tf.concat([op_self_cache, tf.zeros([decoder_args.num_layer, 2, 1,
                                                        tf.shape(memory_tensor)[0],
                                                        decoder_args.hidden_dim], dtype=decoder_args.dtype)], axis=2)

    for i in range(decoder_args.num_layer):
        print("Layer {}".format(i))
        q_w, k_w, v_w = tf.split(decoder_vars[2 + 12 * i], 3, axis=-1)
        q_b, k_b, v_b = tf.split(decoder_vars[3 + 12 * i], 3)
        # For GPT-2, for now re-use the self attention weights for unused x-attention
        # k_w_2, v_w_2 = tf.split(decoder_vars[10 + 20 * i], 2, axis=-1)
        # k_b_2, v_b_2 = tf.split(decoder_vars[11 + 20 * i], 2)
        k_w_2, v_w_2 = tf.split(decoder_vars[2 + 12 * i], 2, axis=-1)
        k_b_2, v_b_2 = tf.split(decoder_vars[3 + 12 * i], 2)
        
        op_result, _, _ = decoder_op_module.decoder(
            inputs, memory_tensor, memory_sequence_length,
            decoder_vars[0 + 12 * i], decoder_vars[1 + 12 * i],
            tf.concat([q_w, k_w, v_w], axis=0),
            q_b, k_b, v_b,
            decoder_vars[4 + 12 * i], 
            decoder_vars[5 + 12 * i], 
            decoder_vars[6 + 12 * i], decoder_vars[7 + 12 * i],
            decoder_vars[0 + 12 * i], decoder_vars[1 + 12 * i],
            k_w_2, k_b_2,
            v_w_2, v_b_2, 
            decoder_vars[4 + 12 * i], decoder_vars[5 + 12 * i],
            decoder_vars[4 + 12 * i], decoder_vars[5 + 12 * i],
            decoder_vars[8 + 12 * i], decoder_vars[9 + 12 * i],
            decoder_vars[10 + 12 * i], decoder_vars[11 + 12 * i],
            op_self_cache[i], op_mem_cache[i],
            psuedo_input,  # add tf_result as input to prevent the OP and TF from parallel execution and lead to error result
            head_num=decoder_args.head_num, 
            size_per_head=decoder_args.size_per_head)
        inputs = op_result

    return op_result, op_self_cache, op_mem_cache

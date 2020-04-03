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
from common import create_initializer

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


def tf_decoder(decoder_args,
               inputs,
               memory,
               memory_sequence_length,
               step,
               cache=None,
               kernel_initializer_range=0.02,
               bias_initializer_range=0):
    memory_mask = None  # has something

    if memory is not None and not tf.contrib.framework.nest.is_sequence(memory):
        memory = (memory,)
        if memory_sequence_length is not None:
            if not tf.contrib.framework.nest.is_sequence(memory_sequence_length):
                memory_sequence_length = (memory_sequence_length,)
            memory_mask = [
                build_sequence_mask(
                    length, num_heads=decoder_args.head_num, maximum_length=tf.shape(m)[1], data_type=decoder_args.dtype)
                for m, length in zip(memory, memory_sequence_length)]

    for l in range(decoder_args.num_layer):
        layer_name = "layer_{}".format(l)
        layer_cache = cache[layer_name] if cache is not None else None

        with tf.variable_scope(layer_name):
            with tf.variable_scope("masked_multi_head"):
                norm_inputs = norm(inputs)
                queries = tf.layers.conv1d(
                    norm_inputs,
                    decoder_args.hidden_dim,
                    1,
                    activation=None,
                    name="query",
                    use_bias=True,
                    bias_initializer=create_initializer(
                        bias_initializer_range, decoder_args.dtype),
                    kernel_initializer=create_initializer(kernel_initializer_range, decoder_args.dtype))

                keys = tf.layers.conv1d(
                    norm_inputs,
                    decoder_args.hidden_dim,
                    1,
                    activation=None,
                    name="key",
                    use_bias=True,
                    bias_initializer=create_initializer(
                        bias_initializer_range, decoder_args.dtype),
                    kernel_initializer=create_initializer(kernel_initializer_range, decoder_args.dtype))

                values = tf.layers.conv1d(
                    norm_inputs,
                    decoder_args.hidden_dim,
                    1,
                    activation=None,
                    name="value",
                    use_bias=True,
                    bias_initializer=create_initializer(
                        bias_initializer_range, decoder_args.dtype),
                    kernel_initializer=create_initializer(kernel_initializer_range, decoder_args.dtype))

                keys = tf.reshape(keys, [decoder_args.batch_size * decoder_args.beam_width,
                                         1, decoder_args.head_num, decoder_args.size_per_head])
                keys = tf.transpose(keys, [0, 2, 1, 3])
                values = tf.reshape(values, [
                                    decoder_args.batch_size * decoder_args.beam_width, 1, decoder_args.head_num, decoder_args.size_per_head])
                values = tf.transpose(values, [0, 2, 1, 3])

                keys = tf.concat([layer_cache["self_keys"], keys], axis=2)
                values = tf.concat(
                    [layer_cache["self_values"], values], axis=2)
                layer_cache["self_keys"] = keys
                layer_cache["self_values"] = values

                queries = tf.reshape(queries, [
                                     decoder_args.batch_size * decoder_args.beam_width, 1, decoder_args.head_num, decoder_args.size_per_head])
                queries = tf.transpose(queries, [0, 2, 1, 3])
                queries *= (decoder_args.size_per_head)**-0.5

                dot = tf.matmul(queries, keys, transpose_b=True)

                attn = tf.cast(tf.nn.softmax(
                    tf.cast(dot, decoder_args.dtype)), dot.dtype)
                context = tf.matmul(attn, values)
                context = tf.transpose(context, [0, 2, 1, 3])
                context = tf.reshape(context, [
                                     decoder_args.batch_size * decoder_args.beam_width, 1, decoder_args.head_num * decoder_args.size_per_head])

                outputs = tf.layers.conv1d(context,
                                           decoder_args.hidden_dim,
                                           1,
                                           activation=None,
                                           use_bias=True,
                                           bias_initializer=create_initializer(
                                               bias_initializer_range, decoder_args.dtype),
                                           kernel_initializer=create_initializer(kernel_initializer_range, decoder_args.dtype))

                # drop_and_add
                input_dim = inputs.get_shape().as_list()[-1]
                output_dim = outputs.get_shape().as_list()[-1]
                if input_dim == output_dim:
                    outputs += inputs
                last_context = outputs

            if memory is not None:
                for i, (mem, mask) in enumerate(zip(memory, memory_mask)):
                    memory_cache = layer_cache["memory"][i] if layer_cache is not None else None

                    with tf.variable_scope("multi_head" if i == 0 else "multi_head_%d" % i):
                        queries = tf.layers.conv1d(
                            norm(last_context),
                            decoder_args.hidden_dim,
                            1,
                            activation=None,
                            name="query",
                            use_bias=True,
                            bias_initializer=create_initializer(
                                bias_initializer_range, decoder_args.dtype),
                            kernel_initializer=create_initializer(kernel_initializer_range, decoder_args.dtype))

                        def _project_and_split():
                            keys = tf.layers.conv1d(
                                mem,
                                decoder_args.hidden_dim,
                                1,
                                activation=None,
                                name="key",
                                use_bias=True,
                                bias_initializer=create_initializer(
                                    bias_initializer_range, decoder_args.dtype),
                                kernel_initializer=create_initializer(kernel_initializer_range, decoder_args.dtype))

                            values = tf.layers.conv1d(
                                mem,
                                decoder_args.hidden_dim,
                                1,
                                activation=None,
                                name="value",
                                use_bias=True,
                                bias_initializer=create_initializer(
                                    bias_initializer_range, decoder_args.dtype),
                                kernel_initializer=create_initializer(kernel_initializer_range, decoder_args.dtype))

                            keys = tf.reshape(keys, [decoder_args.batch_size * decoder_args.beam_width, tf.shape(keys)[1],
                                                     decoder_args.head_num, decoder_args.size_per_head])
                            keys = tf.transpose(keys, [0, 2, 1, 3])
                            values = tf.reshape(values, [decoder_args.batch_size * decoder_args.beam_width, tf.shape(values)[1],
                                                         decoder_args.head_num, decoder_args.size_per_head])
                            values = tf.transpose(values, [0, 2, 1, 3])

                            return keys, values

                        keys, values = tf.cond(
                            tf.equal(
                                tf.shape(memory_cache["memory_keys"])[2], 0),
                            true_fn=_project_and_split,
                            false_fn=lambda: (memory_cache["memory_keys"], memory_cache["memory_values"]))

                        memory_cache["memory_keys"] = keys
                        memory_cache["memory_values"] = values

                        queries = tf.reshape(queries, [decoder_args.batch_size * decoder_args.beam_width, 1,
                                                       decoder_args.head_num, decoder_args.size_per_head])
                        queries = tf.transpose(queries, [0, 2, 1, 3])
                        queries *= (decoder_args.size_per_head)**-0.5

                        dot = tf.matmul(queries, keys, transpose_b=True)

                        dot = tf.cast(tf.cast(dot, decoder_args.dtype) * mask +
                                      ((1.0 - mask) * decoder_args.dtype.min), dot.dtype)

                        attn = tf.cast(tf.nn.softmax(
                            tf.cast(dot, decoder_args.dtype)), dot.dtype)
                        context = tf.matmul(attn, values)
                        context = tf.transpose(context, [0, 2, 1, 3])
                        context = tf.reshape(context, [decoder_args.batch_size * decoder_args.beam_width, 1,
                                                       decoder_args.head_num * decoder_args.size_per_head])
                        context = tf.layers.conv1d(context,
                                                   decoder_args.hidden_dim,
                                                   1,
                                                   activation=None,
                                                   use_bias=True,
                                                   bias_initializer=create_initializer(
                                                       bias_initializer_range, decoder_args.dtype),
                                                   kernel_initializer=create_initializer(kernel_initializer_range, decoder_args.dtype))

                        # drop_and_add
                        input_dim = last_context.get_shape().as_list()[-1]
                        output_dim = context.get_shape().as_list()[-1]
                        if input_dim == output_dim:
                            context += last_context

            with tf.variable_scope("ffn"):
                # forward
                normed_last_context = norm(context)
                input_dim = normed_last_context.get_shape().as_list()[-1]
                inner = tf.layers.conv1d(normed_last_context,
                                         decoder_args.hidden_dim * 4,
                                         1,
                                         activation=tf.nn.relu,
                                         use_bias=True,
                                         bias_initializer=create_initializer(
                                             bias_initializer_range, decoder_args.dtype),
                                         kernel_initializer=create_initializer(kernel_initializer_range, decoder_args.dtype))
                transformed = tf.layers.conv1d(inner,
                                               input_dim,
                                               1,
                                               use_bias=True,
                                               bias_initializer=create_initializer(
                                                   bias_initializer_range, decoder_args.dtype),
                                               kernel_initializer=create_initializer(kernel_initializer_range, decoder_args.dtype))

                # drop_and_add
                input_dim = context.get_shape().as_list()[-1]
                output_dim = transformed.get_shape().as_list()[-1]
                if input_dim == output_dim:
                    transformed += context

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


def init_op_cache(decoder_args):
    self_cache = tf.zeros([decoder_args.num_layer, 2, 0, decoder_args.batch_size * decoder_args.beam_width,
                           decoder_args.hidden_dim], dtype=decoder_args.dtype, name="op_self_caches")
    mem_cache = tf.zeros([decoder_args.num_layer, 2, decoder_args.batch_size * decoder_args.beam_width,
                          decoder_args.max_seq_len, decoder_args.hidden_dim], dtype=decoder_args.dtype, name="op_memory_caches")

    return self_cache, mem_cache


def op_decoder(inputs,
               step,
               memory_tensor,
               memory_sequence_length,
               op_self_cache,
               op_mem_cache,
               psuedo_input,
               decoder_vars,
               decoder_args,
               memory_hidden_dim):

    decoder_op_module = tf.load_op_library(
        os.path.join('./lib/libtf_decoder.so'))

    op_self_cache = tf.concat([op_self_cache, tf.zeros([decoder_args.num_layer, 2, 1,
                                                        decoder_args.batch_size * decoder_args.beam_width,
                                                        decoder_args.hidden_dim], dtype=decoder_args.dtype)], axis=2)

    for i in range(decoder_args.num_layer):
        op_result, _, _ = decoder_op_module.decoder(
            inputs, memory_tensor, memory_sequence_length,
            decoder_vars[0 + 26 * i], decoder_vars[1 + 26 * i],
            decoder_vars[2 + 26 * i], decoder_vars[3 + 26 * i],
            decoder_vars[4 + 26 * i], decoder_vars[5 + 26 * i],
            decoder_vars[6 + 26 * i], decoder_vars[7 + 26 * i],
            decoder_vars[8 + 26 * i], decoder_vars[9 + 26 * i],
            decoder_vars[10 + 26 * i], decoder_vars[11 + 26 * i],
            decoder_vars[12 + 26 * i], decoder_vars[13 + 26 * i],
            decoder_vars[14 + 26 * i], decoder_vars[15 + 26 * i],
            decoder_vars[16 + 26 * i], decoder_vars[17 + 26 * i],
            decoder_vars[18 + 26 * i], decoder_vars[19 + 26 * i],
            decoder_vars[20 + 26 * i], decoder_vars[21 + 26 * i],
            decoder_vars[22 + 26 * i], decoder_vars[23 + 26 * i],
            decoder_vars[24 + 26 * i], decoder_vars[25 + 26 * i],
            op_self_cache[i], op_mem_cache[i],
            psuedo_input,  # add tf_result as input to prevent the OP and TF from parallel execution and lead to error result
            head_num=decoder_args.head_num, 
            size_per_head=decoder_args.size_per_head)
        inputs = op_result

    return op_result, op_self_cache, op_mem_cache

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import device_lib
import tensorflow as tf
import os
import sys
from my_modeling import *

build_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../lib')

transformer_op_module = tf.load_op_library(
    os.path.join(build_path, 'libtf_fastertransformer.so'))


def file_based_input_fn_builder_drop(input_file, seq_length, is_training,
                                     drop_remainder):
    """ Re-implementation of file_based_input_fn_builder function from modeling.py from Google's BERT repository https://github.com/google-research/bert
        with drop_remainder=True.
    """

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        # FASTINFER: drop remainder always
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=True))

        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      dtype=tf.flags.FLAGS.floatx,
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], 
      dtype=tf.flags.FLAGS.floatx,
      initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.flags.FLAGS.floatx)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)

def create_model_squad(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings):
  """Creates a classification model."""
  model = BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  final_hidden = model.get_sequence_output()

  final_hidden_shape = get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  output_weights = tf.get_variable(
      "cls/squad/output_weights", [2, hidden_size],
      dtype=tf.flags.FLAGS.floatx,
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "cls/squad/output_bias", [2],
      dtype=tf.flags.FLAGS.floatx,
      initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2])
  logits = tf.transpose(logits, [2, 0, 1])

  unstacked_logits = tf.unstack(logits, axis=0)

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

  return (start_logits, end_logits)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def fast_transformer_model_trans(input_tensor,
                                 attention_mask=None,
                                 hidden_size=768,
                                 num_hidden_layers=12,
                                 num_attention_heads=12,
                                 intermediate_size=3072,
                                 intermediate_act_fn=gelu,
                                 hidden_dropout_prob=0.1,
                                 attention_probs_dropout_prob=0.1,
                                 initializer_range=0.02,
                                 do_return_all_layers=False,
                                 sequence_length=None):
    """ Re-implementation of transformer_model function from modeling.py from Google's BERT repository https://github.com/google-research/bert
        using FasterTransformer Tensorflow op.

    Multi-headed, multi-layer Transformer from "Attention is All You Need".
    This is almost an exact implementation of the original Transformer encoder.

    See the original paper:
    https://arxiv.org/abs/1706.03762

    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
      attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
        seq_length], with 1 for positions that can be attended to and 0 in
        positions that should not be.
      hidden_size: int. Hidden size of the Transformer.
      num_hidden_layers: int. Number of layers (blocks) in the Transformer.
      num_attention_heads: int. Number of attention heads in the Transformer.
      intermediate_size: int. The size of the "intermediate" (a.k.a., feed
        forward) layer.
      intermediate_act_fn: function. The non-linear activation function to apply
        to the output of the intermediate/feed-forward layer.
      hidden_dropout_prob: float. Dropout probability for the hidden layers.
      attention_probs_dropout_prob: float. Dropout probability of the attention
        probabilities.
      initializer_range: float. Range of the initializer (stddev of truncated
        normal).
      do_return_all_layers: Whether to also return all layers or just the final
        layer.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size], the final
      hidden layer of the Transformer.

    Raises:
      ValueError: A Tensor shape or parameter is invalid.
    """
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    if input_width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                         (input_width, hidden_size))

    # We keep the representation as a 2D tensor to avoid re-shaping it back and
    # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
    # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
    # help the optimizer.
    prev_output = reshape_to_matrix(input_tensor)

    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output

            with tf.variable_scope("attention"):
                attention_heads = []
                with tf.variable_scope("self"):
                    attention_head = attention_layer(
                        from_tensor=layer_input,
                        to_tensor=layer_input,
                        attention_mask=attention_mask,
                        num_attention_heads=num_attention_heads,
                        size_per_head=attention_head_size,
                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=True,
                        batch_size=batch_size,
                        from_seq_length=seq_length,
                        to_seq_length=seq_length)
                    attention_heads.append(attention_head)

                attention_output = None
                if len(attention_heads) == 1:
                    attention_output = attention_heads[0]
                else:
                    # In the case where we have other sequences, we just concatenate
                    # them to the self-attention head before the projection.
                    attention_output = tf.concat(attention_heads, axis=-1)

                # Run a linear projection of `hidden_size` then add a residual
                # with `layer_input`.
                with tf.variable_scope("output"):
                    attention_output = tf.layers.dense(
                        attention_output,
                        hidden_size,
                        kernel_initializer=create_initializer(initializer_range))
                    attention_output = dropout(
                        attention_output, hidden_dropout_prob)
                    attention_output = layer_norm(
                        attention_output + layer_input)

            # The activation is only applied to the "intermediate" hidden layer.
            with tf.variable_scope("intermediate"):
                intermediate_output = tf.layers.dense(
                    attention_output,
                    intermediate_size,
                    activation=intermediate_act_fn,
                    kernel_initializer=create_initializer(initializer_range))

            # Down-project back to `hidden_size` then add the residual.
            with tf.variable_scope("output"):
                layer_output = tf.layers.dense(
                    intermediate_output,
                    hidden_size,
                    kernel_initializer=create_initializer(initializer_range))
                layer_output = dropout(layer_output, hidden_dropout_prob)
                layer_output = layer_norm(layer_output + attention_output)
            # amaxList
            if tf.flags.FLAGS.int8_mode != 0:
                amaxList = tf.get_variable(name="amaxList", shape=[80 + 9*hidden_size + 8], dtype=tf.float32)

            
    # FASTINFER: fast transformer encoder inference
    inputs = input_tensor
    int8_mode = tf.flags.FLAGS.int8_mode
    remove_padding = tf.flags.FLAGS.remove_padding
    allow_gemm_test = tf.flags.FLAGS.allow_gemm_test
    if remove_padding == True:
        inputs, sequence_id_offset = transformer_op_module.build_mask_remove_padding(inputs, sequence_length)
        trt_seq_len = tf.cumsum(tf.concat([[0], sequence_length], axis=0), axis=0)
    else:
        sequence_id_offset = []
        batch = tf.shape(inputs)[0]
        max_seq_len = tf.shape(inputs)[1]
        padding_offset = tf.range(0, batch*max_seq_len, max_seq_len)
        squence_offset_with_padding = sequence_length + padding_offset
        c = tf.concat([padding_offset, squence_offset_with_padding], axis=0)
        c_r = tf.reshape(c, [2, -1])
        t = tf.transpose(c_r)
        trt_seq_len = tf.reshape(t, [-1])
        trt_seq_len = tf.concat([trt_seq_len, [batch*max_seq_len]], axis=0)
    
    graph = tf.get_default_graph()
    for layer_idx in range(num_hidden_layers):
        if int8_mode != 0:
            amaxL = graph.get_tensor_by_name('bert/encoder/layer_%d/amaxList:0' % layer_idx)
        else:
            amaxL = []

        layer_output = transformer_op_module.bert_transformer(
            inputs,
            inputs,
            graph.get_tensor_by_name('bert/encoder/layer_%d/attention/self/query/kernel:0' % layer_idx),
            graph.get_tensor_by_name('bert/encoder/layer_%d/attention/self/query/bias:0' % layer_idx),
            graph.get_tensor_by_name('bert/encoder/layer_%d/attention/self/key/kernel:0' % layer_idx),
            graph.get_tensor_by_name('bert/encoder/layer_%d/attention/self/key/bias:0' % layer_idx),
            graph.get_tensor_by_name('bert/encoder/layer_%d/attention/self/value/kernel:0' % layer_idx),
            graph.get_tensor_by_name('bert/encoder/layer_%d/attention/self/value/bias:0' % layer_idx),
            tf.expand_dims(attention_mask, 1),
            graph.get_tensor_by_name('bert/encoder/layer_%d/attention/output/dense/kernel:0' % layer_idx),
            graph.get_tensor_by_name('bert/encoder/layer_%d/attention/output/dense/bias:0' % layer_idx),
            graph.get_tensor_by_name('bert/encoder/layer_%d/attention/output/LayerNorm/beta:0' % layer_idx),
            graph.get_tensor_by_name('bert/encoder/layer_%d/attention/output/LayerNorm/gamma:0' % layer_idx),
            graph.get_tensor_by_name('bert/encoder/layer_%d/intermediate/dense/kernel:0' % layer_idx),
            graph.get_tensor_by_name('bert/encoder/layer_%d/intermediate/dense/bias:0' % layer_idx),
            graph.get_tensor_by_name('bert/encoder/layer_%d/output/dense/kernel:0' % layer_idx),
            graph.get_tensor_by_name('bert/encoder/layer_%d/output/dense/bias:0' % layer_idx),
            graph.get_tensor_by_name('bert/encoder/layer_%d/output/LayerNorm/beta:0' % layer_idx),
            graph.get_tensor_by_name('bert/encoder/layer_%d/output/LayerNorm/gamma:0' % layer_idx),
            sequence_id_offset,
            amaxL,
            trt_seq_len,
            head_num=num_attention_heads, size_per_head=attention_head_size,
            int8_mode=int8_mode, layer_idx=layer_idx, layer_num = num_hidden_layers,
            allow_gemm_test=allow_gemm_test, remove_padding=remove_padding)

        if remove_padding == True:
            if int8_mode == 0 or layer_idx == num_hidden_layers - 1:
                all_layer_outputs.append(transformer_op_module.rebuild_padding(layer_output, sequence_id_offset, tf.expand_dims(attention_mask, 1)))
            else:
                #for int8 encoder, only the output of last layer
                #can be directly used for downstream tasks
                all_layer_outputs.append(tf.zeros(input_shape))
        else:
            all_layer_outputs.append(layer_output)
        inputs = layer_output
    
    if remove_padding == True:
        layer_output = transformer_op_module.rebuild_padding(layer_output, sequence_id_offset, tf.expand_dims(attention_mask, 1))
            
            
    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = reshape_from_matrix(layer_output, input_shape)
        return final_output

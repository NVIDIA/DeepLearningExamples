# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
import os
import math
import six
from datetime import datetime
import sys
transformer_op_module = tf.load_op_library(os.path.join('./lib/libtf_transformer.so'))

argumentList = sys.argv
batch_size = int(sys.argv[1])
num_layers = int(sys.argv[2])
seq_len = int(sys.argv[3])
print("Argumentlist: batch_size " + str(batch_size) + " num_layers " + str(num_layers) + " seq_len " + str(seq_len))

head_num = 12
size_per_head = 64
hidden_dim = head_num * size_per_head
initializer_range = 0.02
from_data = np.random.randn(batch_size, seq_len, hidden_dim)
from_tensor = tf.convert_to_tensor(from_data, dtype=float)

mask = np.random.randint(2, size=(batch_size, seq_len, seq_len))
attention_mask = tf.convert_to_tensor(mask, dtype=float)

def gelu(x):
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf

def layer_norm(input_tensor, name=None):
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

def create_initializer(initializer_range=0.02):
  return tf.truncated_normal_initializer(stddev=initializer_range)

def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

  if len(from_shape) != len(to_shape):
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  from_tensor_2d = reshape_to_matrix(from_tensor)
  to_tensor_2d = reshape_to_matrix(to_tensor)

  # `query_layer` = [B*F, N*H]
  query_layer = tf.layers.dense(
      from_tensor_2d,
      num_attention_heads * size_per_head,
      activation=query_act,
      name="query",
      use_bias=True,
      bias_initializer = create_initializer(initializer_range),
      kernel_initializer=create_initializer(initializer_range))

  # `key_layer` = [B*T, N*H]
  key_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=key_act,
      name="key",
      use_bias=True,
      bias_initializer = create_initializer(initializer_range),
      kernel_initializer=create_initializer(initializer_range))

  # `value_layer` = [B*T, N*H]
  value_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=value_act,
      name="value",
      use_bias=True,
      bias_initializer = create_initializer(initializer_range),
      kernel_initializer=create_initializer(initializer_range))

  # `query_layer` = [B, N, F, H]
  query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  # `key_layer` = [B, N, T, H]
  key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

  if attention_mask is not None:
    # `attention_mask` = [B, 1, F, T]
    attention_mask = tf.expand_dims(attention_mask, axis=[1])

    adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

    attention_scores += adder

  attention_probs = tf.nn.softmax(attention_scores)


  value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])

  value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

  context_layer = tf.matmul(attention_probs, value_layer)

  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

  if do_return_2d_tensor:
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * size_per_head])
  else:
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * size_per_head])

  return context_layer


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      initializer_range=0.02,
                      do_return_all_layers=False):
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2]

  prev_output = reshape_to_matrix(input_tensor)

  for layer_idx in range(num_hidden_layers):
    with tf.variable_scope("layer_%d" % layer_idx, reuse=tf.AUTO_REUSE):
      layer_input = prev_output
      with tf.variable_scope("attention"):
        with tf.variable_scope("self"):
          attention_head = attention_layer(
              from_tensor=layer_input,
              to_tensor=layer_input,
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length)
          attention_output = attention_head

        with tf.variable_scope("output"):
          attention_output = tf.layers.dense(
              attention_output,
              hidden_size,
              use_bias=True,
              bias_initializer = create_initializer(initializer_range),
              kernel_initializer=create_initializer(initializer_range))
          attention_output = layer_norm(attention_output + layer_input)

      # The activation is only applied to the "intermediate" hidden layer.
      with tf.variable_scope("intermediate"):
        intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            use_bias=True,
            bias_initializer = create_initializer(initializer_range),
            kernel_initializer=create_initializer(initializer_range))

      # Down-project back to `hidden_size` then add the residual.
      with tf.variable_scope("output"):
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            use_bias=True,
            bias_initializer = create_initializer(initializer_range),
            kernel_initializer=create_initializer(initializer_range))
        layer_output = layer_norm(layer_output + attention_output)
        prev_output = layer_output

  return prev_output


def get_shape_list(tensor, expected_rank=None, name=None):
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def transformer_single(input_tensor, params, layer_idx):
  val_off = layer_idx * 16
  output = transformer_op_module.bert_transformer(
      input_tensor, 
      input_tensor, 
      params[val_off + 0], params[val_off + 2], params[val_off + 4], params[val_off + 1], params[val_off + 3], params[val_off + 5], attention_mask, 
      params[val_off + 6], params[val_off + 7], params[val_off + 8], params[val_off + 9], params[val_off + 10], 
      params[val_off + 11], params[val_off + 12], params[val_off + 13], params[val_off + 14], params[val_off + 15], 
      batch_size = batch_size, from_seq_len = seq_len, to_seq_len = seq_len, head_num = head_num, size_per_head = size_per_head)
  return output

def transformer_own(input_tensor, params):
  in_tensor = input_tensor
  for layer_idx in range(num_layers):
    out_tensor = transformer_single(in_tensor, params, layer_idx)
    in_tensor = out_tensor
  return in_tensor
    
output = transformer_model(input_tensor=from_tensor, attention_mask = attention_mask, num_hidden_layers = num_layers, do_return_all_layers=True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
                 
    sess.run(output)
    Model_variables = tf.GraphKeys.GLOBAL_VARIABLES

    idx = 0
    all_vars = tf.get_collection(Model_variables)
    for var in all_vars:
      print (str(idx) + " " + str(var.name) + " " + str(var.shape))
      idx = idx + 1

    params = all_vars
    output_own = transformer_own(from_tensor, params)
    for ite in range(20):
      print("ite " + str(ite))
      try:
        sess.run(output_own)
      except tf.errors.InvalidArgumentError as e:
        print(e)
      except tf.errors.InternalError as e:
        print(e)
      except:
        print("Runtime error")
        

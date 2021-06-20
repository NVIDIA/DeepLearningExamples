# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math
import six
import tensorflow as tf

from tensorflow.python.util import deprecation
from official.modeling import tf_utils


class BertConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02,
               backward_compatible=True):
    """Constructs BertConfig.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
      backward_compatible: Boolean, whether the variables shape are compatible
        with checkpoints converted from TF 1.x BERT.
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range
    self.backward_compatible = backward_compatible

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = BertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.io.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class AlbertConfig(BertConfig):
  """Configuration for `ALBERT`."""

  def __init__(self,
               embedding_size,
               num_hidden_groups=1,
               inner_group_num=1,
               **kwargs):
    """Constructs AlbertConfig.

    Args:
      embedding_size: Size of the factorized word embeddings.
      num_hidden_groups: Number of group for the hidden layers, parameters in
        the same group are shared. Note that this value and also the following
        'inner_group_num' has to be 1 for now, because all released ALBERT
        models set them to 1. We may support arbitary valid values in future.
      inner_group_num: Number of inner repetition of attention and ffn.
      **kwargs: The remaining arguments are the same as above 'BertConfig'.
    """
    super(AlbertConfig, self).__init__(**kwargs)
    self.embedding_size = embedding_size

    # TODO(chendouble): 'inner_group_num' and 'num_hidden_groups' are always 1
    # in the released ALBERT. Support other values in AlbertTransformerEncoder
    # if needed.
    if inner_group_num != 1 or num_hidden_groups != 1:
      raise ValueError("We only support 'inner_group_num' and "
                       "'num_hidden_groups' as 1.")

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `AlbertConfig` from a Python dictionary of parameters."""
    config = AlbertConfig(embedding_size=None, vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config


@deprecation.deprecated(None, "The function should not be used any more.")
def get_bert_model(input_word_ids,
                   input_mask,
                   input_type_ids,
                   config=None,
                   name=None,
                   float_type=tf.float32):
  """Wraps the core BERT model as a keras.Model."""
  bert_model_layer = BertModel(config=config, float_type=float_type, name=name)
  pooled_output, sequence_output = bert_model_layer(input_word_ids, input_mask,
                                                    input_type_ids)
  bert_model = tf.keras.Model(
      inputs=[input_word_ids, input_mask, input_type_ids],
      outputs=[pooled_output, sequence_output])
  return bert_model


class BertModel(tf.keras.layers.Layer):
  """BERT model ("Bidirectional Encoder Representations from Transformers").

  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_word_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  input_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  pooled_output, sequence_output = modeling.BertModel(config=config)(
    input_word_ids=input_word_ids,
    input_mask=input_mask,
    input_type_ids=input_type_ids)
  ...
  ```
  """

  @deprecation.deprecated(
      None, "Please use `nlp.modeling.networks.TransformerEncoder` instead.")
  def __init__(self, config, float_type=tf.float32, **kwargs):
    super(BertModel, self).__init__(**kwargs)
    self.config = (
        BertConfig.from_dict(config)
        if isinstance(config, dict) else copy.deepcopy(config))
    self.float_type = float_type

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.embedding_lookup = EmbeddingLookup(
        vocab_size=self.config.vocab_size,
        embedding_size=self.config.hidden_size,
        initializer_range=self.config.initializer_range,
        dtype=tf.float32,
        name="word_embeddings")
    self.embedding_postprocessor = EmbeddingPostprocessor(
        use_type_embeddings=True,
        token_type_vocab_size=self.config.type_vocab_size,
        use_position_embeddings=True,
        max_position_embeddings=self.config.max_position_embeddings,
        dropout_prob=self.config.hidden_dropout_prob,
        initializer_range=self.config.initializer_range,
        dtype=tf.float32,
        name="embedding_postprocessor")
    self.encoder = Transformer(
        num_hidden_layers=self.config.num_hidden_layers,
        hidden_size=self.config.hidden_size,
        num_attention_heads=self.config.num_attention_heads,
        intermediate_size=self.config.intermediate_size,
        intermediate_activation=self.config.hidden_act,
        hidden_dropout_prob=self.config.hidden_dropout_prob,
        attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
        initializer_range=self.config.initializer_range,
        backward_compatible=self.config.backward_compatible,
        float_type=self.float_type,
        name="encoder")
    self.pooler_transform = tf.keras.layers.Dense(
        units=self.config.hidden_size,
        activation="tanh",
        kernel_initializer=get_initializer(self.config.initializer_range),
        name="pooler_transform")
    super(BertModel, self).build(unused_input_shapes)

  def __call__(self,
               input_word_ids,
               input_mask=None,
               input_type_ids=None,
               **kwargs):
    inputs = tf_utils.pack_inputs([input_word_ids, input_mask, input_type_ids])
    return super(BertModel, self).__call__(inputs, **kwargs)

  def call(self, inputs, mode="bert"):
    """Implements call() for the layer.

    Args:
      inputs: packed input tensors.
      mode: string, `bert` or `encoder`.

    Returns:
      Output tensor of the last layer for BERT training (mode=`bert`) which
      is a float Tensor of shape [batch_size, seq_length, hidden_size] or
      a list of output tensors for encoder usage (mode=`encoder`).
    """
    unpacked_inputs = tf_utils.unpack_inputs(inputs)
    input_word_ids = unpacked_inputs[0]
    input_mask = unpacked_inputs[1]
    input_type_ids = unpacked_inputs[2]

    word_embeddings = self.embedding_lookup(input_word_ids)
    embedding_tensor = self.embedding_postprocessor(
        word_embeddings=word_embeddings, token_type_ids=input_type_ids)
    if self.float_type == tf.float16:
      embedding_tensor = tf.cast(embedding_tensor, tf.float16)
    attention_mask = None
    if input_mask is not None:
      attention_mask = create_attention_mask_from_input_mask(
          input_word_ids, input_mask)

    if mode == "encoder":
      return self.encoder(
          embedding_tensor, attention_mask, return_all_layers=True)

    sequence_output = self.encoder(embedding_tensor, attention_mask)
    first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
    pooled_output = self.pooler_transform(first_token_tensor)

    return (pooled_output, sequence_output)

  def get_config(self):
    config = {"config": self.config.to_dict()}
    base_config = super(BertModel, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class EmbeddingLookup(tf.keras.layers.Layer):
  """Looks up words embeddings for id tensor."""

  def __init__(self,
               vocab_size,
               embedding_size=768,
               initializer_range=0.02,
               **kwargs):
    super(EmbeddingLookup, self).__init__(**kwargs)
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.initializer_range = initializer_range

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.embeddings = self.add_weight(
        "embeddings",
        shape=[self.vocab_size, self.embedding_size],
        initializer=get_initializer(self.initializer_range),
        dtype=self.dtype)
    super(EmbeddingLookup, self).build(unused_input_shapes)

  def call(self, inputs):
    """Implements call() for the layer."""
    input_shape = tf_utils.get_shape_list(inputs)
    flat_input = tf.reshape(inputs, [-1])
    output = tf.gather(self.embeddings, flat_input)
    output = tf.reshape(output, input_shape + [self.embedding_size])
    return output


class EmbeddingPostprocessor(tf.keras.layers.Layer):
  """Performs various post-processing on a word embedding tensor."""

  def __init__(self,
               use_type_embeddings=False,
               token_type_vocab_size=None,
               use_position_embeddings=True,
               max_position_embeddings=512,
               dropout_prob=0.0,
               initializer_range=0.02,
               initializer=None,
               **kwargs):
    super(EmbeddingPostprocessor, self).__init__(**kwargs)
    self.use_type_embeddings = use_type_embeddings
    self.token_type_vocab_size = token_type_vocab_size
    self.use_position_embeddings = use_position_embeddings
    self.max_position_embeddings = max_position_embeddings
    self.dropout_prob = dropout_prob
    self.initializer_range = initializer_range

    if not initializer:
      self.initializer = get_initializer(self.initializer_range)
    else:
      self.initializer = initializer

    if self.use_type_embeddings and not self.token_type_vocab_size:
      raise ValueError("If `use_type_embeddings` is True, then "
                       "`token_type_vocab_size` must be specified.")

  def build(self, input_shapes):
    """Implements build() for the layer."""
    (word_embeddings_shape, _) = input_shapes
    width = word_embeddings_shape.as_list()[-1]
    self.type_embeddings = None
    if self.use_type_embeddings:
      self.type_embeddings = self.add_weight(
          "type_embeddings",
          shape=[self.token_type_vocab_size, width],
          initializer=get_initializer(self.initializer_range),
          dtype=self.dtype)

    self.position_embeddings = None
    if self.use_position_embeddings:
      self.position_embeddings = self.add_weight(
          "position_embeddings",
          shape=[self.max_position_embeddings, width],
          initializer=get_initializer(self.initializer_range),
          dtype=self.dtype)

    self.output_layer_norm = tf.keras.layers.LayerNormalization(
        name="layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32)
    self.output_dropout = tf.keras.layers.Dropout(
        rate=self.dropout_prob, dtype=tf.float32)
    super(EmbeddingPostprocessor, self).build(input_shapes)

  def __call__(self, word_embeddings, token_type_ids=None, **kwargs):
    inputs = tf_utils.pack_inputs([word_embeddings, token_type_ids])
    return super(EmbeddingPostprocessor, self).__call__(inputs, **kwargs)

  def call(self, inputs):
    """Implements call() for the layer."""
    unpacked_inputs = tf_utils.unpack_inputs(inputs)
    word_embeddings = unpacked_inputs[0]
    token_type_ids = unpacked_inputs[1]
    input_shape = tf_utils.get_shape_list(word_embeddings, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = word_embeddings
    if self.use_type_embeddings:
      flat_token_type_ids = tf.reshape(token_type_ids, [-1])
      token_type_embeddings = tf.gather(self.type_embeddings,
                                        flat_token_type_ids)
      token_type_embeddings = tf.reshape(token_type_embeddings,
                                         [batch_size, seq_length, width])
      output += token_type_embeddings

    if self.use_position_embeddings:
      position_embeddings = tf.expand_dims(
          tf.slice(self.position_embeddings, [0, 0], [seq_length, width]),
          axis=0)

      output += position_embeddings

    output = self.output_layer_norm(output)
    output = self.output_dropout(output)

    return output


class Attention(tf.keras.layers.Layer):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.

  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.

  In practice, the multi-headed attention are done with tf.einsum as follows:
    Input_tensor: [BFD]
    Wq, Wk, Wv: [DNH]
    Q:[BFNH] = einsum('BFD,DNH->BFNH', Input_tensor, Wq)
    K:[BTNH] = einsum('BTD,DNH->BTNH', Input_tensor, Wk)
    V:[BTNH] = einsum('BTD,DNH->BTNH', Input_tensor, Wv)
    attention_scores:[BNFT] = einsum('BTNH,BFNH->BNFT', K, Q) / sqrt(H)
    attention_probs:[BNFT] = softmax(attention_scores)
    context_layer:[BFNH] = einsum('BNFT,BTNH->BFNH', attention_probs, V)
    Wout:[DNH]
    Output:[BFD] = einsum('BFNH,DNH>BFD', context_layer, Wout)
  """

  def __init__(self,
               num_attention_heads=12,
               size_per_head=64,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               backward_compatible=False,
               **kwargs):
    super(Attention, self).__init__(**kwargs)
    self.num_attention_heads = num_attention_heads
    self.size_per_head = size_per_head
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.backward_compatible = backward_compatible

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.query_dense = self._projection_dense_layer("query")
    self.key_dense = self._projection_dense_layer("key")
    self.value_dense = self._projection_dense_layer("value")
    self.attention_probs_dropout = tf.keras.layers.Dropout(
        rate=self.attention_probs_dropout_prob)
    super(Attention, self).build(unused_input_shapes)

  def reshape_to_matrix(self, input_tensor):
    """Reshape N > 2 rank tensor to rank 2 tensor for performance."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
      raise ValueError("Input tensor must have at least rank 2."
                       "Shape = %s" % (input_tensor.shape))
    if ndims == 2:
      return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor

  def __call__(self, from_tensor, to_tensor, attention_mask=None, **kwargs):
    inputs = tf_utils.pack_inputs([from_tensor, to_tensor, attention_mask])
    return super(Attention, self).__call__(inputs, **kwargs)

  def call(self, inputs):
    """Implements call() for the layer."""
    (from_tensor, to_tensor, attention_mask) = tf_utils.unpack_inputs(inputs)

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`
    # `query_tensor` = [B, F, N ,H]
    query_tensor = self.query_dense(from_tensor)

    # `key_tensor` = [B, T, N, H]
    key_tensor = self.key_dense(to_tensor)

    # `value_tensor` = [B, T, N, H]
    value_tensor = self.value_dense(to_tensor)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    attention_scores = tf.einsum("BTNH,BFNH->BNFT", key_tensor, query_tensor)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(self.size_per_head)))

    if attention_mask is not None:
      # `attention_mask` = [B, 1, F, T]
      attention_mask = tf.expand_dims(attention_mask, axis=[1])

      # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -10000.0 for masked positions.
      adder = (1.0 - tf.cast(attention_mask, attention_scores.dtype)) * -10000.0

      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.attention_probs_dropout(attention_probs)

    # `context_layer` = [B, F, N, H]
    context_tensor = tf.einsum("BNFT,BTNH->BFNH", attention_probs, value_tensor)

    return context_tensor

  def _projection_dense_layer(self, name):
    """A helper to define a projection layer."""
    return Dense3D(
        num_attention_heads=self.num_attention_heads,
        size_per_head=self.size_per_head,
        kernel_initializer=get_initializer(self.initializer_range),
        output_projection=False,
        backward_compatible=self.backward_compatible,
        name=name)


class Dense3D(tf.keras.layers.Layer):
  """A Dense Layer using 3D kernel with tf.einsum implementation.

  Attributes:
    num_attention_heads: An integer, number of attention heads for each
      multihead attention layer.
    size_per_head: An integer, hidden size per attention head.
    hidden_size: An integer, dimension of the hidden layer.
    kernel_initializer: An initializer for the kernel weight.
    bias_initializer: An initializer for the bias.
    activation: An activation function to use. If nothing is specified, no
      activation is applied.
    use_bias: A bool, whether the layer uses a bias.
    output_projection: A bool, whether the Dense3D layer is used for output
      linear projection.
    backward_compatible: A bool, whether the variables shape are compatible with
      checkpoints converted from TF 1.x.
  """

  def __init__(self,
               num_attention_heads=12,
               size_per_head=72,
               kernel_initializer=None,
               bias_initializer="zeros",
               activation=None,
               use_bias=True,
               output_projection=False,
               backward_compatible=False,
               **kwargs):
    """Inits Dense3D."""
    super(Dense3D, self).__init__(**kwargs)
    self.num_attention_heads = num_attention_heads
    self.size_per_head = size_per_head
    self.hidden_size = num_attention_heads * size_per_head
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.activation = activation
    self.use_bias = use_bias
    self.output_projection = output_projection
    self.backward_compatible = backward_compatible

  @property
  def compatible_kernel_shape(self):
    if self.output_projection:
      return [self.hidden_size, self.hidden_size]
    return [self.last_dim, self.hidden_size]

  @property
  def compatible_bias_shape(self):
    return [self.hidden_size]

  @property
  def kernel_shape(self):
    if self.output_projection:
      return [self.num_attention_heads, self.size_per_head, self.hidden_size]
    return [self.last_dim, self.num_attention_heads, self.size_per_head]

  @property
  def bias_shape(self):
    if self.output_projection:
      return [self.hidden_size]
    return [self.num_attention_heads, self.size_per_head]

  def build(self, input_shape):
    """Implements build() for the layer."""
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError("Unable to build `Dense3D` layer with non-floating "
                      "point (and non-complex) dtype %s" % (dtype,))
    input_shape = tf.TensorShape(input_shape)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError("The last dimension of the inputs to `Dense3D` "
                       "should be defined. Found `None`.")
    self.last_dim = tf.compat.dimension_value(input_shape[-1])
    self.input_spec = tf.keras.layers.InputSpec(
        min_ndim=3, axes={-1: self.last_dim})
    # Determines variable shapes.
    if self.backward_compatible:
      kernel_shape = self.compatible_kernel_shape
      bias_shape = self.compatible_bias_shape
    else:
      kernel_shape = self.kernel_shape
      bias_shape = self.bias_shape

    self.kernel = self.add_weight(
        "kernel",
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        dtype=self.dtype,
        trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(
          "bias",
          shape=bias_shape,
          initializer=self.bias_initializer,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    super(Dense3D, self).build(input_shape)

  def call(self, inputs):
    """Implements ``call()`` for Dense3D.

    Args:
      inputs: A float tensor of shape [batch_size, sequence_length, hidden_size]
        when output_projection is False, otherwise a float tensor of shape
        [batch_size, sequence_length, num_heads, dim_per_head].

    Returns:
      The projected tensor with shape [batch_size, sequence_length, num_heads,
        dim_per_head] when output_projection is False, otherwise [batch_size,
        sequence_length, hidden_size].
    """
    if self.backward_compatible:
      kernel = tf.keras.backend.reshape(self.kernel, self.kernel_shape)
      bias = (
          tf.keras.backend.reshape(self.bias, self.bias_shape)
          if self.use_bias else None)
    else:
      kernel = self.kernel
      bias = self.bias

    if self.output_projection:
      ret = tf.einsum("abcd,cde->abe", inputs, kernel)
    else:
      ret = tf.einsum("abc,cde->abde", inputs, kernel)
    if self.use_bias:
      ret += bias
    if self.activation is not None:
      return self.activation(ret)
    return ret


class Dense2DProjection(tf.keras.layers.Layer):
  """A 2D projection layer with tf.einsum implementation."""

  def __init__(self,
               output_size,
               kernel_initializer=None,
               bias_initializer="zeros",
               activation=None,
               fp32_activation=False,
               **kwargs):
    super(Dense2DProjection, self).__init__(**kwargs)
    self.output_size = output_size
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.activation = activation
    self.fp32_activation = fp32_activation

  def build(self, input_shape):
    """Implements build() for the layer."""
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError("Unable to build `Dense2DProjection` layer with "
                      "non-floating point (and non-complex) "
                      "dtype %s" % (dtype,))
    input_shape = tf.TensorShape(input_shape)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError("The last dimension of the inputs to "
                       "`Dense2DProjection` should be defined. "
                       "Found `None`.")
    last_dim = tf.compat.dimension_value(input_shape[-1])
    self.input_spec = tf.keras.layers.InputSpec(min_ndim=3, axes={-1: last_dim})
    self.kernel = self.add_weight(
        "kernel",
        shape=[last_dim, self.output_size],
        initializer=self.kernel_initializer,
        dtype=self.dtype,
        trainable=True)
    self.bias = self.add_weight(
        "bias",
        shape=[self.output_size],
        initializer=self.bias_initializer,
        dtype=self.dtype,
        trainable=True)
    super(Dense2DProjection, self).build(input_shape)

  def call(self, inputs):
    """Implements call() for Dense2DProjection.

    Args:
      inputs: float Tensor of shape [batch, from_seq_length,
        num_attention_heads, size_per_head].

    Returns:
      A 3D Tensor.
    """
    ret = tf.einsum("abc,cd->abd", inputs, self.kernel)
    ret += self.bias
    if self.activation is not None:
      if self.dtype == tf.float16 and self.fp32_activation:
        ret = tf.cast(ret, tf.float32)
      return self.activation(ret)
    return ret


class TransformerBlock(tf.keras.layers.Layer):
  """Single transformer layer.

  It has two sub-layers. The first is a multi-head self-attention mechanism, and
  the second is a positionwise fully connected feed-forward network.
  """

  def __init__(self,
               hidden_size=768,
               num_attention_heads=12,
               intermediate_size=3072,
               intermediate_activation="gelu",
               hidden_dropout_prob=0.0,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               backward_compatible=False,
               float_type=tf.float32,
               **kwargs):
    super(TransformerBlock, self).__init__(**kwargs)
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.intermediate_activation = tf_utils.get_activation(
        intermediate_activation)
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.backward_compatible = backward_compatible
    self.float_type = float_type

    if self.hidden_size % self.num_attention_heads != 0:
      raise ValueError(
          "The hidden size (%d) is not a multiple of the number of attention "
          "heads (%d)" % (self.hidden_size, self.num_attention_heads))
    self.attention_head_size = int(self.hidden_size / self.num_attention_heads)

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.attention_layer = Attention(
        num_attention_heads=self.num_attention_heads,
        size_per_head=self.attention_head_size,
        attention_probs_dropout_prob=self.attention_probs_dropout_prob,
        initializer_range=self.initializer_range,
        backward_compatible=self.backward_compatible,
        name="self_attention")
    self.attention_output_dense = Dense3D(
        num_attention_heads=self.num_attention_heads,
        size_per_head=int(self.hidden_size / self.num_attention_heads),
        kernel_initializer=get_initializer(self.initializer_range),
        output_projection=True,
        backward_compatible=self.backward_compatible,
        name="self_attention_output")
    self.attention_dropout = tf.keras.layers.Dropout(
        rate=self.hidden_dropout_prob)
    self.attention_layer_norm = (
        tf.keras.layers.LayerNormalization(
            name="self_attention_layer_norm",
            axis=-1,
            epsilon=1e-12,
            # We do layer norm in float32 for numeric stability.
            dtype=tf.float32))
    self.intermediate_dense = Dense2DProjection(
        output_size=self.intermediate_size,
        kernel_initializer=get_initializer(self.initializer_range),
        activation=self.intermediate_activation,
        # Uses float32 so that gelu activation is done in float32.
        fp32_activation=True,
        name="intermediate")
    self.output_dense = Dense2DProjection(
        output_size=self.hidden_size,
        kernel_initializer=get_initializer(self.initializer_range),
        name="output")
    self.output_dropout = tf.keras.layers.Dropout(rate=self.hidden_dropout_prob)
    self.output_layer_norm = tf.keras.layers.LayerNormalization(
        name="output_layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32)
    super(TransformerBlock, self).build(unused_input_shapes)

  def common_layers(self):
    """Explicitly gets all layer objects inside a Transformer encoder block."""
    return [
        self.attention_layer, self.attention_output_dense,
        self.attention_dropout, self.attention_layer_norm,
        self.intermediate_dense, self.output_dense, self.output_dropout,
        self.output_layer_norm
    ]

  def __call__(self, input_tensor, attention_mask=None, **kwargs):
    inputs = tf_utils.pack_inputs([input_tensor, attention_mask])
    return super(TransformerBlock, self).__call__(inputs, **kwargs)

  def call(self, inputs):
    """Implements call() for the layer."""
    (input_tensor, attention_mask) = tf_utils.unpack_inputs(inputs)
    attention_output = self.attention_layer(
        from_tensor=input_tensor,
        to_tensor=input_tensor,
        attention_mask=attention_mask)
    attention_output = self.attention_output_dense(attention_output)
    attention_output = self.attention_dropout(attention_output)
    # Use float32 in keras layer norm and the gelu activation in the
    # intermediate dense layer for numeric stability
    attention_output = self.attention_layer_norm(input_tensor +
                                                 attention_output)
    if self.float_type == tf.float16:
      attention_output = tf.cast(attention_output, tf.float16)
    intermediate_output = self.intermediate_dense(attention_output)
    if self.float_type == tf.float16:
      intermediate_output = tf.cast(intermediate_output, tf.float16)
    layer_output = self.output_dense(intermediate_output)
    layer_output = self.output_dropout(layer_output)
    # Use float32 in keras layer norm for numeric stability
    layer_output = self.output_layer_norm(layer_output + attention_output)
    if self.float_type == tf.float16:
      layer_output = tf.cast(layer_output, tf.float16)
    return layer_output


class Transformer(tf.keras.layers.Layer):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
  """

  def __init__(self,
               num_hidden_layers=12,
               hidden_size=768,
               num_attention_heads=12,
               intermediate_size=3072,
               intermediate_activation="gelu",
               hidden_dropout_prob=0.0,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               backward_compatible=False,
               float_type=tf.float32,
               **kwargs):
    super(Transformer, self).__init__(**kwargs)
    self.num_hidden_layers = num_hidden_layers
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.intermediate_activation = tf_utils.get_activation(
        intermediate_activation)
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.backward_compatible = backward_compatible
    self.float_type = float_type

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.layers = []
    for i in range(self.num_hidden_layers):
      self.layers.append(
          TransformerBlock(
              hidden_size=self.hidden_size,
              num_attention_heads=self.num_attention_heads,
              intermediate_size=self.intermediate_size,
              intermediate_activation=self.intermediate_activation,
              hidden_dropout_prob=self.hidden_dropout_prob,
              attention_probs_dropout_prob=self.attention_probs_dropout_prob,
              initializer_range=self.initializer_range,
              backward_compatible=self.backward_compatible,
              float_type=self.float_type,
              name=("layer_%d" % i)))
    super(Transformer, self).build(unused_input_shapes)

  def __call__(self, input_tensor, attention_mask=None, **kwargs):
    inputs = tf_utils.pack_inputs([input_tensor, attention_mask])
    return super(Transformer, self).__call__(inputs=inputs, **kwargs)

  def call(self, inputs, return_all_layers=False):
    """Implements call() for the layer.

    Args:
      inputs: packed inputs.
      return_all_layers: bool, whether to return outputs of all layers inside
        encoders.

    Returns:
      Output tensor of the last layer or a list of output tensors.
    """
    unpacked_inputs = tf_utils.unpack_inputs(inputs)
    input_tensor = unpacked_inputs[0]
    attention_mask = unpacked_inputs[1]
    output_tensor = input_tensor

    all_layer_outputs = []
    for layer in self.layers:
      output_tensor = layer(output_tensor, attention_mask)
      all_layer_outputs.append(output_tensor)

    if return_all_layers:
      return all_layer_outputs

    return all_layer_outputs[-1]


def get_initializer(initializer_range=0.02):
  """Creates a `tf.initializers.truncated_normal` with the given range.

  Args:
    initializer_range: float, initializer range for stddev.

  Returns:
    TruncatedNormal initializer with stddev = `initializer_range`.
  """
  return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def create_attention_mask_from_input_mask(from_tensor, to_mask):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].

  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
  from_shape = tf_utils.get_shape_list(from_tensor, expected_rank=[2, 3])
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = tf_utils.get_shape_list(to_mask, expected_rank=2)
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]),
      dtype=from_tensor.dtype)

  # We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  #
  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=from_tensor.dtype)

  # Here we broadcast along two dimensions to create the mask.
  mask = broadcast_ones * to_mask

  return mask

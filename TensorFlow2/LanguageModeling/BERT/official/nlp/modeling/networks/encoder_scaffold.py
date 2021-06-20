# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
"""Transformer-based text encoder network."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import inspect
import tensorflow as tf

from official.nlp.modeling import layers


@tf.keras.utils.register_keras_serializable(package='Text')
class EncoderScaffold(tf.keras.Model):
  """Bi-directional Transformer-based encoder network scaffold.

  This network allows users to flexibly implement an encoder similar to the one
  described in "BERT: Pre-training of Deep Bidirectional Transformers for
  Language Understanding" (https://arxiv.org/abs/1810.04805).

  In this network, users can choose to provide a custom embedding subnetwork
  (which will replace the standard embedding logic) and/or a custom hidden layer
  class (which will replace the Transformer instantiation in the encoder). For
  each of these custom injection points, users can pass either a class or a
  class instance. If a class is passed, that class will be instantiated using
  the 'embedding_cfg' or 'hidden_cfg' argument, respectively; if an instance
  is passed, that instance will be invoked. (In the case of hidden_cls, the
  instance will be invoked 'num_hidden_instances' times.

  If the hidden_cls is not overridden, a default transformer layer will be
  instantiated.

  Attributes:
    num_output_classes: The output size of the classification layer.
    classification_layer_initializer: The initializer for the classification
      layer.
    classification_layer_dtype: The dtype for the classification layer.
    embedding_cls: The class or instance to use to embed the input data. This
      class or instance defines the inputs to this encoder. If embedding_cls is
      not set, a default embedding network (from the original BERT paper) will
      be created.
    embedding_cfg: A dict of kwargs to pass to the embedding_cls, if it needs to
      be instantiated. If embedding_cls is not set, a config dict must be
      passed to 'embedding_cfg' with the following values:
      "vocab_size": The size of the token vocabulary.
      "type_vocab_size": The size of the type vocabulary.
      "hidden_size": The hidden size for this encoder.
      "max_seq_length": The maximum sequence length for this encoder.
      "seq_length": The sequence length for this encoder.
      "initializer": The initializer for the embedding portion of this encoder.
      "dropout_rate": The dropout rate to apply before the encoding layers.
      "dtype": (Optional): The dtype of the embedding layers.
    embedding_data: A reference to the embedding weights that will be used to
      train the masked language model, if necessary. This is optional, and only
      needed if (1) you are overriding embedding_cls and (2) are doing standard
      pretraining.
    num_hidden_instances: The number of times to instantiate and/or invoke the
      hidden_cls.
    hidden_cls: The class or instance to encode the input data. If hidden_cls is
      not set, a KerasBERT transformer layer will be used as the encoder class.
    hidden_cfg: A dict of kwargs to pass to the hidden_cls, if it needs to be
      instantiated. If hidden_cls is not set, a config dict must be passed to
      'hidden_cfg' with the following values:
        "num_attention_heads": The number of attention heads. The hidden size
          must be divisible by num_attention_heads.
        "intermediate_size": The intermediate size of the transformer.
        "intermediate_activation": The activation to apply in the transfomer.
        "dropout_rate": The overall dropout rate for the transformer layers.
        "attention_dropout_rate": The dropout rate for the attention layers.
        "kernel_initializer": The initializer for the transformer layers.
        "dtype": The dtype of the transformer.
  """

  def __init__(
      self,
      num_output_classes,
      classification_layer_initializer=tf.keras.initializers.TruncatedNormal(
          stddev=0.02),
      classification_layer_dtype=tf.float32,
      embedding_cls=None,
      embedding_cfg=None,
      embedding_data=None,
      num_hidden_instances=1,
      hidden_cls=layers.Transformer,
      hidden_cfg=None,
      **kwargs):
    print(embedding_cfg)
    self._self_setattr_tracking = False
    self._hidden_cls = hidden_cls
    self._hidden_cfg = hidden_cfg
    self._num_hidden_instances = num_hidden_instances
    self._num_output_classes = num_output_classes
    self._classification_layer_initializer = classification_layer_initializer
    self._embedding_cls = embedding_cls
    self._embedding_cfg = embedding_cfg
    self._embedding_data = embedding_data
    self._kwargs = kwargs

    if embedding_cls:
      if inspect.isclass(embedding_cls):
        self._embedding_network = embedding_cls(embedding_cfg)
      else:
        self._embedding_network = embedding_cls
      inputs = self._embedding_network.inputs
      embeddings, mask = self._embedding_network(inputs)
    else:
      self._embedding_network = None
      word_ids = tf.keras.layers.Input(
          shape=(embedding_cfg['seq_length'],),
          dtype=tf.int32,
          name='input_word_ids')
      mask = tf.keras.layers.Input(
          shape=(embedding_cfg['seq_length'],),
          dtype=tf.int32,
          name='input_mask')
      type_ids = tf.keras.layers.Input(
          shape=(embedding_cfg['seq_length'],),
          dtype=tf.int32,
          name='input_type_ids')
      inputs = [word_ids, mask, type_ids]

      self._embedding_layer = layers.OnDeviceEmbedding(
          vocab_size=embedding_cfg['vocab_size'],
          embedding_width=embedding_cfg['hidden_size'],
          initializer=embedding_cfg['initializer'],
          name='word_embeddings')

      word_embeddings = self._embedding_layer(word_ids)

      # Always uses dynamic slicing for simplicity.
      self._position_embedding_layer = layers.PositionEmbedding(
          initializer=embedding_cfg['initializer'],
          use_dynamic_slicing=True,
          max_sequence_length=embedding_cfg['max_seq_length'])
      position_embeddings = self._position_embedding_layer(word_embeddings)

      type_embeddings = (
          layers.OnDeviceEmbedding(
              vocab_size=embedding_cfg['type_vocab_size'],
              embedding_width=embedding_cfg['hidden_size'],
              initializer=embedding_cfg['initializer'],
              use_one_hot=True,
              name='type_embeddings')(type_ids))

      embeddings = tf.keras.layers.Add()(
          [word_embeddings, position_embeddings, type_embeddings])
      embeddings = (
          tf.keras.layers.LayerNormalization(
              name='embeddings/layer_norm',
              axis=-1,
              epsilon=1e-12,
              dtype=tf.float32)(embeddings))
      embeddings = (
          tf.keras.layers.Dropout(
              rate=embedding_cfg['dropout_rate'], dtype=tf.float32)(embeddings))

      if embedding_cfg.get('dtype') == 'float16':
        embeddings = tf.cast(embeddings, tf.float16)

    attention_mask = layers.SelfAttentionMask()([embeddings, mask])
    data = embeddings

    for _ in range(num_hidden_instances):
      if inspect.isclass(hidden_cls):
        layer = self._hidden_cls(**hidden_cfg)
      else:
        layer = self._hidden_cls
      data = layer([data, attention_mask])

    first_token_tensor = (
        tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(data)
    )
    cls_output = tf.keras.layers.Dense(
        units=num_output_classes,
        activation='tanh',
        kernel_initializer=classification_layer_initializer,
        dtype=classification_layer_dtype,
        name='cls_transform')(
            first_token_tensor)

    super(EncoderScaffold, self).__init__(
        inputs=inputs, outputs=[data, cls_output], **kwargs)

  def get_config(self):
    config_dict = {
        'num_hidden_instances':
            self._num_hidden_instances,
        'num_output_classes':
            self._num_output_classes,
        'classification_layer_initializer':
            self._classification_layer_initializer,
        'embedding_cls':
            self._embedding_network,
        'embedding_cfg':
            self._embedding_cfg,
        'hidden_cfg':
            self._hidden_cfg,
    }
    if inspect.isclass(self._hidden_cls):
      config_dict['hidden_cls_string'] = tf.keras.utils.get_registered_name(
          self._hidden_cls)
    else:
      config_dict['hidden_cls'] = self._hidden_cls

    config_dict.update(self._kwargs)
    return config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    if 'hidden_cls_string' in config:
      config['hidden_cls'] = tf.keras.utils.get_registered_object(
          config['hidden_cls_string'], custom_objects=custom_objects)
      del config['hidden_cls_string']
    return cls(**config)

  def get_embedding_table(self):
    if self._embedding_network is None:
      # In this case, we don't have a custom embedding network and can return
      # the standard embedding data.
      return self._embedding_layer.embeddings

    if self._embedding_data is None:
      raise RuntimeError(('The EncoderScaffold %s does not have a reference '
                          'to the embedding data. This is required when you '
                          'pass a custom embedding network to the scaffold. '
                          'It is also possible that you are trying to get '
                          'embedding data from an embedding scaffold with a '
                          'custom embedding network where the scaffold has '
                          'been serialized and deserialized. Unfortunately, '
                          'accessing custom embedding references after '
                          'serialization is not yet supported.') % self.name)
    else:
      return self._embedding_data

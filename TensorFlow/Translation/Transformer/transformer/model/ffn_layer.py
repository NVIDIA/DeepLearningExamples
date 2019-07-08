# Copyright 2018 MLBenchmark Group. All Rights Reserved.
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
"""Implementation of fully connected network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class FeedFowardNetwork(tf.layers.Layer):
  """Fully connected feedforward network."""

  def __init__(self, params, train):
    super(FeedFowardNetwork, self).__init__()
    self.params = params
    self.train = train

    use_bias = True
    self.filter_dense_layer = tf.layers.Dense(
        self.params.filter_size, use_bias=use_bias, activation=tf.nn.relu, name="filter_layer")
    self.output_dense_layer = tf.layers.Dense(
        self.params.hidden_size, use_bias=use_bias, name="output_layer")

  def call(self, x, padding=None):
    # Retrieve dynamically known shapes
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]

    if padding is not None:
      with tf.name_scope("remove_padding"):
        # Flatten padding to [batch_size*length]
        pad_mask = tf.reshape(padding, [-1])

        nonpad_ids = tf.cast(tf.where(pad_mask < 1e-9), tf.int32)

        # Reshape x to [batch_size*length, hidden_size] to remove padding
        x = tf.reshape(x, [-1, self.params.hidden_size])
        x = tf.gather_nd(x, indices=nonpad_ids)

        # Reshape x from 2 dimensions to 3 dimensions.
        x.set_shape([None, self.params.hidden_size])
        x = tf.expand_dims(x, axis=0)

    output = self.filter_dense_layer(x)
    if self.train:
      output = tf.nn.dropout(output, rate=self.params.relu_dropout)
    output = self.output_dense_layer(output)

    if padding is not None:
      with tf.name_scope("re_add_padding"):
        output = tf.squeeze(output, axis=0)
        output = tf.scatter_nd(
            indices=nonpad_ids,
            updates=output,
            shape=[batch_size * length, self.params.hidden_size]
        )
        output = tf.reshape(output, [batch_size, length, self.params.hidden_size])
    return output


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

import math
import abc
import tensorflow as tf
from utils.reducer import SumReducer

class PositionEncoder(tf.keras.layers.Layer):
    """Base class for position encoders."""

    def __init__(self, reducer=SumReducer(), **kwargs):
        """Initializes the position encoder.
        Args:
          reducer: A :class:`opennmt.layers.Reducer` to merge inputs and position
            encodings.
          **kwargs: Additional layer keyword arguments.
        """
        # super(PositionEncoder, self).__init__(**kwargs)
        super(PositionEncoder, self).__init__(**kwargs)
        self.reducer = reducer

    def call(self, inputs, position=None):  # pylint: disable=arguments-differ
        """Add position encodings to :obj:`inputs`.
        Args:
          inputs: The inputs to encode.
          position: The single position to encode, to use when this layer is called
            step by step.
        Returns:
          A ``tf.Tensor`` whose shape depends on the configured ``reducer``.
        """
        batch_size = tf.shape(inputs)[0]
        timesteps = tf.shape(inputs)[1]
        input_dim = inputs.get_shape().as_list()[-1] # return int 
        positions = tf.range(timesteps) + 1 if position is None else position
        position_encoding = self._encode([positions], input_dim, dtype=inputs.dtype)
        position_encoding = tf.tile(position_encoding, [batch_size, 1, 1])
        return self.reducer([inputs, position_encoding])

    @abc.abstractmethod
    def _encode(self, positions, depth, dtype):
        """Creates position encodings.
        Args:
          positions: The positions to encode of shape :math:`[B, ...]`.
          depth: The encoding depth :math:`D`.
        Returns:
          A ``tf.Tensor`` of shape :math:`[B, ..., D]`.
        """
        raise NotImplementedError()
      
    def _create_position_encoding_table(self, max_seq_len, input_dim, dtype):
      positions = tf.range(max_seq_len) + 1
      self.position_encoding_table = self._encode([positions], input_dim, dtype=dtype)
      self.position_encoding_table = tf.squeeze(self.position_encoding_table)
      return self.position_encoding_table


class SinusoidalPositionEncoder(PositionEncoder):
    """Encodes positions with sine waves as described in
    https://arxiv.org/abs/1706.03762.
    """

    def _encode(self, positions, depth, dtype):
        if depth % 2 != 0:
            raise ValueError("SinusoidalPositionEncoder expects the depth to be divisble "
                             "by 2 but got %d" % depth)

        batch_size = tf.shape(positions)[0]
        positions = tf.cast(positions, tf.float32)

        log_timescale_increment = math.log(10000) / (depth / 2 - 1)
        inv_timescales = tf.exp(
            tf.cast(tf.range(depth / 2), dtype=tf.float32) * -log_timescale_increment)
        inv_timescales = tf.reshape(
            tf.tile(inv_timescales, [batch_size]), [batch_size, -1])
        scaled_time = tf.expand_dims(
            positions, -1) * tf.expand_dims(inv_timescales, 1)
        encoding = tf.concat(
            [tf.sin(scaled_time), tf.cos(scaled_time)], axis=2)
        return tf.cast(encoding, dtype)

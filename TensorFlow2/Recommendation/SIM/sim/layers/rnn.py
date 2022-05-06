# Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf


class VecAttGRUCell(tf.keras.layers.Layer):
    """
    Modification of Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

    Args:
    units: int, The number of units in the GRU cell.
    """

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units

        self._activation = tf.math.tanh

        self._gate_linear = tf.keras.layers.Dense(
            2 * self.units,
            bias_initializer=tf.constant_initializer(1.0),
            kernel_initializer=None,
        )

        self._candidate_linear = tf.keras.layers.Dense(
            self.units,
            bias_initializer=tf.constant_initializer(0.0),
            kernel_initializer=None,
        )

        super(VecAttGRUCell, self).__init__(**kwargs)

    def call(self, inputs_attscore, states):
        """Gated recurrent unit (GRU) with nunits cells."""

        inputs, att_score = inputs_attscore
        state = states[0]

        value = tf.math.sigmoid(self._gate_linear(tf.concat([inputs, state], axis=-1)))
        r, u = tf.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        c = self._activation(
            self._candidate_linear(tf.concat([inputs, r_state], axis=-1))
        )
        u = (1.0 - att_score) * u
        new_h = u * state + (1 - u) * c
        return new_h, [new_h]


class AUGRU(tf.keras.layers.Layer):
    def __init__(self, num_units=None, return_sequence=True, **kwargs):
        self.num_units = num_units
        self.return_sequence = return_sequence
        super(AUGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.internal_rnn = tf.keras.layers.RNN(VecAttGRUCell(self.num_units))
        # Be sure to call this somewhere!
        super(AUGRU, self).build(input_shape)

    def call(self, input_list):
        """
        :param concated_embeds_value: None * field_size * embedding_size
        :return: None*1
        """
        return self.internal_rnn(tuple(input_list))

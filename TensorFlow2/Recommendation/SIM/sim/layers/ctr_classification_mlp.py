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

from functools import partial

import tensorflow as tf


class CTRClassificationMLP(tf.keras.layers.Layer):
    def __init__(
        self,
        layer_sizes=(200,),
        num_outputs=1,
        activation_function=partial(
            tf.keras.layers.PReLU, alpha_initializer=tf.keras.initializers.Constant(0.1)
        ),
        use_bn=False,
        dropout_rate=-1
    ):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.activation_function = activation_function
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate

        if self.use_bn:
            self.batch_norm = tf.keras.layers.BatchNormalization()
        self.layers = []
        for layer_size in self.layer_sizes:
            # add dense layer and activation
            self.layers.append(tf.keras.layers.Dense(layer_size))
            self.layers.append(self.activation_function())

        if self.dropout_rate > 0.0:
            # add dropout between final representation and classification layer
            self.layers.append(tf.keras.layers.Dropout(rate=self.dropout_rate))
        # add the scoring layer
        scoring_layer = tf.keras.layers.Dense(num_outputs, dtype='float32')
        self.layers.append(scoring_layer)

    def call(self, input, training=False):
        if self.use_bn:
            input = self.batch_norm(input, training=training)
        for layer in self.layers:
            input = layer(input, training=training)
        return input

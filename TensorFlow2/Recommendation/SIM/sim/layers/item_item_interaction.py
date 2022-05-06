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


class DotItemItemInteraction(tf.keras.layers.Layer):
    @tf.function
    def call(self, inputs):
        item1, item2 = inputs
        return tf.reduce_sum(item1 * item2, axis=-1)


class DINActivationUnit(tf.keras.layers.Layer):
    def __init__(self):
        super(DINActivationUnit, self).__init__()
        self.dense1 = tf.keras.layers.Dense(80, activation="sigmoid")
        self.dense2 = tf.keras.layers.Dense(40, activation="sigmoid")
        self.linear = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, inputs):
        targets, item = inputs
        items = tf.tile(item, [1, targets.shape[1], 1])

        combined = tf.concat(
            [items, targets, items - targets, items * targets], axis=-1
        )
        output = self.dense1(combined)
        output = self.dense2(output)
        output = self.linear(output)
        # (B, T, 1) -> (B, T)
        output = tf.squeeze(output)
        return output


class DIENAttentionUnit(tf.keras.layers.Layer):
    def __init__(self, embedding_dim):
        """
        NOTE(alexo): this looks very similar to DINActivationUnit.
        Besides the input item adaptation, the remaining part stays the same.
        """
        super(DIENAttentionUnit, self).__init__()
        # Adaptation of input item
        self.item_dense = tf.keras.layers.Dense(embedding_dim)
        self.item_prelu = tf.keras.layers.PReLU(
            alpha_initializer=tf.keras.initializers.Constant(0.1)
        )
        #
        self.dense1 = tf.keras.layers.Dense(80, activation="sigmoid")
        self.dense2 = tf.keras.layers.Dense(40, activation="sigmoid")
        self.linear = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, inputs):
        targets, item = inputs
        item = self.item_dense(item)
        item = self.item_prelu(item)

        items = tf.tile(item, [1, targets.shape[1], 1])

        combined = tf.concat(
            [items, targets, items - targets, items * targets], axis=-1
        )
        output = self.dense1(combined)
        output = self.dense2(output)
        output = self.linear(output)  # unnormalized scores
        # (B, T, 1) -> (B, T)
        output = tf.squeeze(output)
        return output

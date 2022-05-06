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

from sim.layers.item_item_interaction import DIENAttentionUnit
from sim.layers.rnn import AUGRU


@tf.function
def compute_item_sequence_attention(item, sequence, mask, attention_op):
    """
    Computes normalized attention scores between a given sequence and item
    """
    scores_unnormalized = attention_op((sequence, tf.expand_dims(item, axis=1)))

    if mask is not None:
        min_value_for_dtype = scores_unnormalized.dtype.min
        mask = tf.equal(mask, tf.ones_like(mask))
        paddings = tf.ones_like(scores_unnormalized) * min_value_for_dtype
        scores_unnormalized = tf.where(mask, scores_unnormalized, paddings)  # [B, 1, T]

    scores = tf.nn.softmax(scores_unnormalized)

    return scores


class DINItemSequenceInteractionBlock(tf.keras.layers.Layer):
    def __init__(self, item_item_interaction):
        super(DINItemSequenceInteractionBlock, self).__init__()
        self.item_item_interaction = item_item_interaction

    @tf.function
    def call(self, inputs):
        item, item_sequence, mask = inputs
        # compute attention scores between item_sequence and item
        scores = compute_item_sequence_attention(
            item, item_sequence, mask, self.item_item_interaction
        )
        # equivalent to tf.matmul(scores[:,None,:], item_sequence)
        return (
            tf.reduce_sum(tf.expand_dims(scores, axis=-1) * item_sequence, [1]),
            scores,
        )


class DIENItemSequenceInteractionBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_size: int):
        super(DIENItemSequenceInteractionBlock, self).__init__()
        self.hidden_size = hidden_size  # hidden=emb_dim*6
        self.item_item_interaction = DIENAttentionUnit(self.hidden_size)

        self.layer_1 = tf.keras.layers.GRU(self.hidden_size, return_sequences=True)
        self.layer_2 = AUGRU(self.hidden_size)

    @tf.function
    def call(self, inputs):
        """
        Returns:
            - final_seq_repr: final vector representation of the sequence
            - features_layer_1: for auxiliary loss
        """
        item, item_sequence, mask = inputs
        # compute h(1),...,h(T) from e(1),...,e(T)
        features_layer_1 = self.layer_1(item_sequence)
        # compute attention scores between features_layer_1 and item
        attention_scores = compute_item_sequence_attention(
            item, features_layer_1, mask, self.item_item_interaction
        )
        attention_scores = tf.expand_dims(attention_scores, -1)
        # compute h'(T)
        final_seq_repr = self.layer_2([features_layer_1, attention_scores])
        # [B, 1, E] -> [B, E]
        final_seq_repr = tf.squeeze(final_seq_repr)
        return final_seq_repr, features_layer_1

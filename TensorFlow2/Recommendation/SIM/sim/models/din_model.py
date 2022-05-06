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

from sim.layers.item_item_interaction import DINActivationUnit, DotItemItemInteraction
from sim.layers.item_sequence_interaction import DINItemSequenceInteractionBlock
from sim.models.sequential_recommender_model import SequentialRecommenderModel


class DINModel(SequentialRecommenderModel):
    def __init__(
        self,
        feature_spec,
        mlp_hidden_dims=(200, 80),
        embedding_dim=4,
        item_item_interaction="dot",
    ):
        super(DINModel, self).__init__(
            feature_spec, embedding_dim, mlp_hidden_dims
        )
        if item_item_interaction == "dot":
            item_item_interaction_block = DotItemItemInteraction()
        elif item_item_interaction == "activation_unit":
            item_item_interaction_block = DINActivationUnit()

        self.item_seq_interaction = DINItemSequenceInteractionBlock(
            item_item_interaction=item_item_interaction_block
        )

    @tf.function
    def call(
        self,
        inputs,
        training=False
    ):
        user_features = inputs["user_features"]
        target_item_features = inputs["target_item_features"]
        long_sequence_features = inputs["long_sequence_features"]
        short_sequence_features = inputs["short_sequence_features"]
        long_sequence_mask = inputs["long_sequence_mask"]
        short_sequence_mask = inputs["short_sequence_mask"]

        user_embedding = self.embed(user_features)
        target_item_embedding = self.embed(target_item_features)
        long_sequence_embeddings = self.embed(long_sequence_features)
        short_sequence_embeddings = self.embed(short_sequence_features)

        # Concat over time axis
        sequence_embeddings = tf.concat([long_sequence_embeddings, short_sequence_embeddings], axis=1)
        mask = tf.concat([long_sequence_mask, short_sequence_mask], axis=1)

        sequence_embeddings = sequence_embeddings * tf.expand_dims(
            mask, axis=-1
        )

        item_sequence_interaction_embedding, _ = self.item_seq_interaction(
            (target_item_embedding, sequence_embeddings, mask)
        )

        combined_embeddings = tf.concat([
            target_item_embedding, item_sequence_interaction_embedding, user_embedding
        ], -1)

        logits = self.classificationMLP(combined_embeddings, training=training)

        return {"logits": logits}

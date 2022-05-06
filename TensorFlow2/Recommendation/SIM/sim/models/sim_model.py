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

from sim.layers.ctr_classification_mlp import CTRClassificationMLP
from sim.layers.item_item_interaction import DotItemItemInteraction
from sim.layers.item_sequence_interaction import DIENItemSequenceInteractionBlock, DINItemSequenceInteractionBlock
from sim.models.dien_model import compute_auxiliary_probs
from sim.models.sequential_recommender_model import SequentialRecommenderModel


@tf.function
def masked_temporal_mean(sequence_batch, mask):
    masked_sum = tf.reduce_sum(sequence_batch * mask[:, :, None], 1)
    masked_counts = tf.reduce_sum(mask, 1, keepdims=True)
    return masked_sum / (masked_counts + 1.0)


class SIMModel(SequentialRecommenderModel):
    def __init__(self, feature_spec, mlp_hidden_dims, embedding_dim=4, k=50, dropout_rate=-1):
        super(SIMModel, self).__init__(
            feature_spec, embedding_dim
        )
        self.k = k
        self.stage_one_classifier = CTRClassificationMLP(
            layer_sizes=mlp_hidden_dims["stage_1"],
            dropout_rate=dropout_rate
        )
        self.stage_two_classifier = CTRClassificationMLP(
            layer_sizes=mlp_hidden_dims["stage_2"],
            dropout_rate=dropout_rate
        )
        self.stage_two_auxiliary_net = CTRClassificationMLP(
            layer_sizes=mlp_hidden_dims["aux"],
            activation_function=partial(
                tf.keras.layers.Activation, activation="sigmoid"
            ),
            dropout_rate=dropout_rate
        )

        self.stage_one_item_seq_interaction = DINItemSequenceInteractionBlock(
            item_item_interaction=DotItemItemInteraction()
        )
        self.stage_two_item_seq_interaction = DIENItemSequenceInteractionBlock(
            hidden_size=embedding_dim * 6
        )

    def select_top_k_items(self, embeddings, scores):
        top_k = tf.math.top_k(scores, k=self.k)
        top_k_values, top_k_indices = top_k.values, top_k.indices
        top_k_mask = tf.cast(tf.greater(top_k_values, tf.zeros_like(top_k_values)), embeddings.dtype)
        best_k_embeddings = tf.gather(embeddings, top_k_indices, batch_dims=1)
        return best_k_embeddings, top_k_mask

    @tf.function
    def call(
            self,
            inputs,
            compute_aux_loss=True,
            training=False,
    ):
        user_features = inputs["user_features"]
        target_item_features = inputs["target_item_features"]
        long_sequence_features = inputs["long_sequence_features"]
        short_sequence_features = inputs["short_sequence_features"]
        short_neg_sequence_features = inputs["short_neg_sequence_features"]
        long_sequence_mask = inputs["long_sequence_mask"]
        short_sequence_mask = inputs["short_sequence_mask"]

        output_dict = {}

        # GSU Stage
        user_embedding = self.embed(user_features)
        target_item_embedding = self.embed(target_item_features)
        long_sequence_embeddings = self.embed(long_sequence_features)
        long_sequence_embeddings = long_sequence_embeddings * tf.expand_dims(
            long_sequence_mask, axis=-1
        )

        stage_one_interaction_embedding, gsu_scores = self.stage_one_item_seq_interaction(
            (target_item_embedding, long_sequence_embeddings, long_sequence_mask)
        )
        # combine all the stage 1 embeddings
        stage_one_embeddings = tf.concat(
            [target_item_embedding, stage_one_interaction_embedding, user_embedding], -1
        )
        stage_one_logits = self.stage_one_classifier(
            stage_one_embeddings, training=training
        )

        # ESU Stage
        user_embedding = self.embed(user_features)
        target_item_embedding = self.embed(target_item_features)
        short_sequence_embeddings = self.embed(short_sequence_features)
        short_sequence_embeddings = short_sequence_embeddings * tf.expand_dims(
            short_sequence_mask, axis=-1
        )

        # ---- Attention part
        # Take embeddings of k best items produced by GSU at Stage 1
        best_k_long_seq_embeddings, top_k_mask = self.select_top_k_items(
            long_sequence_embeddings, gsu_scores
        )
        # Run attention mechanism to produce a single representation
        att_fea, _ = self.stage_one_item_seq_interaction(
            (target_item_embedding, best_k_long_seq_embeddings, top_k_mask),
        )
        # Take a mean representation of best_k_long_seq_embeddings
        item_his_sum_emb = masked_temporal_mean(best_k_long_seq_embeddings, top_k_mask)
        # ---- DIEN part
        (
            stage_two_interaction_embedding,
            short_features_layer_1,
        ) = self.stage_two_item_seq_interaction(
            (target_item_embedding, short_sequence_embeddings, short_sequence_mask),
        )

        # Compute auxiliary logits for DIEN
        if compute_aux_loss:
            # Embed negative sequence features
            short_neg_sequence_embeddings = self.embed(short_neg_sequence_features)
            short_neg_sequence_embeddings = (
                short_neg_sequence_embeddings
                * tf.expand_dims(short_sequence_mask, axis=-1)
            )

            aux_click_probs = compute_auxiliary_probs(
                self.stage_two_auxiliary_net,
                short_features_layer_1,
                short_sequence_embeddings,
                training=training,
            )
            output_dict["aux_click_probs"] = aux_click_probs

            aux_noclick_probs = compute_auxiliary_probs(
                self.stage_two_auxiliary_net,
                short_features_layer_1,
                short_neg_sequence_embeddings,
                training=training,
            )
            output_dict["aux_noclick_probs"] = aux_noclick_probs

        # combine all the stage 2 embeddings
        stage_two_embeddings = tf.concat(
            [
                att_fea,
                item_his_sum_emb,
                target_item_embedding,
                stage_two_interaction_embedding,
                user_embedding
            ],
            -1,
        )

        stage_two_logits = self.stage_two_classifier(
            stage_two_embeddings, training=training
        )

        output_dict["stage_one_logits"] = stage_one_logits
        output_dict["stage_two_logits"] = stage_two_logits

        return output_dict

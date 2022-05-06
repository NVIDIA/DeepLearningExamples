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
from sim.layers.item_sequence_interaction import DIENItemSequenceInteractionBlock
from sim.models.sequential_recommender_model import SequentialRecommenderModel

EPS = 1e-06
DIEN_ITEM_SEQ_INTERACTION_SIZE = 6  # Value taken from TF1 original code


def compute_auxiliary_probs(auxiliary_net, rnn_states, items_hist, training=False):
    """
    Given h(1),..,h(T) GRU sequence outputs and e(1),..,e(T) encoded user
    sequence or negative user sequence behaviours, compute probabilities
    for auxiliary loss term.

    Args:
        auxiliary_net: model that computes a probability of interaction
        rnn_states: sequence of GRU outputs
        items_hist: sequence of user behaviours or negative user behaviours

    Returns:
        click_prob: clicking probability for each timestep
    """
    # for rnn_states, select h(1),..,h(T-1)
    rnn_states = rnn_states[:, :-1, :]
    # for items_hist, select e(2),..,e(T)
    items_hist = items_hist[:, 1:, :]
    # concatenate over feature dimension
    click_input = tf.concat([rnn_states, items_hist], -1)
    # forward pass
    click_logits = auxiliary_net(click_input, training=training)
    click_probs = tf.nn.sigmoid(click_logits) + EPS
    return tf.squeeze(click_probs, axis=2)


class DIENModel(SequentialRecommenderModel):
    def __init__(
        self,
        feature_spec,
        mlp_hidden_dims,
        embedding_dim=4
    ):
        super(DIENModel, self).__init__(
            feature_spec, embedding_dim, mlp_hidden_dims["classifier"]
        )
        # DIEN block
        self.dien_block = DIENItemSequenceInteractionBlock(
            hidden_size=embedding_dim * DIEN_ITEM_SEQ_INTERACTION_SIZE
        )
        # aux_loss uses an MLP in TF1 code
        self.auxiliary_net = CTRClassificationMLP(
            mlp_hidden_dims["aux"],
            activation_function=partial(
                tf.keras.layers.Activation, activation="sigmoid"
            ),
        )

    @tf.function
    def call(
        self,
        inputs,
        compute_aux_loss=True,
        training=False,
    ):
        user_features = inputs["user_features"]
        target_item_features = inputs["target_item_features"]
        short_sequence_features = inputs["short_sequence_features"]
        short_neg_sequence_features = inputs["short_neg_sequence_features"]
        short_sequence_mask = inputs["short_sequence_mask"]

        output_dict = {}

        user_embedding = self.embed(user_features)
        target_item_embedding = self.embed(target_item_features)
        short_sequence_embeddings = self.embed(short_sequence_features)

        short_sequence_embeddings = short_sequence_embeddings * tf.expand_dims(
            short_sequence_mask, axis=-1
        )

        # Pass sequence_embeddings and target_item_embedding to a DIEN block
        # it needs to output h'(T) for concatenation and h(1),...,h(T) for aux_loss
        final_seq_repr, features_layer_1 = self.dien_block(
            (target_item_embedding, short_sequence_embeddings, short_sequence_mask)
        )

        # short_features_layer_1 = features_layer_1[:, -short_seq_len:, :]

        if compute_aux_loss:
            # Embed negative sequence features
            short_neg_sequence_embeddings = self.embed(short_neg_sequence_features)
            short_neg_sequence_embeddings = short_neg_sequence_embeddings * tf.expand_dims(
                short_sequence_mask, axis=-1
            )

            # compute auxiliary logits
            aux_click_probs = compute_auxiliary_probs(
                self.auxiliary_net,
                features_layer_1,
                short_sequence_embeddings,
                training=training,
            )
            output_dict["aux_click_probs"] = aux_click_probs

            aux_noclick_probs = compute_auxiliary_probs(
                self.auxiliary_net,
                features_layer_1,
                short_neg_sequence_embeddings,
                training=training,
            )
            output_dict["aux_noclick_probs"] = aux_noclick_probs

        combined_embeddings = tf.concat([
            target_item_embedding,
            final_seq_repr,
            user_embedding
        ], -1)

        classification_logits = self.classificationMLP(combined_embeddings)

        output_dict["logits"] = classification_logits

        return output_dict

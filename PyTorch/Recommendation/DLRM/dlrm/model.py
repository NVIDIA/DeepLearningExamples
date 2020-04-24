# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
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


import copy
import json
import math

from absl import logging

import torch
from torch import nn
from typing import List


class Dlrm(nn.Module):
    """Reimplement Facebook's DLRM model

    Original implementation is from https://github.com/facebookresearch/dlrm.

    """

    def __init__(self, num_numerical_features, categorical_feature_sizes, bottom_mlp_sizes, top_mlp_sizes,
                     embedding_dim=32, interaction_op="dot", self_interaction=False, hash_indices=False,
                     base_device="cuda", sigmoid=False):

        # Running everything on gpu by default
        self._base_device = base_device
        self._embedding_device_map = [base_device for _ in range(len(categorical_feature_sizes))]

        super(Dlrm, self).__init__()

        if embedding_dim != bottom_mlp_sizes[-1]:
            raise TypeError("The last bottom MLP layer must have same size as embedding.")

        self._embedding_dim = embedding_dim
        self._interaction_op = interaction_op
        self._self_interaction = self_interaction
        self._hash_indices = hash_indices
        self._categorical_feature_sizes = copy.copy(categorical_feature_sizes)

        # Interactions are among outputs of all the embedding tables and bottom MLP, total number of
        # (num_embedding_tables + 1) vectors with size embdding_dim. ``dot`` product interaction computes dot product
        # between any 2 vectors. ``cat`` interaction concatenate all the vectors together.
        # Output of interaction will have shape [num_interactions, embdding_dim].
        self._num_interaction_inputs = len(categorical_feature_sizes) + 1
        if interaction_op == "dot":
            if self_interaction:
                raise NotImplementedError
            num_interactions = (self._num_interaction_inputs * (self._num_interaction_inputs - 1)) // 2 + embedding_dim
        elif interaction_op == "cat":
            num_interactions = self._num_interaction_inputs * embedding_dim
        else:
            raise TypeError(F"Unknown interaction {interaction_op}.")

        self.embeddings = nn.ModuleList()
        self._create_embeddings(self.embeddings, embedding_dim, categorical_feature_sizes)

        # Create bottom MLP
        bottom_mlp_layers = []
        input_dims = num_numerical_features
        for output_dims in bottom_mlp_sizes:
            bottom_mlp_layers.append(
                nn.Linear(input_dims, output_dims))
            bottom_mlp_layers.append(nn.ReLU(inplace=True))
            input_dims = output_dims
        self.bottom_mlp = nn.Sequential(*bottom_mlp_layers)

        # Create Top MLP
        top_mlp_layers = []

        input_dims = num_interactions
        if self._interaction_op == 'dot':
            input_dims += 1  # pad 1 to be multiple of 8

        for output_dims in top_mlp_sizes[:-1]:
            top_mlp_layers.append(nn.Linear(input_dims, output_dims))
            top_mlp_layers.append(nn.ReLU(inplace=True))
            input_dims = output_dims
        # last Linear layer uses sigmoid
        top_mlp_layers.append(nn.Linear(input_dims, top_mlp_sizes[-1]))
        if sigmoid:
            top_mlp_layers.append(nn.Sigmoid())
        self.top_mlp = nn.Sequential(*top_mlp_layers)

        self._initialize_mlp_weights()
        self._interaction_padding = torch.zeros(1, 1, dtype=torch.float32)
        self.tril_indices = torch.tensor([[i for i in range(len(self.embeddings) + 1) 
                                             for j in range(i + int(self_interaction))],
                                          [j for i in range(len(self.embeddings) + 1) 
                                             for j in range(i + int(self_interaction))]])

    def _interaction(self, 
            bottom_mlp_output: torch.Tensor, 
            embedding_outputs: List[torch.Tensor], 
            batch_size: int) -> torch.Tensor:
        """Interaction

        "dot" interaction is a bit tricky to implement and test. Break it out from forward so that it can be tested
        independently.

        Args:
            bottom_mlp_output (Tensor):
            embedding_outputs (list): Sequence of tensors
            batch_size (int):
        """
        if self._interaction_padding is None:
            self._interaction_padding = torch.zeros(
                batch_size, 1, dtype=bottom_mlp_output.dtype, device=bottom_mlp_output.device)
        concat = torch.cat([bottom_mlp_output] + embedding_outputs, dim=1)
        if self._interaction_op == "dot" and not self._self_interaction:
            concat = concat.view((-1, self._num_interaction_inputs, self._embedding_dim))
            interaction = torch.bmm(concat, torch.transpose(concat, 1, 2))
            interaction_flat = interaction[:, self.tril_indices[0], self.tril_indices[1]]
            # concatenate dense features and interactions
            interaction_padding = self._interaction_padding.expand(batch_size, 1).to(dtype=bottom_mlp_output.dtype)
            interaction_output = torch.cat(
                (bottom_mlp_output, interaction_flat, interaction_padding), dim=1)
        elif self._interaction_op == "cat":
            interaction_output = concat
        else:
            raise NotImplementedError

        return interaction_output

    def _initialize_mlp_weights(self):
        """Initializing weights same as original DLRM"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight.data, 0., math.sqrt(2. / (module.in_features + module.out_features)))
                nn.init.normal_(module.bias.data, 0., math.sqrt(1. /  module.out_features))

        # Explicitly set weight corresponding to zero padded interaction output. They will
        # stay 0 throughout the entire training. An assert can be added to the end of the training
        # to prove it doesn't increase model capacity but just 0 paddings.
        nn.init.zeros_(self.top_mlp[0].weight[:, -1].data)

    @property
    def num_categorical_features(self):
        return len(self._categorical_feature_sizes)

    def extra_repr(self):
        s = (F"interaction_op={self._interaction_op}, self_interaction={self._self_interaction}, "
             F"hash_indices={self._hash_indices}")
        return s
    # pylint:enable=missing-docstring

    @classmethod
    def from_dict(cls, obj_dict, **kwargs):
        """Create from json str"""
        return cls(**obj_dict, **kwargs)

    def _create_embeddings(self, embeddings, embedding_dim, categorical_feature_sizes):
        # Each embedding table has size [num_features, embedding_dim]
        for i, num_features in enumerate(categorical_feature_sizes):
            # Allocate directly on GPU is much faster than allocating on CPU then copying over
            embedding_weight = torch.empty((num_features, embedding_dim), device=self._embedding_device_map[i])
            embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False, sparse=True)

            # Initializing embedding same as original DLRM
            nn.init.uniform_(
                embedding.weight.data,
                -math.sqrt(1. / embedding.num_embeddings),
                math.sqrt(1. / embedding.num_embeddings))

            embeddings.append(embedding)

    def set_devices(self, base_device):
        """Set devices to run the model

        Args:
            base_device (string);
        """
        self._base_device = base_device
        self.bottom_mlp.to(base_device)
        self.top_mlp.to(base_device)
        self._interaction_padding = self._interaction_padding.to(base_device)
        self._embedding_device_map = [base_device for _ in range(self.num_categorical_features)]

        for embedding_id, device in enumerate(self._embedding_device_map):
            logging.info("Place embedding %d on device %s", embedding_id, device)
            self.embeddings[embedding_id].to(device)

    def forward(self, numerical_input, categorical_inputs):
        """

        Args:
            numerical_input (Tensor): with shape [batch_size, num_numerical_features]
            categorical_inputs (Tensor): with shape [batch_size, num_categorical_features]
        """
        batch_size = numerical_input.size()[0]

        # Put indices on the same device as corresponding embedding
        device_indices = []
        for embedding_id, _ in enumerate(self.embeddings):
            device_indices.append(categorical_inputs[:, embedding_id].to(self._embedding_device_map[embedding_id]))

        bottom_mlp_output = self.bottom_mlp(numerical_input)

        # embedding_outputs will be a list of (26 in the case of Criteo) fetched embeddings with shape
        # [batch_size, embedding_size]
        embedding_outputs = []
        for embedding_id, embedding in enumerate(self.embeddings):
            if self._hash_indices:
                device_indices[embedding_id] = device_indices[embedding_id] % embedding.num_embeddings

            embedding_outputs.append(embedding(device_indices[embedding_id]).to(self._base_device))

        interaction_output = self._interaction(bottom_mlp_output, embedding_outputs, batch_size)

        top_mlp_output = self.top_mlp(interaction_output)

        return top_mlp_output

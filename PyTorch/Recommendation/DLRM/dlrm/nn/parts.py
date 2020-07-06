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
import math
from typing import Sequence, Optional, Tuple

import torch
from torch import nn

from dlrm.nn.embeddings import Embeddings
from dlrm.nn.factories import create_embeddings, create_mlp
from dlrm.nn.interactions import Interaction


class DlrmBottom(nn.Module):

    def __init__(
        self,
        num_numerical_features: int,
        categorical_feature_sizes: Sequence[int],
        bottom_mlp_sizes: Optional[Sequence[int]] = None,
        embedding_type: str = "multi_table",
        embedding_dim: int = 128,
        hash_indices: bool = False,
        use_cpp_mlp: bool = False,
        fp16: bool = False,
        device: str = "cuda"
    ):
        super().__init__()
        assert bottom_mlp_sizes is None or embedding_dim == bottom_mlp_sizes[-1], "The last bottom MLP layer must" \
                                                                                  " have same size as embedding."
        self._embedding_dim = embedding_dim
        self._categorical_feature_sizes = copy.copy(categorical_feature_sizes)
        self._fp16 = fp16

        self.embeddings = create_embeddings(
            embedding_type,
            categorical_feature_sizes,
            embedding_dim,
            device,
            hash_indices,
            fp16
        )
        self.mlp = (create_mlp(num_numerical_features, bottom_mlp_sizes, use_cpp_mlp).to(device)
                    if bottom_mlp_sizes else torch.nn.ModuleList())

        self._initialize_embeddings_weights(self.embeddings, categorical_feature_sizes)

    def _initialize_embeddings_weights(self, embeddings: Embeddings, categorical_feature_sizes: Sequence[int]):
        assert len(embeddings.weights) == len(categorical_feature_sizes)

        for size, weight in zip(categorical_feature_sizes, embeddings.weights):
            nn.init.uniform_(
                weight,
                -math.sqrt(1. / size),
                math.sqrt(1. / size)
            )

    @property
    def num_categorical_features(self) -> int:
        return len(self._categorical_feature_sizes)

    @property
    def num_feature_vectors(self) -> int:
        return self.num_categorical_features + int(self.mlp is not None)

    def forward(self, numerical_input, categorical_inputs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            numerical_input (Tensor): with shape [batch_size, num_numerical_features]
            categorical_inputs (Tensor): with shape [batch_size, num_categorical_features]

        Returns:
            Tensor: Concatenated bottom mlp and embedding output in shape [batch, 1 + #embedding, embedding_dim]
        """
        batch_size = categorical_inputs.size()[0]
        bottom_output = []
        bottom_mlp_output = None

        if self.mlp:
            bottom_mlp_output = self.mlp(numerical_input)
            if self._fp16:
                bottom_mlp_output = bottom_mlp_output.half()

            # reshape bottom mlp to concatenate with embeddings
            bottom_output.append(bottom_mlp_output.view(batch_size, 1, -1))

        bottom_output += self.embeddings(categorical_inputs)

        if self._fp16:
            bottom_output = [x.half() if x.dtype != torch.half else x for x in bottom_output]

        if len(bottom_output) == 1:
            return bottom_output[0], bottom_mlp_output

        return torch.cat(bottom_output, dim=1), bottom_mlp_output


class DlrmTop(nn.Module):

    def __init__(self, top_mlp_sizes: Sequence[int], interaction: Interaction, use_cpp_mlp: bool = False):
        super().__init__()

        self.interaction = interaction
        self.mlp = create_mlp(interaction.num_interactions, top_mlp_sizes[:-1], use_cpp_mlp)
        self.out = nn.Linear(top_mlp_sizes[-2], top_mlp_sizes[-1])

        self._initialize_weights()

    def _initialize_weights(self):
        # Explicitly set weight corresponding to zero padded interaction output. They will
        # stay 0 throughout the entire training. An assert can be added to the end of the training
        # to prove it doesn't increase model capacity but just 0 paddings.
        nn.init.zeros_(self.mlp.weights[0][:, -1].data)

    def forward(self, bottom_output, bottom_mlp_output):
        """
        Args:
            bottom_output (Tensor): with shape [batch_size, 1 + #embeddings, embedding_dim]
            bottom_mlp_output (Tensor): with shape [batch_size, embedding_dim]
        """
        interaction_output = self.interaction.interact(bottom_output, bottom_mlp_output)
        return self.out(self.mlp(interaction_output))

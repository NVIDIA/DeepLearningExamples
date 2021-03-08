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

import torch

from dlrm.cuda_ext import dotBasedInteract


class Interaction:

    @property
    def num_interactions(self) -> int:
        raise NotImplementedError()

    def interact(self, bottom_output, bottom_mlp_output):
        """
        :param bottom_output: [batch_size, 1 + #embeddings, embedding_dim]
        :param bottom_mlp_output
        :return:
        """
        raise NotImplementedError()


class DotInteraction(Interaction):

    def __init__(self, embedding_num: int, embedding_dim: int):
        """
        Interactions are among outputs of all the embedding tables and bottom MLP, total number of
        (num_embedding_tables + 1) vectors with size embedding_dim. ``dot`` product interaction computes dot product
        between any 2 vectors. Output of interaction will have shape [num_interactions, embedding_dim].
        """
        self._num_interaction_inputs = embedding_num + 1
        self._embedding_dim = embedding_dim
        self._tril_indices = torch.tensor([[i for i in range(self._num_interaction_inputs)
                                            for _ in range(i)],
                                           [j for i in range(self._num_interaction_inputs)
                                            for j in range(i)]])

    @property
    def num_interactions(self) -> int:
        n = (self._num_interaction_inputs * (self._num_interaction_inputs - 1)) // 2 + self._embedding_dim
        return n + 1  # pad 1 to be multiple of 8

    def interact(self, bottom_output, bottom_mlp_output):
        """
        :param bottom_output: [batch_size, 1 + #embeddings, embedding_dim]
        :param bottom_mlp_output
        :return:
        """
        batch_size = bottom_output.size()[0]

        interaction = torch.bmm(bottom_output, torch.transpose(bottom_output, 1, 2))
        interaction_flat = interaction[:, self._tril_indices[0], self._tril_indices[1]]

        # concatenate dense features and interactions
        zeros_padding = torch.zeros(batch_size, 1, dtype=bottom_output.dtype, device=bottom_output.device)
        interaction_output = torch.cat(
            (bottom_mlp_output, interaction_flat, zeros_padding), dim=1)

        return interaction_output


class CudaDotInteraction(Interaction):

    def __init__(self, dot_interaction: DotInteraction):
        self._dot_interaction = dot_interaction

    @property
    def num_interactions(self):
        return self._dot_interaction.num_interactions

    def interact(self, bottom_output, bottom_mlp_output):
        """
        :param bottom_output: [batch_size, 1 + #embeddings, embedding_dim]
        :param bottom_mlp_output
        :return:
        """
        return dotBasedInteract(bottom_output, bottom_mlp_output)


class CatInteraction(Interaction):

    def __init__(self, embedding_num: int, embedding_dim: int):
        """
        Interactions are among outputs of all the embedding tables and bottom MLP, total number of
        (num_embedding_tables + 1) vectors with size embdding_dim. ``cat`` interaction concatenate all the vectors
        together. Output of interaction will have shape [num_interactions, embedding_dim].
        """
        self._num_interaction_inputs = embedding_num + 1
        self._embedding_dim = embedding_dim

    @property
    def num_interactions(self) -> int:
        return self._num_interaction_inputs * self._embedding_dim

    def interact(self, bottom_output, bottom_mlp_output):
        """
        :param bottom_output: [batch_size, 1 + #embeddings, embedding_dim]
        :param bottom_mlp_output
        :return:
        """
        return bottom_output.view(-1, self.num_interactions)

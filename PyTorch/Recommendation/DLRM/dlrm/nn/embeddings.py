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
from typing import Sequence, List, Iterable

import torch
from absl import logging
from torch import nn

from dlrm import cuda_ext
from dlrm.cuda_ext.fused_gather_embedding import BuckleEmbeddingFusedGatherFunction


class Embeddings(nn.Module):

    def forward(self, categorical_inputs) -> List[torch.Tensor]:
        raise NotImplementedError()

    @property
    def weights(self) -> List[torch.Tensor]:
        """
        Note: output list size should match number of handled categorical features
        """
        raise NotImplementedError()

    def load_weights(self, weights: Iterable[torch.Tensor]):
        raise NotImplementedError()


class MultiTableEmbeddings(Embeddings):

    def __init__(
        self,
        categorical_feature_sizes: Sequence[int],
        embedding_dim: int,
        hash_indices: bool = False,
        device: str = "cuda"
    ):
        super().__init__()
        self._categorical_feature_sizes = copy.copy(categorical_feature_sizes)
        self._base_device = device
        self._embedding_device_map = [device for _ in range(len(categorical_feature_sizes))]

        embeddings = []
        # Each embedding table has size [num_features, embedding_dim]
        for i, num_features in enumerate(categorical_feature_sizes):
            # Allocate directly on GPU is much faster than allocating on CPU then copying over
            embedding_weight = torch.empty((num_features, embedding_dim), device=self._embedding_device_map[i])
            embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False, sparse=True)
            embeddings.append(embedding)

        self.embeddings = nn.ModuleList(embeddings)
        self.hash_indices = hash_indices
        self.embedding_dim = embedding_dim

    def forward(self, categorical_inputs) -> List[torch.Tensor]:
        """
        Args:
            categorical_inputs (Tensor): with shape [batch_size, num_categorical_features]

        Returns:
            Tensor: embedding outputs in shape [batch, embedding_num, embedding_dim]
        """
        # Put indices on the same device as corresponding embedding
        device_indices = []
        for embedding_id, _ in enumerate(self.embeddings):
            device_indices.append(categorical_inputs[:, embedding_id].to(self._embedding_device_map[embedding_id]))

        # embedding_outputs will be a list of (26 in the case of Criteo) fetched embeddings with shape
        # [batch_size, embedding_size]
        embedding_outputs = []
        for embedding_id, embedding in enumerate(self.embeddings):
            if self.hash_indices:
                device_indices[embedding_id] %= embedding.num_embeddings

            embedding_outputs.append(embedding(device_indices[embedding_id]).to(self._base_device).unsqueeze(1))

        return embedding_outputs

    @property
    def weights(self):
        return [embedding.weight.data for embedding in self.embeddings]

    def load_weights(self, weights: Iterable[torch.Tensor]):
        for embedding, weight in zip(self.embeddings, weights):
            embedding.weight.data = weight
            embedding.weight.data.requires_grad_()


class JointEmbedding(Embeddings):
    """Buckle multiple one hot embedding together

    Multiple one hot embedding can be done as one embedding (indexing). Use nn.Embedding to deal with sparse wgrad
    before I fully customizing it.

    Args:
        categorical_feature_sizes (list): A list of integer indicating number of features of each embedding table
        embedding_dim (int): the size of each embedding vector
        device (torch.device): where to create the embedding. Default "cuda"
    """
    def __init__(
        self,
        categorical_feature_sizes: Sequence[int],
        embedding_dim: int,
        device: str = "cuda",
        hash_indices: bool = False
    ):
        super().__init__()
        self._categorical_feature_sizes = copy.copy(categorical_feature_sizes)

        self.register_buffer("offsets", torch.tensor([0] + list(categorical_feature_sizes), device=device).cumsum(0))

        embedding_weight = torch.empty((self.offsets[-1].item(), embedding_dim), device=device)
        self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False, sparse=True)
        self.hash_indices = hash_indices

    # pylint:disable=missing-docstring
    def forward(self, categorical_inputs) -> List[torch.Tensor]:
        if self.hash_indices:
            for cat, size in enumerate(self._categorical_feature_sizes):
                categorical_inputs[:, cat] %= size
                logging.log_first_n(logging.WARNING, F"Hashed indices out of range.", 1)

        return [self.embedding(categorical_inputs + self.offsets[:-1])]

    def extra_repr(self):
        s = F"offsets={self.offsets.cpu().numpy()}"
        return s
    # pylint:enable=missing-docstring

    @property
    def weights(self):
        return [self.embedding.weight.data[self.offsets[cat]:self.offsets[cat + 1]]
                for cat in range(len(self._categorical_feature_sizes))]

    def load_weights(self, weights: Iterable[torch.Tensor]):
        data = self.embedding.weight.data
        offsets = self.offsets

        for cat, weight in zip(range(len(self._categorical_feature_sizes)), weights):
            data[offsets[cat]:offsets[cat + 1]] = weight


class FusedJointEmbedding(Embeddings):
    """
    Buckle multiple one hot embedding together

    Multiple one hot embedding can be done as one embedding (indexing).
    Args:
    categorical_feature_sizes (list): A list of integer indicating number of features of each embedding table
    embedding_dim (int): the size of each embedding vector
    device (torch.device): where to create the embedding. Default "cuda"
    """

    def __init__(
        self,
        categorical_feature_sizes: Sequence[int],
        embedding_dim: int,
        device: str = "cuda",
        hash_indices: bool = False,
        amp_train: bool = False
    ):
        super().__init__()
        self._categorical_feature_sizes = copy.copy(categorical_feature_sizes)

        self.embedding_dim = embedding_dim
        self.amp_train = amp_train
        self.hash_indices = hash_indices

        self.register_buffer("offsets", torch.tensor([0] + categorical_feature_sizes).cumsum(0).to(device))

        self.register_parameter("weight", torch.nn.Parameter(
            torch.empty((self.offsets[-1].item(), embedding_dim), device=device), requires_grad=True))

    def forward(self, categorical_inputs) -> List[torch.Tensor]:
        # Check input has the right shape
        if self.hash_indices:
            for cat, size in enumerate(self._categorical_feature_sizes):
                categorical_inputs[:, cat] %= size
                logging.log_first_n(logging.WARNING, F"Hashed indices out of range.", 1)

        return [BuckleEmbeddingFusedGatherFunction.apply(self.weight, categorical_inputs, self.offsets, self.amp_train)]

    def extra_repr(self):
        return 'embedding_dim={}, categorical_feature_sizes={}, offsets={}'.format(
            self.embedding_dim, self._categorical_feature_sizes, self.offsets)

    @property
    def weights(self) -> List[torch.Tensor]:
        return [self.weight.data[self.offsets[cat]:self.offsets[cat + 1]]
                for cat in range(len(self._categorical_feature_sizes))]

    def load_weights(self, weights: Iterable[torch.Tensor]):
        data = self.weight.data
        offsets = self.offsets

        for cat, weight in zip(range(len(self._categorical_feature_sizes)), weights):
            data[offsets[cat]:offsets[cat + 1]] = weight


class JointSparseEmbedding(Embeddings):

    def __init__(
        self,
        categorical_feature_sizes: List[int],
        embedding_dim: int,
        device: str = "cuda",
        hash_indices: bool = False
    ):
        super().__init__()
        self._categorical_feature_sizes = categorical_feature_sizes
        self.embedding = cuda_ext.JointSparseEmbedding(categorical_feature_sizes, embedding_dim, device)
        self.hash_indices = hash_indices

    def forward(self, categorical_inputs) -> List[torch.Tensor]:
        if self.hash_indices:
            for cat, size in enumerate(self._categorical_feature_sizes):
                categorical_inputs[:, cat] %= size
                logging.log_first_n(logging.WARNING, F"Hashed indices out of range.", 1)

        return [
            self.embedding(categorical_inputs)
        ]

    @property
    def weights(self):
        data = self.embedding.weights.data
        offsets = self.embedding.offsets
        return [data[offsets[cat]:offsets[cat + 1]] for cat in range(len(self._categorical_feature_sizes))]

    def load_weights(self, weights: Iterable[torch.Tensor]):
        data = self.embedding.weights.data
        offsets = self.embedding.offsets

        for cat, weight in zip(range(len(self._categorical_feature_sizes)), weights):
            data[offsets[cat]:offsets[cat + 1]] = weight

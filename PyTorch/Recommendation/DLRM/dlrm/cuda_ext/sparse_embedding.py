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

import torch
from apex import amp
from dlrm.cuda_ext import sparse_gather
from torch import nn
from torch.autograd import Function


class EmbeddingGatherFunction(Function):
    """Customized embedding gather with fused plain SGD"""
    @staticmethod
    def forward(ctx, embedding, indices):
        output = sparse_gather.gather_gpu_fwd(embedding, indices)
        ctx.save_for_backward(indices)
        ctx.num_features = embedding.size(0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        indices = ctx.saved_tensors[0]

        grad_embedding = sparse_gather.gather_gpu_bwd(grad_output, indices, ctx.num_features)

        return grad_embedding, None


class JointSparseEmbedding(nn.Module):
    """Joint multiple one hot embedding together

    Multiple one hot embedding can be done as one embedding (indexing).

    Args:
        categorical_feature_sizes (list): A list of integer indicating number of features of each embedding table
        embedding_dim (int): the size of each embedding vector
        device (torch.device): where to create the embedding. Default "cuda"
    """
    def __init__(self, categorical_feature_sizes, embedding_dim, device="cuda"):
        super(JointSparseEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.categorical_feature_sizes = copy.copy(categorical_feature_sizes)

        self.register_buffer("offsets", torch.tensor([0] + categorical_feature_sizes).cumsum(0).to(device))
        self.weights = torch.nn.Parameter(torch.rand((self.offsets[-1].item(), embedding_dim), device=device))

    def forward(self, categorical_inputs):
        # Check input has the right shape
        assert categorical_inputs.shape[1] == len(self.categorical_feature_sizes)

        embedding_out = embedding_gather(self.weights, categorical_inputs + self.offsets[:-1])

        return embedding_out


embedding_gather = amp.float_function(EmbeddingGatherFunction.apply)

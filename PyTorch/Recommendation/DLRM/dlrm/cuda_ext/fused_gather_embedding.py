# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
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

"""
Fused Buckle Embedding
"""

from absl import logging
import torch
from torch.autograd import Function

from dlrm.cuda_ext import fused_embedding


class BuckleEmbeddingFusedGatherFunction(Function):
    """Customized embedding gather """
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, embedding, indices, offsets, amp_train):
        output = fused_embedding.gather_gpu_fused_fwd(embedding, indices, offsets, amp_train)
        ctx.save_for_backward(embedding, indices, offsets)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        embedding, indices, offsets = ctx.saved_tensors

        logging.log_first_n(logging.WARNING, "Highly specialized embedding for embedding_dim 128", 1)
        grad_weights = fused_embedding.gather_gpu_fused_bwd(embedding, indices, offsets, grad_output)
        return grad_weights, None, None, None


buckle_embedding_fused_gather = BuckleEmbeddingFusedGatherFunction.apply

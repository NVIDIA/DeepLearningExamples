"""
Fused Buckle Embedding
"""

from absl import logging
from apex import amp
from torch.autograd import Function

from dlrm.cuda_ext import fused_embedding


class BuckleEmbeddingFusedGatherFunction(Function):
    """Customized embedding gather """
    @staticmethod
    def forward(ctx, embedding, indices, offsets, amp_train):
        output = fused_embedding.gather_gpu_fused_fwd(embedding, indices, offsets, amp_train)
        ctx.save_for_backward(embedding, indices, offsets)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        embedding, indices, offsets = ctx.saved_tensors

        logging.log_first_n(logging.WARNING, "Highly specialized embedding for embedding_dim 128", 1)
        grad_weights = fused_embedding.gather_gpu_fused_bwd(embedding, indices, offsets, grad_output)
        return grad_weights, None, None, None


buckle_embedding_fused_gather = amp.float_function(BuckleEmbeddingFusedGatherFunction.apply)

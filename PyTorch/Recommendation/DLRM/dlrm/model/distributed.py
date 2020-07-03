from typing import Sequence, Optional

import torch
from torch import nn

from dlrm.nn.factories import create_interaction
from dlrm.nn.parts import DlrmBottom, DlrmTop
from dlrm.utils import distributed as dist


class BottomToTop(torch.autograd.Function):
    """Switch from model parallel to data parallel

    Wrap the communication of doing from bottom model in model parallel fashion to top model in data parallel
    """

    @staticmethod
    def forward(
        ctx,
        local_bottom_outputs: torch.Tensor,
        batch_sizes_per_gpu: Sequence[int],
        vector_dim: int,
        vectors_per_gpu: Sequence[int],
        feature_order: Optional[torch.Tensor] = None,
        device_feature_order: Optional[torch.Tensor] = None
    ):
        """
        Args:
            ctx : Pytorch convention
            local_bottom_outputs (Tensor): Concatenated output of bottom model
            batch_sizes_per_gpu (Sequence[int]):
            vector_dim (int):
            vectors_per_gpu (Sequence[int]): Note, bottom MLP is considered as 1 vector
            device_feature_order:
            feature_order:

        Returns:
            slice_embedding_outputs (Tensor): Patial output from bottom model to feed into data parallel top model
        """
        rank = dist.get_rank()

        ctx.world_size = torch.distributed.get_world_size()
        ctx.batch_sizes_per_gpu = batch_sizes_per_gpu
        ctx.vector_dim = vector_dim
        ctx.vectors_per_gpu = vectors_per_gpu
        ctx.feature_order = feature_order
        ctx.device_feature_order = device_feature_order

        # Buffer shouldn't need to be zero out. If not zero out buffer affecting accuracy, there must be a bug.
        bottom_output_buffer = [torch.empty(
            batch_sizes_per_gpu[rank], n * vector_dim,
            device=local_bottom_outputs.device, dtype=local_bottom_outputs.dtype) for n in vectors_per_gpu]

        torch.distributed.all_to_all(bottom_output_buffer, list(local_bottom_outputs.split(batch_sizes_per_gpu, dim=0)))
        slice_bottom_outputs = torch.cat(bottom_output_buffer, dim=1).view(batch_sizes_per_gpu[rank], -1, vector_dim)

        # feature reordering is just for consistency across different device mapping configurations
        if feature_order is not None and device_feature_order is not None:
            return slice_bottom_outputs[:, feature_order, :]

        return slice_bottom_outputs

    @staticmethod
    def backward(ctx, grad_slice_bottom_outputs):
        rank = dist.get_rank()

        if ctx.feature_order is not None and ctx.device_feature_order is not None:
            grad_slice_bottom_outputs = grad_slice_bottom_outputs[:, ctx.device_feature_order, :]

        grad_local_bottom_outputs = torch.empty(
            sum(ctx.batch_sizes_per_gpu), ctx.vectors_per_gpu[rank] * ctx.vector_dim,
            device=grad_slice_bottom_outputs.device,
            dtype=grad_slice_bottom_outputs.dtype)
        # All to all only takes list while split() returns tuple

        grad_local_bottom_outputs_split = list(grad_local_bottom_outputs.split(ctx.batch_sizes_per_gpu, dim=0))

        split_grads = [t.contiguous() for t in (grad_slice_bottom_outputs.view(ctx.batch_sizes_per_gpu[rank], -1).split(
            [ctx.vector_dim * n for n in ctx.vectors_per_gpu], dim=1))]

        torch.distributed.all_to_all(grad_local_bottom_outputs_split, split_grads)

        return (grad_local_bottom_outputs.view(grad_local_bottom_outputs.shape[0], -1, ctx.vector_dim), None, None,
                None, None, None)


bottom_to_top = BottomToTop.apply


class DistributedDlrm(nn.Module):

    def __init__(
        self,
        vectors_per_gpu: Sequence[int],
        embedding_device_mapping: Sequence[Sequence[int]],
        world_num_categorical_features: int,
        num_numerical_features: int,
        categorical_feature_sizes: Sequence[int],
        bottom_mlp_sizes: Sequence[int],
        top_mlp_sizes: Sequence[int],
        embedding_type: str = "multi_table",
        embedding_dim: int = 128,
        interaction_op: str = "dot",
        hash_indices: bool = False,
        use_cpp_mlp: bool = False,
        fp16: bool = False,
        bottom_features_ordered: bool = False,
        device: str = "cuda"
    ):
        super().__init__()

        self._vectors_per_gpu = vectors_per_gpu
        self._embedding_dim = embedding_dim
        self._interaction_op = interaction_op
        self._hash_indices = hash_indices

        # TODO: take bottom_mlp GPU from device mapping, do not assume it's always first
        self._device_feature_order = torch.tensor(
            [-1] + [i for bucket in embedding_device_mapping for i in bucket], dtype=torch.long, device=device
        ) + 1 if bottom_features_ordered else None
        self._feature_order = self._device_feature_order.argsort() if bottom_features_ordered else None

        interaction = create_interaction(interaction_op, world_num_categorical_features, embedding_dim)

        self.bottom_model = DlrmBottom(
            num_numerical_features, categorical_feature_sizes, bottom_mlp_sizes,
            embedding_type, embedding_dim, hash_indices=hash_indices, use_cpp_mlp=use_cpp_mlp, fp16=fp16, device=device
        )
        self.top_model = DlrmTop(top_mlp_sizes, interaction, use_cpp_mlp=use_cpp_mlp).to(device)

    def extra_repr(self):
        return f"interaction_op={self._interaction_op}, hash_indices={self._hash_indices}"

    # pylint:enable=missing-docstring

    @classmethod
    def from_dict(cls, obj_dict, **kwargs):
        """Create from json str"""
        return cls(**obj_dict, **kwargs)

    def forward(self, numerical_input, categorical_inputs, batch_sizes_per_gpu: Sequence[int]):
        """
        Args:
            numerical_input (Tensor): with shape [batch_size, num_numerical_features]
            categorical_inputs (Tensor): with shape [batch_size, num_categorical_features]
            batch_sizes_per_gpu (Sequence[int]):
        """
        # bottom mlp output may be not present before all to all communication
        bottom_output, _ = self.bottom_model(numerical_input, categorical_inputs)

        from_bottom = bottom_to_top(bottom_output, batch_sizes_per_gpu, self._embedding_dim, self._vectors_per_gpu,
                                    self._feature_order, self._device_feature_order)

        # TODO: take bottom_mlp GPU from device mapping, do not assume it's always first
        bottom_mlp_output = from_bottom[:, 0, :]
        return self.top_model(from_bottom, bottom_mlp_output)

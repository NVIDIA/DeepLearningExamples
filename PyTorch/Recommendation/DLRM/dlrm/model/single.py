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

from typing import Sequence

from torch import nn

from dlrm.nn.factories import create_interaction
from dlrm.nn.parts import DlrmBottom, DlrmTop


class Dlrm(nn.Module):
    """Reimplement Facebook's DLRM model

    Original implementation is from https://github.com/facebookresearch/dlrm.

    """
    def __init__(
        self,
        num_numerical_features: int,
        categorical_feature_sizes: Sequence[int],
        bottom_mlp_sizes: Sequence[int],
        top_mlp_sizes: Sequence[int],
        embedding_type: str = "multi_table",
        embedding_dim: int = 32,
        interaction_op: str = "dot",
        hash_indices: bool = False,
        use_cpp_mlp: bool = False,
        fp16: bool = False,
        base_device: str = "cuda",
    ):
        super().__init__()
        assert embedding_dim == bottom_mlp_sizes[-1], "The last bottom MLP layer must have same size as embedding."

        interaction = create_interaction(interaction_op, len(categorical_feature_sizes), embedding_dim)

        self._interaction_op = interaction_op
        self._hash_indices = hash_indices

        self.bottom_model = DlrmBottom(
            num_numerical_features=num_numerical_features,
            categorical_feature_sizes=categorical_feature_sizes,
            bottom_mlp_sizes=bottom_mlp_sizes,
            embedding_type=embedding_type,
            embedding_dim=embedding_dim,
            hash_indices=hash_indices,
            use_cpp_mlp=use_cpp_mlp,
            fp16=fp16,
            device=base_device
        )
        self.top_model = DlrmTop(top_mlp_sizes, interaction, use_cpp_mlp=use_cpp_mlp).to(base_device)

    def extra_repr(self):
        return f"interaction_op={self._interaction_op}, hash_indices={self._hash_indices}"

    # pylint:enable=missing-docstring
    @classmethod
    def from_dict(cls, obj_dict, **kwargs):
        """Create from json str"""
        return cls(**obj_dict, **kwargs)

    def forward(self, numerical_input, categorical_inputs):
        """

        Args:
            numerical_input (Tensor): with shape [batch_size, num_numerical_features]
            categorical_inputs (Tensor): with shape [batch_size, num_categorical_features]
        """
        bottom_output, bottom_mlp_output = self.bottom_model(numerical_input, categorical_inputs)
        return self.top_model(bottom_output, bottom_mlp_output)

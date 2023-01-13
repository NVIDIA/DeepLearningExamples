# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Layer norm done in fp32 (for fp16 training)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class MaskedGroupNorm(nn.Module):
    """GroupNorm layer which skips padding.

    In wav2vec 2.0 encoder where batch size is small and time dimensio huge,
    this is nearly as fast as nn.GroupNorm.

    Ready for TorchScript, favors composition over inheritance.
    """
    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True,
                 device=None, dtype=None):
        assert num_groups == num_channels, (
            "num_groups != num_channels not yet supported in MaskedGroupNorm")
        super().__init__()
        self._group_norm = nn.GroupNorm(num_groups, num_channels, eps=eps,
                                        affine=affine, device=device,
                                        dtype=dtype)

    def forward(self, x, x_lens):
        var = torch.zeros_like(x[:, :, 0])
        mean = torch.zeros_like(x[:, :, 0])
        for i in range(x.size(0)):
            mean[i] = torch.mean(x[i, :, :x_lens[i]], dim=1)
            var[i] = torch.var(x[i, :, :x_lens[i]], dim=1, unbiased=False)
        out = (x - mean[:, :, None]) / torch.sqrt(var[:, :, None] + self._group_norm.eps)
        if self._group_norm.affine:
            return out * self._group_norm.weight[None, :, None] + self._group_norm.bias[None, :, None]
        else:
            return out


class Fp32MaskedGroupNorm(nn.Module):
    """GroupNorm layer which skips padding.

    In wav2vec 2.0 encoder where batch size is small and time dimensio huge,
    this is nearly as fast as nn.GroupNorm.

    Ready for TorchScript, favors composition over inheritance.
    """
    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True,
                 device=None, dtype=None):
        assert num_groups == num_channels, (
            "num_groups != num_channels not yet supported in MaskedGroupNorm")
        super().__init__()
        self._group_norm = nn.GroupNorm(num_groups, num_channels, eps=eps,
                                        affine=affine, device=device,
                                        dtype=dtype)

        def hook(state_dict, prefix, *args, **kwargs):
            """Renames keys from layers which used inheritance."""
            new_sd = {}
            for k, v in state_dict.items():
                if not k.startswith(prefix):
                    new_sd[k] = v
                else:
                    *pref, param = k.split(".")
                    new_k = ".".join(pref + ["_group_norm", param])
                    new_sd[new_k] = v
            state_dict.clear()
            state_dict.update(new_sd)

        self._register_load_state_dict_pre_hook(hook)

    def forward(self, x, x_lens):
        return self._forward(
            x.float(),
            x_lens,
            self._group_norm.weight.float() if self._group_norm.weight is not None else None,
            self._group_norm.bias.float() if self._group_norm.bias is not None else None,
        ).type_as(x)

    def _forward(self, x, x_lens, weight, bias):
        var = torch.zeros_like(x[:, :, 0])
        mean = torch.zeros_like(x[:, :, 0])
        for i in range(x.size(0)):
            mean[i] = torch.mean(x[i, :, :x_lens[i]], dim=1)
            var[i] = torch.var(x[i, :, :x_lens[i]], dim=1, unbiased=False)
        out = (x - mean[:, :, None]) / torch.sqrt(var[:, :, None] + self._group_norm.eps)
        if self._group_norm.affine:
            return out * weight[None, :, None] + bias[None, :, None]
        else:
            return out

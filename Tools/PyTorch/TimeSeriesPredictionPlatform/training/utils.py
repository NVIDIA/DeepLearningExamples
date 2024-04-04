# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-License-Identifier: Apache-2.0
import dgl
import torch
import numpy as np


def round_dict(input_data, decimal=4):
    rounded_data = {
        key: (np.around(value, decimal) if isinstance(value, (np.floating, float)) else value)
        for key, value in input_data.items()
    }
    return rounded_data


def to_device(batch, device=None):
    if isinstance(batch, torch.Tensor):
        return batch.to(device=device)
    if isinstance(batch, dict):
        return {k: t.to(device=device) if t is not None and t.numel() else None for k, t in batch.items()}
    if isinstance(batch, dgl.DGLGraph):
        return batch.to(device=device)
    elif batch is None:
        return None


def set_seed(seed):
    if seed is None:
        return
    if not isinstance(seed, int):
        raise ValueError(f"Seed has to be an integer or None, but got type {type(seed)}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_optimization_objectives(config, metrics):
    objectives = tuple(v if v == v and v < 2.0**15 else 2.0**15
                       for k, v in metrics.items()
                       if k in config.get('optuna_objectives', [])
                       )
    if len(objectives) == 1:
        return objectives[0]
    elif not objectives:
        return None
    return objectives

# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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


# Copyright 2020 Chengxi Zang
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.


import json
import logging
import numpy as np
import os
from typing import Dict, Tuple

from moflow.config import CODE_TO_BOND, DUMMY_CODE, Config


def _onehot(data: np.ndarray, codes_dict: Dict[int, int], dtype=np.float32) -> np.ndarray:
    shape = [len(codes_dict), *data.shape]
    encoded = np.zeros(shape, dtype=dtype)
    for obj_key, code in codes_dict.items():
        encoded[code, data == obj_key] = 1
    return encoded


def encode_nodes(atomic_nums: np.ndarray, config: Config) -> np.ndarray:
    padded_data = np.full(config.max_num_nodes, DUMMY_CODE, dtype=np.uint8)
    padded_data[:len(atomic_nums)] = atomic_nums
    encoded = _onehot(padded_data, config.dataset_config.atomic_to_code).T
    return encoded


def encode_edges(adj: np.ndarray, config: Config) -> np.ndarray:
    padded_data = np.full((config.max_num_nodes, config.max_num_nodes), DUMMY_CODE, dtype=np.uint8)
    n, m = adj.shape
    assert n == m, 'adjecency matrix should be square'
    padded_data[:n, :n] = adj
    # we already store codes in the file - bond types are rdkit objects
    encoded = _onehot(padded_data, {k:k for k in CODE_TO_BOND})
    return encoded


def transform_fn(data: Tuple[np.ndarray], config: Config) -> Tuple[np.ndarray]:
    node, adj, *labels = data
    node = encode_nodes(node, config)
    adj = encode_edges(adj, config)
    return (node, adj, *labels)


def get_val_ids(config: Config, data_dir: str):
    file_path = os.path.join(data_dir, config.dataset_config.valid_idx_file)
    logging.info('loading train/valid split information from: {}'.format(file_path))
    with open(file_path) as json_data:
        data = json.load(json_data)

    val_ids = [int(idx)-1 for idx in data]
    return val_ids

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
from numba import jit, prange


@jit(nopython=True)
def mas(log_attn_map, width=1):
    # assumes mel x text
    opt = np.zeros_like(log_attn_map)
    log_attn_map = log_attn_map.copy()
    log_attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(log_attn_map)
    log_p[0, :] = log_attn_map[0, :]
    prev_ind = np.zeros_like(log_attn_map, dtype=np.int64)
    for i in range(1, log_attn_map.shape[0]):
        for j in range(log_attn_map.shape[1]):  # for each text dim
            prev_j = np.arange(max(0, j-width), j+1)
            prev_log = np.array([log_p[i-1, prev_idx] for prev_idx in prev_j])

            ind = np.argmax(prev_log)
            log_p[i, j] = log_attn_map[i, j] + prev_log[ind]
            prev_ind[i, j] = prev_j[ind]

    # now backtrack
    curr_text_idx = log_attn_map.shape[1]-1
    for i in range(log_attn_map.shape[0]-1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1
    return opt


@jit(nopython=True)
def mas_width1(log_attn_map):
    """mas with hardcoded width=1"""
    # assumes mel x text
    neg_inf = log_attn_map.dtype.type(-np.inf)
    log_p = log_attn_map.copy()
    log_p[0, 1:] = neg_inf
    for i in range(1, log_p.shape[0]):
        prev_log1 = neg_inf
        for j in range(log_p.shape[1]):
            prev_log2 = log_p[i-1, j]
            log_p[i, j] += max(prev_log1, prev_log2)
            prev_log1 = prev_log2

    # now backtrack
    opt = np.zeros_like(log_p)
    one = opt.dtype.type(1)
    j = log_p.shape[1]-1
    for i in range(log_p.shape[0]-1, 0, -1):
        opt[i, j] = one
        if log_p[i-1, j-1] >= log_p[i-1, j]:
            j -= 1
            if j == 0:
                opt[1:i, j] = one
                break
    opt[0, j] = one
    return opt


@jit(nopython=True, parallel=True)
def b_mas(b_log_attn_map, in_lens, out_lens, width=1):
    assert width == 1
    attn_out = np.zeros_like(b_log_attn_map)

    for b in prange(b_log_attn_map.shape[0]):
        out = mas_width1(b_log_attn_map[b, 0, :out_lens[b], :in_lens[b]])
        attn_out[b, 0, :out_lens[b], :in_lens[b]] = out
    return attn_out

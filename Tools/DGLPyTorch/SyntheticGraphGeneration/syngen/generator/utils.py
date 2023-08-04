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


import cupy as cp
from numba import cuda


WARP_SIZE = 32  # could be 32 or 64


@cuda.jit
def repeat_kernel(repeat_ptr, cumsum_ptr, res, size):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1) / WARP_SIZE
    warp_id = idx / WARP_SIZE
    tid_in_warp = idx % WARP_SIZE

    for i in range(warp_id, size, stride):
        end = cumsum_ptr[i]
        repeat = repeat_ptr[i]
        start = end - repeat

        for j in range(start + tid_in_warp, end, WARP_SIZE):
            res[j] = i


def cuda_repeat(repeats):
    cumsum = repeats.cumsum(0)
    total = cumsum[-1].item()
    size = len(repeats)
    block = 512
    warps_per_block = block // WARP_SIZE
    grid = max((size + warps_per_block - 1) // warps_per_block, 2048)
    res = cp.empty(total, dtype=repeats.dtype)
    repeat_kernel[grid, block](repeats, cumsum, res, size)
    cuda.synchronize()
    return res

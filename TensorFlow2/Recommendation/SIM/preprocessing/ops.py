# Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
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

import logging
from typing import List, Optional

import cudf
import cupy
import numba.cuda
from nvtabular import ops
from nvtabular.dispatch import _build_cudf_list_column as nvt_build_list_column

THREADS = 32

logging.getLogger("numba").setLevel(logging.WARNING)


def list_slice(seq_col, start: int, end: Optional[int] = None):
    """Slices a list column

    This is an nvtabular.ops.ListSlice wrapper that can be used with cuDF or dask-cuDF.

    """
    df = cudf.DataFrame(seq_col)
    col_selector = ops.ColumnSelector(seq_col.name)
    slicer = ops.ListSlice(start, end)
    transformed = slicer.transform(col_selector, df)
    return transformed[seq_col.name]


@numba.cuda.jit
def _calculate_row_sizes(offsets, row_sizes, max_elements):
    rowid = numba.cuda.grid(1)
    if rowid < offsets.size - 1:
        original_row_size = offsets[rowid + 1] - offsets[rowid]

        for i in range(original_row_size):
            row_sizes[1 + offsets[rowid] + i] = min(i + 1, max_elements)


@numba.cuda.jit
def _generate_rows(offsets, chunk_offsets, elements, new_elements, max_elements):
    rowid = numba.cuda.grid(1)
    if rowid < offsets.size - 1:
        original_row_size = offsets[rowid + 1] - offsets[rowid]
        chunk_offset = chunk_offsets[rowid]
        row_offset = 0

        for current_row_size in range(1, original_row_size + 1):
            original_row_offset = offsets[rowid] + max(0, current_row_size - max_elements)
            current_row_size = min(current_row_size, max_elements)
            for i in range(current_row_size):
                new_elements[chunk_offset + row_offset + i] = elements[original_row_offset + i]
            row_offset += current_row_size


@numba.cuda.jit
def _preserve_data(offsets, values, new_values):
    rowid = numba.cuda.grid(1)
    if rowid < offsets.size - 1:
        for i in range(offsets[rowid], offsets[rowid + 1]):
            new_values[i] = values[rowid]


@numba.cuda.jit
def _slice_rjust(max_elements, offsets, elements, new_offsets, new_elements):
    rowid = numba.cuda.grid(1)
    if rowid < new_offsets.size - 1:
        row_size = min(offsets[rowid + 1] - offsets[rowid], max_elements)
        offset = offsets[rowid + 1] - row_size
        new_start = new_offsets[rowid + 1] - row_size

        for i in range(row_size):
            new_elements[new_start + i] = elements[offset + i]


def slice_and_pad_left(seq_col, max_elements, pad_value=0):
    c = seq_col._column
    offsets = c.offsets.values
    elements = c.elements.values

    threads = THREADS
    blocks = (offsets.size + threads - 1) // threads

    new_offsets = cupy.arange(offsets.size, dtype=offsets.dtype) * max_elements

    new_elements = cupy.full(
        new_offsets[-1].item(), fill_value=pad_value, dtype=elements.dtype
    )
    _slice_rjust[blocks, threads](
        max_elements, offsets, elements, new_offsets, new_elements
    )

    new_col = nvt_build_list_column(new_elements, new_offsets)
    return new_col


class ExplodeSequence:
    """
    For each row create a new one with a subsequence of the original list columns.

    Keep at most `max_elements` of elements of a list.

    WARNING: All lists in the same row must have equal lengths!

    """

    def __init__(
        self,
        col_names: List[str],
        keep_cols: List[str],
        max_elements: int,
    ):
        self.col_names = col_names
        self.keep_cols = keep_cols
        self.max_elements = max_elements

        if not self.col_names:
            raise ValueError("`col_names` cannot be empty")

    def transform(self, df: cudf.DataFrame) -> cudf.DataFrame:
        ret = cudf.DataFrame()

        for col_name in self.col_names:
            c = df[col_name]._column
            offsets = c.offsets.values
            elements = c.elements.values

            threads = THREADS
            blocks = (offsets.size + threads - 1) // threads

            lengths = df[col_name].list.len().values

            sizes = cupy.minimum(lengths, self.max_elements)
            sizes = sizes * (sizes + 1) / 2
            truncated = cupy.maximum(lengths - self.max_elements, 0) * self.max_elements
            chunk_sizes = (sizes + truncated).astype(offsets.dtype)

            chunk_offsets = cupy.zeros(len(offsets), dtype=offsets.dtype)
            cupy.cumsum(chunk_sizes, dtype=offsets.dtype, out=chunk_offsets[1:])

            new_offsets_size = int(lengths.sum() + 1)
            new_elements_size = int(chunk_sizes.sum())

            new_offsets = cupy.zeros(new_offsets_size, dtype=offsets.dtype)
            _calculate_row_sizes[blocks, threads](
                offsets, new_offsets, self.max_elements
            )
            new_offsets = cupy.cumsum(new_offsets).astype(offsets.dtype)

            new_elements = cupy.zeros(new_elements_size, dtype=elements.dtype)
            _generate_rows[blocks, threads](
                offsets, chunk_offsets, elements, new_elements, self.max_elements
            )

            col = nvt_build_list_column(new_elements, new_offsets)
            ret[col_name] = col

        for col in self.keep_cols:
            new_values = cupy.zeros(new_offsets_size - 1, dtype=int)
            _preserve_data[blocks, threads](
                offsets, df[col].values, new_values
            )
            ret[col] = new_values

        ret = ret[self.keep_cols + self.col_names]
        return ret


def add_negative_sequence(seq_col, samples):
    c = seq_col._column
    offsets = c.offsets.values
    elements = c.elements.values

    new_offsets = offsets.copy()

    new_elements = cupy.empty_like(elements)
    new_elements = cupy.array(samples.to_gpu_array())

    col = nvt_build_list_column(new_elements, new_offsets)

    return col

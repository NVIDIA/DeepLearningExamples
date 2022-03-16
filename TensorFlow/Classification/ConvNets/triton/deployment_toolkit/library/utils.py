# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from collections import Counter
from typing import Callable, Dict, List

import networkx as nx

from ..core import ShapeSpec


def infer_precision(
    nx_graph: nx.Graph,
    input_names: List[str],
    output_names: List[str],
    get_node_dtype_fn: Callable,
):
    node_dtypes = [nx_graph.nodes[node_name].get("dtype", None) for node_name in nx_graph.nodes]
    node_dtypes = [dt for dt in node_dtypes if dt is None or dt.kind not in ["i", "b"]]
    dtypes_counter = Counter(node_dtypes)
    return dtypes_counter.most_common()[0][0]


def get_shapes_with_dynamic_axes(dataloader, batch_size_dim=0):
    def _set_dynamic_shapes(t, shapes):
        for k, v in t.items():
            shape = list(v.shape)
            for dim, s in enumerate(shape):
                if shapes[k][dim] != -1 and shapes[k][dim] != s:
                    shapes[k][dim] = -1

    ## get all shapes from input and output tensors
    input_shapes = {}
    output_shapes = {}
    for batch in dataloader:
        _, x, y = batch
        for k, v in x.items():
            input_shapes[k] = list(v.shape)
        for k, v in y.items():
            output_shapes[k] = list(v.shape)
        break

    # based on max <max_num_iters> iterations, check which
    # dimensions differ to determine dynamic_axes
    max_num_iters = 100
    for idx, batch in enumerate(dataloader):
        if idx >= max_num_iters:
            break

        _, x, y = batch

        _set_dynamic_shapes(x, input_shapes)
        _set_dynamic_shapes(y, output_shapes)

    return input_shapes, output_shapes


def get_dynamic_axes(dataloader, batch_size_dim=0):
    input_shapes, output_shapes = get_shapes_with_dynamic_axes(dataloader, batch_size_dim)
    all_shapes = {**input_shapes, **output_shapes}
    dynamic_axes = {}

    for k, shape in all_shapes.items():
        for idx, s in enumerate(shape):
            if s == -1:
                dynamic_axes[k] = {idx: k + "_" + str(idx)}

    for k, v in all_shapes.items():
        if k in dynamic_axes:
            dynamic_axes[k].update({batch_size_dim: "batch_size_" + str(batch_size_dim)})
        else:
            dynamic_axes[k] = {batch_size_dim: "batch_size_" + str(batch_size_dim)}

    return dynamic_axes


def get_input_shapes(dataloader, max_batch_size=1) -> Dict[str, ShapeSpec]:
    def init_counters_and_shapes(x, counters, min_shapes, max_shapes):
        for k, v in x.items():
            counters[k] = Counter()
            min_shapes[k] = [float("inf")] * v.ndim
            max_shapes[k] = [float("-inf")] * v.ndim

    counters = {}
    min_shapes: Dict[str, tuple] = {}
    max_shapes: Dict[str, tuple] = {}
    for idx, batch in enumerate(dataloader):
        ids, x, y = batch

        if idx == 0:
            init_counters_and_shapes(x, counters, min_shapes, max_shapes)

        for k, v in x.items():
            shape = v.shape
            counters[k][shape] += 1
            min_shapes[k] = tuple([min(a, b) for a, b in zip(min_shapes[k], shape)])
            max_shapes[k] = tuple([max(a, b) for a, b in zip(max_shapes[k], shape)])

    opt_shapes: Dict[str, tuple] = {}
    for k, v in counters.items():
        opt_shapes[k] = v.most_common(1)[0][0]

    shapes = {}
    for k in opt_shapes.keys():  # same keys in min_shapes and max_shapes
        shapes[k] = ShapeSpec(
            min=(1,) + min_shapes[k][1:],
            max=(max_batch_size,) + max_shapes[k][1:],
            opt=(max_batch_size,) + opt_shapes[k][1:],
        )
    return shapes

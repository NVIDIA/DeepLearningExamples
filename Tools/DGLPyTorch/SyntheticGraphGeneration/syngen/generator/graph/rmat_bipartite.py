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

import math
import os
import subprocess
from os.path import abspath
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np

from syngen.generator.graph.base_graph_generator import BaseBipartiteGraphGenerator
from syngen.generator.graph.fitter import BaseFitter
from syngen.generator.graph.utils import (
    effective_nonsquare_rmat_exact,
    generate_gpu_rmat,
    get_reversed_part,
    graph_to_snap_file,
    rearrange_graph,
    recreate_graph,
)


class RMATBipartiteGenerator(BaseBipartiteGraphGenerator):
    """ Graph generator based on RMAT that generate bipartite graphs
    Args:
        seed (int): Seed to reproduce the results. If None then random seed will be used.
        logdir (str): Directory to store the logging results.
        fitter (BaseFitter): Fitter to be used.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        logdir: str = "./logs",
        fitter: Optional[BaseFitter] = None,
        **kwargs,
    ):
        super().__init__(seed, logdir, fitter)
        self.gpu = True

    def fit(
        self,
        graph: List[Tuple[int, int]],
        src_set: Set[int],
        dst_set: Set[int],
        is_directed: bool,
    ):
        """ Fits generator on the graph
        Args:
            graph (List[Tuple[int, int]]): graph to be fitted on
            src_set (Set[int]): set of source nodes
            dst_set (Set[int]): set of destination nodes
            is_directed (bool): flag indicating whether the graph is directed
        """
        assert graph is not None, "Wrong graph"
        assert isinstance(src_set, set), "Source set must be of type set"
        assert isinstance(dst_set, set), "Destination set must be of type set"
        assert (
            len(src_set & dst_set) == 0
        ), "Source and destination sets must be disjoint"

        lower, upper = rearrange_graph(graph, src_set, dst_set)

        if (
            len(lower) and is_directed
        ):  # No need to fit lower part for undirected graph
            self._fit_dst_src_results = self.fitter.fit(lower)

        if len(upper):
            self._fit_src_dst_results = self.fitter.fit(upper)

        self.logger.log(f"Fit results dst_src: {self._fit_dst_src_results}")
        self.logger.log(f"Fit results src_dst: {self._fit_src_dst_results}")

    def _generate_part(
        self,
        fit_results: Tuple[float, float, float, float],
        part_shape: Tuple[int, int],
        num_edges: int,
        noise: float,
        batch_size: int,
    ):
        if self.gpu:
            return self._generate_part_gpu(
                fit_results=fit_results,
                part_shape=part_shape,
                num_edges=num_edges,
                noise=noise,
            )
        else:
            return self._generate_part_cpu(
                fit_results=fit_results,
                part_shape=part_shape,
                num_edges=num_edges,
                noise=noise,
                batch_size=batch_size,
            )

    def _generate_part_cpu(
        self,
        fit_results: Tuple[float, float, float, float],
        part_shape: Tuple[int, int],
        num_edges: int,
        noise: float,
        batch_size: int,
    ):

        a, b, c, d = fit_results
        theta = np.array([[a, b], [c, d]])
        theta /= a + b + c + d

        part, _, _ = effective_nonsquare_rmat_exact(
            theta,
            num_edges,
            part_shape,
            noise_scaling=noise,
            batch_size=batch_size,
            dtype=np.int64,
            custom_samplers=None,
            generate_back_edges=False,
            remove_selfloops=False,
        )

        return part

    def _generate_part_gpu(
        self,
        fit_results: Tuple[float, float, float, float],
        part_shape: Tuple[int, int],
        num_edges: int,
        noise: float,
    ):

        a, b, c, d = fit_results
        theta = np.array([a, b, c, d])
        theta /= a + b + c + d
        a, b, c, d = theta
        r_scale, c_scale = part_shape

        part = generate_gpu_rmat(
            a,
            b,
            c,
            d,
            r_scale=r_scale,
            c_scale=c_scale,
            n_edges=num_edges,
            noise=noise,
            is_directed=True,
            has_self_loop=True,
        )

        return part

    def generate(
        self,
        num_nodes_src_set: int,
        num_nodes_dst_set: int,
        num_edges_src_dst: int,
        num_edges_dst_src: int,
        is_directed: bool,
        noise: float = 0.5,
        batch_size: int = 1_000_000,
    ):
        """ Generates graph with approximately `num_nodes_src_set`/`num_nodes_dst_set` nodes
         and exactly `num_edges_src_dst`/`num_edges_dst_src` edges from generator
        Args:
            num_nodes_src_set (int): approximate number of source nodes to be generated
            num_nodes_dst_set (int): approximate number of destination nodes to be generated
            num_edges_src_dst (int): exact number of source->destination edges to be generated
            num_edges_dst_src (int): exact number of destination->source to be generated
            is_directed (bool): flag indicating whether the generated graph has to be directed
            noise (float): noise for RMAT generation to get better degree distribution
            batch_size (int): size of the edge chunk that will be generated in one generation step
        Returns:
            new_graph (np.array[int, int]): generated graph
        """
        assert (
            num_nodes_src_set > 0 and num_nodes_dst_set > 0
        ), "Wrong number of nodes"

        assert (
            num_edges_src_dst >= 0 and num_edges_dst_src >= 0
        ), "Wrong number of edges"

        max_edges = num_nodes_src_set * num_nodes_dst_set

        assert (
            num_edges_src_dst < max_edges and num_edges_dst_src < max_edges
        ), "Configuration of nodes nad edges cannot form any graph"

        assert (
            self._fit_src_dst_results or self._fit_dst_src_results
        ), "There are no fit results, \
        call fit method first or load the seeding matrix from the file"

        if (self._fit_dst_src_results is not None) != is_directed:
            requested = "directed" if is_directed else "undirected"
            fitted = "undirected" if requested == "directed" else "directed"
            raise RuntimeError(
                f"Fitted {fitted} graph but requested to generate {requested} one"
            )

        if not is_directed:
            assert (
                num_edges_src_dst == num_edges_dst_src
            ), "For undirected graph expected the same number of edges for each side"

            assert (
                self._fit_dst_src_results is None
            ), "For undirected graph expected only src->dst results to be present"

        log2_row = math.ceil(math.log2(num_nodes_src_set))
        log2_col = math.ceil(math.log2(num_nodes_dst_set))
        part_shape_upper = (log2_row, log2_col)
        part_shape_lower = (log2_col, log2_row)

        offset = int(2 ** log2_row)

        if self._fit_src_dst_results and num_edges_src_dst:
            upper_part = self._generate_part(
                self._fit_src_dst_results,
                part_shape_upper,
                num_edges_src_dst,
                noise,
                batch_size,
            )
        else:
            upper_part = []

        if self._fit_dst_src_results:
            if num_edges_dst_src:
                lower_part = self._generate_part(
                    self._fit_dst_src_results,
                    part_shape_lower,
                    num_edges_dst_src,
                    noise,
                    batch_size,
                )
            else:
                lower_part = []
        elif not is_directed:  # Recreate lower part for undirected graph
            lower_part = get_reversed_part(upper_part)
        else:
            lower_part = []

        new_graph = recreate_graph(lower_part, upper_part, offset)
        return new_graph

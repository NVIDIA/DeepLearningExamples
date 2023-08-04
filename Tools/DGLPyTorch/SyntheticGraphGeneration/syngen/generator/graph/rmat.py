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
from typing import List, Optional, Set, Tuple

import numpy as np

from syngen.generator.graph.fitter import RMATFitter
from syngen.generator.graph.base_graph_generator import BaseGraphGenerator
from syngen.generator.graph.utils import (
    effective_nonsquare_rmat_exact,
    generate_gpu_rmat,
    generate_gpu_chunked_rmat,
)


class RMATGenerator(BaseGraphGenerator):
    """ Graph generator based on RMAT that generate non-partite graphs
    Args:
        seed (int): Seed to reproduce the results. If None then random seed will be used.
        logdir (str): Directory to store the logging results.
        fitter (RMATFitter): RMATFitter to be used.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        logdir: str = "./logs",
        gpu: bool = True,
        fitter: Optional[RMATFitter] = None,
        **kwargs,
    ):
        super().__init__(seed, logdir, gpu)
        self.fitter = fitter or RMATFitter()

    def fit(self, *args, **kwargs):
        """ Fits generator on the graph
        Args:

        """
        self._fit_results = self.fitter.fit(*args, **kwargs)
        self.logger.log(f"Fit results: {self._fit_results}")

    def _generate_part(
        self,
        fit_results: Tuple[float, float, float, float],
        part_shape: Tuple[int, int],
        num_edges: int,
        has_self_loop: bool,
        is_directed: bool,
        noise: float,
        batch_size: int,
        return_node_ids: bool,
        save_path: Optional[str],
    ):
        if self.gpu:
            return self._generate_part_gpu(
                fit_results=fit_results,
                part_shape=part_shape,
                num_edges=num_edges,
                has_self_loop=has_self_loop,
                is_directed=is_directed,
                noise=noise,
                return_node_ids=return_node_ids,
                save_path=save_path,
            )
        else:
            return self._generate_part_cpu(
                fit_results=fit_results,
                part_shape=part_shape,
                num_edges=num_edges,
                has_self_loop=has_self_loop,
                is_directed=is_directed,
                noise=noise,
                batch_size=batch_size,
                return_node_ids=return_node_ids,
            )

    def _generate_part_cpu(
        self,
        fit_results: Tuple[float, float, float, float],
        part_shape: Tuple[int, int],
        num_edges: int,
        has_self_loop: bool,
        is_directed: bool,
        noise: float,
        batch_size: int,
        return_node_ids: bool,
    ):

        a, b, c, d = fit_results
        theta = np.array([[a, b], [c, d]])
        theta /= a + b + c + d

        res = effective_nonsquare_rmat_exact(
            theta,
            num_edges,
            part_shape,
            noise_scaling=noise,
            batch_size=batch_size,
            dtype=np.int64,
            custom_samplers=None,
            generate_back_edges=not is_directed,
            remove_selfloops=not has_self_loop,
            return_node_ids=return_node_ids,
            verbose=self.verbose,
        )
        if return_node_ids:
            return res[0], res[1]
        return res[0]

    def _generate_part_gpu(
        self,
        fit_results: Tuple[float, float, float, float],
        part_shape: Tuple[int, int],
        num_edges: int,
        has_self_loop: bool,
        is_directed: bool,
        noise: float,
        return_node_ids: bool,
        save_path: Optional[str],
        _chunked: bool = True,
    ):

        a, b, c, d = fit_results
        theta = np.array([a, b, c, d])
        theta /= a + b + c + d
        a, b, c, d = theta

        r_scale, c_scale = part_shape

        if _chunked:
            res = generate_gpu_chunked_rmat(
                a,
                b,
                c,
                d,
                r_scale=r_scale,
                c_scale=c_scale,
                n_edges=num_edges,
                noise=noise,
                is_directed=is_directed,
                has_self_loop=has_self_loop,
                return_node_ids=1 if return_node_ids else 0,
                save_path=save_path,
                verbose=self.verbose,
            )
        else:
            res = generate_gpu_rmat(
                a,
                b,
                c,
                d,
                r_scale=r_scale,
                c_scale=c_scale,
                n_edges=num_edges,
                noise=noise,
                is_directed=is_directed,
                has_self_loop=has_self_loop,
                return_node_ids=1 if return_node_ids else 0,
            )
        if return_node_ids:
            return res[0], res[1]
        return res

    def generate(
        self,
        num_nodes: int,
        num_edges: int,
        is_directed: bool,
        has_self_loop: bool,
        noise: float = 0.5,
        batch_size: int = 1_000_000,
        return_node_ids: bool = False,
        save_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """ Generates graph with approximately `num_nodes` nodes and exactly `num_edges` edges from generator
        Args:
            num_nodes (int): approximate number of nodes to be generated
            num_edges(int): exact number of edges to be generated
            is_directed (bool):  flag indicating whether the generated graph has to be directed
            has_self_loop (bool): flag indicating whether to generate self loops
            noise (float): noise for RMAT generation to get better degree distribution
            batch_size (int): size of the edge chunk that will be generated in one generation step (cpu parameter)
            return_node_ids (bool): flag indicating whether the generator has to return nodes_ids as the second output
            save_path (bool): path to store the graph. if specified the method return the number of edges in the graph
        Returns:
            new_graph (np.array[int, int]): generated graph
        """
        assert num_nodes > 0, "Wrong number of nodes"
        assert num_edges > 0, "Wrong number of edges"

        max_edges = (
            num_nodes * num_nodes
            if has_self_loop
            else num_nodes * (num_nodes - 1)
        )
        if is_directed:
            max_edges = max_edges / 2

        assert (
            num_edges < max_edges
        ), "Configuration of nodes and edges cannot form any graph"

        assert (
            self._fit_results
        ), "There are no fit results, call fit method first or load the seeding matrix from the file"

        log2_nodes = math.ceil(math.log2(num_nodes))
        part_shape = (log2_nodes, log2_nodes)

        res = self._generate_part(
            fit_results=self._fit_results,
            part_shape=part_shape,
            num_edges=num_edges,
            has_self_loop=has_self_loop,
            is_directed=is_directed,
            noise=noise,
            batch_size=batch_size,
            return_node_ids=return_node_ids,
            save_path=save_path,
        )

        if return_node_ids:
            return res[0], res[1]
        return res

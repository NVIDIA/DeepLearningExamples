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
import warnings
from typing import List, Optional, Set, Tuple

import numpy as np

from syngen.generator.graph.base_graph_generator import BaseBipartiteGraphGenerator
from syngen.generator.graph.fitter import RMATFitter
from syngen.generator.graph.utils import (
    effective_nonsquare_rmat_exact,
    generate_gpu_rmat,
    get_reversed_part,
    rearrange_graph,
    recreate_graph, generate_gpu_chunked_rmat,
)


class RMATBipartiteGenerator(BaseBipartiteGraphGenerator):
    """ Graph generator based on RMAT that generate bipartite graphs
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

    def fit(
        self,
        graph: List[Tuple[int, int]],
        src_set: Optional[Set[int]],
        dst_set: Optional[Set[int]],
        is_directed: bool,
        transform_graph: bool = True,
    ):
        """ Fits generator on the graph
        Args:
            graph (List[Tuple[int, int]]): graph to be fitted on
            transform_graph (bool): defines if the generator should transform the input graph using src and dst node sets
            src_set (Set[int]): set of source nodes
            dst_set (Set[int]): set of destination nodes
            is_directed (bool): flag indicating whether the graph is directed

        """
        assert graph is not None, "Wrong graph"

        if transform_graph:
            lower, upper = rearrange_graph(graph, src_set, dst_set, assume_unique=True)
        else:
            assert not is_directed
            upper = graph
            lower = []

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
        return_node_ids: bool,
        save_path: Optional[str],
    ):
        if self.gpu:
            return self._generate_part_gpu(
                fit_results=fit_results,
                part_shape=part_shape,
                num_edges=num_edges,
                noise=noise,
                return_node_ids=return_node_ids,
                save_path=save_path,
            )
        else:
            return self._generate_part_cpu(
                fit_results=fit_results,
                part_shape=part_shape,
                num_edges=num_edges,
                noise=noise,
                batch_size=batch_size,
                return_node_ids=return_node_ids,
            )

    def _generate_part_cpu(
        self,
        fit_results: Tuple[float, float, float, float],
        part_shape: Tuple[int, int],
        num_edges: int,
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
            generate_back_edges=False,
            remove_selfloops=False,
            return_node_ids=2 if return_node_ids else 0,
            verbose=self.verbose,
        )
        if return_node_ids:
            return res[0], res[1], res[2]
        return res[0]

    def _generate_part_gpu(
        self,
        fit_results: Tuple[float, float, float, float],
        part_shape: Tuple[int, int],
        num_edges: int,
        noise: float,
        return_node_ids: bool,
        save_path: Optional[str] = None,
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
                is_directed=True,
                has_self_loop=True,
                return_node_ids=2 if return_node_ids else 0,
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
                is_directed=True,
                has_self_loop=True,
                return_node_ids=2 if return_node_ids else 0
            )
        if return_node_ids:
            return res[0], res[1], res[2]
        return res

    def generate(
        self,
        num_nodes_src_set: int,
        num_nodes_dst_set: int,
        num_edges_src_dst: int,
        num_edges_dst_src: int,
        is_directed: bool,
        apply_edge_mirroring = True,
        transform_graph: bool = True,
        noise: float = 0.5,
        batch_size: int = 1_000_000,
        return_node_ids=False,
        save_path: Optional[str] = None,
    ):
        """ Generates graph with approximately `num_nodes_src_set`/`num_nodes_dst_set` nodes
         and exactly `num_edges_src_dst`/`num_edges_dst_src` edges from generator
        Args:
            num_nodes_src_set (int): approximate number of source nodes to be generated
            num_nodes_dst_set (int): approximate number of destination nodes to be generated
            num_edges_src_dst (int): exact number of source->destination edges to be generated
            num_edges_dst_src (int): exact number of destination->source to be generated
            is_directed (bool): flag indicating whether the generated graph has to be directed
            transform_graph (bool): defines if the generator should transform the output graph to avoid node id conflict between src and dst nodes
            noise (float): noise for RMAT generation to get better degree distribution
            batch_size (int): size of the edge chunk that will be generated in one generation step
            return_node_ids (bool): flag indicating whether the generator has to return nodes_ids as the second output
            save_path (bool): path to store the graph. if specified the method return the number of edges in the graph
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

        if apply_edge_mirroring and is_directed:
            warnings.warn('edge mirroring works only for undirected graphs')

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
            upper_part_res = self._generate_part(
                self._fit_src_dst_results,
                part_shape_upper,
                num_edges_src_dst,
                noise,
                batch_size,
                return_node_ids=return_node_ids,
                save_path=save_path,
            )
            if return_node_ids:
                upper_part, upper_part_src_node_ids, upper_part_dst_node_ids = upper_part_res
            else:
                upper_part = upper_part_res
        else:
            upper_part = []

        if self._fit_dst_src_results:
            if save_path is not None:
                raise NotImplementedError('save_path works only for undirected bipartite graphs')
            if num_edges_dst_src:
                lower_part_res = self._generate_part(
                    self._fit_dst_src_results,
                    part_shape_lower,
                    num_edges_dst_src,
                    noise,
                    batch_size,
                    save_path=save_path,
                    return_node_ids=return_node_ids,
                )
                if return_node_ids:
                    lower_part, lower_part_src_node_ids, lower_part_dst_node_ids = lower_part_res
                else:
                    lower_part = lower_part_res
            else:
                lower_part = []
        elif not is_directed and apply_edge_mirroring:  # Recreate lower part for undirected graph
            if return_node_ids:
                lower_part_src_node_ids, lower_part_dst_node_ids = upper_part_dst_node_ids, upper_part_src_node_ids
            lower_part = get_reversed_part(upper_part)
        else:
            lower_part = []

        if transform_graph:
            new_graph = recreate_graph(lower_part, upper_part, offset)
            if return_node_ids:
                lower_part_src_node_ids = lower_part_src_node_ids + offset
                upper_part_dst_node_ids = upper_part_dst_node_ids + offset
                src_node_ids = np.union1d(upper_part_src_node_ids, lower_part_dst_node_ids)
                dst_node_ids = np.union1d(upper_part_dst_node_ids, lower_part_src_node_ids)
        else:
            if apply_edge_mirroring:
                raise NotImplementedError('apply edge mirroring works only with `transform_graph=True`')
            new_graph = upper_part
            if return_node_ids:
                src_node_ids, dst_node_ids = upper_part_src_node_ids, upper_part_dst_node_ids

        if return_node_ids:
            return new_graph, src_node_ids, dst_node_ids
        return new_graph

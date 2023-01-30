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

from typing import Any, List, Optional, Set, Tuple

from syngen.generator.graph.fitter import RandomFitter
from syngen.generator.graph.rmat_bipartite import RMATBipartiteGenerator


class RandomBipartite(RMATBipartiteGenerator):
    """ Graph generator based on erdos-renyi model that generate random bipartite graphs
    Args:
        seed (int):
            Seed to reproduce the results. If None then random seed will be used.
        logdir (str):
            Directory to store the logging results.
            Defaults to ./logs.
        fitter (BaseFitter):
            Fitter to be used.
    """

    def __init__(self, seed: Optional[int] = None, logdir: str = "./logs", **kwargs,):
        fitter = RandomFitter()
        super().__init__(seed, logdir, fitter)
        self.fit()

    def fit(
        self,
        graph: Optional[List[Tuple[int, int]]] = None,
        src_set: Optional[Set[int]] = None,
        dst_set: Optional[Set[int]] = None,
        is_directed: bool = False,
    ):
        """ Fits generator on the graph. For random graph it is graph independent."""

        self._fit_src_dst_results = self.fitter.fit(graph)
        self._fit_dst_src_results = (
            None if not is_directed else self.fitter.fit(graph)
        )

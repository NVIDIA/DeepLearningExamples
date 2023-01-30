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

import time

import matplotlib.pyplot as plt
import pandas as pd

from syngen.analyzer.graph.plotting import (
    plot_clustering_coef_distribution,
    plot_degree_distribution,
    plot_eigenvalue_histogram_distribution,
    plot_eigenvalue_rank_distribution,
    plot_hopplot,
    plot_in_degree_distribution,
    plot_leading_singular_vector_rank,
    plot_out_degree_distribution,
    plot_singular_value_histogram_distribution,
    plot_singular_value_rank_distribution,
    plot_strongly_connected_component_distribution,
    plot_weakly_connected_component_distribution,
)
from syngen.analyzer.graph.stats import (
    get_connectivity,
    get_global_stats,
    get_path_stats,
    get_transitivity,
)
from syngen.analyzer.graph.utils import timed


class AnalysisModule:
    @staticmethod
    def check_assertions(graphs):
        assert len(graphs), "Expected at least 1 graph"
        assert (
            len(set([graph.is_directed for graph in graphs])) == 1
        ), "All graphs have to be directed or undirected"

    @staticmethod
    def maybe_wrap_timer(f, timer, title):
        return timed(f, title) if timer else f

    def compare_graph_stats(
        self,
        *graphs,
        global_stats=True,
        connectivity=True,
        transitivity=True,
        path_stats=True,
        timer=False,
        fast=True,
    ):

        self.check_assertions(graphs)
        results = []
        category_functions = []

        if global_stats:
            category_functions.append(("Global stats", get_global_stats))
        if connectivity:
            category_functions.append(("Connectivity", get_connectivity))
        if transitivity:
            category_functions.append(("Transitivity", get_transitivity))
        if path_stats:
            category_functions.append(("Path stats", get_path_stats))

        for category, F in category_functions:

            start = time.perf_counter()
            stats = [F(G, fast=fast) for G in graphs]
            parsed = [
                tuple(
                    [category, statistic]
                    + [graph_stats[statistic] for graph_stats in stats]
                )
                for statistic in stats[0]
            ]
            results += parsed

            if timer:
                elapsed = time.perf_counter() - start
                print(f'Category "{category}" took {elapsed:.2f}s')

        names = [
            graph.name if graph.name else f"G{i}"
            for i, graph in enumerate(graphs, 1)
        ]
        columns = ["Category", "Statistic"] + names
        return pd.DataFrame(results, columns=columns)

    def compare_graph_plots(self, *graphs, hop_plot_iters=128, timer=False):

        self.check_assertions(graphs)

        is_directed = graphs[0].is_directed

        if is_directed:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            ax1, ax2, ax3, ax4 = ax3, ax4, ax1, ax2
            fig.set_size_inches(18, 6 * 2, forward=True)
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 6, forward=True)

        pdd = self.maybe_wrap_timer(
            plot_degree_distribution, timer, "Degree distribution"
        )
        pidd = self.maybe_wrap_timer(
            plot_in_degree_distribution, timer, "In degree distribution"
        )
        podd = self.maybe_wrap_timer(
            plot_out_degree_distribution, timer, "Out degree distribution"
        )
        ph = self.maybe_wrap_timer(plot_hopplot, timer, "Hop plot")

        if is_directed:
            pidd(ax3, *graphs)
            podd(ax4, *graphs)
        pdd(ax1, *graphs)
        ph(ax2, *graphs, hop_plot_iters=hop_plot_iters)

        return fig

    def compare_graph_dd(self, *graphs, timer=False):

        self.check_assertions(graphs)

        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(18.5, 10.5, forward=True)
        pdd = (
            timed(plot_degree_distribution, "Degree distribution")
            if timer
            else plot_degree_distribution
        )

        pdd(ax1, *graphs)

        return fig

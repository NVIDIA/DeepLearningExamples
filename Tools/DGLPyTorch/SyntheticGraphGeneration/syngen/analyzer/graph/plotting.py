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

import logging
import os
from functools import partial
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh

from syngen.analyzer.graph.graph import safeSNAP
from syngen.utils.types import ColumnType

TMP_NAME = "tmp"


def common_plot(f, ax, *graphs, **kwargs):
    for i, G in enumerate(graphs, 1):
        f(G, i, ax, **kwargs)

    if len(graphs) > 1:
        ax.legend()


def parse_file(plot, filename):
    parsed_filename = f"{plot}.{filename}.tab"
    with open(parsed_filename, "r") as f:
        lines = f.read().splitlines()

    x_values = []
    y_values = []
    for line in lines:
        if len(line) and "#" not in line:
            x, y = line.split()
            x_values.append(float(x))
            y_values.append(float(y))

    return x_values, y_values


def clear_files(plot, filename):
    files_to_clean = [
        f"./{plot}.{filename}.plt",
        f"./{plot}.{filename}.png",
        f"./{plot}.{filename}.tab",
    ]
    for file in files_to_clean:
        try:
            os.remove(file)
        except FileNotFoundError:
            print(f"File {file} attempted to be removed, but not found")


def parse_snap_object(snap_object):
    zipped = [(pair.GetVal1(), pair.GetVal2()) for pair in snap_object]
    x, y = zip(*zipped)
    return x, y


def get_degree_dist(snapGraph):
    return parse_snap_object(snapGraph.GetDegCnt())


def get_in_degree_dist(snapGraph):
    return parse_snap_object(snapGraph.GetInDegCnt())


def get_out_degree_dist(snapGraph):
    return parse_snap_object(snapGraph.GetOutDegCnt())


def get_clustering_coef_dist(snapGraph):
    return parse_snap_object(snapGraph.GetClustCf(True, -1)[1])


def get_strongly_connected_component(snapGraph):
    return parse_snap_object(snapGraph.GetSccSzCnt())


def get_weakly_connected_component(snapGraph):
    return parse_snap_object(snapGraph.GetWccSzCnt())


@safeSNAP
def _add_to_axis_idd(G, i, ax):
    graph_name = G.name or f"Graph {i}"
    title = "Log-log in degree distribution"
    G = G.snapGraph
    x, y = get_in_degree_dist(G)
    ax.set_xscale("log")
    ax.set_xlabel("In degree")
    ax.set_yscale("log")
    ax.set_ylabel("Number of nodes")
    ax.set_title(title)
    ax.scatter(x, y, label=graph_name, s=5)


@safeSNAP
def _add_to_axis_odd(G, i, ax):
    graph_name = G.name or f"Graph {i}"
    title = "Log-log out degree distribution"
    G = G.snapGraph
    x, y = get_out_degree_dist(G)
    ax.set_xscale("log")
    ax.set_xlabel("Out degree")
    ax.set_yscale("log")
    ax.set_ylabel("Number of nodes")
    ax.set_title(title)
    ax.scatter(x, y, label=graph_name, s=5)


@safeSNAP
def _add_to_axis_dd(G, i, ax):
    graph_name = G.name or f"Graph {i}"
    title = "Log-log degree distribution"
    G = G.snapGraph
    x, y = get_degree_dist(G)
    ax.set_xscale("log")
    ax.set_xlabel("Degree")
    ax.set_yscale("log")
    ax.set_ylabel("Number of nodes")
    ax.set_title(title)
    ax.scatter(x, y, label=graph_name, s=5)


@safeSNAP
def _add_to_axis_ccd(G, i, ax):
    graph_name = G.name or f"Graph {i}"
    title = "Log-log distribution of clustering coefficient"
    G = G.snapGraph
    x, y = get_clustering_coef_dist(G)
    ax.set_xscale("log")
    ax.set_xlabel("Degree")
    ax.set_yscale("symlog")
    ax.set_ylabel("Clustering coefficient")
    ax.set_title(title)
    ax.scatter(x, y, label=graph_name, s=5)


@safeSNAP
def _add_to_axis_scc(G, i, ax):
    graph_name = G.name or f"Graph {i}"
    title = "Log-log distribution of sizes of strongly connected components"
    G = G.snapGraph
    x, y = get_strongly_connected_component(G)
    ax.set_xscale("log")
    ax.set_xlabel("Size of strongly connected component")
    ax.set_yscale("symlog")
    ax.set_ylabel("Number of components")
    ax.set_title(title)
    ax.scatter(x, y, label=graph_name, s=5)


@safeSNAP
def _add_to_axis_wcc(G, i, ax):
    is_directed = G.is_directed
    weakly_string = " weakly " if is_directed else " "
    title = (
        f"Log-log distribution of sizes of{weakly_string}connected components"
    )
    graph_name = G.name or f"Graph {i}"
    G = G.snapGraph
    x, y = get_weakly_connected_component(G)
    ax.set_xscale("log")
    ax.set_xlabel(f"Size of{weakly_string}connected component")
    ax.set_yscale("symlog")
    ax.set_ylabel("Number of components")
    ax.set_title(title)
    ax.scatter(x, y, label=graph_name, s=5)


@safeSNAP
def _add_to_axis_hp(G, i, ax, hop_plot_iters=128):
    is_directed = G.is_directed
    graph_name = G.name or f"Graph {i}"
    title = "Hop plot"
    plot = "hop"
    G = G.snapGraph
    G.PlotHops(TMP_NAME, "Hop plot", is_directed, hop_plot_iters)
    num_hops, num_nodes = parse_file(plot=plot, filename=TMP_NAME)
    num_hops = [int(num_hop) for num_hop in num_hops]
    parse_file(plot=plot, filename=TMP_NAME)
    clear_files(plot=plot, filename=TMP_NAME)
    ax.set_xlabel("Number of hops")
    ax.set_ylabel("Number of pairs of nodes")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.plot(num_hops, num_nodes, "--", marker="o", label=graph_name)


@safeSNAP
def _add_to_axis_svr(G, i, ax, num_spectral_values=100):
    graph_name = G.name or f"Graph {i}"
    title = "Singular value rank distribution"
    plot = "sngVal"
    G = G.snapGraph
    G.PlotSngValRank(num_spectral_values, TMP_NAME, title)
    ranks, sin_values = parse_file(plot, filename=TMP_NAME)
    ranks = [int(rank) for rank in ranks]
    parse_file(plot=plot, filename=TMP_NAME)
    clear_files(plot=plot, filename=TMP_NAME)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Singular value")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.plot(
        ranks, sin_values, "--", marker="o", label=graph_name, markersize=5
    )


@safeSNAP
def _add_to_axis_evr(G, i, ax, num_spectral_values=100):
    graph_name = G.name or f"Graph {i}"
    title = "Eigenvalue rank distribution"
    plot = "eigVal"
    G = G.snapGraph
    G.PlotEigValRank(num_spectral_values, TMP_NAME, title)
    ranks, eig_values = parse_file(plot, filename=TMP_NAME)
    ranks = [int(rank) for rank in ranks]
    parse_file(plot=plot, filename=TMP_NAME)
    clear_files(plot=plot, filename=TMP_NAME)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Eigenvalue")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.plot(
        ranks, eig_values, "--", marker="o", label=graph_name, markersize=5
    )


@safeSNAP
def _add_to_axis_svd(G, i, ax, num_spectral_values=100):
    graph_name = G.name or f"Graph {i}"
    title = "Singular value distribution"
    plot = "sngDistr"
    G = G.snapGraph
    G.PlotSngValDistr(num_spectral_values, TMP_NAME, title)
    sin_values, counts = parse_file(plot=plot, filename=TMP_NAME)
    parse_file(plot=plot, filename=TMP_NAME)
    clear_files(plot=plot, filename=TMP_NAME)
    ax.set_xlabel("Singular value")
    ax.set_ylabel("Count")
    ax.set_yscale("symlog")
    ax.set_title(title)
    ax.plot(
        sin_values, counts, "--", marker="o", label=graph_name, markersize=5
    )


@safeSNAP
def _add_to_axis_evd(G, i, ax, num_spectral_values=100):
    graph_name = G.name or f"Graph {i}"
    title = "Eigenvalue distribution"
    plot = "eigDistr"
    G = G.snapGraph
    G.PlotEigValDistr(num_spectral_values, TMP_NAME, title)
    eig_values, counts = parse_file(plot, filename=TMP_NAME)
    parse_file(plot=plot, filename=TMP_NAME)
    clear_files(plot=plot, filename=TMP_NAME)
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Count")
    ax.set_yscale("symlog")
    ax.set_title(title)
    ax.plot(
        eig_values, counts, "--", marker="o", label=graph_name, markersize=5
    )


@safeSNAP
def _add_to_axis_lsv(G, i, ax):
    graph_name = G.name or f"Graph {i}"
    title = "Leading singular vector rank distribution"
    plot = "sngVecL"
    G = G.snapGraph
    G.PlotSngVec(TMP_NAME, title)
    ranks, components = parse_file(plot, filename=TMP_NAME)
    ranks = [int(rank) for rank in ranks]
    parse_file(plot=plot, filename=TMP_NAME)
    clear_files(plot=plot, filename=TMP_NAME)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Component of leading singular vector")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.plot(
        ranks, components, "--", marker="o", label=graph_name, markersize=5
    )


def plot_node_degree_centrality_feat_dist(
    data,
    feat_name_col_info: Dict[str, ColumnType],
    src_col: str = "src",
    dst_col: str = "dst",
):

    # - suppress matplotlib debug logger
    matplotlib_logger = logging.getLogger("matplotlib")
    matplotlib_logger.setLevel(logging.WARNING)

    src_degree = (
        data.groupby(src_col, as_index=False)
        .count()[[src_col, dst_col]]
        .rename(columns={dst_col: "src_degree"})
    )

    # - normalized src_degree
    src_degree_vals = src_degree["src_degree"].values
    normalized_src_degree = (src_degree_vals - np.min(src_degree_vals)) / (
        np.max(src_degree_vals) - np.min(src_degree_vals)
    )
    src_degree.loc[:, "src_degree"] = normalized_src_degree

    # - normalized dst_degree
    dst_degree = (
        data.groupby(dst_col, as_index=False)
        .count()[[src_col, dst_col]]
        .rename(columns={src_col: "dst_degree"})
    )
    dst_degree_vals = dst_degree["dst_degree"].values
    normalized_dst_degree = (dst_degree_vals - np.min(dst_degree_vals)) / (
        np.max(dst_degree_vals) - np.min(dst_degree_vals)
    )

    dst_degree.loc[:, "dst_degree"] = normalized_dst_degree

    # - merge
    data = data.merge(src_degree, how="outer", on=src_col)
    data = data.merge(dst_degree, how="outer", on=dst_col)

    # - normalize continuous columns
    for feat, col_info in feat_name_col_info.items():
        col_type = col_info["type"]
        if col_type == ColumnType.CONTINUOUS:
            vals = data[feat].values
            min_, max_ = np.min(vals), np.max(vals)
            data.loc[:, feat] = (vals - min_) / (max_ - min_)

    # - plot heat maps
    def heat_map(x, y):
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=30)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        return heatmap.T, extent

    nr = 1  # - num plots per row
    fig, axs = plt.subplots(len(feat_name_col_info), nr, figsize=(12, 8))

    c = 0
    for feat in feat_name_col_info:
        if nr * len(feat_name_col_info) == 1:
            heatmap, extent = heat_map(
                data["src_degree"].values, data[feat].values
            )
            axs.imshow(heatmap, extent=extent, origin="lower")
            axs.set_xlabel("src_degree")
            axs.set_ylabel("feat")
        else:
            # - src degree dist
            heatmap, extent = heat_map(
                data["src_degree"].values, data[feat].values
            )
            axs[c].imshow(heatmap, extent=extent, origin="lower")
            axs[c].set_xlabel("src_degree")
            axs[c].set_ylabel("feat")
            c += nr

    return fig


# Degree distribution
plot_degree_distribution = partial(common_plot, _add_to_axis_dd)
# In degree distribution
plot_in_degree_distribution = partial(common_plot, _add_to_axis_idd)
# Out degree distribution
plot_out_degree_distribution = partial(common_plot, _add_to_axis_odd)
# Hop plot
plot_hopplot = partial(common_plot, _add_to_axis_hp)
# Clustering coefficient distribution
plot_clustering_coef_distribution = partial(common_plot, _add_to_axis_ccd)
# Strongly connected component distribution
plot_strongly_connected_component_distribution = partial(
    common_plot, _add_to_axis_scc
)
# Weakly connected component distribution
plot_weakly_connected_component_distribution = partial(
    common_plot, _add_to_axis_wcc
)
# Eigenvalue rank distribution
plot_eigenvalue_rank_distribution = partial(common_plot, _add_to_axis_evr)
# Singular value rank distribution
plot_singular_value_rank_distribution = partial(common_plot, _add_to_axis_svr)
# Eigenvalue rank distribution
plot_eigenvalue_histogram_distribution = partial(common_plot, _add_to_axis_evd)
# Singular value rank distribution
plot_singular_value_histogram_distribution = partial(
    common_plot, _add_to_axis_svd
)
# Leading singular vector rank distribution
plot_leading_singular_vector_rank = partial(common_plot, _add_to_axis_lsv)

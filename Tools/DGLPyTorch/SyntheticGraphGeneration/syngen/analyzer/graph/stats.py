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

from copy import deepcopy
from operator import itemgetter

import numpy as np

from syngen.analyzer.graph.graph import safeSNAP


def get_normalised_cdf(nodes, cdf_points=100, debug=False):
    unique_nodes, unique_nodes_counts = np.unique(nodes, return_counts=True)
    node_degree, node_degree_counts = np.unique(
        unique_nodes_counts, return_counts=True
    )
    if debug:
        print(
            "unique_nodes,unique_nodes_counts",
            unique_nodes,
            unique_nodes_counts,
        )
        print(
            "node_degree,node_degree_counts", node_degree, node_degree_counts
        )
    node_degree_normalized = (
        node_degree / node_degree[-1]
    )  # they are sorted, so [-1] is max
    node_degree_counts_normalized = node_degree_counts / np.sum(
        node_degree_counts
    )  # to have density
    if debug:
        print(
            "node_degree_normalized,node_degree_counts_normalized",
            node_degree_normalized,
            node_degree_counts_normalized,
        )
        plt.plot(node_degree_normalized, node_degree_counts_normalized)
        plt.yscale("log")
        plt.xscale("log")
        plt.title("DD normalized log-log")
        plt.show()
    F = np.cumsum(node_degree_counts_normalized)
    cdf_points_for_F = (np.logspace(0, 1, num=cdf_points) - 1) / 9
    F_normalized = np.zeros(shape=(cdf_points_for_F.shape[0], 2))
    F_normalized[:, 0] = cdf_points_for_F
    for i, p in enumerate(cdf_points_for_F):
        matching_list = F[node_degree_normalized <= p]
        F_normalized[i, 1] = matching_list[-1] if len(matching_list) else 0.0
    if debug:
        print("F_normalized", F_normalized)
        plt.plot(F_normalized[:, 0], F_normalized[:, 1])
        plt.plot(node_degree_normalized, F)
        plt.yscale("log")
        plt.xscale("log")
        plt.title("Normalized CDF of DD normalized log-log ")
        plt.show()
    return F_normalized


# Global stats
@safeSNAP
def get_global_stats(G, *args, **kwargs):
    is_directed = G.is_directed

    G = G.snapGraph
    num_nodes = G.GetNodes()
    num_edges = G.GetEdges()

    density = num_edges / ((num_nodes - 1) * num_nodes) if num_nodes > 1 else 0

    if not is_directed:
        density = 2 * density

    average_degree = num_edges / num_nodes if num_nodes else 0
    self_loops = G.CntSelfEdges()

    zero_degrees = num_nodes - G.CntNonZNodes()
    zero_in_degrees = len(
        [item.GetVal2() for item in G.GetNodeInDegV() if item.GetVal2() == 0]
    )
    zero_out_degrees = len(
        [item.GetVal2() for item in G.GetNodeOutDegV() if item.GetVal2() == 0]
    )
    uniq_bidirectional = G.CntUniqBiDirEdges()
    uniq_undirected = G.CntUniqUndirEdges()
    uniq_directed = G.CntUniqDirEdges()

    return {
        "Nodes": num_nodes,
        "Edges": num_edges,
        "Density": around(density, 4),
        "Average degree": around(average_degree, 2),
        "Zero deg nodes": zero_degrees,
        "Zero in deg nodes": zero_in_degrees,
        "Zero out deg nodes": zero_out_degrees,
        "Self loops": self_loops,
        "Bidirectional edges": uniq_bidirectional,
        "Unique undirected edges": uniq_undirected,
        "Unique directed edges": uniq_directed,
    }


# Connectivity
@safeSNAP
def get_connectivity(G, *args, **kwargs):
    is_directed = G.is_directed
    G = G.snapGraph

    def get_stats(component_dist_snap):
        component_dist = [
            (comp.GetVal1(), comp.GetVal2()) for comp in component_dist_snap
        ]
        if len(component_dist):
            largest_component = max(component_dist, key=itemgetter(0))[0]
        else:
            largest_component = 0
        number_of_components = sum(
            num_component for size, num_component in component_dist
        )
        percent = 100 * largest_component / G.GetNodes()
        return number_of_components, percent

    # Weakly connected components
    number_of_weak_components, percent_of_weak = get_stats(G.GetWccSzCnt())
    is_weakly_connected = number_of_weak_components == 1

    if is_directed:
        # Strongly connected components
        number_of_strong_components, percent_of_strong = get_stats(
            G.GetSccSzCnt()
        )
        is_strongly_connected = number_of_strong_components == 1

        result = {
            "Is strongly connected": is_strongly_connected,
            "Is weakly connected": is_weakly_connected,
            "Number of strongly connected components": number_of_strong_components,
            "Percent of nodes in largest strongly connected component": around(
                percent_of_strong
            ),
            "Number of weakly connected components": number_of_weak_components,
            "Percent of nodes in largest weakly connected component": around(
                percent_of_weak
            ),
        }

    else:
        result = {
            "Is connected": is_weakly_connected,
            "Number of connected components": number_of_weak_components,
            "Percent of nodes in largest component": around(percent_of_weak),
        }

    return result


# Cluster coefficient and triangles
@safeSNAP
def get_transitivity(G, fast=True, *args, **kwargs):
    G = G.snapGraph
    results_dict = {}
    if fast:
        samples = min(G.GetNodes(), int(1e3))
        results_dict["Clustering coefficient"] = G.GetClustCf(samples)
    else:
        cc, ct, op = G.GetClustCfAll()[0]
        results_dict = {
            "Clustering coefficient": cc,
            "Number of closed triangles": ct,
            "Number of open triangles": op,
        }

    return results_dict


# Distances info
@safeSNAP
def get_path_stats(G, *args, **kwargs):
    is_directed = G.is_directed
    G = G.snapGraph

    # Only effective diameter if BFS will be too slow or not accurate
    # approx_eff_diam = G.GetAnfEffDiam()

    num_test_nodes = max(100, G.GetNodes() // 1000)
    approx_eff_diam, _, approx_diam, average_path_length = G.GetBfsEffDiamAll(
        num_test_nodes, is_directed
    )

    return {
        "90% effective diameter": around(approx_eff_diam),
        "Approx. full diameter": approx_diam,
        "Average shortest path length": around(average_path_length),
    }


# Degree similarity
def get_dd_simmilarity_score(edges_original, edges_synthetic, cdf_points=1000):
    F_normalized_original = get_normalised_cdf(
        edges_original, cdf_points=cdf_points, debug=False
    )
    F_normalized_synthetic = get_normalised_cdf(
        edges_synthetic, cdf_points=cdf_points, debug=False
    )
    abs_F = np.abs(F_normalized_original[:, 1] - F_normalized_synthetic[:, 1])
    where_non_zero = F_normalized_original[:, 1] != 0
    error = np.average(
        np.divide(
            abs_F[where_non_zero], F_normalized_original[:, 1][where_non_zero]
        )
    )  # average error of normalized CDFs
    error = min(error, 1)
    if error < 0:
        raise ValueError("Negative values in CDFs!")
    simmilarity_score = 1.0 - error
    return simmilarity_score


def around(number, decimals=2):
    return np.around(number, decimals)

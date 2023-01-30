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

import numpy as np
import similaritymeasures


def get_normalised_cdf(nodes, cdf_points=100):
    unique_nodes, unique_nodes_counts = np.unique(nodes, return_counts=True)
    node_degree, node_degree_counts = np.unique(
        unique_nodes_counts, return_counts=True
    )
    node_degree_normalized = (
        node_degree / node_degree[-1]
    )  # they are sorted, so [-1] is max
    node_degree_counts_normalized = node_degree_counts / np.sum(
        node_degree_counts
    )  # to have density
    F = node_degree_counts_normalized
    cdf_points_for_F = np.array(
        F.shape[0] * (np.logspace(0, 1, num=cdf_points + 1) - 1) / 9,
        dtype=np.int32,
    )
    F_normalized = np.zeros(shape=(cdf_points, 2))
    F_normalized[:, 0] = node_degree_normalized[
        np.array(
            (cdf_points_for_F[0:-1] + cdf_points_for_F[1:]) / 2, dtype=np.int32
        )
    ]
    for i in range(cdf_points_for_F.shape[0] - 1):
        beginning = cdf_points_for_F[i]
        end = cdf_points_for_F[i + 1]
        matching_list = F[beginning:end]
        F_normalized[i, 1] = np.mean(matching_list)
        F_normalized[i, 0] = (
            node_degree_normalized[beginning]
            + (
                node_degree_normalized[end - 1]
                - node_degree_normalized[beginning]
            )
            / 2
        )
    return F_normalized


def get_dd_plot2(data):
    out_dd, in_dd = list(zip(*data))
    out_dd, in_dd = list(out_dd), list(in_dd)
    unique_nodes, unique_nodes_counts = np.unique(out_dd, return_counts=True)
    degree_counts = Counter(unique_nodes_counts)
    x_out, y_out = zip(*degree_counts.items())
    unique_nodes, unique_nodes_counts = np.unique(in_dd, return_counts=True)
    degree_counts = Counter(unique_nodes_counts)
    x_in, y_in = zip(*degree_counts.items())

    return (x_in, y_in), (x_out, y_out)


def get_nan_indicies(*values):
    indicies = None
    for value in values:
        filtered = np.isnan(value)
        current_nan = filtered[:, 0] + filtered[:, 1]
        indicies = current_nan if indicies is None else indicies + current_nan
    return indicies


def remove_nans(*values):
    indicies = get_nan_indicies(*values)
    return tuple(F[~indicies] for F in values)


def get_frechet_score(
    edges_original, edges_to_compare, cdf_points=1000, log=True
):
    F1_normalized = get_normalised_cdf(edges_original, cdf_points=cdf_points)
    F2_normalized = get_normalised_cdf(edges_to_compare, cdf_points=cdf_points)
    F1, F2 = remove_nans(F1_normalized, F2_normalized)
    if log:
        F1 = np.log(F1)
        F2 = np.log(F2)
    score = similaritymeasures.frechet_dist(F1, F2)
    return score


def get_frechet_score_normalized(
    edges_original,
    edges_to_compare,
    edges_normalize,
    cdf_points=1000,
    log=True,
):
    F1_normalized = get_normalised_cdf(edges_original, cdf_points=cdf_points)
    F2_normalized = get_normalised_cdf(edges_to_compare, cdf_points=cdf_points)
    F3_normalized = get_normalised_cdf(edges_normalize, cdf_points=cdf_points)
    F1, F2, F3 = remove_nans(F1_normalized, F2_normalized, F3_normalized)

    if log:
        F1 = np.log(F1)
        F2 = np.log(F2)
        F3 = np.log(F3)

    score = similaritymeasures.frechet_dist(F1, F2)
    worst_score = similaritymeasures.frechet_dist(F1, F3)

    eps = 1e-6
    if worst_score < eps or score >= worst_score:
        normalized_score = 0
    else:
        normalized_score = min(1 - score / worst_score, 1)
    return normalized_score


def get_out_in_dd(edges):
    out_dd = edges[:, 0]
    in_dd = edges[:, 1]
    return out_dd, in_dd


def get_frechet_score_directed(
    edges_original, edges_to_compare, cdf_points=1000, log=True
):
    original_out_dd, original_in_dd = get_out_in_dd(edges_original)
    compare_out_dd, compare_in_dd = get_out_in_dd(edges_to_compare)

    dd_score = get_frechet_score(
        edges_original, edges_to_compare, cdf_points, log
    )
    out_dd_score = get_frechet_score(
        original_out_dd, compare_out_dd, cdf_points, log
    )
    in_dd_score = get_frechet_score(
        original_in_dd, compare_in_dd, cdf_points, log
    )

    return dd_score, out_dd_score, in_dd_score


def get_frechet_score_directed_normalized(
    edges_original,
    edges_to_compare,
    edges_normalize,
    cdf_points=1000,
    log=True,
):
    original_out_dd, original_in_dd = get_out_in_dd(edges_original)
    compare_out_dd, compare_in_dd = get_out_in_dd(edges_to_compare)
    normalize_out_dd, normalize_in_dd = get_out_in_dd(edges_normalize)

    dd_normalized_score = get_frechet_score_normalized(
        edges_original, edges_to_compare, edges_normalize, cdf_points, log
    )
    out_dd_normalized_score = get_frechet_score_normalized(
        original_out_dd, compare_out_dd, normalize_out_dd, cdf_points, log
    )
    in_dd_normalized_score = get_frechet_score_normalized(
        original_in_dd, compare_in_dd, normalize_in_dd, cdf_points, log
    )

    return dd_normalized_score, out_dd_normalized_score, in_dd_normalized_score

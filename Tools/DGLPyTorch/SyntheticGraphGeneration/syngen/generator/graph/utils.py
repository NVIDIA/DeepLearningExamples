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

import os
import logging
import math
import multiprocessing
from datetime import datetime
from functools import partial
from typing import Tuple, Union, Optional

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pylibraft.random import rmat
from scipy import stats

from syngen.utils import NDArray, infer_operator
from syngen.utils.utils import infer_operator
from syngen.utils.io_utils import dump_generated_graph
from syngen.utils.memory_manager import MemoryManager
from syngen.utils.types import NDArray

logger = logging.getLogger(__name__)


def move_ndarray_to_host(ndarray: NDArray):
    if isinstance(ndarray, np.ndarray):
        return ndarray
    elif isinstance(ndarray, cp.ndarray):
        return cp.asnumpy(ndarray)
    else:
        raise ValueError('supports only numpy and cupy ndarrays')


def rearrange_graph(
        edge_list: NDArray,
        src_nodes: NDArray,
        dst_nodes: NDArray,
        assume_unique: bool = False,
) -> Tuple[NDArray, NDArray]:
    """
    Transforms a bipartite graph from edge list format to lower_left and upper_right adjacency matrices.

    Returned matrices are in coordinate list format.

    """
    operator = infer_operator(edge_list)

    if not isinstance(src_nodes, (np.ndarray, cp.ndarray)):
        raise ValueError('src_nodes: expected type NDArray, but %s was passed', type(src_nodes))
    if not isinstance(dst_nodes, (np.ndarray, cp.ndarray)):
        raise ValueError('dst_nodes: expected type NDArray, but %s was passed', type(dst_nodes))

    if not assume_unique:
        src_nodes = operator.unique(src_nodes)
        dst_nodes = operator.unique(dst_nodes)

    if operator.intersect1d(src_nodes, dst_nodes, assume_unique=True).size != 0:
        raise ValueError('node sets cannot intersect')

    edge_list = edge_list.flatten()

    node_set = operator.hstack([src_nodes, dst_nodes])
    pos_to_new_id = operator.argsort(node_set)
    sorted_node_set = node_set[pos_to_new_id]

    pos_in_sorted_nodeset = operator.searchsorted(sorted_node_set, edge_list)

    # need to validate since errors could be ignored
    # https://docs.cupy.dev/en/stable/user_guide/difference.html#out-of-bounds-indices
    message = 'all ids in a graph should be in one of the node sets'
    if operator.any(pos_in_sorted_nodeset == len(sorted_node_set)):
        raise ValueError(message)
    if operator.any(sorted_node_set[pos_in_sorted_nodeset] != edge_list):
        raise ValueError(message)

    edge_list_mapped = pos_to_new_id[pos_in_sorted_nodeset].reshape(-1, 2)

    upper_right = edge_list_mapped[edge_list_mapped[:, 0] < len(src_nodes)]
    upper_right[:, 1] -= len(src_nodes)

    lower_left = edge_list_mapped[edge_list_mapped[:, 0] >= len(src_nodes)]
    lower_left[:, 0] -= len(src_nodes)

    return lower_left, upper_right


def reindex_graph(
        edge_list: NDArray,
        return_counts: bool = False,
) -> Union[NDArray, Tuple[NDArray, int, int]]:
    """
    Reindexes a graph by assigning node ids starting from 0.

    Returns the processed graph and, optionally, number of nodes and number of edges.

    """
    operator = infer_operator(edge_list)

    nodes, inverse_flat = operator.unique(edge_list, return_inverse=True)
    edge_list_reindexed = inverse_flat.reshape(edge_list.shape)

    if return_counts:
        return edge_list_reindexed, len(nodes), len(edge_list)
    else:
        return edge_list_reindexed


def get_reversed_part(part, gpu=False, operator=None):
    operator = operator or (cp if gpu else np)
    new_part = operator.empty_like(part)
    new_part[:, 0] = part[:, 1]
    new_part[:, 1] = part[:, 0]
    return new_part


# Postprocessing
def recreate_graph(lower: NDArray, upper: NDArray, offset: int, gpu=False):
    assert (
            lower is not None and upper is not None
    ), "Upper and lower cannot be None"
    operator = cp if gpu else np
    lower[:, 0] = lower[:, 0] + offset
    upper[:, 1] = upper[:, 1] + offset
    new_graph = operator.concatenate((lower, upper), axis=0)

    return new_graph


def recreate_bipartite_nondirected(graph, row_shape):
    upper = [(row, col + row_shape) for row, col in graph]
    lower = [(col, row) for row, col in upper]
    new_graph = upper + lower
    return new_graph


def to_adj_matrix(graph, shape):
    matrix = np.zeros(shape=shape, dtype=np.bool)
    arr_indicies = np.array(graph)
    matrix[arr_indicies[:, 0], arr_indicies[:, 1]] = 1
    return matrix


def plot_graph_adj(graph, shape):
    graph_adj = to_adj_matrix(graph, shape=shape)
    return plt.imshow(graph_adj, cmap="binary", interpolation="nearest")


def graph_to_snap_file(A, filename):
    np.savetxt(filename, A, fmt="%i", delimiter="\t")


def effective_nonsquare_rmat_approximate(
        theta,
        E,
        A_shape,
        noise_scaling=1.0,
        batch_size=1000,
        dtype=np.int64,
        custom_samplers=None,
        generate_back_edges=False,
        verbose=False,
):
    """ This function generates list of edges using modified RMat approach
    Args:
        theta (np.array): seeding matrix, needs to be shape 2x2
        E (int): number of edges to be generated
        A_shape (tuple): shape of resulting adjacency matrix. numbers has to be powers of 2
                            A_shape should be equal to (ceil(log2(X)),ceil(log2(Y))) X,Y are
                            dimensions of original adjacency
        noise_scaling (float 0..1): noise scaling factor for good degree distribution
        batch_size (int): edges are generated in batches of batch_size size
        dtype (numpy dtype np.int32/np.int64): dtype of nodes id's
        custom_samplers (List[scipy.stats.rv_discrete]): samplers for each step of genration
        process
        generate_back_edges (bool): if True then generated edges will also have "back" edges. Not
        that setting to True for partite graphs makes no sense.
    Returns:
        A (np.array 2 x E): matrix containing in every row a signle edge. Edge is always directed
        0'th column is FROM edge 1st is TO edge
        mtx_shape (tuple) - shape of adjecency matrix (A contains list of edges, this is Adjecency
        metrix shape)
        custom_samplers (List[scipy.stats.rv_discrete]) - list of samplers needed to generate edges
        from the same disctribution for multiple runs of the function
    Description:
            The generation will consist of theta^[n] (x) theta_p^[m] (x) theta_q^[l]
            ^[n] is kronecker power
             (x) is matrix kronecker product
             theta_p (2x1) and theta_q(1x2) are marginals of theta

             This way we can generate rectangular shape of adjecency matrix e.g. for bipatrite
             graphs
    """

    def get_row_col_addres(thetas_n):
        thetas_r = [t.shape[0] for t in thetas_n]
        thetas_c = [t.shape[1] for t in thetas_n]
        row_n = np.prod(thetas_r)  # theta_r**quadrant_sequence.shape[1]
        col_n = np.prod(thetas_c)  # theta_c**quadrant_sequence.shape[1]
        row_adders = np.array(
            [
                int(row_n / thetas_r[i] ** (i + 1)) % row_n
                for i in range(len(thetas_n))
            ]
        )  # there has to be % as we can have thetas_r[i]==1
        col_adders = np.array(
            [
                int(col_n / thetas_c[i] ** (i + 1)) % col_n
                for i in range(len(thetas_n))
            ]
        )
        return row_adders, col_adders, thetas_r, thetas_c, row_n, col_n

    def parse_quadrants(
            quadrant_sequence,
            thetas_n,
            row_adders,
            col_addres,
            thetas_r,
            thetas_c,
            row_n,
            col_n,
            dtype=np.int64,
    ):
        N = len(thetas_n)
        new_edges = np.zeros(
            shape=(quadrant_sequence.shape[0], 2)
        )  # 2 because 0 col=rows_addresses, 1st col = columns
        row_addr = np.array(quadrant_sequence // thetas_c, dtype=dtype)
        col_addr = np.array(quadrant_sequence % thetas_c, dtype=dtype)
        row_adders = np.array(
            [int(row_n / thetas_r[i] ** (i + 1)) % row_n for i in range(N)]
        )  # there has to be % as we can have thetas_r[i]==1
        col_adders = np.array(
            [int(col_n / thetas_c[i] ** (i + 1)) % col_n for i in range(N)]
        )
        new_edges[:, 0] = np.sum(np.multiply(row_addr, row_adders), axis=1)
        new_edges[:, 1] = np.sum(np.multiply(col_addr, col_adders), axis=1)
        return new_edges

    if batch_size > E:  # if bs>E
        batch_size = int(E // 2 * 2)
    if generate_back_edges:
        assert (
                batch_size % 2 == 0 and batch_size >= 2
        ), "batch size has to be odd and >1"
    assert (
            np.abs((np.sum(theta) - 1.0)) < 1e-6
    ), "Theta probabilities has to sum to 1.0"
    assert (theta.shape[0] == 2) and (
            theta.shape[1] == 2
    ), "Only 2x2 seeding matrixes are acceptable"
    assert len(A_shape) == 2, "A_shape needs to be of len 2"

    # get appropriate number of n,m,l always m=0 or l=0 (or both for rectangular adjecency)
    r = A_shape[0]
    c = A_shape[1]
    n = min(r, c)  # theta^[n] (x) theta_p^[m] (x) theta_q^[l]
    m = max(0, r - c)
    # flake8: noqa
    l = max(0, c - r)
    # calc values of marginal theta matrixes
    theta_p = theta.sum(axis=1).reshape((2, -1))  # 2x1
    theta_q = theta.sum(axis=0).reshape((1, -1))  # 1x2
    # get all thetas
    thetas_n = [theta] * n + [theta_p] * m + [theta_q] * l
    # prepare samplers for each of n+m+l steps
    if custom_samplers is None:
        custom_samplers = []
        for i in range(n + m + l):
            theta_n = thetas_n[
                i
            ]  # each of n+m+l steps have their own theta_n which can be theta/theta_p or theta_q +
            # noise
            noise = noise_scaling * np.random.uniform(
                -1, 1, size=theta_n.shape
            )
            noise_to_add = np.multiply(theta_n, noise)
            theta_n = theta_n + noise_to_add
            theta_n = theta_n / np.sum(theta_n)
            cstm_n = "step_" + str(i)
            theta_r = theta_n.shape[0]
            theta_c = theta_n.shape[1]
            xk = tuple(range(theta_r * theta_c))
            pk = theta_n.reshape(-1)
            cstm_s = stats.rv_discrete(name=cstm_n, values=(xk, pk))
            custom_samplers.append(cstm_s)
            # Prepare all batch sizes needed for generation
    if batch_size == 0:
        batch_count = 0  # XXX: why does this happen anyways?
    else:
        batch_count = E // batch_size
    last_batch_size = E - batch_count * batch_size
    if last_batch_size % 2 > 0 and generate_back_edges:
        last_batch_size -= 1
    A = np.zeros((E, 2), dtype=np.int64)
    num_sequences = batch_size
    last_num_sequences = last_batch_size
    if (
            generate_back_edges
    ):  # in case of generating back edges we need to sample just E/2
        last_num_sequences = last_batch_size // 2
        num_sequences = batch_size // 2
        new_back_edges = np.zeros(shape=(num_sequences, 2))
    quadrant_sequence = np.zeros(shape=(num_sequences, n + m + l), dtype=dtype)
    (
        row_adders,
        col_addres,
        thetas_r,
        thetas_c,
        row_n,
        col_n,
    ) = get_row_col_addres(thetas_n)
    # generate sequences of quadrants from previously prepared samplers

    batch_itr = range(batch_count)
    if verbose:
        batch_itr = tqdm(batch_itr)

    for e in batch_itr:
        for i in range(
                n + m + l
        ):  # each steps in generation has its own sampler
            smpl = custom_samplers[i].rvs(size=num_sequences)
            quadrant_sequence[:, i] = smpl
            # produce new edges
        new_edges = parse_quadrants(
            quadrant_sequence,
            thetas_n,
            row_adders,
            col_addres,
            thetas_r,
            thetas_c,
            row_n,
            col_n,
            dtype=dtype,
        )
        if generate_back_edges:
            new_back_edges[:, [0, 1]] = new_edges[:, [1, 0]]  # swap columns
            A[
            e * batch_size: (e + 1) * batch_size: 2, :
            ] = new_edges  # we need interleave so that back edges are "right after" normal edges
            A[
            e * batch_size + 1: (e + 1) * batch_size: 2, :
            ] = new_back_edges
        else:
            A[e * batch_size: (e + 1) * batch_size, :] = new_edges

    # generate last batch
    if last_batch_size > 0:
        for i in range(n + m + l):
            smpl = custom_samplers[i].rvs(size=last_num_sequences)
            quadrant_sequence[:last_num_sequences, i] = smpl
        new_edges = parse_quadrants(
            quadrant_sequence[:last_num_sequences, :],
            thetas_n,
            row_adders,
            col_addres,
            thetas_r,
            thetas_c,
            row_n,
            col_n,
            dtype=dtype,
        )
        if generate_back_edges:
            new_back_edges[:last_num_sequences, [0, 1]] = new_edges[
                                                          :last_num_sequences, [1, 0]
                                                          ]
            # we need interleave so that back edges are "right after" normal edges
            A[
            batch_count * batch_size: batch_count * batch_size
                                      + last_batch_size: 2,
            :,
            ] = new_edges
            # np.concatenate((new_edges,new_back_edges[:last_num_sequences,:]),axis=0)
            A[
            batch_count * batch_size
            + 1: batch_count * batch_size
                 + last_batch_size: 2,
            :,
            ] = new_back_edges[:last_num_sequences, :]
        else:
            A[
            batch_count * batch_size: batch_count * batch_size
                                      + last_batch_size,
            :,
            ] = new_edges
    mtx_shape = (
        np.prod([t.shape[0] for t in thetas_n]),
        np.prod([t.shape[1] for t in thetas_n]),
    )  # shape of resulting adjacency matrix
    return A, mtx_shape, custom_samplers


def effective_nonsquare_rmat_exact(
        theta,
        E,
        A_shape,
        noise_scaling=1.0,
        batch_size=1000,
        dtype=np.int64,
        custom_samplers=None,
        remove_selfloops=False,
        generate_back_edges=False,
        return_node_ids=0,
        verbose=False,
):
    """ This function generates list of edges using modified RMat approach based on effective_nonsuqare_rmat_approximate
    Args:
        theta (np.array): seeding matrix, needs to be shape 2x2
        E (int): number of edges to be generated
        A_shape (tuple): shape of resulting adjacency matrix. numbers has to be powers of 2
                          A_shape should be equal to (ceil(log2(X)),ceil(log2(Y))) X,Y are
                          dimensions of original adjacency
        noise_scaling (float 0..1): noise scaling factor for good degree distribution
        batch_size (int): edges are generated in batches of batch_size size
        dtype (numpy dtype np.int32/np.int64): dtype of nodes id's
        remove_selfloops (bool): If true edges n->n will not be generated. Note that for partite
        graphs this makes no sense
        generate_back_edges (bool): if True then generated edges will also have "back" edges. Not
        that setting to True for partite graphs makes no sense.
    Returns:
        A (np.array 2 x E) - matrix containing in every row a signle edge. Edge is always directed
        0'th column is FROM edge 1st is TO edge
        mtx_shape (tuple) - shape of adjecency matrix (A contains list of edges, this is Adjecency
        metrix shape)
        custom_samplers (List[scipy.stats.rv_discrete]) - list of samplers needed to generate edges
        from the same disctribution for multiple runs of the function
    Description:
            see effective_nonsuqare_rmat_approximate
    """
    heuristics = 1.5
    if verbose:
        print("Getting egdes")
    A, mtx_shape, cs = effective_nonsquare_rmat_approximate(
        theta,
        int(heuristics * E),
        A_shape,
        noise_scaling=noise_scaling,
        batch_size=batch_size,
        dtype=dtype,
        custom_samplers=custom_samplers,
        generate_back_edges=generate_back_edges,
        verbose=verbose,
    )
    if generate_back_edges:
        A = A[
            np.sort(np.unique(A, return_index=True, axis=0)[1])
        ]  # permutation is not needed here
    else:
        if verbose:
            print("Getting unique egdes")
        A = np.unique(A, axis=0)
        if verbose:
            print("Permuting edges")
        perm = np.random.permutation(
            A.shape[0]
        )  # we need to permute it as othervise unique returns edges in order
        A = A[perm]
    if remove_selfloops:
        if verbose:
            print("Removing selfloops")
        A = np.delete(A, np.where(A[:, 0] == A[:, 1]), axis=0)
    E_already_generated = A.shape[0]
    if E_already_generated >= E:
        if return_node_ids == 2:
            return A[:E, :], np.unique(A[:E, :][:, 0]), np.unique(A[:E, :][:, 1]), mtx_shape, cs
        if return_node_ids == 1:
            return A[:E, :], np.unique(A[:E, :]), mtx_shape, cs
        return A[:E, :], mtx_shape, cs
    else:
        while E_already_generated < E:
            if verbose:
                print("Generating some additional edges")
            E_to_generate = int(heuristics * (E - E_already_generated))
            A_next, mtx_shape, cs = effective_nonsquare_rmat_approximate(
                theta,
                E_to_generate,
                A_shape,
                noise_scaling=noise_scaling,
                batch_size=batch_size,
                dtype=dtype,
                custom_samplers=cs,
                verbose=verbose,
            )
            if remove_selfloops:
                A_next = np.delete(
                    A_next, np.where(A_next[:, 0] == A_next[:, 1]), axis=0
                )
            A = np.concatenate((A, A_next), axis=0)
            if generate_back_edges:
                A = A[np.sort(np.unique(A, return_index=True, axis=0)[1])]
            else:
                A = np.unique(A, axis=0)
                perm = np.random.permutation(A.shape[0])
                A = A[perm]
            E_already_generated = A.shape[0]

    if return_node_ids == 2:
        return A[:E, :], np.unique(A[:E, :][:, 0]), np.unique(A[:E, :][:, 1]), mtx_shape, cs
    if return_node_ids == 1:
        return A[:E, :], np.unique(A[:E, :]), mtx_shape, cs
    return A[:E, :], mtx_shape, cs


def cupy_unique_axis0(array):
    # https://stackoverflow.com/questions/58662085/is-there-a-cupy-version-supporting-axis-option-in-cupy-unique-function-any
    sortarr = array[cp.lexsort(array.T[::-1])]
    mask = cp.empty(array.shape[0], dtype=cp.bool_)
    mask[0] = True
    mask[1:] = cp.any(sortarr[1:] != sortarr[:-1], axis=1)
    return sortarr[mask]


def unique_axis0(ar: NDArray) -> NDArray:
    """
    Uniform way of calling operator.unique(ar, axis=0).

    axis != None is not supported in cupy yet.
    This function provides a workaround for one of the cases.

    """
    operator = infer_operator(ar)

    if operator == cp:
        return cupy_unique_axis0(ar)
    else:
        return np.unique(ar, axis=0)


def generate_gpu_rmat(
        a,
        b,
        c,
        d,
        r_scale,
        c_scale,
        n_edges,
        noise=0.5,
        is_directed=False,
        has_self_loop=False,
        return_node_ids=0,
):
    if not is_directed and r_scale != c_scale:
        raise ValueError('undirected generation works only for square adj matrix')

    if not is_directed:
        n_edges = n_edges // 2
    gen_graph = None
    HEURISTIC = 1.2
    edges_to_generate = int(HEURISTIC * n_edges)
    theta_len = max(r_scale, c_scale)

    base_theta = [a, b, c, d]
    if noise > 0:
        full_theta = []
        for i in range(theta_len):
            noise_uniform = noise * np.random.uniform(
                -1, 1, size=len(base_theta)
            )
            noise_to_add = np.multiply(base_theta, noise_uniform)
            theta_n = base_theta + noise_to_add
            theta_n = theta_n / np.sum(theta_n)
            full_theta.append(theta_n)
    else:
        full_theta = base_theta * theta_len

    theta_cpu = np.array(full_theta, dtype=np.float32)
    theta = cp.asarray(theta_cpu)

    while gen_graph is None or gen_graph.shape[0] < n_edges:
        tmp = cp.empty((edges_to_generate, 2), dtype=cp.int32)
        seed = cp.random.randint(0, high=1_000_000, size=None, dtype=int)
        rmat(tmp, theta, r_scale, c_scale, seed=seed)

        # Remove self loops
        if not has_self_loop:
            tmp = tmp[tmp[:, 0] != tmp[:, 1]]

        # Keep only one sided edges
        if not is_directed:
            tmp = tmp[tmp[:, 0] <= tmp[:, 1]]

        if gen_graph is None:
            # Remove duplicates
            gen_graph = cupy_unique_axis0(tmp)
        else:
            gen_graph = cp.concatenate((gen_graph, tmp), axis=0)
            # Remove duplicates
            gen_graph = cupy_unique_axis0(gen_graph)

    gen_graph = gen_graph[:n_edges]
    if not is_directed:
        gen_graph_backward = cp.empty((n_edges, 2), dtype=cp.int32)
        gen_graph_backward[:, 0] = gen_graph[:, 1]
        gen_graph_backward[:, 1] = gen_graph[:, 0]
        gen_graph = cp.concatenate((gen_graph, gen_graph_backward), axis=0)

        gen_graph = cupy_unique_axis0(
            gen_graph
        )

        if not has_self_loop:
            gen_graph = gen_graph[gen_graph[:, 0] != gen_graph[:, 1]]

    if return_node_ids == 2:
        return cp.asnumpy(gen_graph), cp.asnumpy(cp.unique(gen_graph[:, 0])), cp.asnumpy(cp.unique(gen_graph[:, 1]))
    if return_node_ids == 1:
        return cp.asnumpy(gen_graph), cp.asnumpy(cp.unique(gen_graph))
    return cp.asnumpy(gen_graph)


def generate_theta(base_theta, noise, theta_len, is_directed):
    if noise > 0:
        full_theta = []
        for i in range(theta_len):
            noise_uniform = noise * np.random.uniform(
                -1, 1, size=len(base_theta)
            )
            noise_to_add = np.multiply(base_theta, noise_uniform)
            theta_n = base_theta + noise_to_add
            if not is_directed:
                theta_n[2] = theta_n[1]
            theta_n = theta_n / np.sum(theta_n)
            full_theta.append(theta_n)
    else:
        full_theta = [base_theta] * theta_len
    return full_theta


def prepare_chunks(full_theta, r_scale, c_scale, gpu_bytes_to_use, edges_to_generate):
    if r_scale > 32 or c_scale > 32:
        bytes_per_edge = 8
        max_id = 9223372036854775807  # int64 max
    else:
        bytes_per_edge = 4
        max_id = 2147483647  # int32 max
    bytes_to_generate = edges_to_generate * 2 * bytes_per_edge
    skip_theta = 0

    # approximation
    while (bytes_to_generate >> 2 * skip_theta) > gpu_bytes_to_use \
            or (bytes_to_generate >> 2 * skip_theta) > max_id:
        skip_theta += 1

    if skip_theta == 0:
        return [], np.array([edges_to_generate]), full_theta, 0, r_scale, c_scale

    # chunk size is limited by the smaller side of the rectangular graph
    while abs(r_scale - c_scale) > skip_theta:
        skip_theta += 1

    def repeat(a, scale):
        if scale == 1:
            return a
        return np.repeat(np.repeat(a, scale, axis=0), scale, axis=1)

    def tile(a, scale):
        if scale == 1:
            return a
        return np.tile(a, (scale, scale))

    def prepare_prefixes(skip_theta):
        if skip_theta > 0:
            prefix_theta = full_theta[:skip_theta]
            gen_theta_len = max(r_scale, c_scale) - skip_theta
            prefix_edges = np.ones((1 << skip_theta, 1 << skip_theta), dtype=np.float64)
            prefixes = np.zeros((2, 1 << skip_theta, 1 << skip_theta), dtype=np.int32)

            for theta_idx, theta in enumerate(prefix_theta):
                pref_src = np.array([[0, 0], [1, 1]]) << theta_idx
                pref_dst = np.array([[0, 1], [0, 1]]) << theta_idx

                theta = np.array(theta, dtype=np.float64).reshape(2, 2)
                repeat_scale = 1 << (skip_theta - theta_idx - 1)
                tile_scale = 1 << theta_idx
                prefix_edges = prefix_edges * tile(repeat(theta, repeat_scale), tile_scale)

                prefixes[0] = prefixes[0] + tile(repeat(pref_src, repeat_scale), tile_scale)
                prefixes[1] = prefixes[1] + tile(repeat(pref_dst, repeat_scale), tile_scale)

            if r_scale != c_scale:  # probabilities in the rectangular matrix should sum up to 1.0
                r_len = 2 ** (r_scale - gen_theta_len)
                c_len = 2 ** (c_scale - gen_theta_len)
                prefix_edges[:r_len, :c_len] = prefix_edges[:r_len, :c_len] / prefix_edges[:r_len, :c_len].sum()

                prefixes[int(r_scale > c_scale), :r_len, :c_len] = \
                    prefixes[int(r_scale > c_scale), :r_len, :c_len] >> abs(r_scale - c_scale)

            prefix_edges = np.ceil(prefix_edges * edges_to_generate).astype(np.int32).reshape(-1)
            prefixes = prefixes.reshape(2, -1)
        else:
            prefixes = []
            prefix_edges = np.array([edges_to_generate])
        return prefixes, prefix_edges

    prefixes, prefix_edges = prepare_prefixes(skip_theta)

    while prefix_edges.max() * 2 * bytes_per_edge > gpu_bytes_to_use:
        skip_theta += 1
        prefixes, prefix_edges = prepare_prefixes(skip_theta)

    generation_theta = full_theta[skip_theta:]

    return prefixes, prefix_edges, generation_theta, skip_theta, len(generation_theta), len(generation_theta)


def _generate_gpu_chunk_rmat(
        chunk_info,
        prefixes,
        prefix_edges,
        has_self_loop,
        is_directed,
        generation_theta,
        r_log2_nodes,
        c_log2_nodes,
        r_pref_len,
        c_pref_len,
        row_len,
        gpus,
        dtype='int32',
        return_node_ids=0,
        memmap_kwargs: Optional = None,
        chunk_save_path_format: Optional[str] = None):

    chunk_id, chunk_end = chunk_info
    chunk_size = prefix_edges[chunk_id]

    if gpus > 1:
        gpu_id = int(multiprocessing.current_process()._identity[0]) % gpus
    else:
        gpu_id = 0
    theta_cpu = np.array(generation_theta, dtype=np.float32)
    edge_list = None

    is_diagonal_chunk = ((chunk_id // row_len) == (chunk_id % row_len))

    use_memmap = memmap_kwargs is not None
    if use_memmap:
        memmap_outfile = np.load(file=memmap_kwargs['filename'], mmap_mode='r+')

    with cp.cuda.Device(gpu_id):
        theta = cp.asarray(theta_cpu)
        while edge_list is None or edge_list.shape[0] < prefix_edges[chunk_id]:
            tmp = cp.empty((prefix_edges[chunk_id], 2), dtype=dtype)
            seed = cp.random.randint(0, high=1_000_000, size=None, dtype=int)
            rmat(tmp, theta, r_log2_nodes, c_log2_nodes, seed=seed)

            if not has_self_loop and is_diagonal_chunk:
                tmp = tmp[tmp[:, 0] != tmp[:, 1]]

            # undirected diagonal_case
            if not is_directed and is_diagonal_chunk:
                tmp = tmp[tmp[:, 0] <= tmp[:, 1]]
            tmp = cupy_unique_axis0(tmp)

            if edge_list is None:
                edge_list = tmp
            else:
                edge_list = cp.concatenate((edge_list, tmp), axis=0)
                del tmp
                edge_list = cupy_unique_axis0(edge_list)

        if len(prefix_edges) > 1:
            edge_list[:, 0] = (edge_list[:, 0] << r_pref_len) + prefixes[0][chunk_id]
            edge_list[:, 1] = (edge_list[:, 1] << c_pref_len) + prefixes[1][chunk_id]

        edge_list = edge_list[:prefix_edges[chunk_id]]
        if return_node_ids == 2:
            src_nodes_ids = cp.asnumpy(cp.unique(edge_list[:, 0]))
            dst_nodes_ids = cp.asnumpy(cp.unique(edge_list[:, 1]))
        if return_node_ids == 1:
            nodes_ids = cp.asnumpy(cp.unique(edge_list))
        result = cp.asnumpy(edge_list)

        if use_memmap:
            memmap_outfile[chunk_end-chunk_size:chunk_end] = result

        del edge_list

    if chunk_save_path_format is not None:
        dump_generated_graph(chunk_save_path_format.format(chunk_id=chunk_id), result)
        result = len(result)

    if use_memmap:
        result = None

    if return_node_ids == 2:
        return result, src_nodes_ids, dst_nodes_ids
    if return_node_ids == 1:
        return result, nodes_ids
    return result


def generate_gpu_chunked_rmat(
        a,
        b,
        c,
        d,
        r_scale,
        c_scale,
        n_edges,
        noise=0.5,
        is_directed=False,
        has_self_loop=False,
        gpus=None,
        return_node_ids=0,
        save_path: Optional[str] = None,
        verbose: bool = False,
):
    if not is_directed and r_scale != c_scale:
        raise ValueError('undirected generation works only for square adj matrix')

    base_theta = [a, b, c, d]

    theta_len = max(r_scale, c_scale)

    full_theta = generate_theta(base_theta, noise, theta_len, is_directed)
    if gpus is None:
        gpus = MemoryManager().get_available_gpus()
    gpu_bytes_to_use = MemoryManager().get_min_available_across_gpus_memory(gpus=gpus)

    gpu_bytes_to_use = math.floor(gpu_bytes_to_use * 0.10)
    prefixes, prefix_edges, generation_theta, prefix_len, r_log2_nodes, c_log2_nodes = \
        prepare_chunks(full_theta, r_scale, c_scale, gpu_bytes_to_use, n_edges)

    chunk_ids = list(range(len(prefix_edges)))

    row_len = 1 << prefix_len
    r_pref_len = r_scale - len(generation_theta)
    c_pref_len = c_scale - len(generation_theta)

    if not is_directed:  # generate a triangular adj matrix
        chunk_ids = [i for i in chunk_ids if (i // row_len) <= (i % row_len)]
        # reduce the diagonal chunks
        for i in range(prefix_len * 2):
            prefix_edges[i * row_len + i] = prefix_edges[i * row_len + i] // 2

    if r_scale != c_scale:
        chunk_ids = [i for i in chunk_ids if (i // row_len) < 2 ** r_pref_len and (i % row_len) < 2 ** c_pref_len]

    is_single_chunk = len(chunk_ids) == 1

    memmap_kwargs = None
    chunk_save_path_format = None
    use_memmap = False

    if save_path and os.path.isdir(save_path):
        chunk_save_path_format = os.path.join(save_path, 'chunk_{chunk_id}.npy')
    elif save_path and save_path.endswith('.npy'):
        use_memmap = True
        memmap_shape = (sum(prefix_edges[chunk_ids]), 2)
        memmap_dtype = np.uint64 if theta_len > 32 else np.uint32
        memmap_kwargs = dict(
            filename=save_path,
        )
        memmap_outfile = np.lib.format.open_memmap(save_path, dtype=memmap_dtype, shape=memmap_shape, mode='w+')

    dtype = cp.int64 if theta_len > 32 else cp.int32

    _generate_gpu_chunk_rmat_p = partial(
        _generate_gpu_chunk_rmat,
        prefixes=prefixes,
        prefix_edges=prefix_edges,
        has_self_loop=has_self_loop,
        is_directed=is_directed,
        generation_theta=generation_theta,
        r_log2_nodes=r_log2_nodes,
        c_log2_nodes=c_log2_nodes,
        r_pref_len=r_pref_len,
        c_pref_len=c_pref_len,
        row_len=row_len,
        dtype=dtype,
        return_node_ids=return_node_ids,
        chunk_save_path_format=chunk_save_path_format,
        memmap_kwargs=memmap_kwargs,
        gpus=1 if is_single_chunk else gpus,
    )

    if is_single_chunk:
        chunk_res = _generate_gpu_chunk_rmat_p((chunk_ids[0], prefix_edges[chunk_ids[0]]))
        if return_node_ids == 2:
            result, src_node_ids, dst_node_ids = chunk_res
        elif return_node_ids == 1:
            result, node_ids = chunk_res
        else:
            result = chunk_res
        if use_memmap:
            result = memmap_outfile
    else:
        multiprocessing.set_start_method('spawn', force=True)

        sub_res_lists = []
        if return_node_ids == 2:
            src_node_ids_presence = np.full(2**r_scale, False)
            dst_node_ids_presence = np.full(2**c_scale, False)
        elif return_node_ids == 1:
            node_ids_presence = np.full(2**theta_len, False)

        with multiprocessing.Pool(processes=gpus) as pool:

            chunk_res = pool.imap_unordered(_generate_gpu_chunk_rmat_p,
                                            zip(chunk_ids, np.cumsum(prefix_edges[chunk_ids])),
                                            chunksize=(len(chunk_ids)+gpus-1) // gpus )
            if verbose:
                chunk_res = tqdm(chunk_res, total=len(chunk_ids))

            if return_node_ids == 2:
                for res, src_n_ids, dst_n_ids in chunk_res:
                    sub_res_lists.append(res)
                    src_node_ids_presence[src_n_ids] = True
                    dst_node_ids_presence[dst_n_ids] = True
            elif return_node_ids == 1:
                for res, n_ids in chunk_res:
                    sub_res_lists.append(res)
                    node_ids_presence[n_ids] = True
            else:
                sub_res_lists = list(chunk_res)

        if use_memmap:
            result = memmap_outfile
        elif chunk_save_path_format is None:
            result = np.concatenate(sub_res_lists)
        else:
            result = int(np.sum(sub_res_lists))

        if return_node_ids == 2:
            src_node_ids, = np.where(src_node_ids_presence)
            dst_node_ids, = np.where(dst_node_ids_presence)
        elif return_node_ids == 1:
            node_ids, = np.where(node_ids_presence)

    if return_node_ids == 2:
        return result, src_node_ids, dst_node_ids
    if return_node_ids == 1:
        return result, node_ids
    return result


def get_degree_distribution(vertices, gpu=False, operator=None):
    operator = operator or (cp if gpu else np)
    _, degrees = operator.unique(vertices, return_counts=True)
    degree_values, degree_counts = operator.unique(degrees, return_counts=True)
    return degree_values, degree_counts


class BaseLogger:
    """ Base logger class
    Args:
        logdir (str): path to the logging directory
    """

    def __init__(self, logdir: str = "tmp"):
        self.logdir = logdir
        os.makedirs(self.logdir, exist_ok=True)
        currentDateAndTime = datetime.now()
        self.logname = (
            f'{currentDateAndTime.strftime("%Y_%m_%d_%H_%M_%S")}.txt'
        )
        self.logpath = os.path.join(self.logdir, self.logname)
        self.setup_logger()
        self.log("Initialized logger")

    def setup_logger(self):
        """ This function setups logger """
        logging.basicConfig(
            filename=self.logpath,
            filemode="a",
            format="%(asctime)s| %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            level=logging.DEBUG,
        )

    def log(self, msg: str):
        """ This function logs messages in debug mode
        Args:
            msg (str): message to be printed
        """

        logging.debug(msg)


def _reshuffle(X: NDArray, mask: NDArray, max_node_id: int) -> None:
    """
    Shuffles dst nodes of edges specified by idx.

    Preserves degree distribution and keeps edge list sorted.

    """
    operator = infer_operator(X)

    if not operator.any(mask):
        return

    target = X[mask, 1]
    operator.random.shuffle(target)
    X[mask, 1] = target

    src_node_mask = operator.zeros(max_node_id + 1, dtype=operator.bool_)
    src_node_mask[X[mask, 0]] = True

    to_sort_mask = operator.zeros(X.shape[0], dtype=operator.bool_)
    to_sort_mask = src_node_mask[X[:, 0]]

    to_sort = X[to_sort_mask]
    to_sort = to_sort[operator.lexsort(to_sort.T[::-1])]
    X[to_sort_mask] = to_sort


def _find_correct_edges(
    X: NDArray,
    self_loops: bool = False,
    assume_sorted: bool = False,
) -> Tuple[NDArray, NDArray]:
    """ Finds duplicates and self loops in an edge list. """
    operator = infer_operator(X)

    if not assume_sorted:
        X = X[operator.lexsort(X.T[::-1])]

    mask = operator.empty(X.shape[0], dtype=operator.bool_)
    mask[0] = True
    mask[1:] = operator.any(X[1:] != X[:-1], axis=1)

    if not self_loops:
        mask &= X[:, 0] != X[:, 1]

    return X, mask


def postprocess_edge_list(X: NDArray, n_reshuffle: int = 0, self_loops: bool = False) -> NDArray:
    """
    Removes multi-edges and (optionally) self-loops.

    If n_reshuffle > 0 is specified, edges are shuffled between nodes
    so that the degree distribution is preserved and less edges will be removed.
    Assumes node set is reindexed from min_id > 0 to max_id ~ N.

    """
    max_node_id = X.max().item()
    X, mask = _find_correct_edges(X, self_loops=self_loops)

    for _ in range(n_reshuffle):
        _reshuffle(X, ~mask, max_node_id)
        X, mask = _find_correct_edges(X, self_loops=self_loops, assume_sorted=True)

    return X[mask]

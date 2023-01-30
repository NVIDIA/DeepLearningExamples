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

from functools import reduce
from typing import List, Tuple

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from pylibraft.random import rmat
from scipy import stats


def dict_union(*args):
    return reduce(lambda d1, d2: d1.update(d2) or d1, *args, {})


def rearrange_graph(graph, set1, set2):
    new_set1_keys = range(len(set1))
    rearrange_set1 = list(new_set1_keys)
    set1_mapping = dict(zip(set1, rearrange_set1))
    offset = len(set1)
    new_set2_keys = range(offset, len(set2) + offset)
    rearrange_set2 = list(new_set2_keys)
    set2_mapping = dict(zip(set2, rearrange_set2))
    full_mapping = dict_union([set1_mapping, set2_mapping])
    new_graph = [(full_mapping[row], full_mapping[col]) for row, col in graph]

    upper_right = [(x, y - offset) for x, y in new_graph if x in new_set1_keys]
    lower_left = [(x - offset, y) for x, y in new_graph if x in new_set2_keys]

    return lower_left, upper_right


def get_reversed_part(part):
    new_part = np.zeros(shape=part.shape)
    new_part[:, 0] = part[:, 1]
    new_part[:, 1] = part[:, 0]

    return new_part


# Postprocessing
def recreate_graph(lower, upper, offset: int):
    assert (
        lower is not None and upper is not None
    ), "Upper and lower cannot be None"

    lower[:, 0] = lower[:, 0] + offset
    upper[:, 1] = upper[:, 1] + offset
    new_graph = np.concatenate((lower, upper), axis=0)

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
    for e in tqdm.tqdm(range(batch_count)):
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
                e * batch_size : (e + 1) * batch_size : 2, :
            ] = new_edges  # we need interleave so that back edges are "right after" normal edges
            A[
                e * batch_size + 1 : (e + 1) * batch_size : 2, :
            ] = new_back_edges
        else:
            A[e * batch_size : (e + 1) * batch_size, :] = new_edges

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
                batch_count * batch_size : batch_count * batch_size
                + last_batch_size : 2,
                :,
            ] = new_edges
            # np.concatenate((new_edges,new_back_edges[:last_num_sequences,:]),axis=0)
            A[
                batch_count * batch_size
                + 1 : batch_count * batch_size
                + last_batch_size : 2,
                :,
            ] = new_back_edges[:last_num_sequences, :]
        else:
            A[
                batch_count * batch_size : batch_count * batch_size
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
    )
    if generate_back_edges:
        A = A[
            np.sort(np.unique(A, return_index=True, axis=0)[1])
        ]  # permutation is not needed here
    else:
        print("Getting unique egdes")
        A = np.unique(A, axis=0)
        print("Permuting edges")
        perm = np.random.permutation(
            A.shape[0]
        )  # we need to permute it as othervise unique returns edges in order
        A = A[perm]
    if remove_selfloops:
        print("Removing selfloops")
        A = np.delete(A, np.where(A[:, 0] == A[:, 1]), axis=0)
    E_already_generated = A.shape[0]
    if E_already_generated >= E:
        return A[:E, :], mtx_shape, cs
    else:
        while E_already_generated < E:
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

    return A[:E, :], mtx_shape, cs


def cupy_unique_axis0(array):
    # https://stackoverflow.com/questions/58662085/is-there-a-cupy-version-supporting-axis-option-in-cupy-unique-function-any
    sortarr = array[cp.lexsort(array.T[::-1])]
    mask = cp.empty(array.shape[0], dtype=cp.bool_)
    mask[0] = True
    mask[1:] = cp.any(sortarr[1:] != sortarr[:-1], axis=1)
    return sortarr[mask]


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
):
    if not is_directed:
        n_edges = n_edges // 2
    gen_graph = None
    HEURISTIC = 1.2
    edges_to_generate = int(HEURISTIC * n_edges)
    theta_len = max(r_scale, c_scale) * 4

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
        if has_self_loop:
            gen_graph = cupy_unique_axis0(
                gen_graph
            )  # Remove duplicated self_loops

    return cp.asnumpy(gen_graph)

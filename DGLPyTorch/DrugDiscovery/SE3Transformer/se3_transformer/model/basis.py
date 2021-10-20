# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT


from functools import lru_cache
from typing import Dict, List

import e3nn.o3 as o3
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.nvtx import range as nvtx_range

from se3_transformer.runtime.utils import degree_to_dim


@lru_cache(maxsize=None)
def get_clebsch_gordon(J: int, d_in: int, d_out: int, device) -> Tensor:
    """ Get the (cached) Q^{d_out,d_in}_J matrices from equation (8) """
    return o3.wigner_3j(J, d_in, d_out, dtype=torch.float64, device=device).permute(2, 1, 0)


@lru_cache(maxsize=None)
def get_all_clebsch_gordon(max_degree: int, device) -> List[List[Tensor]]:
    all_cb = []
    for d_in in range(max_degree + 1):
        for d_out in range(max_degree + 1):
            K_Js = []
            for J in range(abs(d_in - d_out), d_in + d_out + 1):
                K_Js.append(get_clebsch_gordon(J, d_in, d_out, device))
            all_cb.append(K_Js)
    return all_cb


def get_spherical_harmonics(relative_pos: Tensor, max_degree: int) -> List[Tensor]:
    all_degrees = list(range(2 * max_degree + 1))
    sh = o3.spherical_harmonics(all_degrees, relative_pos, normalize=True)
    return torch.split(sh, [degree_to_dim(d) for d in all_degrees], dim=1)


@torch.jit.script
def get_basis_script(max_degree: int,
                     use_pad_trick: bool,
                     spherical_harmonics: List[Tensor],
                     clebsch_gordon: List[List[Tensor]],
                     amp: bool) -> Dict[str, Tensor]:
    """
    Compute pairwise bases matrices for degrees up to max_degree
    :param max_degree:            Maximum input or output degree
    :param use_pad_trick:         Pad some of the odd dimensions for a better use of Tensor Cores
    :param spherical_harmonics:   List of computed spherical harmonics
    :param clebsch_gordon:        List of computed CB-coefficients
    :param amp:                   When true, return bases in FP16 precision
    """
    basis = {}
    idx = 0
    # Double for loop instead of product() because of JIT script
    for d_in in range(max_degree + 1):
        for d_out in range(max_degree + 1):
            key = f'{d_in},{d_out}'
            K_Js = []
            for freq_idx, J in enumerate(range(abs(d_in - d_out), d_in + d_out + 1)):
                Q_J = clebsch_gordon[idx][freq_idx]
                K_Js.append(torch.einsum('n f, k l f -> n l k', spherical_harmonics[J].float(), Q_J.float()))

            basis[key] = torch.stack(K_Js, 2)  # Stack on second dim so order is n l f k
            if amp:
                basis[key] = basis[key].half()
            if use_pad_trick:
                basis[key] = F.pad(basis[key], (0, 1))  # Pad the k dimension, that can be sliced later

            idx += 1

    return basis


@torch.jit.script
def update_basis_with_fused(basis: Dict[str, Tensor],
                            max_degree: int,
                            use_pad_trick: bool,
                            fully_fused: bool) -> Dict[str, Tensor]:
    """ Update the basis dict with partially and optionally fully fused bases """
    num_edges = basis['0,0'].shape[0]
    device = basis['0,0'].device
    dtype = basis['0,0'].dtype
    sum_dim = sum([degree_to_dim(d) for d in range(max_degree + 1)])

    # Fused per output degree
    for d_out in range(max_degree + 1):
        sum_freq = sum([degree_to_dim(min(d, d_out)) for d in range(max_degree + 1)])
        basis_fused = torch.zeros(num_edges, sum_dim, sum_freq, degree_to_dim(d_out) + int(use_pad_trick),
                                  device=device, dtype=dtype)
        acc_d, acc_f = 0, 0
        for d_in in range(max_degree + 1):
            basis_fused[:, acc_d:acc_d + degree_to_dim(d_in), acc_f:acc_f + degree_to_dim(min(d_out, d_in)),
            :degree_to_dim(d_out)] = basis[f'{d_in},{d_out}'][:, :, :, :degree_to_dim(d_out)]

            acc_d += degree_to_dim(d_in)
            acc_f += degree_to_dim(min(d_out, d_in))

        basis[f'out{d_out}_fused'] = basis_fused

    # Fused per input degree
    for d_in in range(max_degree + 1):
        sum_freq = sum([degree_to_dim(min(d, d_in)) for d in range(max_degree + 1)])
        basis_fused = torch.zeros(num_edges, degree_to_dim(d_in), sum_freq, sum_dim,
                                  device=device, dtype=dtype)
        acc_d, acc_f = 0, 0
        for d_out in range(max_degree + 1):
            basis_fused[:, :, acc_f:acc_f + degree_to_dim(min(d_out, d_in)), acc_d:acc_d + degree_to_dim(d_out)] \
                = basis[f'{d_in},{d_out}'][:, :, :, :degree_to_dim(d_out)]

            acc_d += degree_to_dim(d_out)
            acc_f += degree_to_dim(min(d_out, d_in))

        basis[f'in{d_in}_fused'] = basis_fused

    if fully_fused:
        # Fully fused
        # Double sum this way because of JIT script
        sum_freq = sum([
            sum([degree_to_dim(min(d_in, d_out)) for d_in in range(max_degree + 1)]) for d_out in range(max_degree + 1)
        ])
        basis_fused = torch.zeros(num_edges, sum_dim, sum_freq, sum_dim, device=device, dtype=dtype)

        acc_d, acc_f = 0, 0
        for d_out in range(max_degree + 1):
            b = basis[f'out{d_out}_fused']
            basis_fused[:, :, acc_f:acc_f + b.shape[2], acc_d:acc_d + degree_to_dim(d_out)] = b[:, :, :,
                                                                                              :degree_to_dim(d_out)]
            acc_f += b.shape[2]
            acc_d += degree_to_dim(d_out)

        basis['fully_fused'] = basis_fused

    del basis['0,0']  # We know that the basis for l = k = 0 is filled with a constant
    return basis


def get_basis(relative_pos: Tensor,
              max_degree: int = 4,
              compute_gradients: bool = False,
              use_pad_trick: bool = False,
              amp: bool = False) -> Dict[str, Tensor]:
    with nvtx_range('spherical harmonics'):
        spherical_harmonics = get_spherical_harmonics(relative_pos, max_degree)
    with nvtx_range('CB coefficients'):
        clebsch_gordon = get_all_clebsch_gordon(max_degree, relative_pos.device)

    with torch.autograd.set_grad_enabled(compute_gradients):
        with nvtx_range('bases'):
            basis = get_basis_script(max_degree=max_degree,
                                     use_pad_trick=use_pad_trick,
                                     spherical_harmonics=spherical_harmonics,
                                     clebsch_gordon=clebsch_gordon,
                                     amp=amp)
            return basis

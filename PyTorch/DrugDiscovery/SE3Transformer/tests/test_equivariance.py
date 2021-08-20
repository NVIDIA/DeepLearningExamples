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

import torch

from se3_transformer.model import SE3Transformer
from se3_transformer.model.fiber import Fiber
from tests.utils import get_random_graph, assign_relative_pos, get_max_diff, rot

# Tolerances for equivariance error abs( f(x) @ R  -  f(x @ R) )
TOL = 1e-3
CHANNELS, NODES = 32, 512


def _get_outputs(model, R):
    feats0 = torch.randn(NODES, CHANNELS, 1)
    feats1 = torch.randn(NODES, CHANNELS, 3)

    coords = torch.randn(NODES, 3)
    graph = get_random_graph(NODES)
    if torch.cuda.is_available():
        feats0 = feats0.cuda()
        feats1 = feats1.cuda()
        R = R.cuda()
        coords = coords.cuda()
        graph = graph.to('cuda')
        model.cuda()

    graph1 = assign_relative_pos(graph, coords)
    out1 = model(graph1, {'0': feats0, '1': feats1}, {})
    graph2 = assign_relative_pos(graph, coords @ R)
    out2 = model(graph2, {'0': feats0, '1': feats1 @ R}, {})

    return out1, out2


def _get_model(**kwargs):
    return SE3Transformer(
        num_layers=4,
        fiber_in=Fiber.create(2, CHANNELS),
        fiber_hidden=Fiber.create(3, CHANNELS),
        fiber_out=Fiber.create(2, CHANNELS),
        fiber_edge=Fiber({}),
        num_heads=8,
        channels_div=2,
        **kwargs
    )


def test_equivariance():
    model = _get_model()
    R = rot(*torch.rand(3))
    if torch.cuda.is_available():
        R = R.cuda()
    out1, out2 = _get_outputs(model, R)

    assert torch.allclose(out2['0'], out1['0'], atol=TOL), \
        f'type-0 features should be invariant {get_max_diff(out1["0"], out2["0"])}'
    assert torch.allclose(out2['1'], (out1['1'] @ R), atol=TOL), \
        f'type-1 features should be equivariant {get_max_diff(out1["1"] @ R, out2["1"])}'


def test_equivariance_pooled():
    model = _get_model(pooling='avg', return_type=1)
    R = rot(*torch.rand(3))
    if torch.cuda.is_available():
        R = R.cuda()
    out1, out2 = _get_outputs(model, R)

    assert torch.allclose(out2, (out1 @ R), atol=TOL), \
        f'type-1 features should be equivariant {get_max_diff(out1 @ R, out2)}'


def test_invariance_pooled():
    model = _get_model(pooling='avg', return_type=0)
    R = rot(*torch.rand(3))
    if torch.cuda.is_available():
        R = R.cuda()
    out1, out2 = _get_outputs(model, R)

    assert torch.allclose(out2, out1, atol=TOL), \
        f'type-0 features should be invariant {get_max_diff(out1, out2)}'

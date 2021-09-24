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


from collections import namedtuple
from itertools import product
from typing import Dict

import torch
from torch import Tensor

from se3_transformer.runtime.utils import degree_to_dim

FiberEl = namedtuple('FiberEl', ['degree', 'channels'])


class Fiber(dict):
    """
    Describes the structure of some set of features.
    Features are split into types (0, 1, 2, 3, ...). A feature of type k has a dimension of 2k+1.
    Type-0 features: invariant scalars
    Type-1 features: equivariant 3D vectors
    Type-2 features: equivariant symmetric traceless matrices
    ...

    As inputs to a SE3 layer, there can be many features of the same types, and many features of different types.
    The 'multiplicity' or 'number of channels' is the number of features of a given type.
    This class puts together all the degrees and their multiplicities in order to describe
        the inputs, outputs or hidden features of SE3 layers.
    """

    def __init__(self, structure):
        if isinstance(structure, dict):
            structure = [FiberEl(int(d), int(m)) for d, m in sorted(structure.items(), key=lambda x: x[1])]
        elif not isinstance(structure[0], FiberEl):
            structure = list(map(lambda t: FiberEl(*t), sorted(structure, key=lambda x: x[1])))
        self.structure = structure
        super().__init__({d: m for d, m in self.structure})

    @property
    def degrees(self):
        return sorted([t.degree for t in self.structure])

    @property
    def channels(self):
        return [self[d] for d in self.degrees]

    @property
    def num_features(self):
        """ Size of the resulting tensor if all features were concatenated together """
        return sum(t.channels * degree_to_dim(t.degree) for t in self.structure)

    @staticmethod
    def create(num_degrees: int, num_channels: int):
        """ Create a Fiber with degrees 0..num_degrees-1, all with the same multiplicity """
        return Fiber([(degree, num_channels) for degree in range(num_degrees)])

    @staticmethod
    def from_features(feats: Dict[str, Tensor]):
        """ Infer the Fiber structure from a feature dict """
        structure = {}
        for k, v in feats.items():
            degree = int(k)
            assert len(v.shape) == 3, 'Feature shape should be (N, C, 2D+1)'
            assert v.shape[-1] == degree_to_dim(degree)
            structure[degree] = v.shape[-2]
        return Fiber(structure)

    def __getitem__(self, degree: int):
        """ fiber[degree] returns the multiplicity for this degree """
        return dict(self.structure).get(degree, 0)

    def __iter__(self):
        """ Iterate over namedtuples (degree, channels) """
        return iter(self.structure)

    def __mul__(self, other):
        """
        If other in an int, multiplies all the multiplicities by other.
        If other is a fiber, returns the cartesian product.
        """
        if isinstance(other, Fiber):
            return product(self.structure, other.structure)
        elif isinstance(other, int):
            return Fiber({t.degree: t.channels * other for t in self.structure})

    def __add__(self, other):
        """
        If other in an int, add other to all the multiplicities.
        If other is a fiber, add the multiplicities of the fibers together.
        """
        if isinstance(other, Fiber):
            return Fiber({t.degree: t.channels + other[t.degree] for t in self.structure})
        elif isinstance(other, int):
            return Fiber({t.degree: t.channels + other for t in self.structure})

    def __repr__(self):
        return str(self.structure)

    @staticmethod
    def combine_max(f1, f2):
        """ Combine two fiber by taking the maximum multiplicity for each degree in both fibers """
        new_dict = dict(f1.structure)
        for k, m in f2.structure:
            new_dict[k] = max(new_dict.get(k, 0), m)

        return Fiber(list(new_dict.items()))

    @staticmethod
    def combine_selectively(f1, f2):
        """ Combine two fiber by taking the sum of multiplicities for each degree in the first fiber """
        # only use orders which occur in fiber f1
        new_dict = dict(f1.structure)
        for k in f1.degrees:
            if k in f2.degrees:
                new_dict[k] += f2[k]
        return Fiber(list(new_dict.items()))

    def to_attention_heads(self, tensors: Dict[str, Tensor], num_heads: int):
        # dict(N, num_channels, 2d+1) -> (N, num_heads, -1)
        fibers = [tensors[str(degree)].reshape(*tensors[str(degree)].shape[:-2], num_heads, -1) for degree in
                  self.degrees]
        fibers = torch.cat(fibers, -1)
        return fibers

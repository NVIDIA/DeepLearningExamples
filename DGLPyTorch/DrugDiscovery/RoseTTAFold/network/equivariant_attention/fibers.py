from utils.utils_profiling import * # load before other local modules

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from typing import Dict, List, Tuple


class Fiber(object):
    """A Handy Data Structure for Fibers"""
    def __init__(self, num_degrees: int=None, num_channels: int=None,
                 structure: List[Tuple[int,int]]=None, dictionary=None):
        """
        define fiber structure; use one num_degrees & num_channels OR structure
        OR dictionary

        :param num_degrees: degrees will be [0, ..., num_degrees-1]
        :param num_channels: number of channels, same for each degree
        :param structure: e.g. [(32, 0),(16, 1),(16,2)]
        :param dictionary: e.g. {0:32, 1:16, 2:16}
        """
        if structure:
            self.structure = structure
        elif dictionary:
            self.structure = [(dictionary[o], o) for o in sorted(dictionary.keys())]
        else:
            self.structure = [(num_channels, i) for i in range(num_degrees)]

        self.multiplicities, self.degrees = zip(*self.structure)
        self.max_degree = max(self.degrees)
        self.min_degree = min(self.degrees)
        self.structure_dict = {k: v for v, k in self.structure}
        self.dict = self.structure_dict
        self.n_features = np.sum([i[0] * (2*i[1]+1) for i in self.structure])

        self.feature_indices = {}
        idx = 0
        for (num_channels, d) in self.structure:
            length = num_channels * (2*d + 1)
            self.feature_indices[d] = (idx, idx + length)
            idx += length

    def copy_me(self, multiplicity: int=None):
        s = copy.deepcopy(self.structure)
        if multiplicity is not None:
            # overwrite multiplicities
            s = [(multiplicity, o) for m, o in s]
        return Fiber(structure=s)

    @staticmethod
    def combine(f1, f2):
        new_dict = copy.deepcopy(f1.structure_dict)
        for k, m in f2.structure_dict.items():
            if k in new_dict.keys():
                new_dict[k] += m
            else:
                new_dict[k] = m
        structure = [(new_dict[k], k) for k in sorted(new_dict.keys())]
        return Fiber(structure=structure)

    @staticmethod
    def combine_max(f1, f2):
        new_dict = copy.deepcopy(f1.structure_dict)
        for k, m in f2.structure_dict.items():
            if k in new_dict.keys():
                new_dict[k] = max(m, new_dict[k])
            else:
                new_dict[k] = m
        structure = [(new_dict[k], k) for k in sorted(new_dict.keys())]
        return Fiber(structure=structure)

    @staticmethod
    def combine_selectively(f1, f2):
        # only use orders which occur in fiber f1

        new_dict = copy.deepcopy(f1.structure_dict)
        for k in f1.degrees:
            if k in f2.degrees:
                new_dict[k] += f2.structure_dict[k]
        structure = [(new_dict[k], k) for k in sorted(new_dict.keys())]
        return Fiber(structure=structure)

    @staticmethod
    def combine_fibers(val1, struc1, val2, struc2):
        """
        combine two fibers

        :param val1/2: fiber tensors in dictionary form
        :param struc1/2: structure of fiber
        :return: fiber tensor in dictionary form
        """
        struc_out = Fiber.combine(struc1, struc2)
        val_out = {}
        for k in struc_out.degrees:
            if k in struc1.degrees:
                if k in struc2.degrees:
                    val_out[k] = torch.cat([val1[k], val2[k]], -2)
                else:
                    val_out[k] = val1[k]
            else:
                val_out[k] = val2[k]
            assert val_out[k].shape[-2] == struc_out.structure_dict[k]
        return val_out

    def __repr__(self):
        return f"{self.structure}"



def get_fiber_dict(F, struc, mask=None, return_struc=False):
    if mask is None: mask = struc
    index = 0
    fiber_dict = {}
    first_dims = F.shape[:-1]
    masked_dict = {}
    for o, m in struc.structure_dict.items():
        length = m * (2*o + 1)
        if o in mask.degrees:
            masked_dict[o] = m
            fiber_dict[o] = F[...,index:index + length].view(list(first_dims) + [m, 2*o + 1])
        index += length
    assert F.shape[-1] == index
    if return_struc:
        return fiber_dict, Fiber(dictionary=masked_dict)
    return fiber_dict


def get_fiber_tensor(F, struc):
    some_entry = tuple(F.values())[0]
    first_dims = some_entry.shape[:-2]
    res = some_entry.new_empty([*first_dims, struc.n_features])
    index = 0
    for o, m in struc.structure_dict.items():
        length = m * (2*o + 1)
        res[..., index: index + length] = F[o].view(*first_dims, length)
        index += length
    assert index == res.shape[-1]
    return res


def fiber2tensor(F, structure, squeeze=False):
    if squeeze:
        fibers = [F[f'{i}'].view(*F[f'{i}'].shape[:-2], -1) for i in structure.degrees]
        fibers = torch.cat(fibers, -1)
    else:
        fibers = [F[f'{i}'].view(*F[f'{i}'].shape[:-2], -1, 1) for i in structure.degrees]
        fibers = torch.cat(fibers, -2)
    return fibers


def fiber2head(F, h, structure, squeeze=False):
    if squeeze:
        fibers = [F[f'{i}'].view(*F[f'{i}'].shape[:-2], h, -1) for i in structure.degrees]
        fibers = torch.cat(fibers, -1)
    else:
        fibers = [F[f'{i}'].view(*F[f'{i}'].shape[:-2], h, -1, 1) for i in structure.degrees]
        fibers = torch.cat(fibers, -2)
    return fibers


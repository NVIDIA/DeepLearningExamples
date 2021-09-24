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
from typing import Tuple

import dgl
import pathlib
import torch
from dgl.data import QM9EdgeDataset
from dgl import DGLGraph
from torch import Tensor
from torch.utils.data import random_split, DataLoader, Dataset
from tqdm import tqdm

from se3_transformer.data_loading.data_module import DataModule
from se3_transformer.model.basis import get_basis
from se3_transformer.runtime.utils import get_local_rank, str2bool, using_tensor_cores


def _get_relative_pos(qm9_graph: DGLGraph) -> Tensor:
    x = qm9_graph.ndata['pos']
    src, dst = qm9_graph.edges()
    rel_pos = x[dst] - x[src]
    return rel_pos


def _get_split_sizes(full_dataset: Dataset) -> Tuple[int, int, int]:
    len_full = len(full_dataset)
    len_train = 100_000
    len_test = int(0.1 * len_full)
    len_val = len_full - len_train - len_test
    return len_train, len_val, len_test


class QM9DataModule(DataModule):
    """
    Datamodule wrapping https://docs.dgl.ai/en/latest/api/python/dgl.data.html#qm9edge-dataset
    Training set is 100k molecules. Test set is 10% of the dataset. Validation set is the rest.
    This includes all the molecules from QM9 except the ones that are uncharacterized.
    """

    NODE_FEATURE_DIM = 6
    EDGE_FEATURE_DIM = 4

    def __init__(self,
                 data_dir: pathlib.Path,
                 task: str = 'homo',
                 batch_size: int = 240,
                 num_workers: int = 8,
                 num_degrees: int = 4,
                 amp: bool = False,
                 precompute_bases: bool = False,
                 **kwargs):
        self.data_dir = data_dir  # This needs to be before __init__ so that prepare_data has access to it
        super().__init__(batch_size=batch_size, num_workers=num_workers, collate_fn=self._collate)
        self.amp = amp
        self.task = task
        self.batch_size = batch_size
        self.num_degrees = num_degrees

        qm9_kwargs = dict(label_keys=[self.task], verbose=False, raw_dir=str(data_dir))
        if precompute_bases:
            bases_kwargs = dict(max_degree=num_degrees - 1, use_pad_trick=using_tensor_cores(amp), amp=amp)
            full_dataset = CachedBasesQM9EdgeDataset(bases_kwargs=bases_kwargs, batch_size=batch_size,
                                                     num_workers=num_workers, **qm9_kwargs)
        else:
            full_dataset = QM9EdgeDataset(**qm9_kwargs)

        self.ds_train, self.ds_val, self.ds_test = random_split(full_dataset, _get_split_sizes(full_dataset),
                                                                generator=torch.Generator().manual_seed(0))

        train_targets = full_dataset.targets[self.ds_train.indices, full_dataset.label_keys[0]]
        self.targets_mean = train_targets.mean()
        self.targets_std = train_targets.std()

    def prepare_data(self):
        # Download the QM9 preprocessed data
        QM9EdgeDataset(verbose=True, raw_dir=str(self.data_dir))

    def _collate(self, samples):
        graphs, y, *bases = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        edge_feats = {'0': batched_graph.edata['edge_attr'][..., None]}
        batched_graph.edata['rel_pos'] = _get_relative_pos(batched_graph)
        # get node features
        node_feats = {'0': batched_graph.ndata['attr'][:, :6, None]}
        targets = (torch.cat(y) - self.targets_mean) / self.targets_std

        if bases:
            # collate bases
            all_bases = {
                key: torch.cat([b[key] for b in bases[0]], dim=0)
                for key in bases[0][0].keys()
            }

            return batched_graph, node_feats, edge_feats, all_bases, targets
        else:
            return batched_graph, node_feats, edge_feats, targets

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("QM9 dataset")
        parser.add_argument('--task', type=str, default='homo', const='homo', nargs='?',
                            choices=['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv',
                                     'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C'],
                            help='Regression task to train on')
        parser.add_argument('--precompute_bases', type=str2bool, nargs='?', const=True, default=False,
                            help='Precompute bases at the beginning of the script during dataset initialization,'
                                 ' instead of computing them at the beginning of each forward pass.')
        return parent_parser

    def __repr__(self):
        return f'QM9({self.task})'


class CachedBasesQM9EdgeDataset(QM9EdgeDataset):
    """ Dataset extending the QM9 dataset from DGL with precomputed (cached in RAM) pairwise bases """

    def __init__(self, bases_kwargs: dict, batch_size: int, num_workers: int, *args, **kwargs):
        """
        :param bases_kwargs:  Arguments to feed the bases computation function
        :param batch_size:    Batch size to use when iterating over the dataset for computing bases
        """
        self.bases_kwargs = bases_kwargs
        self.batch_size = batch_size
        self.bases = None
        self.num_workers = num_workers
        super().__init__(*args, **kwargs)

    def load(self):
        super().load()
        # Iterate through the dataset and compute bases (pairwise only)
        # Potential improvement: use multi-GPU and gather
        dataloader = DataLoader(self, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers,
                                collate_fn=lambda samples: dgl.batch([sample[0] for sample in samples]))
        bases = []
        for i, graph in tqdm(enumerate(dataloader), total=len(dataloader), desc='Precomputing QM9 bases',
                             disable=get_local_rank() != 0):
            rel_pos = _get_relative_pos(graph)
            # Compute the bases with the GPU but convert the result to CPU to store in RAM
            bases.append({k: v.cpu() for k, v in get_basis(rel_pos.cuda(), **self.bases_kwargs).items()})
        self.bases = bases  # Assign at the end so that __getitem__ isn't confused

    def __getitem__(self, idx: int):
        graph, label = super().__getitem__(idx)

        if self.bases:
            bases_idx = idx // self.batch_size
            bases_cumsum_idx = self.ne_cumsum[idx] - self.ne_cumsum[bases_idx * self.batch_size]
            bases_cumsum_next_idx = self.ne_cumsum[idx + 1] - self.ne_cumsum[bases_idx * self.batch_size]
            return graph, label, {key: basis[bases_cumsum_idx:bases_cumsum_next_idx] for key, basis in
                                  self.bases[bases_idx].items()}
        else:
            return graph, label

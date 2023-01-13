# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import copy
import numpy as np
from torch.utils.data import DataLoader

from common.fairseq.data import data_utils
from common.helpers import print_once
from common.sampler import DistributedIndicesSampler


def adjust_max_tokens(train_dataset, world_size, args):

    def get_steps_per_epoch(world_size, max_tokens, update_freq):
        train_loader, sampler = get_batch_iterator(
            train_dataset,
            True,
            max_tokens=max_tokens,
            max_sentences=args.batch_size,
            max_positions=(max_tokens, max_tokens),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=world_size,
            shard_id=0,
            num_workers=args.num_workers)

        steps_per_epoch = len(train_loader) // update_freq
        return steps_per_epoch

    steps_ref = get_steps_per_epoch(args.ref_world_size, args.ref_max_tokens, 1)

    min_ = args.ref_max_tokens // 20
    max_ = args.ref_max_tokens * 20

    prev_max_tokens = 0
    align_to = 1000
    while min_ < max_:
        max_tokens = (max_ + min_) // 2 // align_to * align_to  # try to round
        if max_tokens == prev_max_tokens:
            break
        prev_max_tokens = max_tokens
        steps = get_steps_per_epoch(world_size, max_tokens, args.update_freq)
        print_once(f"max_tokens={max_tokens} yields {steps} steps "
                   f"(adjusting for {steps_ref}).")
        if steps == steps_ref:
            break
        elif steps > steps_ref:
            min_ = max_tokens
        else:
            max_ = max_tokens

    args.max_tokens = max_tokens
    args.max_tokens_valid = max_tokens


def filter_indices_by_size(
    indices, dataset, max_positions=None, ignore_invalid_inputs=False
):
    """
    Filter examples that are too large

    Args:
        indices (np.array): original array of sample indices
        dataset (~fairseq.data.FairseqDataset): dataset to batch
        max_positions (optional): max sentence length supported by the
            model (default: None).
        ignore_invalid_inputs (bool, optional): don't raise Exception for
            sentences that are too long (default: False).
    Returns:
        np.array: array of filtered sample indices
    """
    indices, ignored = dataset.filter_indices_by_size(indices, max_positions)
    # TODO: consider removing this function. If `len(ignored) > 0`,
    #  an error is raised in fairseq dataset code, both in sup and unsup case
    if len(ignored) > 0:
        if not ignore_invalid_inputs:
            raise Exception(
                (
                    "Size of sample #{} is invalid (={}) since max_positions={}, "
                    "skip this example with --skip-invalid-size-inputs-valid-test"
                ).format(ignored[0], dataset.size(ignored[0]), max_positions)
            )
        print(
            (
                "WARNING: {:,} samples have invalid sizes and will be skipped, "
                "max_positions={}, first few sample ids={}"
            ).format(len(ignored), max_positions, ignored[:10])
        )
    return indices


def get_batch_iterator(
        dataset,
        training,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        num_concat_batches=1,
):
    # get indices ordered by example size
    with data_utils.numpy_seed(seed):
        indices = dataset.ordered_indices()

    # filter examples that are too large
    if max_positions is not None:
        indices = filter_indices_by_size(
            indices, dataset, max_positions, ignore_invalid_inputs)

    # create mini-batches with given size constraints
    batch_inds, non_grouped_batch_inds = dataset.batch_by_size(
        indices,
        max_tokens=max_tokens,
        max_sentences=max_sentences,
        required_batch_size_multiple=required_batch_size_multiple,
        num_concat_batches=num_concat_batches,
    )

    batch_ids = copy.deepcopy(non_grouped_batch_inds)
    [bi.fill(i) for i, bi in enumerate(batch_ids)]
    inds_ids = zip(np.concatenate(batch_inds), np.concatenate(batch_ids))
    dataset.batch_ids = {idx: batch_idx for idx, batch_idx in inds_ids}

    # Batches are already specified, now we just need to shuffle them
    batch_ind_sampler = DistributedIndicesSampler(batch_inds, shuffle=training,
                                                  num_replicas=num_shards,
                                                  rank=shard_id, seed=seed,
                                                  drop_last=training,
                                                  fillvalue=[])
    loader = DataLoader(
        dataset=dataset,
        collate_fn=dataset.collater,
        batch_sampler=batch_ind_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return loader, batch_ind_sampler

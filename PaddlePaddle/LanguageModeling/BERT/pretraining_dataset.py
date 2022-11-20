# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
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

import random
import h5py
import numpy as np
import paddle
from paddle.io import DataLoader, Dataset
from utils.collate import Stack


def create_pretraining_dataset(args,
                               input_file,
                               data_holders,
                               worker_init=None,
                               places=None):
    train_data = PretrainingDataset(
        input_file=input_file, max_pred_length=args.max_predictions_per_seq)
    train_batch_sampler = paddle.io.BatchSampler(
        train_data, batch_size=args.batch_size, shuffle=True)

    def _collate_data(data, stack_fn=Stack()):
        num_fields = len(data[0])
        out = [None] * num_fields
        [
            input_ids, segment_ids, input_mask, masked_lm_positions,
            masked_lm_labels, next_sentence_labels, masked_lm_scale
        ] = [0, 1, 2, 3, 4, 5, 6]
        for i in (input_ids, segment_ids, input_mask, next_sentence_labels):
            out[i] = stack_fn([x[i] for x in data])
        _, seq_length = out[input_ids].shape
        size = sum(len(x[masked_lm_positions]) for x in data)
        if size % 8 != 0:
            size += 8 - (size % 8)
        out[masked_lm_positions] = np.full(size, 0, dtype=np.int32)
        out[masked_lm_labels] = np.full([size, 1], -1, dtype=np.int64)
        mask_token_num = 0
        for i, x in enumerate(data):
            for j, pos in enumerate(x[masked_lm_positions]):
                out[masked_lm_positions][mask_token_num] = i * seq_length + pos
                out[masked_lm_labels][mask_token_num] = x[masked_lm_labels][j]
                mask_token_num += 1
        # The value of masked_lm_scale is equal to mask_token_num,
        # which would be used to compute average masked_lm_loss.
        out.append(np.asarray([mask_token_num], dtype=np.float32))
        if args.amp and args.use_pure_fp16:
            #out[input_mask] = out[input_mask].astype(np.float16)
            out[masked_lm_scale] = out[masked_lm_scale].astype(np.float16)
        return out

    train_data_loader = DataLoader(
        dataset=train_data,
        places=places,
        feed_list=data_holders,
        batch_sampler=train_batch_sampler,
        collate_fn=_collate_data,
        num_workers=0,
        worker_init_fn=worker_init,
        return_list=False)

    return train_data_loader


def create_pretraining_data_holder():
    input_ids = paddle.static.data(
        name="input_ids", shape=[-1, -1], dtype="int64")
    segment_ids = paddle.static.data(
        name="segment_ids", shape=[-1, -1], dtype="int64")
    input_mask = paddle.static.data(
        name="input_mask", shape=[-1, 1, 1, -1], dtype="int64")
    masked_lm_positions = paddle.static.data(
        name="masked_lm_positions", shape=[-1], dtype="int32")
    masked_lm_labels = paddle.static.data(
        name="masked_lm_labels", shape=[-1, 1], dtype="int64")
    next_sentence_labels = paddle.static.data(
        name="next_sentence_labels", shape=[-1, 1], dtype="int64")
    masked_lm_scale = paddle.static.data(
        name="masked_lm_scale", shape=[-1, 1], dtype="float32")
    return [
        input_ids, segment_ids, input_mask, masked_lm_positions,
        masked_lm_labels, next_sentence_labels, masked_lm_scale
    ]


def select_dataset_file_for_each_worker(files, f_start_id, num_trainers,
                                        trainer_id):
    """
    Spliting the train file according to the worker index.
    """
    num_files = len(files)
    if num_trainers > num_files:
        remainder = num_trainers % num_files
        data_file = files[(
            f_start_id * num_trainers + trainer_id + remainder * f_start_id) %
                          num_files]
    else:
        data_file = files[(f_start_id * num_trainers + trainer_id) % num_files]
    return data_file


class WorkerInitObj:
    "Construct the object with different seed, and the Dataloader will generate the data "
    "with different seed in each worker."

    def __init__(self, seed):
        self.seed = seed

    def __call__(self, pid):
        np.random.seed(seed=self.seed + pid)
        random.seed(self.seed + pid)


class PretrainingDataset(Dataset):
    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = [
            'input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions',
            'masked_lm_ids', 'next_sentence_labels'
        ]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):
        # convert next_sentence_labels (index=5) to np.ndarray type
        [
            input_ids, input_mask, segment_ids, masked_lm_positions,
            masked_lm_ids, next_sentence_labels
        ] = [
            input[index].astype(np.int64)
            if indice < 5 else np.asarray(input[index].astype(np.int64))
            for indice, input in enumerate(self.inputs)
        ]
        # input_mask = (1 - np.reshape(
        #     input_mask.astype(np.float32), [1, 1, input_mask.shape[0]])) * -1e4
        input_mask = np.reshape(input_mask, [1, 1, input_mask.shape[0]])

        index = self.max_pred_length
        padded_mask_indices = (masked_lm_positions == 0).nonzero()[0]
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        else:
            index = self.max_pred_length
        masked_lm_labels = masked_lm_ids[:index]
        masked_lm_positions = masked_lm_positions[:index]
        # softmax_with_cross_entropy enforce last dim size equal 1
        masked_lm_labels = np.expand_dims(masked_lm_labels, axis=-1)
        next_sentence_labels = np.expand_dims(next_sentence_labels, axis=-1)

        return [
            input_ids, segment_ids, input_mask, masked_lm_positions,
            masked_lm_labels, next_sentence_labels
        ]

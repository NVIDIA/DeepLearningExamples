# Copyright (c) 2018, deepakn94, codyaustun, robieta. All rights reserved.
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
#
# -----------------------------------------------------------------------
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

import time
import torch


def create_test_data(test_ratings, test_negs, args):
    test_users = test_ratings[:,0]
    test_pos = test_ratings[:,1].reshape(-1,1)

    # create items with real sample at last position
    num_valid_negative = test_negs.shape[1]
    test_users = test_users.reshape(-1,1).repeat(1, 1 + num_valid_negative)
    test_items = torch.cat((test_negs, test_pos), dim=1)
    del test_ratings, test_negs

    # generate dup mask and real indices for exact same behavior on duplication compare to reference
    # here we need a sort that is stable(keep order of duplicates)
    sorted_items, indices = torch.sort(test_items) # [1,1,1,2], [3,1,0,2]
    sum_item_indices = sorted_items.float()+indices.float()/len(indices[0]) #[1.75,1.25,1.0,2.5]
    indices_order = torch.sort(sum_item_indices)[1] #[2,1,0,3]
    stable_indices = torch.gather(indices, 1, indices_order) #[0,1,3,2]
    # produce -1 mask
    dup_mask = (sorted_items[:,0:-1] == sorted_items[:,1:])
    dup_mask = dup_mask.type(torch.uint8)
    dup_mask = torch.cat((torch.zeros_like(test_pos, dtype=torch.uint8), dup_mask), dim=1)
    dup_mask = torch.gather(dup_mask, 1, stable_indices.sort()[1])
    # produce real sample indices to later check in topk
    sorted_items, indices = (test_items != test_pos).type(torch.uint8).sort()
    sum_item_indices = sorted_items.float()+indices.float()/len(indices[0])
    indices_order = torch.sort(sum_item_indices)[1]
    stable_indices = torch.gather(indices, 1, indices_order)
    real_indices = stable_indices[:,0]

    if args.distributed:
        test_users = torch.chunk(test_users, args.world_size)[args.local_rank]
        test_items = torch.chunk(test_items, args.world_size)[args.local_rank]
        dup_mask = torch.chunk(dup_mask, args.world_size)[args.local_rank]
        real_indices = torch.chunk(real_indices, args.world_size)[args.local_rank]

    test_users = test_users.view(-1).split(args.valid_batch_size)
    test_items = test_items.view(-1).split(args.valid_batch_size)

    return test_users, test_items, dup_mask, real_indices


def prepare_epoch_train_data(train_ratings, nb_items, args):
    # create label
    train_label = torch.ones_like(train_ratings[:,0], dtype=torch.float32)
    neg_label = torch.zeros_like(train_label, dtype=torch.float32)
    neg_label = neg_label.repeat(args.negative_samples)
    train_label = torch.cat((train_label,neg_label))
    del neg_label

    train_users = train_ratings[:,0]
    train_items = train_ratings[:,1]

    train_users_per_worker = len(train_label) / args.world_size
    train_users_begin = int(train_users_per_worker * args.local_rank)
    train_users_end = int(train_users_per_worker * (args.local_rank + 1))

    # prepare data for epoch
    neg_users = train_users.repeat(args.negative_samples)
    neg_items = torch.empty_like(neg_users, dtype=torch.int64).random_(0, nb_items)

    epoch_users = torch.cat((train_users, neg_users))
    epoch_items = torch.cat((train_items, neg_items))

    del neg_users, neg_items

    # shuffle prepared data and split into batches
    epoch_indices = torch.randperm(train_users_end - train_users_begin, device='cuda:{}'.format(args.local_rank))
    epoch_indices += train_users_begin

    epoch_users = epoch_users[epoch_indices]
    epoch_items = epoch_items[epoch_indices]
    epoch_label = train_label[epoch_indices]

    if args.distributed:
        local_batch = args.batch_size // args.world_size
    else:
        local_batch = args.batch_size

    epoch_users = epoch_users.split(local_batch)
    epoch_items = epoch_items.split(local_batch)
    epoch_label = epoch_label.split(local_batch)

    # the last batch will almost certainly be smaller, drop it
    epoch_users = epoch_users[:-1]
    epoch_items = epoch_items[:-1]
    epoch_label = epoch_label[:-1]

    return epoch_users, epoch_items, epoch_label


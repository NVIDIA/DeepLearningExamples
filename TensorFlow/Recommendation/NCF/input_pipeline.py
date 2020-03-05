# -----------------------------------------------------------------------
#
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


import numpy as np
import cupy as cp

def generate_negatives(neg_users, true_mat, item_range, sort=False, use_trick=False):
    """ 
    Generate negative samples for data augmentation
    """
    neg_u = []
    neg_i = []

    # If using the shortcut, generate negative items without checking if the associated
    # user has interacted with it. Speeds up training significantly with very low impact
    # on accuracy.
    if use_trick:
        neg_items = cp.random.randint(0, high=item_range, size=neg_users.shape[0])
        return neg_users, neg_items

    # Otherwise, generate negative items, check if associated user has interacted with it,
    # then generate a new one if true
    while len(neg_users) > 0:
        neg_items = cp.random.randint(0, high=item_range, size=neg_users.shape[0])
        neg_mask = true_mat[neg_users, neg_items]
        neg_u.append(neg_users[neg_mask])
        neg_i.append(neg_items[neg_mask])

        neg_users = neg_users[cp.logical_not(neg_mask)]

    neg_users = cp.concatenate(neg_u)
    neg_items = cp.concatenate(neg_i)

    if not sort:
        return neg_users, neg_items

    sorted_users = cp.sort(neg_users)
    sort_indices = cp.argsort(neg_users)

    return sorted_users, neg_items[sort_indices]

class DataGenerator():
    """
    Class to handle data augmentation
    """
    def __init__(self,
                 seed,
                 hvd_rank,
                 num_users,                 # type: int
                 num_items,                 # type: int
                 neg_mat,                   # type: np.ndarray
                 train_users,               # type: np.ndarray
                 train_items,               # type: np.ndarray
                 train_labels,              # type: np.ndarray
                 train_batch_size,          # type: int
                 train_negative_samples,    # type: int
                 pos_eval_users,            # type: np.ndarray
                 pos_eval_items,            # type: np.ndarray
                 eval_users_per_batch,      # type: int
                 eval_negative_samples,     # type: int
                ):
        # Check input data
        if train_users.shape != train_items.shape:
            raise ValueError(
                "Train shapes mismatch! {} Users vs {} Items!".format(
                    train_users.shape, train_items.shape))
        if pos_eval_users.shape != pos_eval_items.shape:
            raise ValueError(
                "Eval shapes mismatch! {} Users vs {} Items!".format(
                    pos_eval_users.shape, pos_eval_items.shape))
        
        np.random.seed(seed)
        cp.random.seed(seed)
        # Use GPU assigned to the horovod rank
        self.hvd_rank = hvd_rank
        cp.cuda.Device(self.hvd_rank).use()

        self.num_users = num_users
        self.num_items = num_items
        self._neg_mat = neg_mat
        self._train_users = cp.array(train_users)
        self._train_items = cp.array(train_items)
        self._train_labels = cp.array(train_labels)
        self.train_batch_size = train_batch_size
        self._train_negative_samples = train_negative_samples
        self._pos_eval_users = pos_eval_users
        self._pos_eval_items = pos_eval_items
        self.eval_users_per_batch = eval_users_per_batch
        self._eval_negative_samples = eval_negative_samples

        # Eval data
        self.eval_users = None
        self.eval_items = None
        self.dup_mask = None

        # Training data
        self.train_users_batches = None
        self.train_items_batches = None
        self.train_labels_batches = None

    # Augment test data with negative samples
    def prepare_eval_data(self):
        pos_eval_users = cp.array(self._pos_eval_users)
        pos_eval_items = cp.array(self._pos_eval_items)

        neg_mat = cp.array(self._neg_mat)

        neg_eval_users_base = cp.repeat(pos_eval_users, self._eval_negative_samples)

        # Generate negative samples
        test_u_neg, test_i_neg = generate_negatives(neg_users=neg_eval_users_base, true_mat=neg_mat,
                                                    item_range=self.num_items, sort=True, use_trick=False)

        test_u_neg = test_u_neg.reshape((-1, self._eval_negative_samples)).get()
        test_i_neg = test_i_neg.reshape((-1, self._eval_negative_samples)).get()

        test_users = self._pos_eval_users.reshape((-1, 1))
        test_items = self._pos_eval_items.reshape((-1, 1))
        # Combine positive and negative samples
        test_users = np.concatenate((test_u_neg, test_users), axis=1)
        test_items = np.concatenate((test_i_neg, test_items), axis=1)

        # Generate duplicate mask
        ## Stable sort indices by incrementing all values with fractional position
        indices = np.arange(test_users.shape[1]).reshape((1, -1)).repeat(test_users.shape[0], axis=0)
        summed_items = np.add(test_items, indices/test_users.shape[1])
        sorted_indices = np.argsort(summed_items, axis=1)
        sorted_order = np.argsort(sorted_indices, axis=1)
        sorted_items = np.sort(test_items, axis=1)
        ## Generate duplicate mask
        dup_mask = np.equal(sorted_items[:,0:-1], sorted_items[:,1:])
        dup_mask = np.concatenate((dup_mask, np.zeros((test_users.shape[0], 1))), axis=1)
        r_indices = np.arange(test_users.shape[0]).reshape((-1, 1)).repeat(test_users.shape[1], axis=1)
        dup_mask = dup_mask[r_indices, sorted_order].astype(np.float32)

        # Reshape all to (-1) and split into chunks
        batch_size = self.eval_users_per_batch * test_users.shape[1]
        split_indices = np.arange(batch_size, test_users.shape[0]*test_users.shape[1], batch_size)
        self.eval_users = np.split(test_users.reshape(-1), split_indices)
        self.eval_items = np.split(test_items.reshape(-1), split_indices)
        self.dup_mask = np.split(dup_mask.reshape(-1), split_indices)

        # Free GPU memory to make space for Tensorflow
        cp.get_default_memory_pool().free_all_blocks()

    # Augment training data with negative samples
    def prepare_train_data(self):
        batch_size = self.train_batch_size

        is_neg = cp.logical_not(self._train_labels)

        # Do not store verification matrix if using the negatives generation shortcut
        neg_mat = None

        # If there are no negative samples in the local portion of the training data, do nothing
        any_neg = cp.any(is_neg)
        if any_neg:
            self._train_users[is_neg], self._train_items[is_neg] = generate_negatives(
                self._train_users[is_neg], neg_mat, self.num_items, use_trick=True
            )

        shuffled_order = cp.random.permutation(self._train_users.shape[0])
        self._train_users = self._train_users[shuffled_order]
        self._train_items = self._train_items[shuffled_order]
        self._train_labels = self._train_labels[shuffled_order]

        # Manually create batches
        split_indices = np.arange(batch_size, self._train_users.shape[0], batch_size)
        self.train_users_batches = np.split(self._train_users, split_indices)
        self.train_items_batches = np.split(self._train_items, split_indices)
        self.train_labels_batches = np.split(self._train_labels, split_indices)

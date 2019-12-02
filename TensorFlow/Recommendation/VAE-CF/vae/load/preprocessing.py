# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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


import os
from collections import defaultdict
from glob import glob

import pandas as pd
from scipy import sparse
import scipy.sparse as sp
import numpy as np
from scipy.sparse import load_npz, csr_matrix

from vae.load.downloaders import download_movielens
import logging
import json

LOG = logging.getLogger("VAE")

def save_as_npz(m_sp, path):
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    sp.save_npz(path, m_sp)


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def filter_triplets(tp, min_uc=5, min_sc=0):
    # Only keep the triplets for items which were clicked on by at least min_sc users.
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]

    # Only keep the triplets for users who clicked on at least min_uc items
    # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]

    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, usercount, itemcount

def save_id_mappings(cache_dir, show2id, profile2id):
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    for d, filename in [(show2id, 'show2id.json'),
                        (profile2id, 'profile2id.json')]:

        with open(os.path.join(cache_dir, filename), 'w') as f:
            d = {str(k): v for k, v in d.items()}
            json.dump(d, f, indent=4)


def load_and_parse_ML_20M(data_dir, threshold=4):
    """
    Original way of processing ml-20m dataset from VAE for CF paper
	Copyright [2018] [Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara]
	SPDX-License-Identifier: Apache-2.0
	Modifications copyright (C) 2019 Michał Filipiuk, Albert Cieślak, Frederic Grabowski, Radosław Rowicki
    """

    cache_dir = os.path.join(data_dir, "ml-20m/preprocessed")

    train_data_file = os.path.join(cache_dir, "train_data.npz")
    vad_data_true_file = os.path.join(cache_dir, "vad_data_true.npz")
    vad_data_test_file = os.path.join(cache_dir, "vad_data_test.npz")
    test_data_true_file = os.path.join(cache_dir, "test_data_true.npz")
    test_data_test_file = os.path.join(cache_dir, "test_data_test.npz")

    if (os.path.isfile(train_data_file)
       and os.path.isfile(vad_data_true_file)
       and os.path.isfile(vad_data_test_file)
       and os.path.isfile(test_data_true_file)
       and os.path.isfile(test_data_test_file)):

           LOG.info("Already processed, skipping.")
           return load_npz(train_data_file), \
                load_npz(vad_data_true_file), \
                load_npz(vad_data_test_file), \
                load_npz(test_data_true_file), \
                load_npz(test_data_test_file),

    LOG.info("Parsing movielens.")

    source_file = os.path.join(data_dir, "ml-20m/extracted/ml-20m", "ratings.csv")
    if not glob(source_file):
        download_movielens(data_dir=data_dir)

    raw_data = pd.read_csv(source_file)
    raw_data.drop('timestamp', axis=1, inplace=True)

    raw_data = raw_data[raw_data['rating'] >= threshold]
    raw_data, user_activity, item_popularity = filter_triplets(raw_data)

    unique_uid = user_activity.index
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]

    n_users = unique_uid.size
    n_heldout_users = 10000

    true_users = unique_uid[:(n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
    test_users = unique_uid[(n_users - n_heldout_users):]

    train_plays = raw_data.loc[raw_data['userId'].isin(true_users)]

    unique_sid = pd.unique(train_plays['movieId'])

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
    save_id_mappings(cache_dir, show2id, profile2id)

    def split_train_test_proportion(data, test_prop=0.2):
        data_grouped_by_user = data.groupby('userId')
        true_list, test_list = list(), list()

        for i, (_, group) in enumerate(data_grouped_by_user):
            n_items_u = len(group)

            if n_items_u >= 5:
                idx = np.zeros(n_items_u, dtype='bool')
                idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

                true_list.append(group[np.logical_not(idx)])
                test_list.append(group[idx])
            else:
                true_list.append(group)

        data_true = pd.concat(true_list)
        data_test = pd.concat(test_list)

        return data_true, data_test

    vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]

    vad_plays_true, vad_plays_test = split_train_test_proportion(vad_plays)

    test_plays = raw_data.loc[raw_data['userId'].isin(test_users)]
    test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]

    test_plays_true, test_plays_test = split_train_test_proportion(test_plays)

    def numerize(tp):
        uid = tp['userId'].map(lambda x: profile2id[x])
        sid = tp['movieId'].map(lambda x: show2id[x])
        return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

    train_data = numerize(train_plays)
    vad_data_true = numerize(vad_plays_true)
    vad_data_test = numerize(vad_plays_test)
    test_data_true = numerize(test_plays_true)
    test_data_test = numerize(test_plays_test)

    n_items = len(unique_sid)
    def load_train_data(tp):
        n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                  (rows, cols)), dtype='float64',
                                 shape=(n_users, n_items))
        return data

    train_data = load_train_data(train_data)

    def load_true_test_data(tp_true, tp_test):
        start_idx = min(tp_true['uid'].min(), tp_test['uid'].min())
        end_idx = max(tp_true['uid'].max(), tp_test['uid'].max())

        rows_true, cols_true = tp_true['uid'] - start_idx, tp_true['sid']
        rows_test, cols_test = tp_test['uid'] - start_idx, tp_test['sid']

        data_true = sparse.csr_matrix((np.ones_like(rows_true),
                                     (rows_true, cols_true)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
        data_test = sparse.csr_matrix((np.ones_like(rows_test),
                                     (rows_test, cols_test)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
        return data_true, data_test

    vad_data_true, vad_data_test = load_true_test_data(vad_data_true, vad_data_test)

    test_data_true, test_data_test = load_true_test_data(test_data_true, test_data_test)

    save_as_npz(train_data, train_data_file)
    save_as_npz(vad_data_true, vad_data_true_file)
    save_as_npz(vad_data_test, vad_data_test_file)
    save_as_npz(test_data_true, test_data_true_file)
    save_as_npz(test_data_test, test_data_test_file)

    return train_data, vad_data_true, vad_data_test, test_data_true, test_data_test


def filter_data(data, min_users=1, min_items=5):
    """

    :param data: input matrix
    :param min_users: only keep items, that were clicked by at least min_users
    :param min_items: only keep users, that clicked at least min_items
    :return: filtered matrix
    """

    col_count = defaultdict(lambda: 0)
    for col in data.nonzero()[1]:
        col_count[col] += 1

    filtered_col = [k for k, v in col_count.items() if v >= min_users]
    filtered_data_c = data[:, filtered_col]
    del data

    row_count = defaultdict(lambda: 0)
    for row in filtered_data_c.nonzero()[0]:
        row_count[row] += 1

    filtered_row = [k for k, v in row_count.items() if v >= min_items]
    filtered_data_r = filtered_data_c[filtered_row, :]
    del filtered_data_c

    return filtered_data_r


def split_into_train_val_test(data, val_ratio, test_ratio):
    """

    :param data: input matrix
    :param val_ratio: Ratio of validation users to all users
    :param test_ratio: Ratio of test users to all users
    :return: Tuple of 3 matrices : {train_matrix, val_matrix, test_matrix}
    """

    assert val_ratio + test_ratio < 1
    train_ratio = 1 - val_ratio - test_ratio
    rows_count = data.shape[0]

    idx = np.random.permutation(range(rows_count))
    train_users_count = int(np.rint(rows_count * train_ratio))
    val_users_count = int(np.rint(rows_count * val_ratio))
    seperator = train_users_count + val_users_count

    train_matrix = data[idx[:train_users_count]]
    val_matrix = data[idx[train_users_count:seperator]]
    test_matrix = data[idx[seperator:]]

    return train_matrix, val_matrix, test_matrix


def split_movies_into_train_test(data, train_ratio):
    """
    Splits data into 2 matrices. The users stay the same, but the items are being split by train_ratio
    :param data: input matrix
    :param train_ratio: Ratio of input items to all items
    :return: tuple of 2 matrices: {train_matrix, test_matrix}
    """
    rows_count, columns_count = data.shape

    train_rows = list()
    train_columns = list()
    test_rows = list()
    test_columns = list()

    for i in range(rows_count):
        user_movies = data.getrow(i).nonzero()[1]
        np.random.shuffle(user_movies)

        movies_count = len(user_movies)
        train_count = int(np.floor(movies_count * train_ratio))
        test_count = movies_count - train_count

        train_movies = user_movies[:train_count]
        test_movies = user_movies[train_count:]

        train_rows += ([i] * train_count)
        train_columns += list(train_movies)

        test_rows += ([i] * test_count)
        test_columns += list(test_movies)

    train_matrix = csr_matrix(([1] * len(train_rows), (train_rows, train_columns)), shape=(rows_count, columns_count))
    test_matrix = csr_matrix(([1] * len(test_rows), (test_rows, test_columns)), shape=(rows_count, columns_count))

    return train_matrix, test_matrix


def remove_items_that_doesnt_occure_in_train(train_matrix, val_matrix, test_matrix):
    """
    Remove items that don't occure in train matrix
    :param train_matrix: training data
    :param val_matrix: validation data
    :param test_matrix: test data
    :return: Input matrices without some items
    """
    item_occure = defaultdict(lambda: False)
    for col in train_matrix.nonzero()[1]:
        item_occure[col] = True

    non_empty_items = [k for k, v in item_occure.items() if v == True]

    return train_matrix[:, non_empty_items], val_matrix[:, non_empty_items], test_matrix[:, non_empty_items]

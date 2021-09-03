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
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from argparse import ArgumentParser
import pandas as pd
import numpy as np
from load import implicit_load
from convert import save_feature_spec, _TestNegSampler, TEST_0, TEST_1, TRAIN_0, TRAIN_1
import torch
import os

USER_COLUMN = 'user_id'
ITEM_COLUMN = 'item_id'


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default='/data/ml-20m/ratings.csv',
                        help='Path to reviews CSV file from MovieLens')
    parser.add_argument('--output', type=str, default='/data',
                        help='Output directory for train and test files')
    parser.add_argument('--valid_negative', type=int, default=100,
                        help='Number of negative samples for each positive test example')
    parser.add_argument('--seed', '-s', type=int, default=1,
                        help='Manually set random seed for torch')
    parser.add_argument('--test', type=str, help='select modification to be applied to the set')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    print("Loading raw data from {}".format(args.path))
    df = implicit_load(args.path, sort=False)

    if args.test == 'less_user':
        to_drop = set(list(df[USER_COLUMN].unique())[-100:])
        df = df[~df[USER_COLUMN].isin(to_drop)]
    if args.test == 'less_item':
        to_drop = set(list(df[ITEM_COLUMN].unique())[-100:])
        df = df[~df[ITEM_COLUMN].isin(to_drop)]
    if args.test == 'more_user':
        sample = df.sample(frac=0.2).copy()
        sample[USER_COLUMN] = sample[USER_COLUMN] + 10000000
        df = df.append(sample)
        users = df[USER_COLUMN]
        df = df[users.isin(users[users.duplicated(keep=False)])]  # make sure something remains in the train set
    if args.test == 'more_item':
        sample = df.sample(frac=0.2).copy()
        sample[ITEM_COLUMN] = sample[ITEM_COLUMN] + 10000000
        df = df.append(sample)

    print("Mapping original user and item IDs to new sequential IDs")
    df[USER_COLUMN] = pd.factorize(df[USER_COLUMN])[0]
    df[ITEM_COLUMN] = pd.factorize(df[ITEM_COLUMN])[0]

    user_cardinality = df[USER_COLUMN].max() + 1
    item_cardinality = df[ITEM_COLUMN].max() + 1

    # Need to sort before popping to get last item
    df.sort_values(by='timestamp', inplace=True)

    # clean up data
    del df['rating'], df['timestamp']
    df = df.drop_duplicates()  # assuming it keeps order

    # Test set is the last interaction for a given user
    grouped_sorted = df.groupby(USER_COLUMN, group_keys=False)
    test_data = grouped_sorted.tail(1).sort_values(by=USER_COLUMN)
    # Train set is all interactions but the last one
    train_data = grouped_sorted.apply(lambda x: x.iloc[:-1])

    sampler = _TestNegSampler(train_data.values, args.valid_negative)
    test_negs = sampler.generate().cuda()
    if args.valid_negative > 0:
        test_negs = test_negs.reshape(-1, args.valid_negative)
    else:
        test_negs = test_negs.reshape(test_data.shape[0], 0)

    if args.test == 'more_pos':
        mask = np.random.rand(len(test_data)) < 0.5
        sample = test_data[mask].copy()
        sample[ITEM_COLUMN] = sample[ITEM_COLUMN] + 5
        test_data = test_data.append(sample)
        test_negs_copy = test_negs[mask]
        test_negs = torch.cat((test_negs, test_negs_copy), dim=0)
    if args.test == 'less_pos':
        mask = np.random.rand(len(test_data)) < 0.5
        test_data = test_data[mask]
        test_negs = test_negs[mask]

    # Reshape train set into user,item,label tabular and save
    train_ratings = torch.from_numpy(train_data.values).cuda()
    train_labels = torch.ones_like(train_ratings[:, 0:1], dtype=torch.float32)
    torch.save(train_ratings, os.path.join(args.output, TRAIN_0))
    torch.save(train_labels, os.path.join(args.output, TRAIN_1))

    # Reshape test set into user,item,label tabular and save
    # All users have the same number of items, items for a given user appear consecutively
    test_ratings = torch.from_numpy(test_data.values).cuda()
    test_users_pos = test_ratings[:, 0:1]  # slicing instead of indexing to keep dimensions
    test_items_pos = test_ratings[:, 1:2]
    test_users = test_users_pos.repeat_interleave(args.valid_negative + 1, dim=0)
    test_items = torch.cat((test_items_pos.reshape(-1, 1), test_negs), dim=1).reshape(-1, 1)
    positive_labels = torch.ones_like(test_users_pos, dtype=torch.float32)
    negative_labels = torch.zeros_like(test_users_pos, dtype=torch.float32).repeat(1, args.valid_negative)
    test_labels = torch.cat((positive_labels, negative_labels), dim=1).reshape(-1, 1)
    dtypes = {'user': str(test_users.dtype), 'item': str(test_items.dtype), 'label': str(test_labels.dtype)}
    test_tensor = torch.cat((test_users, test_items), dim=1)
    torch.save(test_tensor, os.path.join(args.output, TEST_0))
    torch.save(test_labels, os.path.join(args.output, TEST_1))

    if args.test == 'other_names':
        dtypes = {'user_2': str(test_users.dtype),
                  'item_2': str(test_items.dtype),
                  'label_2': str(test_labels.dtype)}
        save_feature_spec(user_cardinality=user_cardinality, item_cardinality=item_cardinality, dtypes=dtypes,
                          test_negative_samples=args.valid_negative, output_path=args.output + '/feature_spec.yaml',
                          user_feature_name='user_2',
                          item_feature_name='item_2',
                          label_feature_name='label_2')
    else:
        save_feature_spec(user_cardinality=user_cardinality, item_cardinality=item_cardinality, dtypes=dtypes,
                          test_negative_samples=args.valid_negative, output_path=args.output + '/feature_spec.yaml')


if __name__ == '__main__':
    main()

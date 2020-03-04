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

from argparse import ArgumentParser
import pandas as pd
from load import implicit_load
import torch
import tqdm

MIN_RATINGS = 20
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
    return parser.parse_args()


class _TestNegSampler:
    def __init__(self, train_ratings, nb_neg):
        self.nb_neg = nb_neg
        self.nb_users = int(train_ratings[:, 0].max()) + 1
        self.nb_items = int(train_ratings[:, 1].max()) + 1

        # compute unique ids for quickly created hash set and fast lookup
        ids = (train_ratings[:, 0] * self.nb_items) + train_ratings[:, 1]
        self.set = set(ids)

    def generate(self, batch_size=128*1024):
        users = torch.arange(0, self.nb_users).reshape([1, -1]).repeat([self.nb_neg, 1]).transpose(0, 1).reshape(-1)

        items = [-1] * len(users)

        random_items = torch.LongTensor(batch_size).random_(0, self.nb_items).tolist()
        print('Generating validation negatives...')
        for idx, u in enumerate(tqdm.tqdm(users.tolist())):
            if not random_items:
                random_items = torch.LongTensor(batch_size).random_(0, self.nb_items).tolist()
            j = random_items.pop()
            while u * self.nb_items + j in self.set:
                if not random_items:
                    random_items = torch.LongTensor(batch_size).random_(0, self.nb_items).tolist()
                j = random_items.pop()

            items[idx] = j
        items = torch.LongTensor(items)
        return items


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    print("Loading raw data from {}".format(args.path))
    df = implicit_load(args.path, sort=False)

    print("Filtering out users with less than {} ratings".format(MIN_RATINGS))
    grouped = df.groupby(USER_COLUMN)
    df = grouped.filter(lambda x: len(x) >= MIN_RATINGS)

    print("Mapping original user and item IDs to new sequential IDs")
    df[USER_COLUMN] = pd.factorize(df[USER_COLUMN])[0]
    df[ITEM_COLUMN] = pd.factorize(df[ITEM_COLUMN])[0]

    # Need to sort before popping to get last item
    df.sort_values(by='timestamp', inplace=True)

    # clean up data
    del df['rating'], df['timestamp']
    df = df.drop_duplicates() # assuming it keeps order

    # now we have filtered and sorted by time data, we can split test data out
    grouped_sorted = df.groupby(USER_COLUMN, group_keys=False)
    test_data = grouped_sorted.tail(1).sort_values(by='user_id')
    # need to pop for each group
    train_data = grouped_sorted.apply(lambda x: x.iloc[:-1])

    # Note: no way to keep reference training data ordering because use of python set and multi-process
    # It should not matter since it will be later randomized again
    # save train and val data that is fixed.
    train_ratings = torch.from_numpy(train_data.values)
    torch.save(train_ratings, args.output+'/train_ratings.pt')
    test_ratings = torch.from_numpy(test_data.values)
    torch.save(test_ratings, args.output+'/test_ratings.pt')

    sampler = _TestNegSampler(train_ratings.cpu().numpy(), args.valid_negative)
    test_negs = sampler.generate().cuda()
    test_negs = test_negs.reshape(-1, args.valid_negative)
    torch.save(test_negs, args.output+'/test_negatives.pt')

if __name__ == '__main__':
    main()

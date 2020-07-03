# Copyright (c) 2020 NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import time
from typing import Tuple, Optional

from torch.utils.data import DataLoader

from dlrm.data.datasets import CriteoBinDataset
from dlrm.data.factories import create_dataset_factory


def get_data_loaders(flags, device_mapping: Optional[dict] = None) -> Tuple[DataLoader, DataLoader]:
    dataset_factory = create_dataset_factory(flags, device_mapping=device_mapping)

    dataset_train, dataset_test = dataset_factory.create_datasets()
    train_sampler = dataset_factory.create_sampler(dataset_train) if flags.shuffle_batch_order else None
    collate_fn = dataset_factory.create_collate_fn()

    data_loader_train = dataset_factory.create_data_loader(dataset_train, collate_fn=collate_fn, sampler=train_sampler)
    data_loader_test = dataset_factory.create_data_loader(dataset_test, collate_fn=collate_fn)
    return data_loader_train, data_loader_test


if __name__ == '__main__':
    print('Dataloader benchmark')

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--steps', type=int, default=1000)
    args = parser.parse_args()

    dataset = CriteoBinDataset(data_file=args.file, batch_size=args.batch_size)

    begin = time.time()
    for i in range(args.steps):
        _ = dataset[i]
    end = time.time()

    step_time = (end - begin) / args.steps
    throughput = args.batch_size / step_time

    print(f'Mean step time: {step_time:.6f} [s]')
    print(f'Mean throughput: {throughput:,.0f} [samples / s]')

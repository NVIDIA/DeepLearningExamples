#!/usr/bin/env python3

# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

import glob

import tensorflow as tf
from triton.tf_dataloader import eval_input_fn


def get_dataloader_fn(
        *,
        data_pattern: str,
        batch_size: int,
):
    files_path = (glob.glob(data_pattern))
    assert len(files_path), "Expected at least 1 parquet file, found 0"
    with tf.device('/cpu:0'):
        input_fn = eval_input_fn(
            files_path=files_path,
            records_batch_size=batch_size,
        )

    def _get_dataloader():
        for x, y, ids in input_fn:
            ids = ids.numpy()
            x = {name: tensor.numpy() for name, tensor in x.items()}
            y = {'wide_deep_model': y.numpy()}

            yield ids, x, y

    return _get_dataloader


def main():
    import argparse

    parser = argparse.ArgumentParser(description="short_description")
    parser.add_argument("--data_pattern", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    args = parser.parse_args()

    dataloader_fn = get_dataloader_fn(data_pattern=args.data_pattern,
                                      batch_size=args.batch_size)

    for i, (ids, x, y) in enumerate(dataloader_fn()):
        print(x, y)


if __name__ == "__main__":
    main()

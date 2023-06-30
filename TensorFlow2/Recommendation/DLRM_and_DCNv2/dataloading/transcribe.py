# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
# author: Tomasz Grel (tgrel@nvidia.com)


import os
import argparse
from .feature_spec import FeatureSpec
from .dataloader import create_input_pipelines
from .split_tfrecords_multihot_dataset import SplitTFRecordsDataset
from .raw_binary_dataset import TfRawBinaryDataset


def parse_args():
    p = argparse.ArgumentParser(description="Transcribe from one dataset format to another")
    p.add_argument('--src_dataset_path', default='synthetic_dataset', type=str, help='Path to the source directory')
    p.add_argument('--src_dataset_type', default='tf_raw',
                      choices=['tf_raw', 'synthetic', 'binary_multihot', 'tfrecords_multihot', 'nvt', 'split_tfrecords'],
                      help='The type of the source dataset')
    p.add_argument('--src_feature_spec', default='feature_spec.yaml', type=str, help='Feature spec filename')
    p.add_argument('--src_batch_size', default=65536, type=int, help='Batch size of the source dataset')
    p.add_argument('--src_synthetic_dataset_use_feature_spec', action='store_true',
                        help='Use feature spec for the synthetic dataset')

    p.add_argument('--dst_dataset_path', default='synthetic_dataset', type=str, help='Path to the destination directory')
    p.add_argument('--dst_prebatch_size', default=65536, type=int, help='Prebatch size for the dst dataset')
    p.add_argument('--dst_feature_spec', type=str, default='feature_spec.yaml',
                        help='Dst feature spec filename')
    p.add_argument('--dst_dataset_type', default='split_tfrecords',
                      choices=['tf_raw', 'synthetic', 'binary_multihot', 'tfrecords_multihot', 'nvt', 'split_tfrecords'],
                      help='The type of the source dataset')

    p.add_argument('--max_batches_train', default=-1, type=int,
                        help='Max number of train batches to transcribe. Passing -1 will transcribe all the data.')
    p.add_argument('--max_batches_test', default=-1, type=int,
                        help='Max number of test batches to transcribe. Passing -1 will transcribe all the data.')
    p.add_argument('--train_only', action='store_true', default=False, help='Only transcribe the train dataset.')
    return p.parse_args()


def main():
    args = parse_args()

    fspec_path = os.path.join(args.src_dataset_path, args.src_feature_spec)
    feature_spec = FeatureSpec.from_yaml(fspec_path)
    table_ids = list(range(len(feature_spec.get_categorical_sizes())))

    src_train, src_test = create_input_pipelines(dataset_type=args.src_dataset_type, dataset_path=args.src_dataset_path,
                                                 train_batch_size=args.src_batch_size,
                                                 test_batch_size=args.src_batch_size,
                                                 table_ids=table_ids, feature_spec=args.src_feature_spec,
                                                 rank=0, world_size=1)

    os.makedirs(args.dst_dataset_path, exist_ok=True)

    if args.dst_dataset_type == 'split_tfrecords':
        SplitTFRecordsDataset.generate(src_train=src_train, src_test=src_test, feature_spec=feature_spec,
                                       dst_dir=args.dst_dataset_path, dst_feature_spec=args.dst_feature_spec,
                                       prebatch_size=args.dst_prebatch_size, max_batches_train=args.max_batches_train,
                                       max_batches_test=args.max_batches_test)
    elif args.dst_dataset_type == 'tf_raw':
        TfRawBinaryDataset.generate(src_train=src_train, src_test=src_test, feature_spec=feature_spec,
                                    dst_dir=args.dst_dataset_path, dst_feature_spec=args.dst_feature_spec,
                                    max_batches_train=args.max_batches_train, max_batches_test=args.max_batches_test)

    else:
        raise ValueError(f'Unimplemented dst_dataset_type: {args.dst_dataset_type}')


    print('Done.')


if __name__ == '__main__':
    main()

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


from . import dataloader
import argparse
import os
import time

import tensorflow as tf
import horovod.tensorflow as hvd
from .feature_spec import FeatureSpec

def compute_bytes_per_batch(batch):
    bytes_per_dtype = dict(
        float16=2,
        int32=4,
        int8=1
    )

    (numerical, categorical), label = batch
    numerical_bytes = numerical.shape[0] * numerical.shape[1] * bytes_per_dtype[numerical.dtype.name]

    categorical_bytes = []
    for c in categorical:
        if hasattr(c, 'flat_values'):
            # ragged tensor
            values = c.flat_values
            values_bytes = values.shape[0] * bytes_per_dtype[values.dtype.name]
            categorical_bytes.append(values_bytes)
        else:
            # dense tensor
            num_bytes = c.shape[0] * c.shape[1] * bytes_per_dtype[c.dtype.name]
            categorical_bytes.append(num_bytes)
    categorical_bytes = sum(categorical_bytes)

    label_bytes = label.shape[0] * bytes_per_dtype[label.dtype.name]
    return numerical_bytes + categorical_bytes + label_bytes


def main():
    parser = argparse.ArgumentParser(description="Benchmark a dataloader")
    parser.add_argument('--dataset_path', default='synthetic_dataset', type=str,
                        help='Path to the destination directory')
    parser.add_argument('--dataset_type', type=str, choices=['tf_raw', 'split_tfrecords'])
    parser.add_argument('--batch_size', default=65536, type=int, help='Batch size')
    parser.add_argument('--xla', default=False, action='store_true', help='Batch size')
    parser.add_argument('--amp', default=False, action='store_true', help='Batch size')
    parser.add_argument('--run_eagerly', default=False, action='store_true', help='Batch size')
    parser.add_argument('--tfdata_debug', default=False, action='store_true', help='Batch size')
    parser.add_argument('--feature_spec', type=str, default='feature_spec.yaml',
                        help='Filename of the feature spec describing the dataset')
    parser.add_argument('--max_batches', type=int, default=100,
                        help='Stop after this many batches, even if there is still some data to be read')
    parser.add_argument('--warmup_steps', type=int, default=5,
                        help='Number of warmup steps that are not benchmarked')
    parser.add_argument('--sleep', type=int, default=0,
                        help='Sleep for this many seconds after creating the dataloader. For debug only.')

    args = parser.parse_args()

    args.synthetic_dataset_use_feature_spec = False
    args.valid_batch_size = args.batch_size

    if args.dataset_type == 'nvt' and not args.run_eagerly:
        raise ValueError('NVT dataloader does not support graph mode. Please specify --run_eagerly to use it.')

    if args.xla:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=fusible'

    hvd.init()

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if args.dataset_type != 'nvt':
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    visible_gpus = []
    if gpus:
        visible_gpus = gpus[hvd.local_rank()]
    tf.config.experimental.set_visible_devices(visible_gpus, 'GPU')

    if args.amp:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

    tf.config.run_functions_eagerly(args.run_eagerly)
    if args.tfdata_debug:
        tf.data.experimental.enable_debug_mode()

    fspec_path = os.path.join(args.dataset_path, args.feature_spec)
    feature_spec = FeatureSpec.from_yaml(fspec_path)

    table_ids = list(range(len(feature_spec.get_categorical_sizes())))

    table_ids = table_ids[hvd.rank()::hvd.size()]

    print('Creating the pipelines')
    train_pipeline, validation_pipeline = dataloader.create_input_pipelines(args, table_ids=table_ids,
                                                                            rank=hvd.rank(),
                                                                            world_size=hvd.size())

    print('Benchmarking...')

    it = iter(train_pipeline.op())

    reduce_input = tf.convert_to_tensor([0], dtype=tf.float32, name='reduce_input')

    @tf.function
    def step():
        device = '/GPU:0'
        with tf.device(device):
            b = next(it)
            _ = hvd.allreduce(reduce_input, name='barrier')
            return

    for i in range(args.warmup_steps):
        print('warmup step:', i)
        l = step()

    rank = hvd.rank()
    if args.sleep != 0:
        print('sleeping...')
        time.sleep(args.sleep)

    begin = time.time()
    current = begin
    for idx in range(args.max_batches):
        l = step()
        new  = time.time()
        if rank == 0:
            print(f'batch: {idx}, step time: {current - new:.3f}')
        current = new

    end = time.time()

    print('Benchmark done')
    num_batches = (idx + 1)
    elapsed = (end - begin)
    batches_per_second = num_batches / elapsed
    samples_per_second = batches_per_second * args.batch_size

    if rank == 0:
        print(f'Batches per second: {batches_per_second:.2e}')
        print(f'Samples per second: {samples_per_second:.2e}')


if __name__ == '__main__':
    main()

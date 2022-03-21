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
#
# author: Tomasz Grel (tgrel@nvidia.com)


import tensorflow as tf
from collections import deque
import math
import horovod.tensorflow as hvd
from collections import namedtuple
import itertools
import numpy as np

class BroadcastingInitializer(tf.keras.initializers.Initializer):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __call__(self, *args, **kwargs):
        weights = self.wrapped(*args, **kwargs)
        weights = hvd.broadcast(weights, root_rank=0, name='BroadcastingInitializer')
        return weights

    def get_config(self):
        return {}


def is_multinode():
    if not hvd.is_homogeneous():
        return ValueError('Only homogeneous clusters are currently supported')

    return hvd.cross_size() > 1


def argsort(sequence, reverse: bool = False):
    idx_pairs = [(x, i) for i, x in enumerate(sequence)]
    sorted_pairs = sorted(idx_pairs, key=lambda pair: pair[0], reverse=reverse)
    return [i for _, i in sorted_pairs]


def distribute_to_buckets(sizes, buckets_num):
    def sum_sizes(indices):
        return sum(sizes[i] for i in indices)

    max_bucket_size = math.ceil(len(sizes) / buckets_num)
    idx_sorted = deque(argsort(sizes, reverse=True))
    buckets = [[] for _ in range(buckets_num)]
    final_buckets = []

    while idx_sorted:
        bucket = buckets[0]
        bucket.append(idx_sorted.popleft())

        if len(bucket) == max_bucket_size:
            final_buckets.append(buckets.pop(0))

        buckets.sort(key=sum_sizes)

    final_buckets += buckets

    return final_buckets


MultiGpuMetadata = namedtuple('MultiGpuMetadata',
                              ['bottom_mlp_ranks','rank_to_categorical_ids',
                               'rank_to_feature_count', 'sort_order'])


def _find_mapping(target, source):
    """
    Find a mapping, such that target == source[mapping]
    """
    assert len(target) == len(source)

    mapping = []
    for x in target:
        j = source.index(x)
        mapping.append(j)
    return mapping


def get_device_mapping(embedding_sizes, num_gpus, data_parallel_bottom_mlp,
                       columnwise_split, num_numerical_features):
    """Get device mappings for hybrid parallelism

    Bottom MLP running on device 0. Embeddings will be distributed across among all the devices.

    Optimal solution for partitioning set of N embedding tables into K devices to minimize maximal subset sum
    is an NP-hard problem. Additionally, embedding tables distribution should be nearly uniform due to the performance
    constraints. Therefore, suboptimal greedy approach with max bucket size is used.

    Args:
        embedding_sizes (Sequence[int]): embedding tables sizes
        num_gpus (int): Default 8.

    Returns:
        device_mapping (dict):
    """
    if num_numerical_features == 0:
        bottom_mlp_ranks = []
    elif data_parallel_bottom_mlp:
        bottom_mlp_ranks = list(range(num_gpus))
    else:
        bottom_mlp_ranks = [0]


    singlegpu_order = distribute_to_buckets(embedding_sizes, 1)[0]
    if not data_parallel_bottom_mlp or num_gpus == 1:
        # one more feature because of the bottom mlp
        singlegpu_order = [-1] + singlegpu_order

    print('singleGPU order: ', singlegpu_order)

    if columnwise_split:
        gpu_buckets = num_gpus * [distribute_to_buckets(embedding_sizes, 1)[0]]

        vectors_per_gpu = [len(bucket) for bucket in gpu_buckets]

        if num_numerical_features > 0:
            vectors_per_gpu[0] += 1  # count bottom mlp

        # [-1] for the model parallel bottom mlp at first position
        flattened_buckets = [] if data_parallel_bottom_mlp else [-1]
        flattened_buckets.extend(gpu_buckets[0])
        sort_order = _find_mapping(target=singlegpu_order, source=flattened_buckets)

        print(f'bottom mlp ranks: {bottom_mlp_ranks}\n'
              f'gpu buckets: {gpu_buckets}\n'
              f'vectors_per_gpu: {vectors_per_gpu}\n',
              f'sort_order: {sort_order}')

        return MultiGpuMetadata(bottom_mlp_ranks=bottom_mlp_ranks,
                                rank_to_categorical_ids=gpu_buckets,
                                rank_to_feature_count=vectors_per_gpu,
                                sort_order=sort_order)

    if num_gpus > 4 and not data_parallel_bottom_mlp and num_numerical_features > 0:
        # for higher no. of GPUs, make sure the one with bottom mlp has no embeddings
        gpu_buckets = distribute_to_buckets(embedding_sizes, num_gpus - 1)  # leave one device out for the bottom MLP
        gpu_buckets.insert(0, [])
    else:
        gpu_buckets = distribute_to_buckets(embedding_sizes, num_gpus)

    vectors_per_gpu = [len(bucket) for bucket in gpu_buckets]

    if not data_parallel_bottom_mlp:
        for rank in bottom_mlp_ranks:
            vectors_per_gpu[rank] += 1  # count bottom mlp

    # [-1] for the model parallel bottom mlp at first position
    flattened_buckets = [] if data_parallel_bottom_mlp else [-1]
    flattened_buckets.extend(itertools.chain(*gpu_buckets))

    print('tablewise-multiGPU order: ', flattened_buckets)
    sort_order = _find_mapping(target=singlegpu_order, source=flattened_buckets)

    if hvd.rank() == 0:
        print(f'bottom mlp ranks: {bottom_mlp_ranks}\n'
              f'gpu buckets: {gpu_buckets}\n'
              f'vectors_per_gpu: {vectors_per_gpu}\n',
              f'sort_order: {sort_order}')

    return MultiGpuMetadata(bottom_mlp_ranks=bottom_mlp_ranks,
                            rank_to_categorical_ids=gpu_buckets,
                            rank_to_feature_count=vectors_per_gpu,
                            sort_order=sort_order)
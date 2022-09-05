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


import cupy
import horovod.tensorflow as hvd
import tensorflow as tf
from nvtabular.loader.tensorflow import KerasSequenceLoader
from data.outbrain.defaults import LABEL_CHANNEL, MAP_FEATURE_CHANNEL, NUMERICAL_CHANNEL, ONEHOT_CHANNEL, \
    MULTIHOT_CHANNEL

cupy.random.seed(None)


def seed_fn():
    min_int, max_int = tf.int32.limits
    max_rand = max_int // hvd.size()

    # Generate a seed fragment on each worker
    seed_fragment = cupy.random.randint(0, max_rand).get()

    # Aggregate seed fragments from all Horovod workers
    seed_tensor = tf.constant(seed_fragment)
    reduced_seed = hvd.allreduce(seed_tensor, name="shuffle_seed", op=hvd.mpi_ops.Sum)

    return reduced_seed % max_rand


def get_dataset(feature_spec, mapping, batch_size, buffer_size=0.1, parts_per_chunk=1,
                map_channel_enabled=False, shuffle=True):

    data_paths = feature_spec.get_paths_by_mapping(mapping)
    label_names = feature_spec.get_names_by_channel(LABEL_CHANNEL)
    cat_names = feature_spec.get_names_by_channel(ONEHOT_CHANNEL) + feature_spec.get_names_by_channel(MULTIHOT_CHANNEL)
    cont_names = feature_spec.get_names_by_channel(NUMERICAL_CHANNEL)
    if map_channel_enabled:
        cat_names += feature_spec.get_names_by_channel(MAP_FEATURE_CHANNEL)

    tf_dataset = KerasSequenceLoader(
        data_paths,
        batch_size=batch_size,
        label_names=label_names,
        cat_names=cat_names,
        cont_names=cont_names,
        engine="parquet",
        shuffle=shuffle,
        buffer_size=buffer_size,
        parts_per_chunk=parts_per_chunk,
        global_size=hvd.size(),
        global_rank=hvd.rank(),
        seed_fn=seed_fn,
    )

    return tf_dataset

def make_padding_function(multihot_hotness_dict):
    @tf.function(experimental_relax_shapes=True)
    def pad_batch(batch):
        batch = batch.copy()
        for feature, hotness in multihot_hotness_dict.items():
            multihot_tuple = batch[feature]
            values = multihot_tuple[0][:, 0]
            row_lengths = multihot_tuple[1][:, 0]
            padded = tf.RaggedTensor.from_row_lengths(
                values, row_lengths, validate=False
            ).to_tensor(default_value=-1, shape=[None, hotness])
            batch[feature] = padded
        return batch

    return pad_batch

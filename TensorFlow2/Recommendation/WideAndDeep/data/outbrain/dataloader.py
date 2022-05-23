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
from data.outbrain.features import CATEGORICAL_COLUMNS, MULTIHOT_COLUMNS, NUMERIC_COLUMNS
from nvtabular.loader.tensorflow import KerasSequenceLoader

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


def train_input_fn(
        train_paths, records_batch_size, buffer_size=0.1, parts_per_chunk=1, shuffle=True
):
    train_dataset_tf = KerasSequenceLoader(
        train_paths,
        batch_size=records_batch_size,
        label_names=["clicked"],
        cat_names=CATEGORICAL_COLUMNS,
        cont_names=NUMERIC_COLUMNS,
        engine="parquet",
        shuffle=shuffle,
        buffer_size=buffer_size,
        parts_per_chunk=parts_per_chunk,
        global_size=hvd.size(),
        global_rank=hvd.rank(),
        seed_fn=seed_fn,
    )

    return train_dataset_tf


def eval_input_fn(
        valid_paths, records_batch_size, buffer_size=0.1, parts_per_chunk=1, shuffle=False
):
    valid_dataset_tf = KerasSequenceLoader(
        valid_paths,
        batch_size=records_batch_size,
        label_names=["clicked"],
        cat_names=CATEGORICAL_COLUMNS + ["display_id"],
        cont_names=NUMERIC_COLUMNS,
        engine="parquet",
        shuffle=shuffle,
        buffer_size=buffer_size,
        parts_per_chunk=parts_per_chunk,
        global_size=hvd.size(),
        global_rank=hvd.rank(),
        seed_fn=seed_fn,
    )

    return valid_dataset_tf


@tf.function(experimental_relax_shapes=True)
def pad_batch(batch):
    batch = batch.copy()
    for feature, hotness in MULTIHOT_COLUMNS.items():
        multihot_tuple = batch[feature]
        values = multihot_tuple[0][:, 0]
        row_lengths = multihot_tuple[1][:, 0]
        padded = tf.RaggedTensor.from_row_lengths(
            values, row_lengths, validate=False
        ).to_tensor(default_value=-1, shape=[None, hotness])
        batch[feature] = padded
    return batch

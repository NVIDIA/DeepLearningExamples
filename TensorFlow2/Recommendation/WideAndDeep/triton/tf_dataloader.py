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

from functools import partial

import numpy as np
import pandas as pd
import tensorflow as tf
from data.outbrain.features import get_features_keys, MULTIHOT_COLUMNS


def prepare_df(df):
    for multihot_key, value in MULTIHOT_COLUMNS.items():
        multihot_col = df.pop(multihot_key)
        for i in range(value):
            df[f"{multihot_key}_{i}"] = multihot_col.apply(
                lambda x: x[i] if len(x) > i else -1
            )
        df[f"{multihot_key}_nnzs"] = multihot_col.apply(lambda x: len(x))

    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.integer):
            df[col] = df[col].astype(np.int32)

        if np.issubdtype(df[col].dtype, np.floating):
            df[col] = df[col].astype(np.float32)

    return df


def _merge_multihots(*multihots, axis=1):
    expanded = [tf.expand_dims(multihot, axis) for multihot in multihots]
    concatenated = tf.concat(expanded, axis)
    reshaped = tf.reshape(concatenated, [-1])
    mask = tf.math.not_equal(reshaped, -1)
    filtered = tf.boolean_mask(reshaped, mask)

    return tf.reshape(filtered, [-1, 1])


def _filter_batch(elem):
    label = elem.pop("clicked")
    label = tf.reshape(label, [-1, 1])
    disp_id = elem.pop("display_id")

    for multihot_key, value in MULTIHOT_COLUMNS.items():
        multihot_values = [elem.pop(f"{multihot_key}_{i}") for i in range(value)]
        multihot_nnzs = elem.pop(f"{multihot_key}_nnzs")

        values = _merge_multihots(*multihot_values)
        row_lengths = multihot_nnzs
        values = tf.reshape(values, [-1])
        row_lengths = tf.reshape(row_lengths, [-1])

        x = tf.RaggedTensor.from_row_lengths(
            values, row_lengths, validate=False
        ).to_tensor(default_value=-1, shape=[None, value])

        elem[f"{multihot_key}"] = x

    features = get_features_keys()

    elem = {
        key: (
            tf.reshape(value, [-1, 1])
            if "list" not in key
            else tf.reshape(value, [-1, MULTIHOT_COLUMNS[key]])
        )
        for key, value in elem.items()
        if key in features or "list" in key
    }

    return elem, label, disp_id


def eval_input_fn(files_path, records_batch_size):
    frames = []
    for file in files_path:
        frames.append(pd.read_parquet(file))

    if len(frames) > 1:
        df = pd.concat(frames)
    else:
        df = frames[0]

    full_df = prepare_df(df)
    dataset = tf.data.Dataset.from_tensor_slices(dict(full_df))
    dataset = dataset.batch(batch_size=records_batch_size, drop_remainder=False)
    dataset = dataset.map(map_func=partial(_filter_batch), num_parallel_calls=None)
    dataset = dataset.prefetch(buffer_size=2)

    return dataset

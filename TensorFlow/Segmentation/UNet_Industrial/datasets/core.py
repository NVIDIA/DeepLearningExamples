#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
# ==============================================================================

import os
from abc import ABC, abstractmethod

import math

import tensorflow as tf

__all__ = ["BaseDataset"]


class BaseDataset(ABC):

    authorized_normalization_methods = [None, "zero_centered", "zero_one"]

    def __init__(self, data_dir):
        self.data_dir = data_dir

        if not os.path.exists(data_dir):
            raise FileNotFoundError("The dataset directory `%s` does not exist." % data_dir)

    @staticmethod
    def _count_steps(iter_unit, num_samples, num_iter, global_batch_size):

        if iter_unit not in ["batch", "epoch"]:
            raise ValueError("Invalid `iter_unit` value: %s" % iter_unit)

        if iter_unit == 'epoch':
            num_steps = (num_samples // global_batch_size) * num_iter
            num_epochs = num_iter

        else:
            num_steps = num_iter
            num_epochs = math.ceil(num_steps / (num_samples // global_batch_size))

        return num_steps, num_epochs

    @abstractmethod
    def dataset_name(self):
        raise NotImplementedError

    @abstractmethod
    def get_dataset_runtime_specs(self, training, iter_unit, num_iter, global_batch_size):
        # return filenames, num_samples, num_steps, num_epochs
        raise NotImplementedError

    @abstractmethod
    def dataset_fn(
        self,
        batch_size,
        training,
        input_shape,
        mask_shape,
        num_threads,
        use_gpu_prefetch,
        normalize_data_method,
        only_defective_images,
        augment_data,
        seed=None
    ):

        if normalize_data_method not in BaseDataset.authorized_normalization_methods:
            raise ValueError(
                'Unknown `normalize_data_method`: %s - Authorized: %s' %
                (normalize_data_method, BaseDataset.authorized_normalization_methods)
            )

    def synth_dataset_fn(
        self,
        batch_size,
        training,
        input_shape,
        mask_shape,
        num_threads,
        use_gpu_prefetch,
        normalize_data_method,
        only_defective_images,
        augment_data,
        seed=None
    ):

        if normalize_data_method not in BaseDataset.authorized_normalization_methods:
            raise ValueError(
                'Unknown `normalize_data_method`: %s - Authorized: %s' %
                (normalize_data_method, BaseDataset.authorized_normalization_methods)
            )

        input_shape = [batch_size] + list(input_shape)
        mask_shape = [batch_size] + list(mask_shape)

        # Convert the inputs to a Dataset
        if normalize_data_method is None:
            mean_val = 127.5

        elif normalize_data_method == "zero_centered":
            mean_val = 0

        else:
            mean_val = 0.5

        inputs = tf.truncated_normal(
            input_shape, dtype=tf.float32, mean=mean_val, stddev=1, seed=seed, name='synth_inputs'
        )
        masks = tf.truncated_normal(mask_shape, dtype=tf.float32, mean=0.01, stddev=0.1, seed=seed, name='synth_masks')
        labels = tf.random_uniform([batch_size], minval=0, maxval=1, dtype=tf.int32, name='synthetic_labels')

        dataset = tf.data.Dataset.from_tensors(((inputs, masks), labels))

        dataset = dataset.cache()
        dataset = dataset.repeat()

        dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

        if use_gpu_prefetch:
            dataset.apply(tf.data.experimental.prefetch_to_device(device="/gpu:0", buffer_size=batch_size * 8))

        return dataset

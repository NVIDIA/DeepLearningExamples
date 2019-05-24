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

""" Dataset class encapsulates the data loading"""
import math
import os
import multiprocessing

import tensorflow as tf
import numpy as np
from PIL import Image, ImageSequence


class Dataset():
    """Load, separate and prepare the data for training and prediction"""

    def __init__(self, data_dir, batch_size, augment=False, gpu_id=0, num_gpus=1, seed=0):
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._augment = augment

        self._seed = seed

        self._train_images = \
            self._load_multipage_tiff(os.path.join(self._data_dir, 'train-volume.tif'))
        self._train_masks = \
            self._load_multipage_tiff(os.path.join(self._data_dir, 'train-labels.tif'))
        self._test_images = \
            self._load_multipage_tiff(os.path.join(self._data_dir, 'test-volume.tif'))

        self._num_gpus = num_gpus
        self._gpu_id = gpu_id

    def _load_multipage_tiff(self, path):
        """Load tiff images containing many images in the channel dimension"""
        return np.array([np.array(p) for p in ImageSequence.Iterator(Image.open(path))])

    def _normalize_inputs(self, inputs):
        """Normalize inputs"""
        inputs = tf.expand_dims(tf.cast(inputs, tf.float32), -1)

        # Center around zero
        inputs = tf.divide(inputs, 127.5) - 1

        inputs = tf.image.resize_images(inputs, (392, 392))

        return tf.image.resize_image_with_crop_or_pad(inputs, 572, 572)

    def _normalize_labels(self, labels):
        """Normalize labels"""
        labels = tf.expand_dims(tf.cast(labels, tf.float32), -1)
        labels = tf.divide(labels, 255)

        labels = tf.image.resize_images(labels, (388, 388))
        labels = tf.image.resize_image_with_crop_or_pad(labels, 572, 572)

        cond = tf.less(labels, 0.5 * tf.ones(tf.shape(labels)))
        labels = tf.where(cond, tf.zeros(tf.shape(labels)), tf.ones(tf.shape(labels)))

        return tf.one_hot(tf.squeeze(tf.cast(labels, tf.int32)), 2)

    def _preproc_samples(self, inputs, labels, augment=True):
        """Preprocess samples and perform random augmentations"""
        inputs = self._normalize_inputs(inputs)
        labels = self._normalize_labels(labels)

        if self._augment and augment:
            # Horizontal flip
            h_flip = tf.random_uniform([]) > 0.5
            inputs = tf.cond(h_flip, lambda: tf.image.flip_left_right(inputs), lambda: inputs)
            labels = tf.cond(h_flip, lambda: tf.image.flip_left_right(labels), lambda: labels)

            # Vertical flip
            v_flip = tf.random_uniform([]) > 0.5
            inputs = tf.cond(v_flip, lambda: tf.image.flip_up_down(inputs), lambda: inputs)
            labels = tf.cond(v_flip, lambda: tf.image.flip_up_down(labels), lambda: labels)

            # Prepare for batched transforms
            inputs = tf.expand_dims(inputs, 0)
            labels = tf.expand_dims(labels, 0)

            # Elastic deformation

            alpha = tf.random.uniform([], minval=0, maxval=34)

            # Create random vector flows
            delta_x = tf.random.uniform([1, 4, 4, 1], minval=-1, maxval=1)
            delta_y = tf.random.uniform([1, 4, 4, 1], minval=-1, maxval=1)

            # Build 2D flow and apply
            flow = tf.concat([delta_x, delta_y], axis=-1) * alpha
            inputs = tf.contrib.image.dense_image_warp(inputs,
                                                       tf.image.resize_images(flow, (572, 572)))
            labels = tf.contrib.image.dense_image_warp(labels,
                                                       tf.image.resize_images(flow, (572, 572)))

            # Rotation invariance

            # Rotate by random angle\
            radian = tf.random_uniform([], maxval=360) * math.pi / 180
            inputs = tf.contrib.image.rotate(inputs, radian)
            labels = tf.contrib.image.rotate(labels, radian)

            # Shift invariance

            # Random crop and resize
            left = tf.random_uniform([]) * 0.3
            right = 1 - tf.random_uniform([]) * 0.3
            top = tf.random_uniform([]) * 0.3
            bottom = 1 - tf.random_uniform([]) * 0.3

            inputs = tf.image.crop_and_resize(inputs, [[top, left, bottom, right]], [0], (572, 572))
            labels = tf.image.crop_and_resize(labels, [[top, left, bottom, right]], [0], (572, 572))

            # Gray value variations

            # Adjust brightness and keep values in range
            inputs = tf.image.random_brightness(inputs, max_delta=0.2)
            inputs = tf.clip_by_value(inputs, clip_value_min=-1, clip_value_max=1)

            inputs = tf.squeeze(inputs, 0)
            labels = tf.squeeze(labels, 0)

        # Bring back labels to network's output size and remove interpolation artifacts
        labels = tf.image.resize_image_with_crop_or_pad(labels, target_width=388, target_height=388)
        cond = tf.less(labels, 0.5 * tf.ones(tf.shape(labels)))
        labels = tf.where(cond, tf.zeros(tf.shape(labels)), tf.ones(tf.shape(labels)))

        return (inputs, labels)

    def train_fn(self):
        """Input function for training"""
        dataset = tf.data.Dataset.from_tensor_slices(
            (self._train_images, self._train_masks))
        dataset = dataset.shuffle(self._batch_size * 3)
        dataset = dataset.repeat()
        dataset = dataset.shard(self._num_gpus, self._gpu_id)
        dataset = dataset.apply(
            tf.data.experimental.map_and_batch(map_func=self._preproc_samples,
                                               batch_size=self._batch_size,
                                               num_parallel_calls=multiprocessing.cpu_count()))
        dataset = dataset.prefetch(self._batch_size)

        return dataset

    def test_fn(self):
        """Input function for testing"""
        dataset = tf.data.Dataset.from_tensor_slices(
            (self._test_images))
        dataset = dataset.map(self._normalize_inputs)
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.prefetch(self._batch_size)

        return dataset

    def synth_fn(self):
        """Synthetic data function for testing"""
        inputs = tf.truncated_normal((572, 572, 1), dtype=tf.float32, mean=127.5, stddev=1, seed=self._seed,
                                     name='synth_inputs')
        masks = tf.truncated_normal((388, 388, 2), dtype=tf.float32, mean=0.01, stddev=0.1, seed=self._seed,
                                    name='synth_masks')

        dataset = tf.data.Dataset.from_tensors((inputs, masks))

        dataset = dataset.cache()
        dataset = dataset.repeat()
        dataset = dataset.batch(self._batch_size)

        dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

        return dataset
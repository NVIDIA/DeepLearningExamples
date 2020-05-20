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
import multiprocessing
import os
from collections import deque

import numpy as np
import tensorflow as tf
from PIL import Image, ImageSequence


class Dataset:
    """Load, separate and prepare the data for training and prediction"""

    def __init__(self, data_dir, batch_size, fold, augment=False, gpu_id=0, num_gpus=1, seed=0):
        if not os.path.exists(data_dir):
            raise FileNotFoundError('Cannot find data dir: {}'.format(data_dir))
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._augment = augment

        self._seed = seed

        images = self._load_multipage_tiff(os.path.join(self._data_dir, 'train-volume.tif'))
        masks = self._load_multipage_tiff(os.path.join(self._data_dir, 'train-labels.tif'))
        self._test_images = \
            self._load_multipage_tiff(os.path.join(self._data_dir, 'test-volume.tif'))

        train_indices, val_indices = self._get_val_train_indices(len(images), fold)
        self._train_images = images[train_indices]
        self._train_masks = masks[train_indices]
        self._val_images = images[val_indices]
        self._val_masks = masks[val_indices]

        self._num_gpus = num_gpus
        self._gpu_id = gpu_id

    @property
    def train_size(self):
        return len(self._train_images)

    @property
    def eval_size(self):
        return len(self._val_images)

    @property
    def test_size(self):
        return len(self._test_images)

    def _load_multipage_tiff(self, path):
        """Load tiff images containing many images in the channel dimension"""
        return np.array([np.array(p) for p in ImageSequence.Iterator(Image.open(path))])

    def _get_val_train_indices(self, length, fold, ratio=0.8):
        assert 0 < ratio <= 1, "Train/total data ratio must be in range (0.0, 1.0]"
        np.random.seed(self._seed)
        indices = np.arange(0, length, 1, dtype=np.int)
        np.random.shuffle(indices)
        if fold is not None:
            indices = deque(indices)
            indices.rotate(fold * int((1.0 - ratio) * length))
            indices = np.array(indices)
            train_indices = indices[:int(ratio * len(indices))]
            val_indices = indices[int(ratio * len(indices)):]
        else:
            train_indices = indices
            val_indices = []
        return train_indices, val_indices

    def _normalize_inputs(self, inputs):
        """Normalize inputs"""
        inputs = tf.expand_dims(tf.cast(inputs, tf.float32), -1)

        # Center around zero
        inputs = tf.divide(inputs, 127.5) - 1
        # Resize to match output size
        inputs = tf.image.resize(inputs, (388, 388))

        return tf.image.resize_with_crop_or_pad(inputs, 572, 572)

    def _normalize_labels(self, labels):
        """Normalize labels"""
        labels = tf.expand_dims(tf.cast(labels, tf.float32), -1)
        labels = tf.divide(labels, 255)

        # Resize to match output size
        labels = tf.image.resize(labels, (388, 388))
        labels = tf.image.resize_with_crop_or_pad(labels, 572, 572)

        cond = tf.less(labels, 0.5 * tf.ones(tf.shape(input=labels)))
        labels = tf.where(cond, tf.zeros(tf.shape(input=labels)), tf.ones(tf.shape(input=labels)))

        return tf.one_hot(tf.squeeze(tf.cast(labels, tf.int32)), 2)

    @tf.function
    def _preproc_samples(self, inputs, labels, augment=True):
        """Preprocess samples and perform random augmentations"""
        inputs = self._normalize_inputs(inputs)
        labels = self._normalize_labels(labels)

        if self._augment and augment:
            # Horizontal flip
            h_flip = tf.random.uniform([]) > 0.5
            inputs = tf.cond(pred=h_flip, true_fn=lambda: tf.image.flip_left_right(inputs), false_fn=lambda: inputs)
            labels = tf.cond(pred=h_flip, true_fn=lambda: tf.image.flip_left_right(labels), false_fn=lambda: labels)

            # Vertical flip
            v_flip = tf.random.uniform([]) > 0.5
            inputs = tf.cond(pred=v_flip, true_fn=lambda: tf.image.flip_up_down(inputs), false_fn=lambda: inputs)
            labels = tf.cond(pred=v_flip, true_fn=lambda: tf.image.flip_up_down(labels), false_fn=lambda: labels)

            # Prepare for batched transforms
            inputs = tf.expand_dims(inputs, 0)
            labels = tf.expand_dims(labels, 0)

            # Random crop and resize
            left = tf.random.uniform([]) * 0.3
            right = 1 - tf.random.uniform([]) * 0.3
            top = tf.random.uniform([]) * 0.3
            bottom = 1 - tf.random.uniform([]) * 0.3

            inputs = tf.image.crop_and_resize(inputs, [[top, left, bottom, right]], [0], (572, 572))
            labels = tf.image.crop_and_resize(labels, [[top, left, bottom, right]], [0], (572, 572))

            # Gray value variations

            # Adjust brightness and keep values in range
            inputs = tf.image.random_brightness(inputs, max_delta=0.2)
            inputs = tf.clip_by_value(inputs, clip_value_min=-1, clip_value_max=1)

            inputs = tf.squeeze(inputs, 0)
            labels = tf.squeeze(labels, 0)

        # Bring back labels to network's output size and remove interpolation artifacts
        labels = tf.image.resize_with_crop_or_pad(labels, target_width=388, target_height=388)
        cond = tf.less(labels, 0.5 * tf.ones(tf.shape(input=labels)))
        labels = tf.where(cond, tf.zeros(tf.shape(input=labels)), tf.ones(tf.shape(input=labels)))

        return inputs, labels

    def train_fn(self, drop_remainder=False):
        """Input function for training"""
        dataset = tf.data.Dataset.from_tensor_slices(
            (self._train_images, self._train_masks))
        dataset = dataset.shard(self._num_gpus, self._gpu_id)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(self._batch_size * 3)
        dataset = dataset.map(self._preproc_samples,
                              num_parallel_calls=multiprocessing.cpu_count()//self._num_gpus)
        dataset = dataset.batch(self._batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(self._batch_size)

        return dataset

    def eval_fn(self, count, drop_remainder=False):
        """Input function for validation"""
        dataset = tf.data.Dataset.from_tensor_slices(
            (self._val_images, self._val_masks))
        dataset = dataset.repeat(count=count)
        dataset = dataset.map(self._preproc_samples,
                              num_parallel_calls=multiprocessing.cpu_count())
        dataset = dataset.batch(self._batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(self._batch_size)

        return dataset

    def test_fn(self, count, drop_remainder=False):
        """Input function for testing"""
        dataset = tf.data.Dataset.from_tensor_slices(
            self._test_images)
        dataset = dataset.repeat(count=count)
        dataset = dataset.map(self._normalize_inputs)
        dataset = dataset.batch(self._batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(self._batch_size)

        return dataset

    def synth_fn(self):
        """Synthetic data function for testing"""
        inputs = tf.random.truncated_normal((572, 572, 1), dtype=tf.float32, mean=127.5, stddev=1, seed=self._seed,
                                            name='synth_inputs')
        masks = tf.random.truncated_normal((388, 388, 2), dtype=tf.float32, mean=0.01, stddev=0.1, seed=self._seed,
                                           name='synth_masks')

        dataset = tf.data.Dataset.from_tensors((inputs, masks))

        dataset = dataset.cache()
        dataset = dataset.repeat()
        dataset = dataset.batch(self._batch_size)

        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

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

import json
import math
import multiprocessing
import os

import SimpleITK as sitk
import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
from scipy import stats


def parse_nifti(path, dtype, dst_size, interpolator, normalization=None, modality=None):
    sitk_image = load_image(path)
    sitk_image = resize_image(sitk_image,
                              dst_size=dst_size,
                              interpolator=interpolator)

    image = sitk_to_np(sitk_image)

    if modality and 'CT' not in modality:
        if normalization:
            image = stats.zscore(image, axis=None)
    elif modality:
        raise NotImplementedError

    return image


def make_ref_image(img_path, dst_size, interpolator):
    ref_image = load_image(img_path)

    ref_image = resize_image(ref_image, dst_size=dst_size,
                             interpolator=interpolator)
    return sitk_to_np(ref_image) / np.max(ref_image) * 255


def make_interpolator(interpolator):
    if interpolator == 'linear':
        return sitk.sitkLinear
    else:
        raise ValueError("Unknown interpolator type")


def load_image(img_path):
    image = sitk.ReadImage(img_path)

    if image.GetDimension() == 4:
        image = sitk.GetImageFromArray(sitk.GetArrayFromImage(image)[-1, :, :, :])

    if image.GetPixelID() != sitk.sitkFloat32:
        return sitk.Cast(image, sitk.sitkFloat32)

    return image


def sitk_to_np(sitk_img):
    return np.transpose(sitk.GetArrayFromImage(sitk_img), [2, 1, 0])


def resize_image(sitk_img,
                 dst_size=(128, 128, 64),
                 interpolator=sitk.sitkNearestNeighbor):
    reference_image = sitk.Image(dst_size, sitk_img.GetPixelIDValue())
    reference_image.SetOrigin(sitk_img.GetOrigin())
    reference_image.SetDirection(sitk_img.GetDirection())
    reference_image.SetSpacing(
        [sz * spc / nsz for nsz, sz, spc in zip(dst_size, sitk_img.GetSize(), sitk_img.GetSpacing())])

    return sitk.Resample(sitk_img, reference_image, sitk.Transform(3, sitk.sitkIdentity), interpolator)


class MSDJsonParser:
    def __init__(self, json_path):
        with open(json_path) as f:
            data = json.load(f)

            self._labels = data.get('labels')
            self._x_train = [os.path.join(os.path.dirname(json_path), p['image']) for p in data.get('training')]
            self._y_train = [os.path.join(os.path.dirname(json_path), p['label']) for p in data.get('training')]
            self._x_test = [os.path.join(os.path.dirname(json_path), p) for p in data.get('test')]
            self._modality = [data.get('modality')[k] for k in data.get('modality').keys()]

    @property
    def labels(self):
        return self._labels

    @property
    def x_train(self):
        return self._x_train

    @property
    def y_train(self):
        return self._y_train

    @property
    def x_test(self):
        return self._x_test

    @property
    def modality(self):
        return self._modality


def make_split(json_parser, train_split, split_seed=0):
    np.random.seed(split_seed)

    train_size = int(len(json_parser.x_train) * train_split)

    return np.array(json_parser.x_train)[:train_size], np.array(json_parser.y_train)[:train_size], \
           np.array(json_parser.x_train)[train_size:], np.array(json_parser.y_train)[train_size:]


class MSDDataset(object):
    def __init__(self, json_path,
                 dst_size=[128, 128, 64],
                 seed=None,
                 interpolator=None,
                 data_normalization=None,
                 batch_size=1,
                 train_split=1.0,
                 split_seed=0):
        self._json_parser = MSDJsonParser(json_path)
        self._interpolator = make_interpolator(interpolator)

        self._ref_image = make_ref_image(img_path=self._json_parser.x_test[0],
                                         dst_size=dst_size,
                                         interpolator=self._interpolator)

        np.random.seed(split_seed)

        self._train_img, self._train_label, \
        self._eval_img, self._eval_label = make_split(self._json_parser, train_split)
        self._test_img = np.array(self._json_parser.x_test)

        self._dst_size = dst_size

        self._seed = seed
        self._batch_size = batch_size
        self._train_split = train_split
        self._data_normalization = data_normalization

        np.random.seed(self._seed)

    @property
    def labels(self):
        return self._json_parser.labels

    @property
    def train_steps(self):
        global_batch_size = hvd.size() * self._batch_size

        return math.ceil(
            len(self._train_img) / global_batch_size)

    @property
    def eval_steps(self):
        return math.ceil(len(self._eval_img) / self._batch_size)

    @property
    def test_steps(self):
        return math.ceil(len(self._test_img) / self._batch_size)

    def _parse_image(self, img):
        return parse_nifti(path=img,
                           dst_size=self._dst_size,
                           dtype=tf.float32,
                           interpolator=self._interpolator,
                           normalization=self._data_normalization,
                           modality=self._json_parser.modality)

    def _parse_label(self, label):
        return parse_nifti(path=label,
                           dst_size=self._dst_size,
                           dtype=tf.int32,
                           interpolator=sitk.sitkNearestNeighbor)

    def _augment(self, x, y):
        # Horizontal flip
        h_flip = tf.random_uniform([]) > 0.5
        x = tf.cond(h_flip, lambda: tf.image.flip_left_right(x), lambda: x)
        y = tf.cond(h_flip, lambda: tf.image.flip_left_right(y), lambda: y)

        # Vertical flip
        v_flip = tf.random_uniform([]) > 0.5
        x = tf.cond(v_flip, lambda: tf.image.flip_up_down(x), lambda: x)
        y = tf.cond(v_flip, lambda: tf.image.flip_up_down(y), lambda: y)

        return x, y

    def _img_generator(self, collection):
        for element in collection:
            yield self._parse_image(element)

    def _label_generator(self, collection):
        for element in collection:
            yield self._parse_label(element)

    def train_fn(self, augment):
        images = tf.data.Dataset.from_generator(generator=lambda: self._img_generator(self._train_img),
                                                output_types=tf.float32,
                                                output_shapes=(32, 32, 32))
        labels = tf.data.Dataset.from_generator(generator=lambda: self._label_generator(self._train_label),
                                                output_types=tf.int32,
                                                output_shapes=(32, 32, 32))

        dataset = tf.data.Dataset.zip((images, labels))

        dataset = dataset.cache()

        dataset = dataset.repeat()

        dataset = dataset.shuffle(buffer_size=self._batch_size * 2,
                                  reshuffle_each_iteration=True,
                                  seed=self._seed)
        dataset = dataset.shard(hvd.size(), hvd.rank())

        if augment:
            dataset = dataset.apply(
                tf.data.experimental.map_and_batch(map_func=self._augment,
                                                   batch_size=self._batch_size,
                                                   drop_remainder=True,
                                                   num_parallel_calls=multiprocessing.cpu_count()))
        else:
            dataset = dataset.batch(self._batch_size)

        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def eval_fn(self):
        images = tf.data.Dataset.from_generator(generator=lambda: self._img_generator(self._eval_img),
                                                output_types=tf.float32,
                                                output_shapes=(32, 32, 32))
        labels = tf.data.Dataset.from_generator(generator=lambda: self._label_generator(self._eval_label),
                                                output_types=tf.int32,
                                                output_shapes=(32, 32, 32))
        dataset = tf.data.Dataset.zip((images, labels))

        dataset = dataset.cache()

        dataset = dataset.batch(self._batch_size, drop_remainder=True)

        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def test_fn(self, count=1):
        dataset = tf.data.Dataset.from_generator(generator=lambda: self._img_generator(self._test_img),
                                                 output_types=tf.float32,
                                                 output_shapes=(32, 32, 32))

        dataset = dataset.cache()

        dataset = dataset.repeat(count=count)

        dataset = dataset.batch(self._batch_size, drop_remainder=True)

        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

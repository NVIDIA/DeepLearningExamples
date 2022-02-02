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

""" Data loader """
import os

import numpy as np
import horovod.tensorflow as hvd
import tensorflow as tf

from dataset.transforms import NormalizeImages, OneHotLabels, apply_transforms, PadXYZ, RandomCrop3D, \
    RandomHorizontalFlip, RandomBrightnessCorrection, CenterCrop, \
    apply_test_transforms, Cast

CLASSES = {0: "tumor_core", 1: "peritumoral_edema", 2: "enhancing_tumor"}


def cross_validation(arr: np.ndarray, fold_idx: int, n_folds: int):
    """ Split data into folds for training and evaluation

    :param arr: Collection items to split
    :param fold_idx: Index of crossvalidation fold
    :param n_folds: Total number of folds
    :return: Train and Evaluation folds
    """
    if fold_idx < 0 or fold_idx >= n_folds:
        raise ValueError('Fold index has to be [0, n_folds). Received index {} for {} folds'.format(fold_idx, n_folds))

    _folders = np.array_split(arr, n_folds)

    return np.concatenate(_folders[:fold_idx] + _folders[fold_idx + 1:]), _folders[fold_idx]


class Dataset: # pylint: disable=R0902
    """ Class responsible for the data loading during training, inference and evaluation """

    def __init__(self, data_dir, batch_size=2, input_shape=(128, 128, 128), # pylint: disable=R0913
                 fold_idx=0, n_folds=5, seed=0, params=None):
        """ Creates and configures the dataset

        :param data_dir: Directory where the data is stored
        :param batch_size: Number of pairs to be provided by batch
        :param input_shape: Dimension of the input to the model
        :param fold_idx: Fold index for crossvalidation
        :param n_folds: Total number of folds in crossvalidation
        :param seed: Random seed
        :param params: Dictionary with additional configuration parameters
        """
        self._folders = np.array([os.path.join(data_dir, path) for path in os.listdir(data_dir)
                                  if path.endswith(".tfrecords")])
        assert len(self._folders) > 0, "No matching data found at {}".format(data_dir)
        self._train, self._eval = cross_validation(self._folders, fold_idx=fold_idx, n_folds=n_folds)
        self._input_shape = input_shape
        self._data_dir = data_dir
        self.params = params

        self._batch_size = batch_size
        self._seed = seed

        self._xshape = (240, 240, 155, 4)
        self._yshape = (240, 240, 155)

    def parse(self, serialized):
        """ Parse TFRecord

        :param serialized: Serialized record for a particular example
        :return: sample, label, mean and std of intensities
        """
        features = {
            'X': tf.io.FixedLenFeature([], tf.string),
            'Y': tf.io.FixedLenFeature([], tf.string),
            'mean': tf.io.FixedLenFeature([4], tf.float32),
            'stdev': tf.io.FixedLenFeature([4], tf.float32)
        }

        parsed_example = tf.io.parse_single_example(serialized=serialized,
                                                    features=features)

        sample = tf.io.decode_raw(parsed_example['X'], tf.uint8)
        sample = tf.cast(tf.reshape(sample, self._xshape), tf.uint8)
        label = tf.io.decode_raw(parsed_example['Y'], tf.uint8)
        label = tf.cast(tf.reshape(label, self._yshape), tf.uint8)

        mean = parsed_example['mean']
        stdev = parsed_example['stdev']

        return sample, label, mean, stdev

    def parse_x(self, serialized):
        """ Parse only the sample in a TFRecord with sample and label

        :param serialized:
        :return: sample, mean and std of intensities
        """
        features = {'X': tf.io.FixedLenFeature([], tf.string),
                    'Y': tf.io.FixedLenFeature([], tf.string),
                    'mean': tf.io.FixedLenFeature([4], tf.float32),
                    'stdev': tf.io.FixedLenFeature([4], tf.float32)}

        parsed_example = tf.io.parse_single_example(serialized=serialized,
                                                    features=features)

        sample = tf.io.decode_raw(parsed_example['X'], tf.uint8)
        sample = tf.cast(tf.reshape(sample, self._xshape), tf.uint8)

        mean = parsed_example['mean']
        stdev = parsed_example['stdev']

        return sample, mean, stdev

    def train_fn(self):
        """ Create dataset for training """
        if 'debug' in self.params.exec_mode:
            return self.synth_train_fn()

        assert len(self._train) > 0, "Training data not found."

        dataset = tf.data.TFRecordDataset(filenames=self._train)

        dataset = dataset.shard(hvd.size(), hvd.rank())
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=self._batch_size * 8, seed=self._seed)
        dataset = dataset.repeat()

        dataset = dataset.map(self.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        transforms = [
            RandomCrop3D(self._input_shape),
            RandomHorizontalFlip() if self.params.augment else None,
            Cast(dtype=tf.float32),
            NormalizeImages(),
            RandomBrightnessCorrection() if self.params.augment else None,
            OneHotLabels(n_classes=4),
        ]

        dataset = dataset.map(
            map_func=lambda x, y, mean, stdev: apply_transforms(x, y, mean, stdev, transforms=transforms),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.batch(batch_size=self._batch_size,
                                drop_remainder=True)

        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        if self._batch_size == 1:
            options = dataset.options()
            options.experimental_optimization.map_and_batch_fusion = False
            dataset = dataset.with_options(options)

        return dataset

    def eval_fn(self):
        """ Create dataset for evaluation """
        dataset = tf.data.TFRecordDataset(filenames=self._eval)
        assert len(self._eval) > 0, "Evaluation data not found. Did you specify --fold flag?"

        dataset = dataset.cache()
        dataset = dataset.map(self.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        transforms = [
            CenterCrop((224, 224, 155)),
            Cast(dtype=tf.float32),
            NormalizeImages(),
            OneHotLabels(n_classes=4),
            PadXYZ()
        ]

        dataset = dataset.map(
            map_func=lambda x, y, mean, stdev: apply_transforms(x, y, mean, stdev, transforms=transforms),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size=self._batch_size,
                                drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def test_fn(self):
        """ Create dataset for inference """
        if 'debug' in self.params.exec_mode:
            return self.synth_predict_fn()

        count = 1 if not self.params.benchmark \
            else 2 * self.params.warmup_steps * self.params.batch_size // self.test_size

        dataset = tf.data.TFRecordDataset(filenames=self._eval)
        assert len(self._eval) > 0, "Evaluation data not found. Did you specify --fold flag?"

        dataset = dataset.repeat(count)
        dataset = dataset.map(self.parse_x, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        transforms = [
            CenterCrop((224, 224, 155)),
            Cast(dtype=tf.float32),
            NormalizeImages(),
            PadXYZ((224, 224, 160))
        ]

        dataset = dataset.map(
            map_func=lambda x, mean, stdev: apply_test_transforms(x, mean, stdev, transforms=transforms),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size=self._batch_size,
                                drop_remainder=self.params.benchmark)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def export_fn(self):
        """ Create dataset for calibrating and exporting """
        dataset = tf.data.TFRecordDataset(filenames=self._eval)
        assert len(self._eval) > 0, "Evaluation data not found. Did you specify --fold flag?"

        dataset = dataset.repeat(1)
        dataset = dataset.map(self.parse_x, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        transforms = [
            CenterCrop((224, 224, 155)),
            Cast(dtype=tf.float32),
            NormalizeImages(),
            PadXYZ((224, 224, 160))
        ]

        dataset = dataset.map(
            map_func=lambda x, mean, stdev: apply_test_transforms(x, mean, stdev, transforms=transforms),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size=self._batch_size,
                                drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def synth_train_fn(self):
        """ Synthetic data function for training """
        inputs = tf.random.uniform(self._xshape, dtype=tf.int32, minval=0, maxval=255, seed=self._seed,
                                   name='synth_inputs')
        masks = tf.random.uniform(self._yshape, dtype=tf.int32, minval=0, maxval=4, seed=self._seed,
                                  name='synth_masks')
        mean = tf.random.uniform((4,), dtype=tf.float32, minval=0, maxval=255, seed=self._seed)
        stddev = tf.random.uniform((4,), dtype=tf.float32, minval=0, maxval=1, seed=self._seed)

        dataset = tf.data.Dataset.from_tensors((inputs, masks))
        dataset = dataset.repeat()

        transforms = [
            Cast(dtype=tf.uint8),
            RandomCrop3D((128, 128, 128)),
            RandomHorizontalFlip() if self.params.augment else None,
            Cast(dtype=tf.float32),
            NormalizeImages(),
            RandomBrightnessCorrection() if self.params.augment else None,
            OneHotLabels(n_classes=4),
        ]

        dataset = dataset.map(map_func=lambda x, y: apply_transforms(x, y, mean, stddev, transforms),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def synth_predict_fn(self):
        """Synthetic data function for testing"""
        inputs = tf.random.truncated_normal((224, 224, 160, 4), dtype=tf.float32, mean=0.0, stddev=1.0, seed=self._seed,
                                     name='synth_inputs')

        count = 2 * self.params.warmup_steps

        dataset = tf.data.Dataset.from_tensors(inputs)
        dataset = dataset.repeat(count)
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    @property
    def train_size(self):
        """ Number of pairs in the training set """
        return len(self._train)

    @property
    def eval_size(self):
        """ Number of pairs in the validation set """
        return len(self._eval)

    @property
    def test_size(self):
        """ Number of pairs in the test set """
        return len(self._eval)

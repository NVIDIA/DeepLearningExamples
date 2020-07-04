# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import os

import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf

from dataset.transforms import NormalizeImages, OneHotLabels, apply_transforms, PadXYZ, RandomCrop3D, \
    RandomHorizontalFlip, RandomGammaCorrection, RandomVerticalFlip, RandomBrightnessCorrection, CenterCrop, \
    apply_test_transforms, Cast

CLASSES = {0: "TumorCore", 1: "PeritumoralEdema", 2: "EnhancingTumor"}


def cross_validation(x: np.ndarray, fold_idx: int, n_folds: int):
    if fold_idx < 0 or fold_idx >= n_folds:
        raise ValueError('Fold index has to be [0, n_folds). Received index {} for {} folds'.format(fold_idx, n_folds))

    _folders = np.array_split(x, n_folds)

    return np.concatenate(_folders[:fold_idx] + _folders[fold_idx + 1:]), _folders[fold_idx]


class Dataset:
    def __init__(self, data_dir, batch_size=2, fold_idx=0, n_folds=5, seed=0, pipeline_factor=1, params=None):
        self._folders = np.array([os.path.join(data_dir, path) for path in os.listdir(data_dir)])
        self._train, self._eval = cross_validation(self._folders, fold_idx=fold_idx, n_folds=n_folds)
        self._pipeline_factor = pipeline_factor
        self._data_dir = data_dir
        self.params = params

        self._batch_size = batch_size
        self._seed = seed

        self._xshape = (240, 240, 155, 4)
        self._yshape = (240, 240, 155)

    def parse(self, serialized):
        features = {
            'X': tf.io.FixedLenFeature([], tf.string),
            'Y': tf.io.FixedLenFeature([], tf.string),
            'mean': tf.io.FixedLenFeature([4], tf.float32),
            'stdev': tf.io.FixedLenFeature([4], tf.float32)
        }

        parsed_example = tf.io.parse_single_example(serialized=serialized,
                                                    features=features)

        x = tf.io.decode_raw(parsed_example['X'], tf.uint8)
        x = tf.cast(tf.reshape(x, self._xshape), tf.uint8)
        y = tf.io.decode_raw(parsed_example['Y'], tf.uint8)
        y = tf.cast(tf.reshape(y, self._yshape), tf.uint8)

        mean = parsed_example['mean']
        stdev = parsed_example['stdev']

        return x, y, mean, stdev

    def parse_x(self, serialized):
        features = {'X': tf.io.FixedLenFeature([], tf.string),
                    'Y': tf.io.FixedLenFeature([], tf.string),
                    'mean': tf.io.FixedLenFeature([4], tf.float32),
                    'stdev': tf.io.FixedLenFeature([4], tf.float32)}

        parsed_example = tf.io.parse_single_example(serialized=serialized,
                                                    features=features)

        x = tf.io.decode_raw(parsed_example['X'], tf.uint8)
        x = tf.cast(tf.reshape(x, self._xshape), tf.uint8)

        mean = parsed_example['mean']
        stdev = parsed_example['stdev']

        return x, mean, stdev

    def train_fn(self):
        assert len(self._train) > 0, "Training data not found."

        ds = tf.data.TFRecordDataset(filenames=self._train)

        ds = ds.shard(hvd.size(), hvd.rank())
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=self._batch_size * 8, seed=self._seed)
        ds = ds.repeat()

        ds = ds.map(self.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        transforms = [
            RandomCrop3D((128, 128, 128)),
            RandomHorizontalFlip() if self.params.augment else None,
            Cast(dtype=tf.float32),
            NormalizeImages(),
            RandomBrightnessCorrection() if self.params.augment else None,
            OneHotLabels(n_classes=4),
        ]

        ds = ds.map(map_func=lambda x, y, mean, stdev: apply_transforms(x, y, mean, stdev, transforms=transforms),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds = ds.batch(batch_size=self._batch_size,
                      drop_remainder=True)

        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return ds

    def eval_fn(self):
        ds = tf.data.TFRecordDataset(filenames=self._eval)
        assert len(self._eval) > 0, "Evaluation data not found. Did you specify --fold flag?"

        ds = ds.cache()
        ds = ds.map(self.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        transforms = [
            CenterCrop((224, 224, 155)),
            Cast(dtype=tf.float32),
            NormalizeImages(),
            OneHotLabels(n_classes=4),
            PadXYZ()
        ]

        ds = ds.map(map_func=lambda x, y, mean, stdev: apply_transforms(x, y, mean, stdev, transforms=transforms),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size=self._batch_size,
                      drop_remainder=False)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return ds

    def test_fn(self, count=1, drop_remainder=False):
        ds = tf.data.TFRecordDataset(filenames=self._eval)
        assert len(self._eval) > 0, "Evaluation data not found. Did you specify --fold flag?"

        ds = ds.repeat(count)
        ds = ds.map(self.parse_x, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        transforms = [
            CenterCrop((224, 224, 155)),
            Cast(dtype=tf.float32),
            NormalizeImages(),
            PadXYZ((224, 224, 160))
        ]

        ds = ds.map(map_func=lambda x, mean, stdev: apply_test_transforms(x, mean, stdev, transforms=transforms),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size=self._batch_size,
                      drop_remainder=drop_remainder)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return ds

    def synth_train_fn(self):
        """Synthetic data function for testing"""
        inputs = tf.random_uniform(self._xshape, dtype=tf.int32, minval=0, maxval=255, seed=self._seed,
                                   name='synth_inputs')
        masks = tf.random_uniform(self._yshape, dtype=tf.int32, minval=0, maxval=4, seed=self._seed,
                                  name='synth_masks')

        ds = tf.data.Dataset.from_tensors((inputs, masks))
        ds = ds.repeat()

        transforms = [
            Cast(dtype=tf.uint8),
            RandomCrop3D((128, 128, 128)),
            RandomHorizontalFlip() if self.params.augment else None,
            Cast(dtype=tf.float32),
            NormalizeImages(),
            RandomBrightnessCorrection() if self.params.augment else None,
            OneHotLabels(n_classes=4),
        ]

        ds = ds.map(map_func=lambda x, y: apply_transforms(x, y, transforms),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(self._batch_size)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return ds

    def synth_predict_fn(self, count=1):
        """Synthetic data function for testing"""
        inputs = tf.truncated_normal((64, 64, 64, 4), dtype=tf.float32, mean=0.0, stddev=1.0, seed=self._seed,
                                     name='synth_inputs')

        ds = tf.data.Dataset.from_tensors(inputs)
        ds = ds.repeat(count)
        ds = ds.batch(self._batch_size)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return ds

    @property
    def train_size(self):
        return len(self._train)

    @property
    def eval_size(self):
        return len(self._eval)

    @property
    def test_size(self):
        return len(self._eval)


def main():
    from time import time
    hvd.init()

    dataset = Dataset(data_dir='/data/BraTS19_tfrecord', batch_size=3)

    it = dataset.test().make_initializable_iterator()

    sess = tf.Session()
    sess.run(it.initializer)

    next_element = it.get_next()

    t0 = time()
    cnt = 0
    # while True:
    import matplotlib.pyplot as plt
    import numpy.ma as ma
    for i in range(200):
        t0 = time()
        # if i == 20:
        #     t0 = time()

        res = sess.run(next_element)
        a = res[0]
        a = a[0, :, :, 80, 0]
        a = ma.masked_array(a, mask=a == 0)
        # plt.imshow(a.astype(np.uint8))
        plt.imshow(a)
        plt.colorbar()
        plt.savefig("/opt/project/img.png")

        # print()
        print(time() - t0)


if __name__ == '__main__':
    main()

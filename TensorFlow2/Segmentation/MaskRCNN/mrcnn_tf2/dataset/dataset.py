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
""" Data loading and processing.

Defines dataset class that exports input functions of Mask-RCNN
for training and evaluation using Estimator API.

The train_fn includes training data for category classification,
bounding box regression, and number of positive examples to normalize
the loss during training.

"""
import glob
import logging
import os

import tensorflow as tf

from mrcnn_tf2.dataset.dataset_parser import dataset_parser

TRAIN_SPLIT_PATTERN = 'train*.tfrecord'
EVAL_SPLIT_PATTERN = 'val*.tfrecord'
TRAIN_SPLIT_SAMPLES = 118287


class Dataset:
    """ Load and preprocess the coco dataset. """

    def __init__(self, params):
        """ Configures dataset. """
        self._params = params

        self._train_files = glob.glob(os.path.join(self._params.data_dir, TRAIN_SPLIT_PATTERN))
        self._eval_files = glob.glob(os.path.join(self._params.data_dir, EVAL_SPLIT_PATTERN))

        self._logger = logging.getLogger('dataset')

    def train_fn(self, batch_size):
        """ Input function for training. """
        data = tf.data.TFRecordDataset(self._train_files)

        data = data.cache()
        data = data.shuffle(buffer_size=4096, reshuffle_each_iteration=True, seed=self._params.seed)
        data = data.repeat()

        data = data.map(
            lambda x: dataset_parser(
                value=x,
                mode='train',
                params=self._params,
                use_instance_mask=self._params.include_mask,
                seed=self._params.seed
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        data = data.batch(batch_size=batch_size, drop_remainder=True)

        if self._params.use_synthetic_data:
            self._logger.info("Using fake dataset loop")
            data = data.take(1).cache().repeat()

        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        data = data.with_options(self._data_options)

        return data

    def eval_fn(self, batch_size):
        """ Input function for validation. """
        data = tf.data.TFRecordDataset(self._eval_files)

        if self._params.eval_samples:
            self._logger.info(f'Amount of samples limited to {self._params.eval_samples}')
            data = data.take(self._params.eval_samples)

        data = data.cache()

        data = data.map(
            lambda x: dataset_parser(
                value=x,
                # dataset parser expects mode to be PREDICT even for evaluation
                mode='eval',
                params=self._params,
                use_instance_mask=self._params.include_mask,
                seed=self._params.seed
            ),
            num_parallel_calls=16
        )

        data = data.batch(batch_size=batch_size, drop_remainder=True)

        if self._params.use_synthetic_data:
            self._logger.info("Using fake dataset loop")
            data = data.take(1).cache().repeat()
            data = data.take(5000 // batch_size)

        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        # FIXME: This is a walkaround for a bug and should be removed as soon as the fix is merged
        # http://nvbugs/2967052 [V100][JoC][MaskRCNN][TF1] performance regression with 1 GPU
        data = data.apply(tf.data.experimental.prefetch_to_device('/gpu:0', buffer_size=1))

        return data

    @property
    def train_size(self):
        """ Size of the train dataset. """
        return TRAIN_SPLIT_SAMPLES

    @property
    def _data_options(self):
        """ Constructs tf.data.Options for this dataset. """
        data_options = tf.data.Options()

        data_options.experimental_optimization.parallel_batch = True
        data_options.experimental_slack = True
        data_options.experimental_threading.max_intra_op_parallelism = 1
        data_options.experimental_optimization.map_parallelization = True

        map_vectorization_options = tf.data.experimental.MapVectorizationOptions()
        map_vectorization_options.enabled = True
        map_vectorization_options.use_choose_fastest = True
        data_options.experimental_optimization.map_vectorization = map_vectorization_options

        return data_options

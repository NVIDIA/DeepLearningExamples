# Lint as: python3
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
# ==============================================================================
"""Dataset utilities for vision tasks using TFDS and tf.data.Dataset."""
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import os
from typing import Any, List, Optional, Tuple, Mapping, Union
import functools
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

from utils import augment, preprocessing, Dali
import horovod.tensorflow.keras as hvd
import nvidia.dali.plugin.tf as dali_tf





AUGMENTERS = {
    'autoaugment': augment.AutoAugment,
    'randaugment': augment.RandAugment,
}

class Dataset:
  """An object for building datasets.

  Allows building various pipelines fetching examples, preprocessing, etc.
  Maintains additional state information calculated from the dataset, i.e.,
  training set split, batch size, and number of steps (batches).
  """

  def __init__(self, 
  data_dir,
  index_file_dir,
  split='train',
  num_classes=None,
  image_size=224,
  num_channels=3,
  batch_size=128,
  dtype='float32',
  one_hot=False,
  use_dali=False,
  augmenter=None,
  shuffle_buffer_size=10000,
  file_shuffle_buffer_size=1024,
  cache=False,
  mean_subtract=False,
  standardize=False,
  augmenter_params=None,
  mixup_alpha=0.0):
    """Initialize the builder from the config."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError('Cannot find data dir: {}'.format(data_dir))
    if one_hot and num_classes is None:
        raise FileNotFoundError('Number of classes is required for one_hot')
    self._data_dir = data_dir
    self._split = split
    self._image_size = image_size
    self._num_classes = num_classes
    self._num_channels = num_channels
    self._batch_size = batch_size
    self._dtype = dtype
    self._one_hot = one_hot
    self._augmenter_name = augmenter
    self._shuffle_buffer_size = shuffle_buffer_size
    self._file_shuffle_buffer_size = file_shuffle_buffer_size
    self._cache = cache
    self._mean_subtract = mean_subtract
    self._standardize = standardize
    self._index_file = index_file_dir
    self._use_dali = use_dali
    self.mixup_alpha = mixup_alpha
    
    self._num_gpus = hvd.size()

    if self._augmenter_name is not None:
      augmenter = AUGMENTERS.get(self._augmenter_name, None)
      params = augmenter_params or {}
      self._augmenter = augmenter(**params) if augmenter is not None else None
    else:
      self._augmenter = None

  def mixup(self, batch_size, alpha, images, labels):
    """Applies Mixup regularization to a batch of images and labels.
    [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
      Mixup: Beyond Empirical Risk Minimization.
      ICLR'18, https://arxiv.org/abs/1710.09412
    Arguments:
      batch_size: The input batch size for images and labels.
      alpha: Float that controls the strength of Mixup regularization.
      images: A batch of images of shape [batch_size, ...]
      labels: A batch of labels of shape [batch_size, num_classes]
    Returns:
      A tuple of (images, labels) with the same dimensions as the input with
      Mixup regularization applied.
    """
    # Mixup of images will be performed on device later
    if alpha == 0.0:
      images_mix_weight = tf.ones([batch_size, 1, 1, 1])
      return (images, images_mix_weight), labels

    mix_weight = tf.compat.v1.distributions.Beta(alpha, alpha).sample([batch_size, 1])
    mix_weight = tf.maximum(mix_weight, 1. - mix_weight)
    images_mix_weight = tf.reshape(mix_weight, [batch_size, 1, 1, 1])
    # Mixup on a single batch is implemented by taking a weighted sum with the
    # same batch in reverse.
    labels_mix = labels * mix_weight + labels[::-1] * (1. - mix_weight)
    return (images, images_mix_weight), labels_mix


  @property
  def is_training(self) -> bool:
    """Whether this is the training set."""
    return self._split == 'train'

  @property
  def global_batch_size(self) -> int:
    """The batch size, multiplied by the number of replicas (if configured)."""
    return self._batch_size * self._num_gpus

  @property
  def local_batch_size(self):
    """The base unscaled batch size."""
    return self._batch_size

  @property
  def dtype(self) -> tf.dtypes.DType:
    """Converts the config's dtype string to a tf dtype.

    Returns:
      A mapping from string representation of a dtype to the `tf.dtypes.DType`.

    Raises:
      ValueError if the config's dtype is not supported.

    """
    dtype_map = {
        'float32': tf.float32,
        'bfloat16': tf.bfloat16,
        'float16': tf.float16,
        'fp32': tf.float32,
        'bf16': tf.bfloat16,
    }
    try:
      return dtype_map[self._dtype]
    except:
      raise ValueError('{} provided key. Invalid DType provided. Supported types: {}'.format(self._dtype,
          dtype_map.keys()))

  @property
  def image_size(self) -> int:
    """The size of each image (can be inferred from the dataset)."""
    return int(self._image_size)

  @property
  def num_channels(self) -> int:
    """The number of image channels (can be inferred from the dataset)."""
    return int(self._num_channels)

  @property
  def num_classes(self) -> int:
    """The number of classes (can be inferred from the dataset)."""
    return int(self._num_classes)

  @property
  def num_steps(self) -> int:
    """The number of classes (can be inferred from the dataset)."""
    return int(self._num_steps)

  def build(self) -> tf.data.Dataset:
    """Construct a dataset end-to-end and return it.

    Args:
      input_context: An optional context provided by `tf.distribute` for
        cross-replica training.

    Returns:
      A TensorFlow dataset outputting batched images and labels.
    """
    if self._use_dali:
        print("Using dali for {train} dataloading".format(train = "training" if self.is_training else "validation"))
        tfrec_filenames = sorted(tf.io.gfile.glob(os.path.join(self._data_dir, '%s-*' % self._split)))
        tfrec_idx_filenames = sorted(tf.io.gfile.glob(os.path.join(self._index_file, '%s-*' % self._split)))

        # # Create pipeline
        dali_pipeline = Dali.DaliPipeline(tfrec_filenames=tfrec_filenames,
        tfrec_idx_filenames=tfrec_idx_filenames,
        height=self._image_size,
        width=self._image_size,
        batch_size=self.local_batch_size,
        num_threads=1,
        device_id=hvd.local_rank(),
        shard_id=hvd.rank(),
        num_gpus=hvd.size(),
        num_classes=self.num_classes,
        deterministic=False,
        dali_cpu=False,
        training=self.is_training)

        # Define shapes and types of the outputs
        shapes = (
            (self.local_batch_size, self._image_size, self._image_size, 3),
            (self.local_batch_size, self._num_classes))
        dtypes = (
            tf.float32,
            tf.float32)

        # Create dataset
        dataset = dali_tf.DALIDataset(
            pipeline=dali_pipeline,
            batch_size=self.local_batch_size,
            output_shapes=shapes,
            output_dtypes=dtypes,
            device_id=hvd.local_rank())
        # if self.is_training and self._augmenter:
        #     print('Augmenting with {}'.format(self._augmenter))
        #     dataset.unbatch().map(self.augment_pipeline, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(self.local_batch_size)
        return dataset
    else:
        print("Using tf native pipeline for {train} dataloading".format(train = "training" if self.is_training else "validation"))
        dataset = self.load_records()
        dataset = self.pipeline(dataset)

        return dataset

  # def augment_pipeline(self, image, label) -> Tuple[tf.Tensor, tf.Tensor]:
  #   image = self._augmenter.distort(image)
  #   return image, label


  def load_records(self) -> tf.data.Dataset:
    """Return a dataset loading files with TFRecords."""
    if self._data_dir is None:
        raise ValueError('Dataset must specify a path for the data files.')

    file_pattern = os.path.join(self._data_dir,
                                  '{}*'.format(self._split))
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
    return dataset

  def pipeline(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Build a pipeline fetching, shuffling, and preprocessing the dataset.

    Args:
      dataset: A `tf.data.Dataset` that loads raw files.

    Returns:
      A TensorFlow dataset outputting batched images and labels.
    """
    if self._num_gpus > 1:
      dataset = dataset.shard(self._num_gpus, hvd.rank())

    if self.is_training:
      # Shuffle the input files.
      dataset.shuffle(buffer_size=self._file_shuffle_buffer_size)

    if self.is_training and not self._cache:
      dataset = dataset.repeat()

    # Read the data from disk in parallel
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=10,
        block_length=1,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if self._cache:
      dataset = dataset.cache()

    if self.is_training:
      dataset = dataset.shuffle(self._shuffle_buffer_size)
      dataset = dataset.repeat()

    # Parse, pre-process, and batch the data in parallel
    preprocess = self.parse_record
    dataset = dataset.map(preprocess,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if self._num_gpus > 1:
      # The batch size of the dataset will be multiplied by the number of
      # replicas automatically when strategy.distribute_datasets_from_function
      # is called, so we use local batch size here.
      dataset = dataset.batch(self.local_batch_size,
                              drop_remainder=self.is_training)
    else:
      dataset = dataset.batch(self.global_batch_size,
                              drop_remainder=self.is_training)

    # Apply Mixup
    mixup_alpha = self.mixup_alpha if self.is_training else 0.0
    dataset = dataset.map(
        functools.partial(self.mixup, self.local_batch_size, mixup_alpha),
        num_parallel_calls=64)

    # Prefetch overlaps in-feed with training
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

  def parse_record(self, record: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Parse an ImageNet record from a serialized string Tensor."""
    keys_to_features = {
        'image/encoded':
            tf.io.FixedLenFeature((), tf.string, ''),
        'image/format':
            tf.io.FixedLenFeature((), tf.string, 'jpeg'),
        'image/class/label':
            tf.io.FixedLenFeature([], tf.int64, -1),
        'image/class/text':
            tf.io.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin':
            tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin':
            tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax':
            tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax':
            tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/class/label':
            tf.io.VarLenFeature(dtype=tf.int64),
    }

    parsed = tf.io.parse_single_example(record, keys_to_features)

    label = tf.reshape(parsed['image/class/label'], shape=[1])
    label = tf.cast(label, dtype=tf.int32)

    # Subtract one so that labels are in [0, 1000)
    label -= 1

    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])
    image, label = self.preprocess(image_bytes, label)

    return image, label

  def preprocess(self, image: tf.Tensor, label: tf.Tensor
                ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply image preprocessing and augmentation to the image and label."""
    if self.is_training:
      image = preprocessing.preprocess_for_train(
          image,
          image_size=self._image_size,
          mean_subtract=self._mean_subtract,
          standardize=self._standardize,
          dtype=self.dtype,
          augmenter=self._augmenter)
    else:
      image = preprocessing.preprocess_for_eval(
          image,
          image_size=self._image_size,
          num_channels=self._num_channels,
          mean_subtract=self._mean_subtract,
          standardize=self._standardize,
          dtype=self.dtype)

    label = tf.cast(label, tf.int32)
    if self._one_hot:
      label = tf.one_hot(label, self.num_classes)
      label = tf.reshape(label, [self.num_classes])

    return image, label

  @classmethod
  def from_params(cls, *args, **kwargs):
    """Construct a dataset builder from a default config and any overrides."""
    config = DatasetConfig.from_args(*args, **kwargs)
    return cls(config)



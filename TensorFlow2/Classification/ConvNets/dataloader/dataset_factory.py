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

from dataloader import augment
from dataloader import preprocessing
from dataloader import Dali

import horovod.tensorflow.keras as hvd
import nvidia.dali.plugin.tf as dali_tf





AUGMENTERS = {
    'autoaugment': augment.AutoAugment,
    'randaugment': augment.RandAugment,
}

def cutmix_mask(alpha, h, w):
  """[summary]
  Returns image mask of size wxh for CutMix where the masked region is one 
  and bakground is zero. To create the mask, we first sample the top-left
  corner of the masked region and then determine its width and height by 
  sampling a scale ratio from the beta distribution parameterized by alpha. 
  The masked region determined above is painted white and then zero-padded
  to have width w and height h.
  
  Args:
      alpha ([float]): used to sample a scale ratio
      h ([integer]): width of the mask image
      w ([integer]): height of the mask image

  Returns:
      [type]: [description]
  """
    
  if alpha == 0:
    return tf.zeros((h,w,1))

  r_x = tf.random.uniform([], 0, w, tf.int32)
  r_y = tf.random.uniform([], 0, h, tf.int32)
  area = tf.compat.v1.distributions.Beta(alpha, alpha).sample()
  patch_ratio = tf.cast(tf.math.sqrt(1 - area), tf.float32)
  r_w = tf.cast(patch_ratio * tf.cast(w, tf.float32), tf.int32)
  r_h = tf.cast(patch_ratio * tf.cast(h, tf.float32), tf.int32)
  bbx1 = tf.clip_by_value(tf.cast(r_x - r_w // 2, tf.int32), 0, w)
  bby1 = tf.clip_by_value(tf.cast(r_y - r_h // 2, tf.int32), 0, h)
  bbx2 = tf.clip_by_value(tf.cast(r_x + r_w // 2, tf.int32), 0, w)
  bby2 = tf.clip_by_value(tf.cast(r_y + r_h // 2, tf.int32), 0, h)

  # Create the binary mask.
  pad_left = bbx1
  pad_top = bby1
  pad_right = tf.maximum(w - bbx2, 0)
  pad_bottom = tf.maximum(h - bby2, 0)
  r_h = bby2 - bby1
  r_w = bbx2 - bbx1

  mask = tf.pad(
      tf.ones((r_h, r_w)),
      paddings=[[pad_top, pad_bottom], [pad_left, pad_right]],
      mode='CONSTANT',
      constant_values=0)
  mask.set_shape((h, w))
  return mask[..., None]  # Add channel dim.
  
def mixup(batch_size, alpha, images, labels, defer_img_mixing):
  """Applies Mixup regularization to a batch of images and labels.
  [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
    Mixup: Beyond Empirical Risk Minimization.
    ICLR'18, https://arxiv.org/abs/1710.09412
  Arguments:
    batch_size: The input batch size for images and labels.
    alpha: Float that controls the strength of Mixup regularization.
    images: A batch of images of shape [batch_size, ...]
    labels: A batch of labels of shape [batch_size, num_classes]
    defer_img_mixing: If true, labels are mixed in this function but image 
    mixing is postponed until the data arrives on the compute device. This 
    can accelerate the data pipeline. Note that it is the user's responsibility
    to implement image mixing in the module that defines the forward pass of the
    network. To ensure that the subsequent on-device image mixing is consistent
    with label mixing performed here, this function returns the mixing weights
    as well.
  Returns:
    A tuple of ((images, mix_weights), labels) with the same dimensions as the input with
    Mixup regularization applied.
  """
  if alpha == 0.0:
    # returning 1s as mixing weights means to mixup
    return (images, tf.ones((batch_size,1,1,1))), labels

  mix_weight = tf.compat.v1.distributions.Beta(alpha, alpha).sample([batch_size, 1])
  mix_weight = tf.maximum(mix_weight, 1. - mix_weight)
  img_weight = tf.cast(tf.reshape(mix_weight, [batch_size, 1, 1, 1]), images.dtype)
  labels_weight = tf.cast(mix_weight, labels.dtype)
  # Mixup: taking a weighted sum with the same batch in reverse.
  labels_mix = labels * labels_weight + labels[::-1] * (1. - labels_weight)
  if not defer_img_mixing:
    images_mix = images * img_weight + images[::-1] * (1. - img_weight)
  else:
    # postpone image mixing
    images_mix = images
  return (images_mix, img_weight),  labels_mix

def cutmix(images, labels, masks, defer_img_mixing):
  """[summary]
  Applies CutMix regularization to a batch of images and labels.
  Reference: https://arxiv.org/pdf/1905.04899.pdf

  Args:
      images: a Tensor of batched images
      labels: a Tensor of batched labels.
      masks: a Tensor of batched masks.
      defer_img_mixing: If true, labels are mixed in this function but image 
      mixing is postponed until the data arrives on the compute device. This 
      can accelerate the data pipeline. Note that it is the user's responsibility
      to implement image mixing in the module that defines the forward pass of the
      network. To ensure that the subsequent on-device image mixing is consistent
      with label mixing performed here, this function returns the mixing masks
      as well.

  Returns:
      A tuple of ((images, mix_masks), labels)
  """
  
  mix_area = tf.reduce_sum(masks) / tf.cast(tf.size(masks), masks.dtype)
  mix_area = tf.cast(mix_area, labels.dtype)
  mixed_label = (1. - mix_area) * labels + mix_area * labels[::-1]
  masks = tf.cast(masks, images.dtype)
  if not defer_img_mixing:
    mixed_images = (1. - masks) * images + masks * images[::-1]
  else:
    # postpone image mixing
    mixed_images = images
  return (mixed_images, masks), mixed_label

def mixing(batch_size, mixup_alpha, cutmix_alpha, defer_img_mixing, features, labels):
  """Applies mixing regularization to a batch of images and labels. If both
  mixup and cutmix requested, the batch is halved followed by applying
  mixup on one half and cutmix on the other half. 
  
  Arguments:
    batch_size: The input batch size for images and labels.
    mixup_alpha: Float that controls the strength of Mixup regularization.
    cutmix_alpha: FLoat that controls the strength of Cutmix regularization.
    defer_img_mixing: If true, the image mixing ops will be postponed.
    labels: a dict of batched labels.
  Returns:
    A new dict of features with updated images and labels with the same
    dimensions as the input.
  """
  image = features['image']
  label = labels['label']
  mix_masks = features['cutmix_mask']
  
  if mixup_alpha and cutmix_alpha:
    # split the batch half-half, and apply mixup and cutmix for each half.
    bs = batch_size // 2
    (img1, mix_weights), lab1 = mixup(bs, mixup_alpha, image[:bs], label[:bs], defer_img_mixing)
    (img2, mix_masks), lab2 = cutmix(image[bs:], label[bs:], mix_masks[bs:], defer_img_mixing)

    image = tf.concat([img1, img2], axis=0)
    label = tf.concat([lab1, lab2], axis=0)
    
  elif mixup_alpha:
    # only mixup
    (image, mix_weights), label = mixup(batch_size, mixup_alpha, image, label, defer_img_mixing)
    # mix_masks = tf.zeros_like(mix_masks) -> mix_masks is already all 0s (see cutmix fn)
    
  elif cutmix_alpha:
    # only cutmix
    (image, mix_masks), label = cutmix(image, label, mix_masks, defer_img_mixing)
    mix_weights = tf.ones((batch_size,1,1,1)) # 1s mean no mixup
  
  else:
    # mix_masks = tf.zeros_like(mix_masks) -> mix_masks is already all 0s (see cutmix fn)
    mix_weights = tf.ones((batch_size,1,1,1)) # 1s mean no mixup
  
  features['image'] = image  
  features['mixup_weight'] = mix_weights
  features['cutmix_mask'] = mix_masks
  return features, label

def mixing_lite(images, mixup_weights, cutmix_masks, batch_size, do_mixup, do_cutmix):
  """[summary]
  This function, which  is a simplified version of the mixing function (see above), 
  will be called in the model module when the user wishes to perform image mixing 
  on-device (defer_image_mixing=True). 
  
  Note: the logic here must be identical to that of the mixing fn above.

  Args:
      images: a Tensor of batched images.
      mixup_weights: a Tensor of batched mixup weights.
      cutmix_masks: a Tensor of batched cutmix masks.
      batch_size: static batch size.
      do_mixup: boolean, to determine if mixup is needed
      do_cutmix: boolean, to determine if cutmix is needed

  Returns:
      a Tensor of batched MIXED images
  """
  if do_mixup and do_cutmix:
    # split the batch half-half, and apply mixup and cutmix for each half.
    bs = batch_size // 2
    images_mixup = images[:bs] * mixup_weights + images[:bs][::-1] * (1. - mixup_weights)
    images_cutmix = images[bs:] * (1. - cutmix_masks) * + images[bs:][::-1] * cutmix_masks
    
    # concat order must be consistent with mixing fn
    return tf.concat([images_mixup, images_cutmix], axis=0) 
    
  elif do_mixup:
    return images * mixup_weights + images[::-1] * (1. - mixup_weights)

  elif do_cutmix:
    return images * (1. - cutmix_masks) + images[::-1] * cutmix_masks
  
  else:
    return images

  
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
  cutmix_alpha=0.0,
  mixup_alpha=0.0,
  defer_img_mixing=True,
  hvd_size=None,
  disable_map_parallelization=False
  ):
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
    self.cutmix_alpha = cutmix_alpha
    self.defer_img_mixing = defer_img_mixing
    self.disable_map_parallelization = disable_map_parallelization
    self._num_gpus = hvd.size() if not hvd_size else hvd_size
    

    if self._augmenter_name is not None:
      augmenter = AUGMENTERS.get(self._augmenter_name, None)
      params = augmenter_params or {}
      self._augmenter = augmenter(**params) if augmenter is not None else None
    else:
      self._augmenter = None
  
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

  def set_shapes(self, batch_size, features, labels):
    """Statically set the batch_size dimension."""
    features['image'].set_shape(features['image'].get_shape().merge_with(
        tf.TensorShape([batch_size, None, None, None])))
    labels['label'].set_shape(labels['label'].get_shape().merge_with(
        tf.TensorShape([batch_size, None])))
    return features, labels
  
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
    # This can help resolve OOM issues when using only 1 GPU for training
    options = tf.data.Options()
    options.experimental_optimization.map_parallelization = (not self.disable_map_parallelization)
    dataset = dataset.with_options(options)
    
    if self._num_gpus > 1:
      # For multi-host training, we want each hosts to always process the same
      # subset of files.  Each host only sees a subset of the entire dataset,
      # allowing us to cache larger datasets in memory.
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

    # apply Mixup/CutMix only during training, if requested in the data pipeline,
    # otherwise they will be applied in the model module on device
    mixup_alpha = self.mixup_alpha if self.is_training else 0.0
    cutmix_alpha = self.cutmix_alpha if self.is_training else 0.0
    dataset = dataset.map(
        functools.partial(mixing, self.local_batch_size, mixup_alpha, cutmix_alpha, self.defer_img_mixing),
        num_parallel_calls=64)
    

    # Assign static batch size dimension
    # dataset = dataset.map(
    #     functools.partial(self.set_shapes, batch_size),
    #     num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    
    
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

    # populate features and labels dict
    features = dict()
    labels = dict()
    features['image'] = image
    features['is_tr_split'] = self.is_training
    if self.cutmix_alpha:
      features['cutmix_mask'] = cutmix_mask(self.cutmix_alpha, self._image_size, self._image_size)
    else:
      features['cutmix_mask'] = tf.zeros((self._image_size, self._image_size,1))
    labels['label'] = label  
    return features, labels

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

  # @classmethod
  # def from_params(cls, *args, **kwargs):
  #   """Construct a dataset builder from a default config and any overrides."""
  #   config = DatasetConfig.from_args(*args, **kwargs)
  #   return cls(config)



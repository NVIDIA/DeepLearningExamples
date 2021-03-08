#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def nearest_upsampling(data, scale):
  """Nearest neighbor upsampling implementation.

  Args:
    data: A tensor with a shape of [batch, height_in, width_in, channels].
    scale: An integer multiple to scale resolution of input data.

  Returns:
    data_up: A tensor with a shape of
      [batch, height_in*scale, width_in*scale, channels]. Same dtype as input
      data.
  """
  with tf.name_scope('nearest_upsampling'):
      bs, h, w, c = tf.unstack(tf.shape(data))

      # Use reshape to quickly upsample the input.
      # The nearest pixel is selected implicitly via broadcasting.
      # data = tf.reshape(data, [bs, h, 1, w, 1, c]) * tf.ones([1, 1, scale, 1, scale, 1], dtype=data.dtype)

      # Instead of broadcasting with a 6-d tensor, we're using stacking here
      # for TfLite compatibity.
      output = tf.stack([data] * scale, axis=3)
      output = tf.stack([output] * scale, axis=2)
      return tf.reshape(output, [bs, h * scale, w * scale, c])

  return tf.reshape(data, [bs, h * scale, w * scale, c])


def selective_crop_and_resize(features,
                              boxes,
                              box_levels,
                              boundaries,
                              output_size=7,
                              is_gpu_inference=False):
  """Crop and resize boxes on a set of feature maps.

  Given multiple features maps indexed by different levels, and a set of boxes
  where each box is mapped to a certain level, it selectively crops and resizes
  boxes from the corresponding feature maps to generate the box features.

  We follow the ROIAlign technique (see https://arxiv.org/pdf/1703.06870.pdf,
  figure 3 for reference). Specifically, for each feature map, we select an
  (output_size, output_size) set of pixels corresponding to the box location,
  and then use bilinear interpolation to select the feature value for each
  pixel.

  For performance, we perform the gather and interpolation on all layers as a
  single operation. This is op the multi-level features are first stacked and
  gathered into [2*output_size, 2*output_size] feature points. Then bilinear
  interpolation is performed on the gathered feature points to generate
  [output_size, output_size] RoIAlign feature map.

  Here is the step-by-step algorithm:
    1. The multi-level features are gathered into a
       [batch_size, num_boxes, output_size*2, output_size*2, num_filters]
       Tensor. The Tensor contains four neighboring feature points for each
       vertice in the output grid.
    2. Compute the interpolation kernel of shape
       [batch_size, num_boxes, output_size*2, output_size*2]. The last 2 axis
       can be seen as stacking 2x2 interpolation kernels for all vertices in the
       output grid.
    3. Element-wise multiply the gathered features and interpolation kernel.
       Then apply 2x2 average pooling to reduce spatial dimension to
       output_size.

  Args:
    features: a 5-D tensor of shape
      [batch_size, num_levels, max_height, max_width, num_filters] where
      cropping and resizing are based.
    boxes: a 3-D tensor of shape [batch_size, num_boxes, 4] encoding the
      information of each box w.r.t. the corresponding feature map.
      boxes[:, :, 0:2] are the grid position in (y, x) (float) of the top-left
      corner of each box. boxes[:, :, 2:4] are the box sizes in (h, w) (float)
        in terms of the number of pixels of the corresponding feature map size.
    box_levels: a 3-D tensor of shape [batch_size, num_boxes, 1] representing
      the 0-based corresponding feature level index of each box.
    boundaries: a 3-D tensor of shape [batch_size, num_boxes, 2] representing
      the boundary (in (y, x)) of the corresponding feature map for each box.
      Any resampled grid points that go beyond the bounary will be clipped.
    output_size: a scalar indicating the output crop size.
    is_gpu_inference: whether to build the model for GPU inference.

  Returns:
    features_per_box: a 5-D tensor of shape
      [batch_size, num_boxes, output_size, output_size, num_filters]
      representing the cropped features.
  """
  (batch_size, num_levels, max_feature_height, max_feature_width,
   num_filters) = features.get_shape().as_list()
  _, num_boxes, _ = boxes.get_shape().as_list()

  # Compute the grid position w.r.t. the corresponding feature map.
  box_grid_x = []
  box_grid_y = []
  for i in range(output_size):
    box_grid_x.append(boxes[:, :, 1:2] +
                      (i + 0.5) * boxes[:, :, 3:4] / output_size)
    box_grid_y.append(boxes[:, :, 0:1] +
                      (i + 0.5) * boxes[:, :, 2:3] / output_size)
  box_grid_x = tf.concat(box_grid_x, axis=-1)
  box_grid_y = tf.concat(box_grid_y, axis=-1)

  # Compute indices for gather operation.
  box_grid_y0 = tf.floor(box_grid_y)
  box_grid_x0 = tf.floor(box_grid_x)
  box_grid_x0 = tf.maximum(0., box_grid_x0)
  box_grid_y0 = tf.maximum(0., box_grid_y0)
  box_gridx0x1 = tf.stack([
      tf.minimum(box_grid_x0, boundaries[:, :, 1:2]),
      tf.minimum(box_grid_x0 + 1, boundaries[:, :, 1:2])
  ],
                          axis=3)
  box_gridy0y1 = tf.stack([
      tf.minimum(box_grid_y0, boundaries[:, :, 0:1]),
      tf.minimum(box_grid_y0 + 1, boundaries[:, :, 0:1])
  ],
                          axis=3)

  x_indices = tf.reshape(box_gridx0x1, [batch_size, num_boxes, output_size * 2])
  y_indices = tf.reshape(box_gridy0y1, [batch_size, num_boxes, output_size * 2])

  # If using GPU for inference, delay the cast until when Gather ops show up
  # since GPU inference supports float point better.
  # TODO(laigd): revisit this when newer versions of GPU libraries is released.
  indices_dtype = tf.float32 if is_gpu_inference else tf.int32

  if not is_gpu_inference:
    x_indices = tf.cast(x_indices, tf.int32)
    y_indices = tf.cast(y_indices, tf.int32)

  height_dim_offset = max_feature_width
  level_dim_offset = max_feature_height * height_dim_offset
  batch_dim_offset = num_levels * level_dim_offset

  batch_dim_indices = (
      tf.reshape(tf.range(batch_size, dtype=indices_dtype) * batch_dim_offset, [batch_size, 1, 1, 1]) *
      tf.ones([1, num_boxes, output_size * 2, output_size * 2], dtype=indices_dtype)
  )

  box_level_indices = (
      tf.reshape(box_levels * level_dim_offset, [batch_size, num_boxes, 1, 1]) *
      tf.ones([1, 1, output_size * 2, output_size * 2], dtype=indices_dtype)
  )

  height_indices = (
      tf.reshape(y_indices * height_dim_offset, [batch_size, num_boxes, output_size * 2, 1]) *
      tf.ones([1, 1, 1, output_size * 2], dtype=indices_dtype)
  )

  width_indices = (
      tf.reshape(x_indices, [batch_size, num_boxes, 1, output_size * 2]) *
      tf.ones([1, 1, output_size * 2, 1], dtype=indices_dtype)
  )

  # TODO(hongjunchoi): Remove the need for temporary variables as
  # temporary variables with

  if True:
      batch_dim_indices = tf.cast(batch_dim_indices, tf.float32)
      box_level_indices = tf.cast(box_level_indices, tf.float32)
      height_indices = tf.cast(height_indices, tf.float32)
      width_indices = tf.cast(width_indices, tf.float32)

      indices = tf.add_n([
          batch_dim_indices,
          box_level_indices,
          height_indices,
          width_indices,
      ])

      indices = tf.cast(indices, tf.int32)

  else:  # TODO: Restore this API int32 dtype will be supported on GPUs.
      indices = tf.add_n([
          batch_dim_indices,
          box_level_indices,
          height_indices,
          width_indices,
      ])

  if batch_size == 1:
    # Special handling for single batch input to make it friendly for GPU
    # inference.
    indices = tf.reshape(indices, [1, -1])

    if is_gpu_inference:
      indices = tf.cast(indices, dtype=tf.int32)

    features = tf.reshape(features, [1, -1, num_filters])
    # Cast should happen at last since GPU has better support for floating point
    # operations.
    features_per_box = tf.gather(features, indices, axis=1)

  else:
    indices = tf.reshape(indices, [-1])

    if is_gpu_inference:
      indices = tf.cast(indices, dtype=tf.int32)

    features = tf.reshape(features, [-1, num_filters])
    features_per_box = tf.gather(features, indices)

  features_per_box = tf.reshape(
      features_per_box,
      [batch_size, num_boxes, output_size * 2, output_size * 2, num_filters]
  )

  # The RoIAlign feature f can be computed by bilinear interpolation of four
  # neighboring feature points f0, f1, f2, and f3.
  # f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
  #                       [f10, f11]]
  # f(y, x) = (hy*hx)f00 + (hy*lx)f01 + (ly*hx)f10 + (lx*ly)f11
  # f(y, x) = w00*f00 + w01*f01 + w10*f10 + w11*f11
  ly = box_grid_y - box_grid_y0
  lx = box_grid_x - box_grid_x0
  hy = 1.0 - ly
  hx = 1.0 - lx
  kernel_x = tf.reshape(tf.stack([hx, lx], axis=3), [batch_size, num_boxes, 1, output_size * 2])
  kernel_y = tf.reshape(tf.stack([hy, ly], axis=3), [batch_size, num_boxes, output_size * 2, 1])

  # Use implicit broadcast to generate the interpolation kernel. The
  # multiplier `4` is for avg pooling.
  interpolation_kernel = kernel_y * kernel_x * 4

  # Interpolate the gathered features with computed interpolation kernels.
  features_per_box *= tf.cast(tf.expand_dims(interpolation_kernel, axis=4), dtype=features_per_box.dtype)
  features_per_box = tf.reshape(
      features_per_box,
      [batch_size * num_boxes, output_size * 2, output_size * 2, num_filters]
  )
  features_per_box = tf.nn.avg_pool2d(features_per_box, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
  features_per_box = tf.reshape(features_per_box, [batch_size, num_boxes, output_size, output_size, num_filters])

  return features_per_box


def multilevel_crop_and_resize(features,
                               boxes,
                               output_size=7,
                               is_gpu_inference=False):
  """Crop and resize on multilevel feature pyramid.

  Generate the (output_size, output_size) set of pixels for each input box
  by first locating the box into the correct feature level, and then cropping
  and resizing it using the correspoding feature map of that level.

  Args:
    features: A dictionary with key as pyramid level and value as features. The
      features are in shape of [batch_size, height_l, width_l, num_filters].
    boxes: A 3-D Tensor of shape [batch_size, num_boxes, 4]. Each row represents
      a box with [y1, x1, y2, x2] in un-normalized coordinates.
    output_size: A scalar to indicate the output crop size.
    is_gpu_inference: whether to build the model for GPU inference.

  Returns:
    A 5-D tensor representing feature crop of shape
    [batch_size, num_boxes, output_size, output_size, num_filters].
  """
  with tf.name_scope('multilevel_crop_and_resize'):
      levels = features.keys()
      min_level = min(levels)
      max_level = max(levels)
      _, max_feature_height, max_feature_width, _ = (
          features[min_level].get_shape().as_list())

      # Stack feature pyramid into a features_all of shape
      # [batch_size, levels, height, width, num_filters].
      features_all = []
      for level in range(min_level, max_level + 1):
        features_all.append(tf.image.pad_to_bounding_box(features[level], 0, 0, max_feature_height, max_feature_width))

      features_all = tf.stack(features_all, axis=1)

      # Assign boxes to the right level.
      box_width = tf.squeeze(boxes[:, :, 3:4] - boxes[:, :, 1:2], axis=-1)
      box_height = tf.squeeze(boxes[:, :, 2:3] - boxes[:, :, 0:1], axis=-1)

      areas_sqrt = tf.sqrt(box_height * box_width)

      levels = tf.math.floordiv(tf.math.log(tf.divide(areas_sqrt, 224.0)), tf.math.log(2.0)) + 4.0

      if not is_gpu_inference:
        levels = tf.cast(levels, dtype=tf.int32)

      # Map levels between [min_level, max_level].
      levels = tf.minimum(
          float(max_level) if is_gpu_inference else max_level,
          tf.maximum(levels, float(min_level) if is_gpu_inference else min_level)
      )

      # Project box location and sizes to corresponding feature levels.
      scale_to_level = tf.cast(
          tf.pow(tf.constant(2.0), levels if is_gpu_inference else tf.cast(levels, tf.float32)),
          dtype=boxes.dtype
      )

      boxes /= tf.expand_dims(scale_to_level, axis=2)

      box_width /= scale_to_level
      box_height /= scale_to_level

      boxes = tf.concat(
          [boxes[:, :, 0:2],
          tf.expand_dims(box_height, -1),
          tf.expand_dims(box_width, -1)],
          axis=-1
      )

      # Map levels to [0, max_level-min_level].
      levels -= min_level
      level_strides = tf.pow([[2.0]], levels if is_gpu_inference else tf.cast(levels, tf.float32))

      boundary = tf.cast(
          tf.concat(
              [
                  tf.expand_dims([[tf.cast(max_feature_height, tf.float32)]] / level_strides - 1, axis=-1),
                  tf.expand_dims([[tf.cast(max_feature_width, tf.float32)]] / level_strides - 1, axis=-1),
              ],
              axis=-1
          ),
          boxes.dtype
      )

  return selective_crop_and_resize(features_all, boxes, levels, boundary, output_size, is_gpu_inference)

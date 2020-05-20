#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Util functions to manipulate boxes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports

import numpy as np
import tensorflow as tf


BBOX_XFORM_CLIP = np.log(1000. / 16.)
NMS_TILE_SIZE = 512


def bbox_overlap(boxes, gt_boxes):
  """Calculates the overlap between proposal and ground truth boxes.

  Some `gt_boxes` may have been padded.  The returned `iou` tensor for these
  boxes will be -1.

  Args:
    boxes: a tensor with a shape of [batch_size, N, 4]. N is the number of
      proposals before groundtruth assignment (e.g., rpn_post_nms_topn). The
      last dimension is the pixel coordinates in [ymin, xmin, ymax, xmax] form.
    gt_boxes: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES, 4]. This
      tensor might have paddings with a negative value.
  Returns:
    iou: a tensor with as a shape of [batch_size, N, MAX_NUM_INSTANCES].
  """

  with tf.name_scope('bbox_overlap'):

      bb_y_min, bb_x_min, bb_y_max, bb_x_max = tf.split(value=boxes, num_or_size_splits=4, axis=2)
      gt_y_min, gt_x_min, gt_y_max, gt_x_max = tf.split(value=gt_boxes, num_or_size_splits=4, axis=2)

      # Calculates the intersection area.
      i_xmin = tf.maximum(bb_x_min, tf.transpose(a=gt_x_min, perm=[0, 2, 1]))
      i_xmax = tf.minimum(bb_x_max, tf.transpose(a=gt_x_max, perm=[0, 2, 1]))
      i_ymin = tf.maximum(bb_y_min, tf.transpose(a=gt_y_min, perm=[0, 2, 1]))
      i_ymax = tf.minimum(bb_y_max, tf.transpose(a=gt_y_max, perm=[0, 2, 1]))
      i_area = tf.maximum((i_xmax - i_xmin), 0) * tf.maximum((i_ymax - i_ymin), 0)

      # Calculates the union area.
      bb_area = (bb_y_max - bb_y_min) * (bb_x_max - bb_x_min)
      gt_area = (gt_y_max - gt_y_min) * (gt_x_max - gt_x_min)
      # Adds a small epsilon to avoid divide-by-zero.
      u_area = bb_area + tf.transpose(a=gt_area, perm=[0, 2, 1]) - i_area + 1e-8

      # Calculates IoU.
      iou = i_area / u_area

      # Fills -1 for padded ground truth boxes.
      padding_mask = tf.less(i_xmin, tf.zeros_like(i_xmin))
      iou = tf.where(padding_mask, -tf.ones_like(iou), iou)

  return iou


def top_k(scores, k, boxes_list):
  """A wrapper that returns top-k scores and correponding boxes.

  This functions selects the top-k scores and boxes as follows.

  indices = argsort(scores)[:k]
  scores = scores[indices]
  outputs = []
  for boxes in boxes_list:
    outputs.append(boxes[indices, :])
  return scores, outputs

  Args:
    scores: a tensor with a shape of [batch_size, N]. N is the number of scores.
    k: an integer for selecting the top-k elements.
    boxes_list: a list containing at least one element. Each element has a shape
      of [batch_size, N, 4].
  Returns:
    scores: the selected top-k scores with a shape of [batch_size, k].
    outputs: the list containing the corresponding boxes in the order of the
      input `boxes_list`.
  """
  assert isinstance(boxes_list, list)
  assert boxes_list  # not empty list

  batch_size, _ = scores.get_shape().as_list()

  scores, top_k_indices = tf.nn.top_k(scores, k=k)
  outputs = []
  for boxes in boxes_list:
    if batch_size == 1:
      boxes = tf.squeeze(tf.gather(boxes, top_k_indices, axis=1), axis=1)
    else:
      boxes_index_offsets = tf.range(batch_size) * tf.shape(input=boxes)[1]
      boxes_indices = tf.reshape(
          top_k_indices + tf.expand_dims(boxes_index_offsets, 1), [-1])
      boxes = tf.reshape(
          tf.gather(tf.reshape(boxes, [-1, 4]), boxes_indices),
          [batch_size, -1, 4])
    outputs.append(boxes)
  return scores, outputs


def _self_suppression(iou, _, iou_sum):
  batch_size = tf.shape(input=iou)[0]
  can_suppress_others = tf.cast(
      tf.reshape(tf.reduce_max(input_tensor=iou, axis=1) <= 0.5, [batch_size, -1, 1]), iou.dtype)
  iou_suppressed = tf.reshape(
      tf.cast(tf.reduce_max(input_tensor=can_suppress_others * iou, axis=1) <= 0.5, iou.dtype),
      [batch_size, -1, 1]) * iou
  iou_sum_new = tf.reduce_sum(input_tensor=iou_suppressed, axis=[1, 2])
  return [
      iou_suppressed,
      tf.reduce_any(input_tensor=iou_sum - iou_sum_new > 0.5), iou_sum_new
  ]


def _cross_suppression(boxes, box_slice, iou_threshold, inner_idx):
  batch_size = tf.shape(input=boxes)[0]
  new_slice = tf.slice(boxes, [0, inner_idx * NMS_TILE_SIZE, 0],
                       [batch_size, NMS_TILE_SIZE, 4])
  iou = bbox_overlap(new_slice, box_slice)
  ret_slice = tf.expand_dims(
      tf.cast(tf.reduce_all(input_tensor=iou < iou_threshold, axis=[1]), box_slice.dtype),
      2) * box_slice
  return boxes, ret_slice, iou_threshold, inner_idx + 1


def _suppression_loop_body(boxes, iou_threshold, output_size, idx):
  """Process boxes in the range [idx*NMS_TILE_SIZE, (idx+1)*NMS_TILE_SIZE).

  Args:
    boxes: a tensor with a shape of [batch_size, anchors, 4].
    iou_threshold: a float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    output_size: an int32 tensor of size [batch_size]. Representing the number
      of selected boxes for each batch.
    idx: an integer scalar representing induction variable.

  Returns:
    boxes: updated boxes.
    iou_threshold: pass down iou_threshold to the next iteration.
    output_size: the updated output_size.
    idx: the updated induction variable.
  """
  num_tiles = tf.shape(input=boxes)[1] // NMS_TILE_SIZE
  batch_size = tf.shape(input=boxes)[0]

  # Iterates over tiles that can possibly suppress the current tile.
  box_slice = tf.slice(boxes, [0, idx * NMS_TILE_SIZE, 0],
                       [batch_size, NMS_TILE_SIZE, 4])
  _, box_slice, _, _ = tf.while_loop(
      cond=lambda _boxes, _box_slice, _threshold, inner_idx: inner_idx < idx,
      body=_cross_suppression, loop_vars=[boxes, box_slice, iou_threshold,
                           tf.constant(0)])

  # Iterates over the current tile to compute self-suppression.
  iou = bbox_overlap(box_slice, box_slice)
  mask = tf.expand_dims(
      tf.reshape(tf.range(NMS_TILE_SIZE), [1, -1]) > tf.reshape(
          tf.range(NMS_TILE_SIZE), [-1, 1]), 0)
  iou *= tf.cast(tf.logical_and(mask, iou >= iou_threshold), iou.dtype)
  suppressed_iou, _, _ = tf.while_loop(
      cond=lambda _iou, loop_condition, _iou_sum: loop_condition, body=_self_suppression,
      loop_vars=[iou, tf.constant(True),
       tf.reduce_sum(input_tensor=iou, axis=[1, 2])])
  suppressed_box = tf.reduce_sum(input_tensor=suppressed_iou, axis=1) > 0
  box_slice *= tf.expand_dims(1.0 - tf.cast(suppressed_box, box_slice.dtype), 2)

  # Uses box_slice to update the input boxes.
  mask = tf.reshape(
      tf.cast(tf.equal(tf.range(num_tiles), idx), boxes.dtype), [1, -1, 1, 1])
  boxes = tf.tile(tf.expand_dims(
      box_slice, [1]), [1, num_tiles, 1, 1]) * mask + tf.reshape(
          boxes, [batch_size, num_tiles, NMS_TILE_SIZE, 4]) * (1 - mask)
  boxes = tf.reshape(boxes, [batch_size, -1, 4])

  # Updates output_size.
  output_size += tf.reduce_sum(
      input_tensor=tf.cast(tf.reduce_any(input_tensor=box_slice > 0, axis=[2]), tf.int32), axis=[1])
  return boxes, iou_threshold, output_size, idx + 1


def sorted_non_max_suppression_padded(scores,
                                      boxes,
                                      max_output_size,
                                      iou_threshold):
  """A wrapper that handles non-maximum suppression.

  Assumption:
    * The boxes are sorted by scores unless the box is a dot (all coordinates
      are zero).
    * Boxes with higher scores can be used to suppress boxes with lower scores.

  The overal design of the algorithm is to handle boxes tile-by-tile:

  boxes = boxes.pad_to_multiply_of(tile_size)
  num_tiles = len(boxes) // tile_size
  output_boxes = []
  for i in range(num_tiles):
    box_tile = boxes[i*tile_size : (i+1)*tile_size]
    for j in range(i - 1):
      suppressing_tile = boxes[j*tile_size : (j+1)*tile_size]
      iou = bbox_overlap(box_tile, suppressing_tile)
      # if the box is suppressed in iou, clear it to a dot
      box_tile *= _update_boxes(iou)
    # Iteratively handle the diagnal tile.
    iou = _box_overlap(box_tile, box_tile)
    iou_changed = True
    while iou_changed:
      # boxes that are not suppressed by anything else
      suppressing_boxes = _get_suppressing_boxes(iou)
      # boxes that are suppressed by suppressing_boxes
      suppressed_boxes = _get_suppressed_boxes(iou, suppressing_boxes)
      # clear iou to 0 for boxes that are suppressed, as they cannot be used
      # to suppress other boxes any more
      new_iou = _clear_iou(iou, suppressed_boxes)
      iou_changed = (new_iou != iou)
      iou = new_iou
    # remaining boxes that can still suppress others, are selected boxes.
    output_boxes.append(_get_suppressing_boxes(iou))
    if len(output_boxes) >= max_output_size:
      break

  Args:
    scores: a tensor with a shape of [batch_size, anchors].
    boxes: a tensor with a shape of [batch_size, anchors, 4].
    max_output_size: a scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non max suppression.
    iou_threshold: a float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.

  Returns:
    nms_scores: a tensor with a shape of [batch_size, anchors]. It has same
      dtype as input scores.
    nms_proposals: a tensor with a shape of [batch_size, anchors, 4]. It has
      same dtype as input boxes.
  """
  batch_size = tf.shape(input=boxes)[0]
  num_boxes = tf.shape(input=boxes)[1]
  pad = tf.cast(
      tf.math.ceil(tf.cast(num_boxes, tf.float32) / NMS_TILE_SIZE),
      tf.int32) * NMS_TILE_SIZE - num_boxes
  boxes = tf.pad(tensor=tf.cast(boxes, tf.float32), paddings=[[0, 0], [0, pad], [0, 0]])
  scores = tf.pad(tensor=tf.cast(scores, tf.float32), paddings=[[0, 0], [0, pad]])
  num_boxes += pad

  def _loop_cond(unused_boxes, unused_threshold, output_size, idx):
    return tf.logical_and(
        tf.reduce_min(input_tensor=output_size) < max_output_size,
        idx < num_boxes // NMS_TILE_SIZE)

  selected_boxes, _, output_size, _ = tf.while_loop(
      cond=_loop_cond, body=_suppression_loop_body, loop_vars=[
          boxes, iou_threshold,
          tf.zeros([batch_size], tf.int32),
          tf.constant(0)
      ])
  idx = num_boxes - tf.cast(
      tf.nn.top_k(
          tf.cast(tf.reduce_any(input_tensor=selected_boxes > 0, axis=[2]), tf.int32) *
          tf.expand_dims(tf.range(num_boxes, 0, -1), 0), max_output_size)[0],
      tf.int32)
  idx = tf.minimum(idx, num_boxes - 1)
  idx = tf.reshape(
      idx + tf.reshape(tf.range(batch_size) * num_boxes, [-1, 1]), [-1])
  boxes = tf.reshape(
      tf.gather(tf.reshape(boxes, [-1, 4]), idx),
      [batch_size, max_output_size, 4])
  boxes = boxes * tf.cast(
      tf.reshape(tf.range(max_output_size), [1, -1, 1]) < tf.reshape(
          output_size, [-1, 1, 1]), boxes.dtype)
  scores = tf.reshape(
      tf.gather(tf.reshape(scores, [-1, 1]), idx),
      [batch_size, max_output_size])
  scores = scores * tf.cast(
      tf.reshape(tf.range(max_output_size), [1, -1]) < tf.reshape(
          output_size, [-1, 1]), scores.dtype)
  return scores, boxes


def encode_boxes(boxes, anchors, weights=None):
  """Encode boxes to targets.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    anchors: a tensor whose shape is the same as `boxes` representing the
      coordinates of anchors in ymin, xmin, ymax, xmax order.
    weights: None or a list of four float numbers used to scale coordinates.

  Returns:
    encoded_boxes: a tensor whose shape is the same as `boxes` representing the
      encoded box targets.
  """
  with tf.name_scope('encode_box'):
      boxes = tf.cast(boxes, dtype=anchors.dtype)

      y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=-1)

      # y_min = boxes[..., 0:1]
      # x_min = boxes[..., 1:2]
      # y_max = boxes[..., 2:3]
      # x_max = boxes[..., 3:4]

      box_h = y_max - y_min + 1.0
      box_w = x_max - x_min + 1.0
      box_yc = y_min + 0.5 * box_h
      box_xc = x_min + 0.5 * box_w

      anchor_ymin, anchor_xmin, anchor_ymax, anchor_xmax = tf.split(anchors, 4, axis=-1)

      # anchor_ymin = anchors[..., 0:1]
      # anchor_xmin = anchors[..., 1:2]
      # anchor_ymax = anchors[..., 2:3]
      # anchor_xmax = anchors[..., 3:4]

      anchor_h = anchor_ymax - anchor_ymin + 1.0
      anchor_w = anchor_xmax - anchor_xmin + 1.0
      anchor_yc = anchor_ymin + 0.5 * anchor_h
      anchor_xc = anchor_xmin + 0.5 * anchor_w

      encoded_dy = (box_yc - anchor_yc) / anchor_h
      encoded_dx = (box_xc - anchor_xc) / anchor_w
      encoded_dh = tf.math.log(box_h / anchor_h)
      encoded_dw = tf.math.log(box_w / anchor_w)

      if weights:
        encoded_dy *= weights[0]
        encoded_dx *= weights[1]
        encoded_dh *= weights[2]
        encoded_dw *= weights[3]

      encoded_boxes = tf.concat([encoded_dy, encoded_dx, encoded_dh, encoded_dw], axis=-1)
  return encoded_boxes


def decode_boxes(encoded_boxes, anchors, weights=None):
  """Decode boxes.

  Args:
    encoded_boxes: a tensor whose last dimension is 4 representing the
      coordinates of encoded boxes in ymin, xmin, ymax, xmax order.
    anchors: a tensor whose shape is the same as `boxes` representing the
      coordinates of anchors in ymin, xmin, ymax, xmax order.
    weights: None or a list of four float numbers used to scale coordinates.

  Returns:
    encoded_boxes: a tensor whose shape is the same as `boxes` representing the
      decoded box targets.
  """
  with tf.name_scope('decode_box'):

      encoded_boxes = tf.cast(encoded_boxes, dtype=anchors.dtype)

      dy, dx, dh, dw = tf.split(encoded_boxes, 4, axis=-1)

      # dy = encoded_boxes[..., 0:1]
      # dx = encoded_boxes[..., 1:2]
      # dh = encoded_boxes[..., 2:3]
      # dw = encoded_boxes[..., 3:4]

      if weights:
        dy /= weights[0]
        dx /= weights[1]
        dh /= weights[2]
        dw /= weights[3]

      dh = tf.minimum(dh, BBOX_XFORM_CLIP)
      dw = tf.minimum(dw, BBOX_XFORM_CLIP)

      anchor_ymin, anchor_xmin, anchor_ymax, anchor_xmax = tf.split(anchors, 4, axis=-1)

      # anchor_ymin = anchors[..., 0:1]
      # anchor_xmin = anchors[..., 1:2]
      # anchor_ymax = anchors[..., 2:3]
      # anchor_xmax = anchors[..., 3:4]

      anchor_h = anchor_ymax - anchor_ymin + 1.0
      anchor_w = anchor_xmax - anchor_xmin + 1.0
      anchor_yc = anchor_ymin + 0.5 * anchor_h
      anchor_xc = anchor_xmin + 0.5 * anchor_w

      decoded_boxes_yc = dy * anchor_h + anchor_yc
      decoded_boxes_xc = dx * anchor_w + anchor_xc
      decoded_boxes_h = tf.exp(dh) * anchor_h
      decoded_boxes_w = tf.exp(dw) * anchor_w

      decoded_boxes_ymin = decoded_boxes_yc - 0.5 * decoded_boxes_h
      decoded_boxes_xmin = decoded_boxes_xc - 0.5 * decoded_boxes_w
      decoded_boxes_ymax = decoded_boxes_ymin + decoded_boxes_h - 1.0
      decoded_boxes_xmax = decoded_boxes_xmin + decoded_boxes_w - 1.0

      decoded_boxes = tf.concat(
          [decoded_boxes_ymin, decoded_boxes_xmin, decoded_boxes_ymax, decoded_boxes_xmax],
          axis=-1
      )

  return decoded_boxes


def clip_boxes(boxes, height, width):
  """Clip boxes.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    height: an integer, a scalar or a tensor such as all but the last dimensions
      are the same as `boxes`. The last dimension is 1. It represents the height
      of the image.
    width: an integer, a scalar or a tensor such as all but the last dimensions
      are the same as `boxes`. The last dimension is 1. It represents the width
      of the image.

  Returns:
    clipped_boxes: a tensor whose shape is the same as `boxes` representing the
      clipped boxes.
  """
  with tf.name_scope('clip_box'):
      y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=-1)

      # y_min = boxes[..., 0:1]
      # x_min = boxes[..., 1:2]
      # y_max = boxes[..., 2:3]
      # x_max = boxes[..., 3:4]

      height = tf.cast(height, dtype=boxes.dtype)
      width = tf.cast(width, dtype=boxes.dtype)

      clipped_y_min = tf.maximum(tf.minimum(y_min, height - 1.0), 0.0)
      clipped_y_max = tf.maximum(tf.minimum(y_max, height - 1.0), 0.0)
      clipped_x_min = tf.maximum(tf.minimum(x_min, width - 1.0), 0.0)
      clipped_x_max = tf.maximum(tf.minimum(x_max, width - 1.0), 0.0)

      clipped_boxes = tf.concat([clipped_y_min, clipped_x_min, clipped_y_max, clipped_x_max], axis=-1)

  return clipped_boxes


def filter_boxes(boxes, scores, min_size, height, width, scale):
  """Filter out boxes that are too small.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    scores: a tensor such as all but the last dimensions are the same as
      `boxes`. The last dimension is 1. It represents the scores.
    min_size: an integer specifying the minimal size.
    height: an integer, a scalar or a tensor such as all but the last dimensions
      are the same as `boxes`. The last dimension is 1. It represents the height
      of the image.
    width: an integer, a scalar or a tensor such as all but the last dimensions
      are the same as `boxes`. The last dimension is 1. It represents the width
      of the image.
    scale: an integer, a scalar or a tensor such as all but the last dimensions
      are the same as `boxes`. The last dimension is 1. It represents the scale
      of the image.

  Returns:
    filtered_boxes: a tensor whose shape is the same as `boxes` representing the
      filtered boxes.
    filtered_scores: a tensor whose shape is the same as `scores` representing
      the filtered scores.
  """
  with tf.name_scope('filter_box'):
      y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=-1)

      # y_min = boxes[..., 0:1]
      # x_min = boxes[..., 1:2]
      # y_max = boxes[..., 2:3]
      # x_max = boxes[..., 3:4]

      h = y_max - y_min + 1.0
      w = x_max - x_min + 1.0
      yc = y_min + h / 2.0
      xc = x_min + w / 2.0

      height = tf.cast(height, dtype=boxes.dtype)
      width = tf.cast(width, dtype=boxes.dtype)
      scale = tf.cast(scale, dtype=boxes.dtype)

      min_size = tf.cast(tf.maximum(min_size, 1), dtype=boxes.dtype)

      size_mask = tf.logical_and(
          tf.greater_equal(h, min_size * scale),
          tf.greater_equal(w, min_size * scale)
      )

      center_mask = tf.logical_and(tf.less(yc, height), tf.less(xc, width))
      selected_mask = tf.logical_and(size_mask, center_mask)

      filtered_scores = tf.where(selected_mask, scores, tf.zeros_like(scores))
      filtered_boxes = tf.cast(selected_mask, dtype=boxes.dtype) * boxes

  return filtered_boxes, filtered_scores


def to_normalized_coordinates(boxes, height, width):
  """Converted absolute box coordinates to normalized ones.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    height: an integer, a scalar or a tensor such as all but the last dimensions
      are the same as `boxes`. The last dimension is 1. It represents the height
      of the image.
    width: an integer, a scalar or a tensor such as all but the last dimensions
      are the same as `boxes`. The last dimension is 1. It represents the width
      of the image.

  Returns:
    normalized_boxes: a tensor whose shape is the same as `boxes` representing
      the boxes in normalized coordinates.
  """
  with tf.name_scope('normalize_box'):
      height = tf.cast(height, dtype=boxes.dtype)
      width = tf.cast(width, dtype=boxes.dtype)

      y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=-1)

      y_min = y_min / height
      x_min = x_min / width
      y_max = y_max / height
      x_max = x_max / width

      # y_min = boxes[..., 0:1] / height
      # x_min = boxes[..., 1:2] / width
      # y_max = boxes[..., 2:3] / height
      # x_max = boxes[..., 3:4] / width

      normalized_boxes = tf.concat([y_min, x_min, y_max, x_max], axis=-1)

  return normalized_boxes


def to_absolute_coordinates(boxes, height, width):
  """Converted normalized box coordinates to absolute ones.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    height: an integer, a scalar or a tensor such as all but the last dimensions
      are the same as `boxes`. The last dimension is 1. It represents the height
      of the image.
    width: an integer, a scalar or a tensor such as all but the last dimensions
      are the same as `boxes`. The last dimension is 1. It represents the width
      of the image.

  Returns:
    absolute_boxes: a tensor whose shape is the same as `boxes` representing the
      boxes in absolute coordinates.
  """
  with tf.name_scope('denormalize_box'):
      height = tf.cast(height, dtype=boxes.dtype)
      width = tf.cast(width, dtype=boxes.dtype)

      y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=-1)

      y_min = y_min * height
      x_min = x_min * width
      y_max = y_max * height
      x_max = x_max * width

      # y_min = boxes[..., 0:1] * height
      # x_min = boxes[..., 1:2] * width
      # y_max = boxes[..., 2:3] * height
      # x_max = boxes[..., 3:4] * width

      absolute_boxes = tf.concat([y_min, x_min, y_max, x_max], axis=-1)

  return absolute_boxes

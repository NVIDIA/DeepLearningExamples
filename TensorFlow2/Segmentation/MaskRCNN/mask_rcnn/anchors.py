#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Mask-RCNN anchor definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import numpy as np
import tensorflow as tf

from mask_rcnn.object_detection import argmax_matcher
from mask_rcnn.object_detection import balanced_positive_negative_sampler
from mask_rcnn.object_detection import box_list
from mask_rcnn.object_detection import faster_rcnn_box_coder
from mask_rcnn.object_detection import region_similarity_calculator
from mask_rcnn.object_detection import target_assigner


def _generate_anchor_configs(min_level, max_level, num_scales, aspect_ratios):
  """Generates mapping from output level to a list of anchor configurations.

  A configuration is a tuple of (num_anchors, scale, aspect_ratio).

  Args:
      min_level: integer number of minimum level of the output feature pyramid.
      max_level: integer number of maximum level of the output feature pyramid.
      num_scales: integer number representing intermediate scales added
        on each level. For instances, num_scales=2 adds two additional
        anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: list of tuples representing the aspect raito anchors added
        on each level. For instances, aspect_ratios =
        [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.
  Returns:
    anchor_configs: a dictionary with keys as the levels of anchors and
      values as a list of anchor configuration.
  """
  anchor_configs = {}
  for level in range(min_level, max_level + 1):
    anchor_configs[level] = []
    for scale_octave in range(num_scales):
      for aspect in aspect_ratios:
        anchor_configs[level].append(
            (2**level, scale_octave / float(num_scales), aspect))
  return anchor_configs


def _generate_anchor_boxes(image_size, anchor_scale, anchor_configs):
  """Generates multiscale anchor boxes.

  Args:
    image_size: integer number of input image size. The input image has the
      same dimension for width and height. The image_size should be divided by
      the largest feature stride 2^max_level.
    anchor_scale: float number representing the scale of size of the base
      anchor to the feature stride 2^level.
    anchor_configs: a dictionary with keys as the levels of anchors and
      values as a list of anchor configuration.
  Returns:
    anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
      feature levels.
  Raises:
    ValueError: input size must be the multiple of largest feature stride.
  """
  boxes_all = []
  for _, configs in anchor_configs.items():
    boxes_level = []
    for config in configs:
      stride, octave_scale, aspect = config
      if image_size[0] % stride != 0 or image_size[1] % stride != 0:
        raise ValueError('input size must be divided by the stride.')
      base_anchor_size = anchor_scale * stride * 2**octave_scale
      anchor_size_x_2 = base_anchor_size * aspect[0] / 2.0
      anchor_size_y_2 = base_anchor_size * aspect[1] / 2.0

      x = np.arange(stride / 2, image_size[1], stride)
      y = np.arange(stride / 2, image_size[0], stride)
      xv, yv = np.meshgrid(x, y)
      xv = xv.reshape(-1)
      yv = yv.reshape(-1)

      boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                         yv + anchor_size_y_2, xv + anchor_size_x_2))
      boxes = np.swapaxes(boxes, 0, 1)
      boxes_level.append(np.expand_dims(boxes, axis=1))
    # concat anchors on the same level to the reshape NxAx4
    boxes_level = np.concatenate(boxes_level, axis=1)
    boxes_all.append(boxes_level.reshape([-1, 4]))

  anchor_boxes = np.vstack(boxes_all)
  return anchor_boxes


class Anchors(object):
  """Mask-RCNN Anchors class."""

  def __init__(self, min_level, max_level, num_scales, aspect_ratios, anchor_scale, image_size):
    """Constructs multiscale Mask-RCNN anchors.

    Args:
      min_level: integer number of minimum level of the output feature pyramid.
      max_level: integer number of maximum level of the output feature pyramid.
      num_scales: integer number representing intermediate scales added
        on each level. For instances, num_scales=2 adds two additional
        anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: list of tuples representing the aspect raito anchors added
        on each level. For instances, aspect_ratios =
        [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.
      anchor_scale: float number representing the scale of size of the base
        anchor to the feature stride 2^level.
      image_size: integer number of input image size. The input image has the
        same dimension for width and height. The image_size should be divided by
        the largest feature stride 2^max_level.
    """
    self.min_level = min_level
    self.max_level = max_level
    self.num_scales = num_scales
    self.aspect_ratios = aspect_ratios
    self.anchor_scale = anchor_scale
    self.image_size = image_size
    self.config = self._generate_configs()
    self.boxes = self._generate_boxes()

  def _generate_configs(self):
    """Generate configurations of anchor boxes."""
    return _generate_anchor_configs(self.min_level, self.max_level,
                                    self.num_scales, self.aspect_ratios)

  def _generate_boxes(self):
    """Generates multiscale anchor boxes."""
    boxes = _generate_anchor_boxes(self.image_size, self.anchor_scale,
                                   self.config)
    boxes = tf.convert_to_tensor(value=boxes, dtype=tf.float32)
    return boxes

  def get_anchors_per_location(self):
    return self.num_scales * len(self.aspect_ratios)

  def get_unpacked_boxes(self):
    return self.unpack_labels(self.boxes)

  def unpack_labels(self, labels):
    """Unpacks an array of labels into multiscales labels."""
    labels_unpacked = OrderedDict()
    count = 0
    for level in range(self.min_level, self.max_level + 1):
      feat_size0 = int(self.image_size[0] / 2**level)
      feat_size1 = int(self.image_size[1] / 2**level)
      steps = feat_size0 * feat_size1 * self.get_anchors_per_location()
      indices = tf.range(count, count + steps)
      count += steps
      labels_unpacked[level] = tf.reshape(
          tf.gather(labels, indices), [feat_size0, feat_size1, -1])
    return labels_unpacked


class AnchorLabeler(object):
  """Labeler for multiscale anchor boxes."""

  def __init__(self, anchors, num_classes, match_threshold=0.7,
               unmatched_threshold=0.3, rpn_batch_size_per_im=256,
               rpn_fg_fraction=0.5):
    """Constructs anchor labeler to assign labels to anchors.

    Args:
      anchors: an instance of class Anchors.
      num_classes: integer number representing number of classes in the dataset.
      match_threshold: a float number between 0 and 1 representing the
        lower-bound threshold to assign positive labels for anchors. An anchor
        with a score over the threshold is labeled positive.
      unmatched_threshold: a float number between 0 and 1 representing the
        upper-bound threshold to assign negative labels for anchors. An anchor
        with a score below the threshold is labeled negative.
      rpn_batch_size_per_im: a integer number that represents the number of
        sampled anchors per image in the first stage (region proposal network).
      rpn_fg_fraction: a float number between 0 and 1 representing the fraction
        of positive anchors (foreground) in the first stage.
    """
    similarity_calc = region_similarity_calculator.IouSimilarity()
    matcher = argmax_matcher.ArgMaxMatcher(
        match_threshold,
        unmatched_threshold=unmatched_threshold,
        negatives_lower_than_unmatched=True,
        force_match_for_each_row=True)
    box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()

    self._target_assigner = target_assigner.TargetAssigner(
        similarity_calc, matcher, box_coder)
    self._anchors = anchors
    self._match_threshold = match_threshold
    self._unmatched_threshold = unmatched_threshold
    self._rpn_batch_size_per_im = rpn_batch_size_per_im
    self._rpn_fg_fraction = rpn_fg_fraction
    self._num_classes = num_classes

  def _get_rpn_samples(self, match_results):
    """Computes anchor labels.

    This function performs subsampling for foreground (fg) and background (bg)
    anchors.
    Args:
      match_results: A integer tensor with shape [N] representing the
        matching results of anchors. (1) match_results[i]>=0,
        meaning that column i is matched with row match_results[i].
        (2) match_results[i]=-1, meaning that column i is not matched.
        (3) match_results[i]=-2, meaning that column i is ignored.
    Returns:
      score_targets: a integer tensor with the a shape of [N].
        (1) score_targets[i]=1, the anchor is a positive sample.
        (2) score_targets[i]=0, negative. (3) score_targets[i]=-1, the anchor is
        don't care (ignore).
    """
    sampler = (
        balanced_positive_negative_sampler.BalancedPositiveNegativeSampler(
            positive_fraction=self._rpn_fg_fraction, is_static=False))
    # indicator includes both positive and negative labels.
    # labels includes only positives labels.
    # positives = indicator & labels.
    # negatives = indicator & !labels.
    # ignore = !indicator.
    indicator = tf.greater(match_results, -2)
    labels = tf.greater(match_results, -1)

    samples = sampler.subsample(
        indicator, self._rpn_batch_size_per_im, labels)
    positive_labels = tf.where(
        tf.logical_and(samples, labels),
        tf.constant(2, dtype=tf.int32, shape=match_results.shape),
        tf.constant(0, dtype=tf.int32, shape=match_results.shape))
    negative_labels = tf.where(
        tf.logical_and(samples, tf.logical_not(labels)),
        tf.constant(1, dtype=tf.int32, shape=match_results.shape),
        tf.constant(0, dtype=tf.int32, shape=match_results.shape))
    ignore_labels = tf.fill(match_results.shape, -1)

    return (ignore_labels + positive_labels + negative_labels,
            positive_labels, negative_labels)

  def label_anchors(self, gt_boxes, gt_labels):
    """Labels anchors with ground truth inputs.

    Args:
      gt_boxes: A float tensor with shape [N, 4] representing groundtruth boxes.
        For each row, it stores [y0, x0, y1, x1] for four corners of a box.
      gt_labels: A integer tensor with shape [N, 1] representing groundtruth
        classes.
    Returns:
      score_targets_dict: ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors]. The height_l and width_l
        represent the dimension of class logits at l-th level.
      box_targets_dict: ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors * 4]. The height_l and
        width_l represent the dimension of bounding box regression output at
        l-th level.
    """
    gt_box_list = box_list.BoxList(gt_boxes)
    anchor_box_list = box_list.BoxList(self._anchors.boxes)

    # cls_targets, cls_weights, box_weights are not used
    _, _, box_targets, _, matches = self._target_assigner.assign(
        anchor_box_list, gt_box_list, gt_labels)

    # score_targets contains the subsampled positive and negative anchors.
    score_targets, _, _ = self._get_rpn_samples(matches.match_results)

    # Unpack labels.
    score_targets_dict = self._anchors.unpack_labels(score_targets)
    box_targets_dict = self._anchors.unpack_labels(box_targets)

    return score_targets_dict, box_targets_dict

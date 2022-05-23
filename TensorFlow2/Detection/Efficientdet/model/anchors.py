# Copyright 2020 Google Research. All Rights Reserved.
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
"""Anchor definition."""
import collections
import numpy as np
import tensorflow as tf

from utils import model_utils
from object_detection import argmax_matcher
from object_detection import box_list
from object_detection import faster_rcnn_box_coder
from object_detection import region_similarity_calculator
from object_detection import target_assigner

MAX_DETECTION_POINTS = 5000


def decode_box_outputs(pred_boxes, anchor_boxes):
  """Transforms relative regression coordinates to absolute positions.

  Network predictions are normalized and relative to a given anchor; this
  reverses the transformation and outputs absolute coordinates for the input
  image.

  Args:
    pred_boxes: predicted box regression targets.
    anchor_boxes: anchors on all feature levels.
  Returns:
    outputs: bounding boxes.
  """
  anchor_boxes = tf.cast(anchor_boxes, pred_boxes.dtype)
  ycenter_a = (anchor_boxes[..., 0] + anchor_boxes[..., 2]) / 2
  xcenter_a = (anchor_boxes[..., 1] + anchor_boxes[..., 3]) / 2
  ha = anchor_boxes[..., 2] - anchor_boxes[..., 0]
  wa = anchor_boxes[..., 3] - anchor_boxes[..., 1]
  ty, tx, th, tw = tf.unstack(pred_boxes, num=4, axis=-1)

  w = tf.math.exp(tw) * wa
  h = tf.math.exp(th) * ha
  ycenter = ty * ha + ycenter_a
  xcenter = tx * wa + xcenter_a
  ymin = ycenter - h / 2.
  xmin = xcenter - w / 2.
  ymax = ycenter + h / 2.
  xmax = xcenter + w / 2.
  return tf.stack([ymin, xmin, ymax, xmax], axis=-1)


def _generate_anchor_configs(feat_sizes, min_level, max_level, num_scales,
                             aspect_ratios):
  """Generates mapping from output level to a list of anchor configurations.

  A configuration is a tuple of (num_anchors, scale, aspect_ratio).

  Args:
      feat_sizes: list of dict of integer numbers of feature map sizes.
      min_level: integer number of minimum level of the output feature pyramid.
      max_level: integer number of maximum level of the output feature pyramid.
      num_scales: integer number representing intermediate scales added
        on each level. For instances, num_scales=2 adds two additional
        anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: list of tuples representing the aspect ratio anchors added
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
            ((feat_sizes[0]['height'] / float(feat_sizes[level]['height']),
              feat_sizes[0]['width'] / float(feat_sizes[level]['width'])),
             scale_octave / float(num_scales), aspect))
  return anchor_configs


def _generate_anchor_boxes(image_size, anchor_scale, anchor_configs):
  """Generates multiscale anchor boxes.

  Args:
    image_size: tuple of integer numbers of input image size.
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
      base_anchor_size_x = anchor_scale * stride[1] * 2**octave_scale
      base_anchor_size_y = anchor_scale * stride[0] * 2**octave_scale
      anchor_size_x_2 = base_anchor_size_x * aspect[0] / 2.0
      anchor_size_y_2 = base_anchor_size_y * aspect[1] / 2.0

      x = np.arange(stride[1] / 2, image_size[1], stride[1])
      y = np.arange(stride[0] / 2, image_size[0], stride[0])
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
  """RetinaNet Anchors class."""

  def __init__(self, min_level, max_level, num_scales, aspect_ratios,
               anchor_scale, image_size):
    """Constructs multiscale RetinaNet anchors.

    Args:
      min_level: integer number of minimum level of the output feature pyramid.
      max_level: integer number of maximum level of the output feature pyramid.
      num_scales: integer number representing intermediate scales added
        on each level. For instances, num_scales=2 adds two additional
        anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: list of tuples representing the aspect ratio anchors added
        on each level. For instances, aspect_ratios =
        [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.
      anchor_scale: float number representing the scale of size of the base
        anchor to the feature stride 2^level.
      image_size: integer number or tuple of integer number of input image size.
    """
    self.min_level = min_level
    self.max_level = max_level
    self.num_scales = num_scales
    self.aspect_ratios = aspect_ratios
    self.anchor_scale = anchor_scale
    self.image_size = model_utils.parse_image_size(image_size)
    self.feat_sizes = model_utils.get_feat_sizes(image_size, max_level)
    self.config = self._generate_configs()
    self.boxes = self._generate_boxes()

  def _generate_configs(self):
    """Generate configurations of anchor boxes."""
    return _generate_anchor_configs(self.feat_sizes, self.min_level,
                                    self.max_level, self.num_scales,
                                    self.aspect_ratios)

  def _generate_boxes(self):
    """Generates multiscale anchor boxes."""
    boxes = _generate_anchor_boxes(self.image_size, self.anchor_scale,
                                   self.config)
    boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
    return boxes

  def get_anchors_per_location(self):
    return self.num_scales * len(self.aspect_ratios)


class AnchorLabeler(object):
  """Labeler for multiscale anchor boxes."""

  def __init__(self, anchors, num_classes, match_threshold=0.5):
    """Constructs anchor labeler to assign labels to anchors.

    Args:
      anchors: an instance of class Anchors.
      num_classes: integer number representing number of classes in the dataset.
      match_threshold: float number between 0 and 1 representing the threshold
        to assign positive labels for anchors.
    """
    similarity_calc = region_similarity_calculator.IouSimilarity()
    matcher = argmax_matcher.ArgMaxMatcher(
        match_threshold,
        unmatched_threshold=match_threshold,
        negatives_lower_than_unmatched=True,
        force_match_for_each_row=True)
    box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()

    self._target_assigner = target_assigner.TargetAssigner(
        similarity_calc, matcher, box_coder)
    self._anchors = anchors
    self._match_threshold = match_threshold
    self._num_classes = num_classes

  def _unpack_labels(self, labels):
    """Unpacks an array of labels into multiscales labels."""
    labels_unpacked = collections.OrderedDict()
    anchors = self._anchors
    count = 0
    for level in range(anchors.min_level, anchors.max_level + 1):
      feat_size = anchors.feat_sizes[level]
      steps = feat_size['height'] * feat_size[
          'width'] * anchors.get_anchors_per_location()
      indices = tf.range(count, count + steps)
      count += steps
      labels_unpacked[level] = tf.reshape(
          tf.gather(labels, indices),
          [feat_size['height'], feat_size['width'], -1])
    return labels_unpacked

  def label_anchors(self, gt_boxes, gt_labels):
    """Labels anchors with ground truth inputs.

    Args:
      gt_boxes: A float tensor with shape [N, 4] representing groundtruth boxes.
        For each row, it stores [y0, x0, y1, x1] for four corners of a box.
      gt_labels: A integer tensor with shape [N, 1] representing groundtruth
        classes.
    Returns:
      cls_targets_dict: ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors]. The height_l and width_l
        represent the dimension of class logits at l-th level.
      box_targets_dict: ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors * 4]. The height_l and
        width_l represent the dimension of bounding box regression output at
        l-th level.
      num_positives: scalar tensor storing number of positives in an image.
    """
    gt_box_list = box_list.BoxList(gt_boxes)
    anchor_box_list = box_list.BoxList(self._anchors.boxes)

    # cls_weights, box_weights are not used
    cls_targets, _, box_targets, _, matches = self._target_assigner.assign(
        anchor_box_list, gt_box_list, gt_labels)

    # class labels start from 1 and the background class = -1
    cls_targets -= 1
    cls_targets = tf.cast(cls_targets, tf.int32)

    # Unpack labels.
    cls_targets_dict = self._unpack_labels(cls_targets)
    box_targets_dict = self._unpack_labels(box_targets)
    num_positives = tf.reduce_sum(
        tf.cast(tf.not_equal(matches.match_results, -1), tf.float32))

    return cls_targets_dict, box_targets_dict, num_positives
